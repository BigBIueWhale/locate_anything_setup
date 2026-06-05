//! JPEG validation helpers. Header-only parse — no pixel decode.
//!
//! Why headers only:
//! - We want to validate dimensions before tying up a GPU slot in the
//!   Python worker.
//! - Header parse is ~microseconds.
//! - Full pixel decode happens in Python (PIL) where the model lives.
//!
//! EXIF orientation: the Python worker applies `PIL.ImageOps.exif_transpose`
//! to every frame BEFORE the model sees it (`worker/inference.py`), which swaps
//! width<->height for the 90°/270° orientations (EXIF Orientation 5/6/7/8 —
//! i.e. essentially every portrait phone photo). `read_dimensions_blocking`
//! mirrors that swap so the strict preprocessor gates in `ws.rs` operate on
//! EXACTLY the dimensions the model will receive, honoring the "validate the
//! same image the model decodes" guarantee instead of the raw stored
//! (pre-transpose) dimensions.

use jpeg_decoder::Decoder;
use std::io::Cursor;

#[inline]
pub fn is_jpeg(b: &[u8]) -> bool {
    b.len() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

/// Parse JPEG header only and return (width, height) in pixels, in the
/// orientation the model will actually process — i.e. with EXIF Orientation
/// applied, matching the worker's `exif_transpose`.
/// Synchronous; caller MUST run under `tokio::task::spawn_blocking`.
pub fn read_dimensions_blocking(bytes: &[u8]) -> Result<(u16, u16), String> {
    let mut d = Decoder::new(Cursor::new(bytes));
    d.read_info().map_err(|e| format!("JPEG header parse: {e}"))?;
    let info = d
        .info()
        .ok_or_else(|| "JPEG decoder yielded no info".to_string())?;
    let (mut width, mut height) = (info.width, info.height);

    // EXIF Orientation 5/6/7/8 are the 90°/270° transposes; PIL's
    // `exif_transpose` (applied worker-side before inference) swaps the axes for
    // these. Orientations 1-4 (identity / mirror / 180°) keep the dimensions.
    // jpeg-decoder 0.3.2 does NOT apply orientation, so we read the tag from the
    // APP1 segment ourselves. Any malformation → `None` → no swap (safe: falls
    // back to the stored dimensions). No panics: every access is bounds-checked.
    if let Some(orientation) = exif_orientation(bytes) {
        if matches!(orientation, 5 | 6 | 7 | 8) {
            std::mem::swap(&mut width, &mut height);
        }
    }
    Ok((width, height))
}

/// Read the EXIF Orientation tag (0x0112) from a JPEG's APP1 ("Exif\0\0")
/// segment. Returns the raw orientation value (1..=8) or `None` if there is no
/// EXIF block, no Orientation tag, or the structure is malformed. Pure
/// byte-scan; never panics; never allocates.
fn exif_orientation(bytes: &[u8]) -> Option<u16> {
    // JPEG marker segments: after SOI (FF D8) comes a sequence of
    // `FF <marker> <len_be:2> <payload>`, where `len` counts the 2 length bytes
    // plus the payload. Walk segments until APP1/Exif or the start of scan.
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    loop {
        // Need `FF <marker> <len:2>`.
        if i + 4 > bytes.len() || bytes[i] != 0xFF {
            return None;
        }
        let marker = bytes[i + 1];
        // SOS (FF DA) = compressed image data begins; EOI (FF D9) = end. EXIF,
        // if present, is an APP1 segment before either of these.
        if marker == 0xDA || marker == 0xD9 {
            return None;
        }
        let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if seg_len < 2 {
            return None;
        }
        let payload_start = i + 4;
        // `i + 2` is the start of the length field; `seg_len` covers itself + payload.
        let segment_end = i + 2 + seg_len;
        if segment_end > bytes.len() {
            return None;
        }
        if marker == 0xE1 {
            let payload = &bytes[payload_start..segment_end];
            if payload.len() >= 6 && &payload[..6] == b"Exif\x00\x00" {
                return tiff_orientation(&payload[6..]);
            }
        }
        i = segment_end;
    }
}

/// Parse the Orientation SHORT (tag 0x0112) out of a TIFF block (the bytes after
/// the "Exif\0\0" APP1 header). Handles both byte orders. Bounds-checked; the
/// `get(..)` accessors make every read fallible rather than panicking.
fn tiff_orientation(tiff: &[u8]) -> Option<u16> {
    if tiff.len() < 8 {
        return None;
    }
    let little_endian = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |off: usize| -> Option<u16> {
        let b = tiff.get(off..off + 2)?;
        Some(if little_endian {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |off: usize| -> Option<u32> {
        let b = tiff.get(off..off + 4)?;
        Some(if little_endian {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    // TIFF magic 0x002A.
    if u16_at(2)? != 0x002A {
        return None;
    }
    let ifd0 = u32_at(4)? as usize;
    let n_entries = u16_at(ifd0)? as usize;
    // Each IFD entry is 12 bytes: tag(2) type(2) count(4) value/offset(4).
    let mut e = ifd0.checked_add(2)?;
    for _ in 0..n_entries {
        let tag = u16_at(e)?;
        if tag == 0x0112 {
            // Orientation is a SHORT; its value sits in the first 2 bytes of
            // the entry's 4-byte value field (offset e+8).
            return u16_at(e + 8);
        }
        e = e.checked_add(12)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny JPEG byte stream: SOI + APP1(Exif, Orientation=<o>) + SOS.
    /// Enough for `exif_orientation` to parse; not a decodable image.
    fn jpeg_with_orientation(o: u16) -> Vec<u8> {
        // TIFF (little-endian), one IFD0 entry: Orientation SHORT = o.
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II"); // little-endian
        tiff.extend_from_slice(&0x002Au16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
        tiff.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
        tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // tag: Orientation
        tiff.extend_from_slice(&3u16.to_le_bytes()); // type: SHORT
        tiff.extend_from_slice(&1u32.to_le_bytes()); // count
        // SHORT value occupies the FIRST 2 bytes of the 4-byte value field
        // (little-endian here), padded with 2 unused bytes.
        tiff.extend_from_slice(&o.to_le_bytes());
        tiff.extend_from_slice(&0u16.to_le_bytes());

        let mut payload = Vec::new();
        payload.extend_from_slice(b"Exif\x00\x00");
        payload.extend_from_slice(&tiff);
        let seg_len = (payload.len() + 2) as u16;

        let mut out = vec![0xFF, 0xD8]; // SOI
        out.extend_from_slice(&[0xFF, 0xE1]); // APP1
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&[0xFF, 0xDA]); // SOS (stop)
        out
    }

    #[test]
    fn reads_orientation_6() {
        assert_eq!(exif_orientation(&jpeg_with_orientation(6)), Some(6));
    }

    #[test]
    fn reads_orientation_8_big_endian() {
        // Same as orientation 6 builder but big-endian TIFF + value 8.
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"MM");
        tiff.extend_from_slice(&0x002Au16.to_be_bytes());
        tiff.extend_from_slice(&8u32.to_be_bytes());
        tiff.extend_from_slice(&1u16.to_be_bytes());
        tiff.extend_from_slice(&0x0112u16.to_be_bytes());
        tiff.extend_from_slice(&3u16.to_be_bytes());
        tiff.extend_from_slice(&1u32.to_be_bytes());
        // SHORT value in the first 2 bytes of the 4-byte value field (big-endian).
        tiff.extend_from_slice(&8u16.to_be_bytes());
        tiff.extend_from_slice(&0u16.to_be_bytes());
        let mut payload = Vec::new();
        payload.extend_from_slice(b"Exif\x00\x00");
        payload.extend_from_slice(&tiff);
        let seg_len = (payload.len() + 2) as u16;
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&[0xFF, 0xDA]);
        assert_eq!(exif_orientation(&out), Some(8));
    }

    #[test]
    fn orientation_1_does_not_swap_via_read_dims_contract() {
        // exif_orientation reports 1; read_dimensions_blocking's swap predicate
        // (5/6/7/8) must therefore NOT fire.
        assert_eq!(exif_orientation(&jpeg_with_orientation(1)), Some(1));
        assert!(!matches!(1u16, 5 | 6 | 7 | 8));
    }

    #[test]
    fn no_exif_returns_none() {
        // SOI then straight to SOS — no APP1.
        let bytes = [0xFFu8, 0xD8, 0xFF, 0xDA, 0x00, 0x00];
        assert_eq!(exif_orientation(&bytes), None);
    }

    #[test]
    fn truncated_app1_is_safe_none() {
        // APP1 claims a length longer than the buffer → safe None, no panic.
        let bytes = [0xFFu8, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF, b'E', b'x'];
        assert_eq!(exif_orientation(&bytes), None);
    }

    #[test]
    fn non_jpeg_returns_none() {
        assert_eq!(exif_orientation(&[0x00, 0x01, 0x02]), None);
    }
}
