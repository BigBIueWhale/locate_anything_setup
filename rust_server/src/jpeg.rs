//! JPEG validation helpers. Header-only parse — no pixel decode.
//!
//! Why headers only:
//! - We want to validate dimensions before tying up a GPU slot in the
//!   Python worker.
//! - Header parse is ~microseconds.
//! - Full pixel decode happens in Python (PIL) where the model lives.

use jpeg_decoder::Decoder;
use std::io::Cursor;

#[inline]
pub fn is_jpeg(b: &[u8]) -> bool {
    b.len() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

/// Parse JPEG header only and return (width, height) in pixels.
/// Synchronous; caller MUST run under `tokio::task::spawn_blocking`.
pub fn read_dimensions_blocking(bytes: &[u8]) -> Result<(u16, u16), String> {
    let mut d = Decoder::new(Cursor::new(bytes));
    d.read_info().map_err(|e| format!("JPEG header parse: {e}"))?;
    let info = d
        .info()
        .ok_or_else(|| "JPEG decoder yielded no info".to_string())?;
    Ok((info.width, info.height))
}
