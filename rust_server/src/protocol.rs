//! Wire protocol — both the Rust↔Python IPC and the WebSocket external API.
//!
//! ## External (client ↔ Rust server) — WebSocket /v1/stream
//!
//! Stop-and-wait per connection: client opens WS, immediately sends a Frame,
//! reads exactly one Result or Error response, optionally sends the next
//! Frame, and so on. There is no handshake — the WS opens "hot." Capabilities
//! are fetched out of band via `GET /v1/capabilities`.
//!
//! Every WebSocket binary message is a Frame:
//!   `[ 4 bytes BE u32 header_len ] [ header_len bytes UTF-8 JSON ] [ JPEG bytes ]`
//!
//! ## Internal (Rust ↔ Python) — Unix domain socket /tmp/la.sock
//!
//! Length-prefixed frames using LengthDelimitedCodec (4-byte BE u32).
//! Each request is TWO consecutive frames sent by the Rust side:
//!   1) Header JSON (`InferHeader` for frames; `{"kind":"capabilities"|"info"}`
//!      for control queries).
//!   2) JPEG bytes (zero-length for capability/info queries).
//!
//! The Python worker responds with ONE frame: a JSON body. The body either
//! describes a successful inference (forwarded verbatim as a `type:"result"`
//! WS message) or an error (forwarded verbatim as a `type:"error"` WS
//! message). The Rust frontend does no shape translation.

use serde::{Deserialize, Serialize};

/// The JSON header the client sends with every WebSocket Frame.
/// Mirrored as the payload of the Rust→Python "request header" frame, with
/// `kind` added by the Rust side so the worker can dispatch between
/// frame inference and control queries on a single UDS.
///
/// **Every field is required.** Missing-or-malformed → WS Close 1008.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct InferHeader {
    /// Client-assigned correlation id. Echoed unchanged in the response.
    /// 1..=256 chars.
    pub frame_id: String,

    /// One of the seven canonical LocateAnything-3B prompts. The
    /// single-source-of-truth catalog of allowed templates lives in
    /// `worker/prompts.py`; this Rust crate enforces them at the WS edge
    /// via `prompt_validator::validate`. 1..=MAX_PROMPT_CHARS chars.
    pub prompt: String,

    /// Exactly one of "fast", "hybrid", "slow". No server default — every
    /// Frame commits to a mode.
    pub generation_mode: String,

    /// Declared JPEG payload size in bytes. Must equal the actual JPEG bytes
    /// that follow the header in the binary frame.
    pub jpeg_len: usize,
}

/// Minimum allowed image dimension (per side, in pixels). Below this
/// the model has zero useful spatial information.
pub const MIN_IMAGE_DIM: u16 = 32;

/// Maximum allowed prompt length in CHARACTERS. The actual
/// tokenizer.model_max_length is 16384 *tokens*, which is always ≥ the
/// equivalent character count for ASCII / Western prompts. This is a
/// generous-but-finite character cap so we can reject 100 MB JSON
/// payloads at the network layer.
pub const MAX_PROMPT_CHARS: usize = 16384;
