//! Wire protocol — both the Rust↔Python IPC and the WebSocket external API.
//!
//! ## External (client ↔ Rust server) — WebSocket /v1/stream
//!
//! Every WebSocket message is a binary frame consisting of:
//!   `[ 4 bytes BE u32 header_len ] [ header_len bytes UTF-8 JSON ] [ optional binary payload ]`
//!
//! The JSON header carries the message type and metadata. Binary payload
//! (when present) is a JPEG byte string.
//!
//! ## Internal (Rust ↔ Python) — Unix domain socket /tmp/la.sock
//!
//! Length-prefixed frames using LengthDelimitedCodec (4-byte BE u32).
//! Each request is TWO consecutive frames sent by the Rust side:
//!   1) Header JSON (`InferHeader` below).
//!   2) JPEG bytes (may be zero-length for capability/info queries).
//!
//! The Python worker responds with ONE frame:
//!   - JSON `InferResponse` (see Python worker schema docs).
//!
//! Versioning: a `protocol_version` integer is part of every Hello and
//! every Capabilities response. Bumps are major-only — the worker and
//! server are deployed together, so we never see version skew in
//! production.

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FrameKind {
    /// Client→Server: opens a session, declares client capabilities.
    Hello,
    /// Client→Server: a single image frame to be processed. Every Frame
    /// is self-contained — its header carries its own prompt and
    /// generation_mode. There is no separate Configure/Cancel — every
    /// piece of state the model needs is on the Frame itself.
    Frame,
    /// Server→Client: response to a Hello.
    Capabilities,
    /// Server→Client: bounding-box result for a frame.
    Result,
    /// Server→Client: per-frame or server-state error.
    Error,
    /// Server→Client: 1 Hz advisory beacon.
    Beacon,
}

/// The JSON header the client sends with every Frame.
/// Mirrored as the payload of the Rust→Python "request header" frame.
///
/// **Every field is required.** There are no defaults, no nullable
/// inputs on the inference path. Missing-or-malformed → per-frame
/// `invalid_request` error.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct InferHeader {
    #[serde(rename = "type")]
    pub kind: FrameKind,

    /// Client-assigned correlation id, REQUIRED. The server uses this
    /// as the sole linkage primitive between Frames and Results/Errors.
    /// Echoed unchanged in every response that pertains to this frame.
    pub frame_id: String,

    /// Free-form session identifier the client chooses. Echoed back in
    /// every Result/Error. The server does NOT use it for any decision;
    /// it exists purely so client-side logs can group "which frames were
    /// sent in which session." Allowed to be the empty string but the
    /// field must be present.
    pub session_id: String,

    /// One of the LocateAnything prompts (e.g. `Locate all the instances
    /// that matches the following description: drone.`). Length 1..=16384
    /// (the model's tokenizer.model_max_length cap, before tokenization).
    pub prompt: String,

    /// Required. Exactly one of "fast", "hybrid", "slow". The server
    /// has NO default — clients must commit to a mode per frame.
    pub generation_mode: String,

    /// Declared JPEG payload size in bytes. Must equal the actual
    /// JPEG bytes that follow the header in the binary frame.
    pub jpeg_len: usize,

    /// Must be exactly "RGB" — there is no other supported color space.
    /// The model's processor expects RGB; we reject everything else
    /// rather than auto-convert (auto-conversion is a hidden assumption
    /// about the client's intent).
    pub image_color_space: String,

    /// Must be exactly "jpeg" today — no other encoding is supported.
    pub image_encoding: String,
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

/// Sent from client to server first, as a Text WebSocket message,
/// immediately after the WS upgrade. Every field is required. No defaults.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct HelloMessage {
    #[serde(rename = "type")]
    pub kind: FrameKind,            // must be Hello
    pub protocol_version: u32,
    pub client_id: String,
    pub session_id: String,         // may be the empty string, but the field must appear
}

/// A client error frame is allowed but optional — used for protocol breaks
/// before any frame_id has been assigned. The server emits its own error
/// frames; this type exists for symmetry.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(dead_code)] // documents the protocol; clients use this shape too
pub struct ErrorMessage<'a> {
    #[serde(rename = "type")]
    pub kind: FrameKind, // must be Error
    pub code: u16,
    pub error_type: &'a str,
    pub message: &'a str,
    #[serde(default)]
    pub frame_id: Option<&'a str>,
    pub retriable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}
