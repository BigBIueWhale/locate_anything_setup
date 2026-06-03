//! WebSocket /v1/stream handler — stop-and-wait per connection.
//!
//! Flow per WebSocket:
//!   1. Client opens WS. No handshake — capabilities are out of band
//!      (`GET /v1/capabilities`).
//!   2. Client sends a Frame binary message:
//!      `[ 4-byte BE u32 header_len ][ header JSON ][ JPEG bytes ]`.
//!   3. Server runs one inference and replies with exactly one typed
//!      [`Response`] Text frame (`boxes` XOR `points` XOR `abstained` XOR
//!      `error`) — never both, never multiples.
//!   4. Repeat from step 2 until the client closes (or the server
//!      shuts down, in which case we send a clean Close 1001).
//!
//! Error surfaces — exactly two, per docs/CLIENT_PROTOCOL.md:
//!   * **Framing-fatal → Close(1008)**: the binary framing is unparseable
//!     so the next Frame cannot be located in the bytestream. Header
//!     length-prefix wrong, header JSON unparseable (including an unknown
//!     field, a malformed `request`, or a `generation_mode` that is not one
//!     of fast/hybrid/slow — serde rejects these at parse time), frame_id
//!     missing — all collapse the connection because there is no way to send
//!     a correlated error.
//!   * **Per-frame → `Response::Error` Text(`{type:"error", frame_id, code,
//!     message}`)**: framing is intact but the content of THIS Frame is
//!     rejected. `code` is one of the A.2 `ErrorCode`s: `invalid_request`
//!     (typed-slot validation failed, or jpeg_len ≠ payload), `invalid_image`
//!     (signature/decode/dimension/patch-cap), or `internal` (a runtime
//!     fault). The WebSocket stays open and the next Frame proceeds normally
//!     — the client can correct and retry.
//!
//! The model is stateless across calls; many WebSockets handle many
//! clients concurrently via asyncio inside the worker, but each individual
//! WS is strictly sequential — one Frame in, one Result out, repeat.

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, instrument, warn};

use crate::ipc;
use crate::jpeg;
use crate::prompt_validator;
use crate::protocol::{ErrorBody, ErrorCode, InferHeader, Response, MIN_IMAGE_DIM};
use crate::state::AppState;

/// Cadence at which the server sends WebSocket ping control frames to
/// the client. The client's tungstenite stack auto-responds with Pong;
/// no application-level handling required on the client. We need this
/// because Linux TCP keepalive defaults to ~2h of silence before
/// probing — far too long to notice a dead client.
const PING_INTERVAL: Duration = Duration::from_secs(15);
/// Reader-side hard idle timeout. If no WebSocket message (data OR
/// pong) arrives for this long, the connection is considered dead and
/// closed. PING_INTERVAL × 4 leaves room for a few missed pings before
/// declaring the peer gone.
const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Close codes per RFC 6455 §7.4. We don't issue a Close(1000) ourselves
/// — a clean shutdown by the client is observed as ws_rx returning None.
const CLOSE_GOING_AWAY: u16        = 1001;
const CLOSE_POLICY_VIOLATION: u16  = 1008;
const CLOSE_SERVER_ERROR: u16      = 1011;

pub async fn ws_route(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let max = state.args.max_jpeg_bytes + 64 * 1024;
    ws.max_message_size(max)
        .max_frame_size(max)
        .on_upgrade(move |socket| handle_ws(socket, state))
}

#[instrument(skip_all)]
async fn handle_ws(socket: WebSocket, state: Arc<AppState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Open the dedicated UDS connection to the Python worker. One WS = one
    // worker conn = one inference at a time (worker serializes on its own
    // lock; this WS just drives a single pipeline through it).
    let mut conn = match ipc::WorkerConn::connect(&state.args.worker_socket).await {
        Ok(c) => c,
        Err(e) => {
            send_close(
                &mut ws_tx,
                CLOSE_SERVER_ERROR,
                &format!("worker_unavailable: {e}"),
            ).await;
            return;
        }
    };

    let mut ping = tokio::time::interval(PING_INTERVAL);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    ping.tick().await; // eat the immediate first tick

    let max_jpeg_bytes = state.args.max_jpeg_bytes;
    let max_image_dim = state.args.max_image_dim;
    let shutdown = state.shutdown.clone();

    loop {
        // Read the next WebSocket message OR drive a ping OR honor shutdown.
        // We only act on Binary; Text/Pong/Ping refresh the idle timer.
        let bytes = loop {
            tokio::select! {
                biased;
                _ = shutdown.notified() => {
                    send_close(&mut ws_tx, CLOSE_GOING_AWAY,
                               "server shutting down").await;
                    return;
                }
                _ = ping.tick() => {
                    if ws_tx.send(Message::Ping(Bytes::new())).await.is_err() {
                        debug!("ping send failed — peer is gone");
                        return;
                    }
                }
                res = tokio::time::timeout(READ_IDLE_TIMEOUT, ws_rx.next()) => {
                    let msg = match res {
                        Ok(Some(Ok(m))) => m,
                        Ok(Some(Err(e))) => { warn!(error=?e, "WS recv error"); return; }
                        Ok(None) => return,
                        Err(_) => {
                            warn!(timeout=?READ_IDLE_TIMEOUT,
                                  "WS reader idle timeout — closing connection");
                            send_close(&mut ws_tx, CLOSE_GOING_AWAY,
                                       "read idle timeout").await;
                            return;
                        }
                    };
                    match msg {
                        Message::Binary(b) => break b,
                        // Text is not a valid frame in this protocol but
                        // we tolerate it (and Pings/Pongs) so the idle
                        // timer refreshes naturally.
                        Message::Text(_) | Message::Ping(_) | Message::Pong(_) => continue,
                        Message::Close(_) => return,
                    }
                }
            }
        };

        // Parse + validate the Frame. Three outcomes:
        //   Pending → forward to worker
        //   FatalFraming → Close(1008), connection cannot continue
        //   PerFrame    → emit a typed `Response::Error`, keep WS open
        let pending = match process_binary(bytes, max_jpeg_bytes, max_image_dim).await {
            BinaryOutcome::Pending(p) => p,
            BinaryOutcome::FatalFraming(reason) => {
                send_close(&mut ws_tx, CLOSE_POLICY_VIOLATION, &reason).await;
                return;
            }
            BinaryOutcome::PerFrame { frame_id, code, message } => {
                let payload = serialize_response(&Response::Error(ErrorBody {
                    frame_id,
                    code,
                    message,
                }));
                if ws_tx.send(Message::Text(payload.into())).await.is_err() {
                    return;
                }
                continue;
            }
        };

        // Forward to the worker. The reply is already a typed `Response`
        // (a boxes/points/abstained success OR a Response::Error — egress
        // schema-enforced in ipc::infer). We serialize it straight to the
        // client; frame_id is a typed field on each variant, so there is no
        // post-hoc stamping. A transport/protocol error closes the WS because
        // the framed UDS may now be desynced.
        let payload = match conn.infer(
            &pending.header.frame_id,
            &pending.prompt,
            pending.prompt_task,
            pending.header.generation_mode,
            pending.header.jpeg_len,
            pending.jpeg,
        ).await {
            Ok(response) => serialize_response(&response),
            Err(e) => {
                // Worker transport/protocol error. The framed UDS may now be
                // desynced, so we close the WS rather than try to resync on
                // this connection. The next WS will get a fresh worker conn.
                send_close(&mut ws_tx, CLOSE_SERVER_ERROR,
                           &format!("worker: {e}")).await;
                return;
            }
        };
        if ws_tx.send(Message::Text(payload.into())).await.is_err() {
            return;
        }
    }
}

struct PendingFrame {
    header: InferHeader,
    jpeg: Bytes,
    /// The COMPILED trained prompt string, assembled from the client's typed
    /// `header.request` by `prompt_validator::compile`. Forwarded to the
    /// worker as the IPC header's `prompt` field (the typed request never
    /// crosses the UDS).
    prompt: String,
    /// The `prompt_task` wire-name for the compiled prompt (one of
    /// `prompt_validator::TemplateKind::wire_name`). Forwarded to the worker
    /// for trained-correct task→shape routing of off-shape model output. See
    /// worker/inference.py::EXPECTED_SHAPE.
    prompt_task: &'static str,
}

/// The three possible outcomes of binary-frame parsing/validation.
enum BinaryOutcome {
    /// Ready to forward to the worker.
    Pending(PendingFrame),
    /// Framing is unparseable — close the WebSocket with reason text.
    /// Used when there is no recoverable frame_id to correlate against
    /// (length prefix bad, header JSON unparseable, frame_id missing).
    FatalFraming(String),
    /// Framing is intact but the content of this Frame is rejected.
    /// The Frame is dropped; the WS stays open for the next Frame. `code`
    /// is one of the four A.2 `ErrorCode`s (no longer a numeric status).
    PerFrame { frame_id: String, code: ErrorCode, message: String },
}

/// Validate a single WebSocket binary message and produce a BinaryOutcome.
///
/// Failures BEFORE we successfully parse a usable `frame_id` are
/// `FatalFraming` (we have no correlation key for a per-frame error).
/// Failures AFTER `frame_id` is in hand are `PerFrame` — the client gets
/// a typed error message and the next Frame proceeds normally.
async fn process_binary(
    bytes: Bytes,
    max_jpeg_bytes: usize,
    max_image_dim: u16,
) -> BinaryOutcome {
    // ---- 1. Length-prefix sanity (fatal — can't locate next frame) ----
    if bytes.len() < 4 {
        return BinaryOutcome::FatalFraming(format!(
            "WS binary frame is {} bytes but a 4-byte BE u32 length-prefix \
             header is required (see docs/CLIENT_PROTOCOL.md)",
            bytes.len()
        ));
    }
    let header_len = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if header_len == 0 {
        return BinaryOutcome::FatalFraming(
            "header_len prefix is 0; the JSON header is mandatory".into(),
        );
    }
    if 4 + header_len > bytes.len() {
        return BinaryOutcome::FatalFraming(format!(
            "declared header_len={} extends past total binary frame size {} \
             (header_len must fit AND leave room for the JPEG payload)",
            header_len, bytes.len()
        ));
    }
    let header_slice = &bytes[4..4 + header_len];

    // ---- 2. JSON header parse (fatal — no frame_id yet) ---------------
    let header: InferHeader = match serde_json::from_slice(header_slice) {
        Ok(h) => h,
        Err(e) => {
            return BinaryOutcome::FatalFraming(format!(
                "header JSON parse failed: {e}. Required keys: frame_id, \
                 request (a typed {{\"task\":...}} object), generation_mode \
                 (\"fast\"|\"hybrid\"|\"slow\"), jpeg_len. Extra keys rejected. \
                 See docs/CLIENT_PROTOCOL.md.",
            ));
        }
    };

    // ---- 3. frame_id (fatal — we need it to correlate per-frame errors) -
    if header.frame_id.is_empty() {
        return BinaryOutcome::FatalFraming(
            "header.frame_id is empty; frame_id is required as the response \
             correlation primitive — without it we cannot send a per-frame \
             error".into(),
        );
    }
    if header.frame_id.len() > 256 {
        return BinaryOutcome::FatalFraming(format!(
            "header.frame_id length {} > 256 chars (bounded to keep log \
             lines manageable)",
            header.frame_id.len()
        ));
    }
    let frame_id = header.frame_id.clone();

    // From here on we have a usable frame_id — every error becomes a
    // per-frame `type:"error"` message and the WS stays open.

    // ---- 4. Compile the typed request → the exact trained prompt. The
    //         builder validates the TYPED slot values (NFC, no control chars,
    //         no surrounding whitespace, no `</c>`, no trailing '.', ≤200
    //         chars, categories 1..=10 / no-comma, point/category no-comma)
    //         per the locked A.1 contract, REJECTING (never normalizing) any
    //         violation. It also enforces the compiled-prompt MAX_PROMPT_CHARS
    //         cap. The returned wire-name is forwarded to the worker as
    //         `prompt_task` so the parser can route off-shape model output per
    //         the trained task→shape contract (prompt_validator::
    //         TemplateKind::wire_name + worker/inference.py::EXPECTED_SHAPE).
    //
    //         NOTE: generation_mode is no longer string-checked here — serde's
    //         typed `GenerationMode` enum rejected any non-{fast,hybrid,slow}
    //         value at header-parse time above (a fatal-framing error). -----
    let (prompt, prompt_task) = match prompt_validator::compile(&header.request) {
        Ok(compiled) => compiled,
        Err(e) => {
            return BinaryOutcome::PerFrame {
                frame_id,
                code: ErrorCode::InvalidRequest,
                message: format!("request rejected: {}", e.message()),
            };
        }
    };

    // ---- 5. JPEG payload size + signature ------------------------------
    let jpeg = bytes.slice(4 + header_len..);
    if jpeg.len() != header.jpeg_len {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidRequest,
            message: format!(
                "header.jpeg_len={} != actual payload length {} (these must \
                 match exactly — mismatch indicates a framing bug)",
                header.jpeg_len, jpeg.len()
            ),
        };
    }
    if jpeg.is_empty() {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: "JPEG payload is zero bytes (a Frame must carry a JPEG image)".into(),
        };
    }
    if jpeg.len() > max_jpeg_bytes {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "JPEG payload {} bytes exceeds server cap LA_MAX_JPEG_BYTES={} \
                 (configurable in scripts/lib/versions.sh)",
                jpeg.len(), max_jpeg_bytes
            ),
        };
    }
    if !jpeg::is_jpeg(&jpeg) {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "payload first bytes [{:02X}, {:02X}, {:02X}] are not the JPEG \
                 SOI marker FF D8 FF; we accept only baseline JPEG with the \
                 standard signature",
                jpeg.first().copied().unwrap_or(0),
                jpeg.get(1).copied().unwrap_or(0),
                jpeg.get(2).copied().unwrap_or(0),
            ),
        };
    }

    // ---- 6. JPEG dimensions (off-thread; jpeg_decoder is sync) ---------
    let jpeg_for_check = jpeg.clone();
    let dims = tokio::task::spawn_blocking(move || {
        jpeg::read_dimensions_blocking(&jpeg_for_check)
    })
    .await;
    let (w, h) = match dims {
        Ok(Ok(d)) => d,
        Ok(Err(s)) => return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "JPEG header parse failed: {s} (the SOI marker matched but the \
                 JPEG structure is malformed — check the encoder output)"
            ),
        },
        Err(e) => return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::Internal,
            message: format!(
                "spawn_blocking for jpeg_decoder join error: {e} (this is a \
                 tokio runtime issue, not a client error)"
            ),
        },
    };
    if w < MIN_IMAGE_DIM || h < MIN_IMAGE_DIM {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "image dimensions {}x{} below MIN_IMAGE_DIM={} (a useful input \
                 must occupy at least one LLM token in the model's 28px grid)",
                w, h, MIN_IMAGE_DIM
            ),
        };
    }
    if w > max_image_dim || h > max_image_dim {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "image dimensions {}x{} exceed LA_MAX_IMAGE_DIM={} \
                 (configurable in scripts/lib/versions.sh; the model's preprocessor \
                 rescales anything above ~2240px square to fit its 25600-patch cap)",
                w, h, max_image_dim
            ),
        };
    }

    // Strict trained-correct preprocessor gates. The model's
    // image_processing_locateanything.py:rescale() enforces THREE constraints;
    // we mirror them bit-for-bit at the network edge so the client never
    // gets a result for a silently-modified image.
    //
    //   (a) `(W // 14) * (H // 14) <= in_token_limit (=25,600)` — line 52
    //       of the model's rescale(). Above this, the preprocessor would
    //       internally BICUBIC-rescale to fit; the client's frame_id would
    //       then refer to a different spatial frame than the one returned.
    //
    //   (b) `W // 14 < 512` AND `H // 14 < 512` — line 68 of rescale().
    //       The MoonViT positional embedding is a 64×64 base learnable
    //       embedding bicubic-interpolated up to the runtime grid; 512
    //       patches per side is the documented "Exceed pos emb" hard cap
    //       (image_processing_locateanything.py line 68-69). Beyond this
    //       the preprocessor raises a Python ValueError — we want a clean
    //       client-side rejection at the WS edge instead.
    //
    //   (NB: the formula uses FLOOR-DIV on the raw 14-px patch grid, NOT
    //   ceil-div on the merged 28-px grid. We had this wrong in a prior
    //   revision — verified against NVIDIA's code at the SHA pin.)
    //
    // At the current LA_MAX_IMAGE_DIM=2240, both checks are dormant
    // (2240/14 = 160 per side → 25,600 patches square / 160 < 512), so
    // these gates protect future cap raises and unusual aspect ratios.
    const PATCH_PX: u64        = 14;          // from preprocessor_config.json
    const IN_TOKEN_LIMIT: u64  = 25_600;      // from preprocessor_config.json
    const POS_EMB_PATCH_CAP: u64 = 512;       // from model's rescale() line 68
    let w_patches = w as u64 / PATCH_PX;
    let h_patches = h as u64 / PATCH_PX;
    let n_patches = w_patches * h_patches;
    if n_patches > IN_TOKEN_LIMIT {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "image dimensions {}x{} produce {} ViT patches \
                 ((W // {}) × (H // {})), exceeding the trained \
                 `in_token_limit = {}`. The model's preprocessor would \
                 internally BICUBIC-downscale to fit, producing detections \
                 relative to a smaller image than the one you sent — we \
                 refuse this rather than silently scale. Reduce dimensions \
                 so (W // {}) × (H // {}) ≤ {}.",
                w, h, n_patches,
                PATCH_PX, PATCH_PX,
                IN_TOKEN_LIMIT,
                PATCH_PX, PATCH_PX, IN_TOKEN_LIMIT
            ),
        };
    }
    if w_patches >= POS_EMB_PATCH_CAP || h_patches >= POS_EMB_PATCH_CAP {
        return BinaryOutcome::PerFrame {
            frame_id,
            code: ErrorCode::InvalidImage,
            message: format!(
                "image dimensions {}x{} would map to a {}×{} patch grid; the \
                 MoonViT positional embedding's bicubic-interpolation cap is \
                 {} patches per side (= {} px), per the model's preprocessor \
                 at image_processing_locateanything.py line 68 (\"Exceed pos \
                 emb\"). Reduce each dimension to < {} px.",
                w, h, w_patches, h_patches,
                POS_EMB_PATCH_CAP, POS_EMB_PATCH_CAP * PATCH_PX,
                POS_EMB_PATCH_CAP * PATCH_PX
            ),
        };
    }

    BinaryOutcome::Pending(PendingFrame { header, jpeg, prompt, prompt_task })
}

/// Serialize a typed [`Response`] to the JSON Text frame sent to the client.
/// `frame_id` and the `type` tag are typed fields on the variant — no
/// post-hoc stamping. serde serialization of these types is infallible in
/// practice (no NaN/Inf reach our f32/f64 — they come from valid wire JSON or
/// finite-constructed errors); the unreachable error arm still emits a
/// guaranteed-valid `internal` error frame rather than panicking.
fn serialize_response(response: &Response) -> String {
    match serde_json::to_string(response) {
        Ok(s) => s,
        Err(e) => {
            // Should be unreachable for our concrete types. Emit a minimal,
            // hand-rolled error frame so the client still receives a typed
            // error rather than a dropped/garbled message.
            let frame_id = match response {
                Response::Boxes(b) => b.frame_id.as_str(),
                Response::Points(p) => p.frame_id.as_str(),
                Response::Abstained(a) => a.frame_id.as_str(),
                Response::Error(er) => er.frame_id.as_str(),
            };
            serde_json::json!({
                "type":     "error",
                "frame_id": frame_id,
                "code":     ErrorCode::Internal.as_wire(),
                "message":  format!("internal: failed to serialize response: {e}"),
            }).to_string()
        }
    }
}

/// Send a WebSocket Close frame with a specific code and reason, then
/// close the sink. Reason length is capped at 123 UTF-8 bytes per
/// RFC 6455 §5.5.1 (payload of close = 2-byte code + ≤123-byte reason).
async fn send_close(
    ws_tx: &mut SplitSink<WebSocket, Message>,
    code: u16,
    reason: &str,
) {
    let mut bounded = String::new();
    for c in reason.chars() {
        let needed = c.len_utf8();
        if bounded.len() + needed > 123 { break; }
        bounded.push(c);
    }
    let close_frame = CloseFrame { code, reason: bounded.into() };
    let _ = ws_tx.send(Message::Close(Some(close_frame))).await;
    let _ = ws_tx.close().await;
}
