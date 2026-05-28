//! WebSocket /v1/stream handler — bidi binary protocol.
//!
//! Flow:
//!   1. Client opens WS.
//!   2. Client sends `Hello` JSON (Text frame). Server validates and
//!      replies with `Capabilities` JSON.
//!   3. Client sends one or more `Frame` binary messages:
//!      `[ 4-byte BE u32 header_len ][ header JSON ][ JPEG bytes ]`.
//!   4. Server emits one `Result` Text per Frame, in submission order
//!      per connection.
//!   5. Server also emits a 1 Hz `Beacon` Text (advisory only).
//!
//! Backpressure: bounded mpsc → reader task stops draining the WS → TCP
//! flow control pushes back. The server NEVER drops frames silently.

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::error::ServerError;
use crate::ipc;
use crate::jpeg;
use crate::protocol::{
    FrameKind, HelloMessage, InferHeader, MAX_PROMPT_CHARS, MIN_IMAGE_DIM, PROTOCOL_VERSION,
};
use crate::state::AppState;

const BEACON_INTERVAL: Duration = Duration::from_secs(1);
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

/// Close codes per RFC 6455 §7.4.
const CLOSE_NORMAL: u16            = 1000;
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

    // -------- Step 1: hello handshake --------
    let hello_msg = match tokio::time::timeout(Duration::from_secs(10), ws_rx.next()).await {
        Ok(Some(Ok(Message::Text(t)))) => t,
        _ => {
            send_error_and_close(
                &mut ws_tx,
                "hello_required",
                "first WebSocket frame must be a JSON Hello",
                CLOSE_POLICY_VIOLATION,
            ).await;
            return;
        }
    };
    let hello: HelloMessage = match serde_json::from_str::<HelloMessage>(hello_msg.as_str()) {
        Ok(h) => {
            if h.kind != FrameKind::Hello {
                send_error_and_close(&mut ws_tx, "hello_invalid",
                    &format!("Hello.type must be \"hello\"; got {:?}", h.kind),
                    CLOSE_POLICY_VIOLATION).await;
                return;
            }
            if h.protocol_version != PROTOCOL_VERSION {
                send_error_and_close(&mut ws_tx, "hello_invalid",
                    &format!(
                        "Hello.protocol_version={} but this server speaks v{}. \
                         Clients must match exactly — there is no compatibility \
                         downgrade.",
                        h.protocol_version, PROTOCOL_VERSION
                    ),
                    CLOSE_POLICY_VIOLATION).await;
                return;
            }
            if h.client_id.is_empty() {
                send_error_and_close(&mut ws_tx, "hello_invalid",
                    "Hello.client_id is empty (any non-empty string is fine; \
                     used in server-side logs to identify the connection)",
                    CLOSE_POLICY_VIOLATION).await;
                return;
            }
            h
        }
        Err(e) => {
            send_error_and_close(&mut ws_tx, "hello_invalid",
                &format!(
                    "Hello JSON parse failed: {e}. Required fields: \
                     type=\"hello\", protocol_version={}, client_id (non-empty), \
                     session_id (may be empty). #[serde(deny_unknown_fields)] \
                     is on — no extra keys.",
                    PROTOCOL_VERSION
                ),
                CLOSE_POLICY_VIOLATION).await;
            return;
        }
    };

    // Reply with capabilities (fetched live from the worker).
    let caps_payload = match ipc::query_capabilities(&state.args.worker_socket).await {
        Ok(mut v) => {
            if let serde_json::Value::Object(ref mut map) = v {
                map.insert("type".into(), serde_json::json!("capabilities"));
            }
            v.to_string()
        }
        Err(e) => {
            warn!(error=?e, "capabilities query failed");
            send_error_and_close(&mut ws_tx, "worker_unavailable",
                &e.to_string(), CLOSE_SERVER_ERROR).await;
            return;
        }
    };
    if ws_tx.send(Message::Text(caps_payload.into())).await.is_err() {
        return;
    }
    info!(client_id = %hello.client_id, "WS hello complete");

    // -------- Step 2: dedicated worker conn + bounded channels --------
    let conn = match ipc::WorkerConn::connect(&state.args.worker_socket).await {
        Ok(c) => c,
        Err(e) => {
            send_error_and_close(&mut ws_tx, "worker_unavailable",
                &e.to_string(), CLOSE_SERVER_ERROR).await;
            return;
        }
    };

    let max_inflight = state.args.max_inflight.max(1);
    let (frame_tx, mut frame_rx) = mpsc::channel::<PendingFrame>(max_inflight);
    let (out_tx, mut out_rx) = mpsc::channel::<String>(max_inflight + 4);

    let stats = Arc::new(Stats::new(max_inflight));
    // Hand the beacon a clone of frame_tx so it can read live capacity().
    stats.set_frame_tx(frame_tx.clone());

    // -------- Step 3: processor task --------
    //
    // Invariant: the LengthDelimitedCodec stream `conn` is bidirectionally
    // framed and MUST stay synced between Rust and Python. ANY error on
    // either send_frame or recv_frame leaves the position indeterminate
    // (a partial read could have consumed a header but not its payload).
    // We therefore reset `conn` on any error before the next iteration —
    // the cost is one UDS reconnect, the gain is no silent poisoning of
    // subsequent frames on the same WS.
    let proc_stats = stats.clone();
    let proc_out_tx = out_tx.clone();
    let proc_socket_path = state.args.worker_socket.clone();
    let processor = tokio::spawn(async move {
        let mut conn_opt: Option<ipc::WorkerConn> = Some(conn);
        while let Some(pending) = frame_rx.recv().await {
            // If the previous iteration desynced the connection, the
            // option is None and we open a fresh one. If reconnect
            // fails, surface a worker_unavailable error on this frame
            // and loop again — the next iteration will retry.
            let conn = match conn_opt.as_mut() {
                Some(c) => c,
                None => {
                    match ipc::WorkerConn::connect(&proc_socket_path).await {
                        Ok(c) => { conn_opt = Some(c); conn_opt.as_mut().unwrap() }
                        Err(e) => {
                            let err = ServerError::WorkerUnavailable(format!(
                                "UDS reconnect failed: {e}"
                            ));
                            let body = error_frame(&err, Some(&pending.header.frame_id));
                            if proc_out_tx.send(body).await.is_err() { break; }
                            continue;
                        }
                    }
                }
            };
            proc_stats.inflight.fetch_add(1, Ordering::Relaxed);
            let result = conn.infer(&pending.header, pending.jpeg).await;
            proc_stats.inflight.fetch_sub(1, Ordering::Relaxed);

            let payload = match result {
                Ok(v) => {
                    // The worker uses a single response schema with `ok:bool`.
                    // We translate it into the externally-documented two-shape
                    // taxonomy here: ok=true → type:"result", ok=false →
                    // type:"error". The client never sees `ok:false` on a
                    // type:"result" frame, so the documented contract is
                    // ALL the client ever needs to handle.
                    match v.get("ok").and_then(|x| x.as_bool()) {
                        Some(true) => {
                            let body = build_result_body(v, &pending.header.frame_id);
                            proc_stats.completed.fetch_add(1, Ordering::Relaxed);
                            proc_stats.store_last(&pending.header.frame_id);
                            body
                        }
                        Some(false) => {
                            error!(frame_id=%pending.header.frame_id,
                                   "worker reported per-frame error");
                            build_worker_error_body(v, &pending.header.frame_id)
                        }
                        None => {
                            let e = ServerError::WorkerProtocol(format!(
                                "worker response is missing the required `ok` boolean field; \
                                 received keys: {:?}",
                                v.as_object().map(|m| m.keys().collect::<Vec<_>>())
                            ));
                            // A protocol-violating response means we cannot
                            // trust the framing position either; drop conn.
                            conn_opt = None;
                            error_frame(&e, Some(&pending.header.frame_id))
                        }
                    }
                }
                Err(e) => {
                    error!(error=?e, frame_id=%pending.header.frame_id,
                           "infer transport error — dropping WorkerConn to resync");
                    // Drop the framed connection — the next iteration will
                    // reconnect a fresh one before reading again.
                    conn_opt = None;
                    error_frame(&e, Some(&pending.header.frame_id))
                }
            };
            if proc_out_tx.send(payload).await.is_err() {
                break;
            }
        }
        debug!("processor task exit");
    });

    // -------- Step 4: writer task (results + beacons + pings → WS) --------
    let writer_stats = stats.clone();
    let writer_shutdown = state.shutdown.clone();
    let writer = tokio::spawn(async move {
        let mut beacon = tokio::time::interval(BEACON_INTERVAL);
        beacon.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut pinger = tokio::time::interval(PING_INTERVAL);
        pinger.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Eat the first immediate tick of each interval so we don't ping
        // / beacon at t=0 before any state exists.
        beacon.tick().await;
        pinger.tick().await;
        let mut close_code = CLOSE_NORMAL;
        let mut close_reason = "normal";
        loop {
            tokio::select! {
                biased;
                _ = writer_shutdown.notified() => {
                    close_code = CLOSE_GOING_AWAY;
                    close_reason = "server shutting down";
                    break;
                }
                Some(text) = out_rx.recv() => {
                    if ws_tx.send(Message::Text(text.into())).await.is_err() {
                        close_code = CLOSE_SERVER_ERROR;
                        close_reason = "send failed";
                        break;
                    }
                }
                _ = beacon.tick() => {
                    let b = writer_stats.beacon_json();
                    if ws_tx.send(Message::Text(b.into())).await.is_err() {
                        close_code = CLOSE_SERVER_ERROR;
                        close_reason = "beacon send failed";
                        break;
                    }
                }
                _ = pinger.tick() => {
                    if ws_tx.send(Message::Ping(Bytes::new())).await.is_err() {
                        // The client side is gone; we don't bother with
                        // a Close frame since the send already failed.
                        debug!("ping send failed — peer is gone");
                        return;
                    }
                }
                else => break,
            }
        }
        send_close(&mut ws_tx, close_code, close_reason).await;
        debug!("writer task exit");
    });

    // -------- Step 5: reader task --------
    //
    // Reader loop with both an explicit per-message READ_IDLE_TIMEOUT
    // (catches half-open TCP that even server pings can't keep alive)
    // and a shutdown-watch (exits early when the server is shutting
    // down). Either condition closes the connection cleanly.
    let max_jpeg_bytes = state.args.max_jpeg_bytes;
    let max_image_dim = state.args.max_image_dim;
    let reader_shutdown = state.shutdown.clone();
    loop {
        let next = tokio::select! {
            biased;
            _ = reader_shutdown.notified() => {
                info!("reader: shutdown notified — exiting");
                break;
            }
            res = tokio::time::timeout(READ_IDLE_TIMEOUT, ws_rx.next()) => res,
        };
        let msg = match next {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(e))) => { warn!(error=?e, "WS recv error"); break; }
            Ok(None) => break,
            Err(_) => {
                warn!(timeout=?READ_IDLE_TIMEOUT,
                      "WS reader idle timeout — closing connection");
                break;
            }
        };
        match msg {
            Message::Binary(bytes) => {
                let outcome = process_binary(
                    bytes, max_jpeg_bytes, max_image_dim,
                ).await;
                match outcome {
                    Ok(pf) => {
                        // `frame_tx.send(...).await` blocks when the bounded
                        // channel is at capacity — TCP backpressure point.
                        // The beacon reads queue depth via mpsc capacity,
                        // not via a side-counter.
                        if frame_tx.send(pf).await.is_err() {
                            break;
                        }
                    }
                    Err((err, frame_id)) => {
                        // Per-frame validation error → push directly to writer.
                        let body = error_frame(&err, frame_id.as_deref());
                        if out_tx.send(body).await.is_err() {
                            break;
                        }
                    }
                }
            }
            // Pong messages from the client (response to OUR pings) and
            // Pings (which axum auto-pongs) just refresh the read timer
            // by virtue of having arrived.
            Message::Text(_) | Message::Ping(_) | Message::Pong(_) => {}
            Message::Close(_) => break,
        }
    }

    drop(frame_tx);
    drop(out_tx);
    let _ = tokio::join!(processor, writer);
    info!("WS connection closed");
}

struct PendingFrame {
    header: InferHeader,
    jpeg: Bytes,
}

struct Stats {
    inflight: AtomicU64,
    completed: AtomicU64,
    last_frame_id: Mutex<Option<String>>,
    /// Snapshot of the channel's send-side capacity at construction time
    /// (i.e., `max_inflight`). The beacon publishes queue depth as
    /// `max_inflight - current_capacity` — the only way to derive true
    /// in-buffer count from tokio's `Sender::capacity()`.
    max_inflight: u64,
    /// Live handle to the frame-tx side. We keep it here purely so the
    /// beacon thread can ask for `capacity()`.
    frame_tx_capacity: Mutex<Option<mpsc::Sender<PendingFrame>>>,
}

impl Stats {
    fn new(max_inflight: usize) -> Self {
        Self {
            inflight: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            last_frame_id: Mutex::new(None),
            max_inflight: max_inflight as u64,
            frame_tx_capacity: Mutex::new(None),
        }
    }
    fn beacon_json(&self) -> String {
        let inflight = self.inflight.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let last = self.last_frame_id.lock().unwrap().clone();
        let queue_depth = {
            // Channel may have been dropped; if so, depth is 0.
            let guard = self.frame_tx_capacity.lock().unwrap();
            guard
                .as_ref()
                .map(|tx| self.max_inflight.saturating_sub(tx.capacity() as u64))
                .unwrap_or(0)
        };
        serde_json::json!({
            "type": "beacon",
            "queue_depth": queue_depth,
            "inflight": inflight,
            "completed_total": completed,
            "last_completed_frame_id": last,
            "model_state": "ready",
            "max_inflight": self.max_inflight,
        })
        .to_string()
    }
    fn store_last(&self, fid: &str) {
        *self.last_frame_id.lock().unwrap() = Some(fid.to_string());
    }
    fn set_frame_tx(&self, tx: mpsc::Sender<PendingFrame>) {
        *self.frame_tx_capacity.lock().unwrap() = Some(tx);
    }
}

/// Validate a single WebSocket binary message and produce a PendingFrame
/// ready for the worker. Returns the failing ServerError plus the
/// frame_id if it was parseable from the header.
async fn process_binary(
    bytes: Bytes,
    max_jpeg_bytes: usize,
    max_image_dim: u16,
) -> Result<PendingFrame, (ServerError, Option<String>)> {
    // ---- 1. Length-prefix sanity --------------------------------------
    if bytes.len() < 4 {
        return Err((ServerError::InvalidRequest(format!(
            "WS binary frame is {} bytes but a 4-byte BE u32 length-prefix \
             header is required (see docs/CLIENT_PROTOCOL.md#frame-binary)",
            bytes.len()
        )), None));
    }
    let header_len = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    if header_len == 0 {
        return Err((ServerError::InvalidRequest(
            "header_len prefix is 0; the JSON header is mandatory".into(),
        ), None));
    }
    if 4 + header_len > bytes.len() {
        return Err((ServerError::InvalidRequest(format!(
            "declared header_len={} extends past total binary frame size {} \
             (header_len must fit AND leave room for the JPEG payload)",
            header_len, bytes.len()
        )), None));
    }
    let header_slice = &bytes[4..4 + header_len];

    // ---- 2. JSON header parse (deny_unknown_fields enforced by serde) -
    let header: InferHeader = match serde_json::from_slice(header_slice) {
        Ok(h) => h,
        Err(e) => return Err((ServerError::InvalidRequest(format!(
            "header JSON parse failed: {e}. Every Frame header must be a \
             JSON object with exactly these keys: type, frame_id, \
             session_id, prompt, generation_mode, jpeg_len, \
             image_color_space, image_encoding. See \
             docs/CLIENT_PROTOCOL.md#frame-binary.",
        )), None)),
    };
    let fid = Some(header.frame_id.clone());

    // ---- 3. Field-by-field validation ----------------------------------
    if header.kind != FrameKind::Frame {
        return Err((ServerError::InvalidRequest(format!(
            "header.type={:?} but binary frames must declare type=\"frame\". \
             The Hello message is the only Text WebSocket message in the \
             protocol — there is no Configure or Cancel.",
            header.kind
        )), fid));
    }
    if header.frame_id.is_empty() {
        return Err((ServerError::InvalidRequest(
            "header.frame_id is empty; frame_id is required and is the only \
             correlation primitive between this Frame and its Result/Error".into(),
        ), fid));
    }
    if header.frame_id.len() > 256 {
        return Err((ServerError::InvalidRequest(format!(
            "header.frame_id length {} > 256 chars (excessive id length \
             rejected to bound log-line size)",
            header.frame_id.len()
        )), fid));
    }
    if header.prompt.is_empty() {
        return Err((ServerError::InvalidRequest(
            "header.prompt is empty; the model requires a non-empty prompt. \
             See /v1/capabilities.preset_prompts for valid prompt forms.".into(),
        ), fid));
    }
    if header.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err((ServerError::InvalidRequest(format!(
            "header.prompt length {} chars > MAX_PROMPT_CHARS={} (the model's \
             tokenizer.model_max_length is 16384 tokens — even on ASCII this \
             cap is generous). See docs/MODEL_CAPABILITIES.md.",
            header.prompt.chars().count(),
            MAX_PROMPT_CHARS,
        )), fid));
    }
    if !matches!(header.generation_mode.as_str(), "fast" | "hybrid" | "slow") {
        return Err((ServerError::InvalidRequest(format!(
            "header.generation_mode={:?} is not one of \"fast\" | \"hybrid\" \
             | \"slow\". There is no default — every Frame must commit to a \
             mode. See docs/MODEL_CAPABILITIES.md#generation-modes.",
            header.generation_mode
        )), fid));
    }
    if header.image_color_space != "RGB" {
        return Err((ServerError::InvalidImage(format!(
            "header.image_color_space={:?} but the only supported value is \
             \"RGB\" (the model's processor expects RGB and the server \
             will NOT auto-convert from any other color space)",
            header.image_color_space
        )), fid));
    }
    if header.image_encoding != "jpeg" {
        return Err((ServerError::InvalidImage(format!(
            "header.image_encoding={:?} but the only supported value is \
             \"jpeg\"",
            header.image_encoding
        )), fid));
    }

    // ---- 4. Payload size + JPEG header validation ---------------------
    let jpeg = bytes.slice(4 + header_len..);
    if jpeg.len() != header.jpeg_len {
        return Err((ServerError::InvalidImage(format!(
            "header.jpeg_len={} != actual payload length {} (these must \
             match exactly so that mid-stream framing errors are caught \
             at the boundary, not as a confusing decode failure later)",
            header.jpeg_len, jpeg.len()
        )), fid));
    }
    if jpeg.is_empty() {
        return Err((ServerError::InvalidImage(
            "JPEG payload is zero bytes (a Frame must carry a JPEG image)".into(),
        ), fid));
    }
    if jpeg.len() > max_jpeg_bytes {
        return Err((ServerError::InvalidImage(format!(
            "JPEG payload {} bytes exceeds server cap LA_MAX_JPEG_BYTES={} \
             (configurable in scripts/lib/versions.sh)",
            jpeg.len(), max_jpeg_bytes
        )), fid));
    }
    if !jpeg::is_jpeg(&jpeg) {
        return Err((ServerError::InvalidImage(format!(
            "payload first bytes [{:02X}, {:02X}, {:02X}] are not the JPEG \
             SOI marker FF D8 FF; we accept only baseline JPEG with the \
             standard signature",
            jpeg.first().copied().unwrap_or(0),
            jpeg.get(1).copied().unwrap_or(0),
            jpeg.get(2).copied().unwrap_or(0),
        )), fid));
    }

    let jpeg_for_check = jpeg.clone();
    let dims = tokio::task::spawn_blocking(move || {
        jpeg::read_dimensions_blocking(&jpeg_for_check)
    })
    .await;
    let (w, h) = match dims {
        Ok(Ok(d)) => d,
        Ok(Err(s)) => return Err((ServerError::InvalidImage(format!(
            "JPEG header parse failed: {s} (the SOI marker matched but the \
             JPEG structure is malformed — check the encoder output)"
        )), fid)),
        Err(e) => return Err((ServerError::Internal(format!(
            "spawn_blocking for jpeg_decoder join error: {e} (this is a \
             tokio runtime issue, not a client error)"
        )), fid)),
    };
    if w < MIN_IMAGE_DIM || h < MIN_IMAGE_DIM {
        return Err((ServerError::InvalidImage(format!(
            "image dimensions {}x{} below MIN_IMAGE_DIM={} (a useful input \
             must occupy at least one LLM token in the model's 28px grid)",
            w, h, MIN_IMAGE_DIM
        )), fid));
    }
    if w > max_image_dim || h > max_image_dim {
        return Err((ServerError::InvalidImage(format!(
            "image dimensions {}x{} exceed LA_MAX_IMAGE_DIM={} \
             (configurable in scripts/lib/versions.sh; the model's preprocessor \
             rescales anything above ~2240px square to fit its 25600-patch cap)",
            w, h, max_image_dim
        )), fid));
    }

    Ok(PendingFrame { header, jpeg })
}

/// Build the type:"result" body from a successful worker response.
/// Strips the worker's transport-layer `ok` flag and stamps the canonical
/// shape (type=result, frame_id=<client>, …other worker keys passed through).
fn build_result_body(mut v: serde_json::Value, frame_id: &str) -> String {
    if let serde_json::Value::Object(ref mut map) = v {
        map.remove("ok");
        map.insert("type".into(), serde_json::json!("result"));
        map.insert("frame_id".into(), serde_json::json!(frame_id));
    }
    v.to_string()
}

/// Build the type:"error" body from a worker {ok:false, code, error_type, …}
/// response. Keeps the worker's `error_type` taxonomy (e.g. "invalid_image",
/// "worker_error") intact so the client uses ONE error schema across the
/// whole stack.
fn build_worker_error_body(v: serde_json::Value, frame_id: &str) -> String {
    let code = v.get("code").and_then(|x| x.as_u64()).unwrap_or(500) as u16;
    let error_type = v
        .get("error_type")
        .and_then(|x| x.as_str())
        .unwrap_or("worker_error");
    let message = v
        .get("message")
        .and_then(|x| x.as_str())
        .unwrap_or("worker returned ok:false with no message field");
    let retriable = v
        .get("retriable")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    serde_json::json!({
        "type":       "error",
        "code":       code,
        "error_type": error_type,
        "message":    message,
        "retriable":  retriable,
        "frame_id":   frame_id,
    })
    .to_string()
}

fn make_proto_error(error_type: &str, msg: &str) -> String {
    serde_json::json!({
        "type": "error",
        "code": 400,
        "error_type": error_type,
        "message": msg,
        "retriable": false,
        "frame_id": serde_json::Value::Null,
    })
    .to_string()
}

/// Send an error JSON frame, then a Close frame with the given code,
/// then close the sink. Used in the hello-rejection paths where we
/// want the client to see both the structured error and a meaningful
/// WebSocket close code.
async fn send_error_and_close(
    ws_tx: &mut SplitSink<WebSocket, Message>,
    error_type: &str,
    msg: &str,
    code: u16,
) {
    let _ = ws_tx
        .send(Message::Text(make_proto_error(error_type, msg).into()))
        .await;
    send_close(ws_tx, code, error_type).await;
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

fn error_frame(err: &ServerError, frame_id: Option<&str>) -> String {
    serde_json::json!({
        "type":       "error",
        "code":       err.status().as_u16(),
        "error_type": err.error_type(),
        "message":    err.to_string(),
        "retriable":  matches!(err, ServerError::WorkerUnavailable(_)),
        "frame_id":   frame_id,
    })
    .to_string()
}
