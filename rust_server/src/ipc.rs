//! Rust → Python worker IPC over Unix domain socket.
//!
//! Wire format: `tokio_util::codec::LengthDelimitedCodec` with default
//! 4-byte big-endian u32 length prefix. The Rust side sends:
//!     [ length(4) ][ header JSON ]
//!     [ length(4) ][ JPEG bytes  ]
//! Python responds with:
//!     [ length(4) ][ response JSON ]
//!
//! Frame inference: the header JSON carries `kind:"frame"` plus
//! `frame_id`, the COMPILED `prompt` string, the `prompt_task` wire-name,
//! the serialized `generation_mode`, and `jpeg_len`. NOTE: the worker
//! receives the assembled prompt STRING (built by `prompt_validator::compile`
//! from the client's typed request) — the typed `PromptRequest` never crosses
//! the UDS; the prompt/prompt_task/generation_mode header shape is unchanged.
//! Control queries (`{"kind":"capabilities"}` / `{"kind":"info"}`) carry the
//! kind only and follow with an empty-length JPEG frame so framing stays
//! uniform.
//!
//! Worker response envelope: every inference body carries a top-level
//! `ok:bool` discriminator (internal to the UDS hop; never reaches the
//! client). The rest of the body is the client-facing A.2 shape — a flat
//! tagged union on `type` — which the Rust edge DESERIALIZES into the typed
//! [`crate::protocol::Response`]. Egress is therefore schema-enforced: a
//! drifted worker field (a stale `off_shape_count`, a missing `latency_ms`,
//! a duplicate key, an `ok`/`type` contradiction) fails loud here instead of
//! being forwarded blindly. `query_capabilities`/`query_info` stay
//! `serde_json::Value` — they are control-plane, deliberately untyped.

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use tokio::net::UnixStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::error::ServerError;
use crate::protocol::{GenerationMode, Response};

pub struct WorkerConn {
    framed: Framed<UnixStream, LengthDelimitedCodec>,
}

impl WorkerConn {
    pub async fn connect(path: &Path) -> Result<Self, std::io::Error> {
        let stream = UnixStream::connect(path).await?;
        let codec = LengthDelimitedCodec::builder()
            .length_field_length(4)         // 4-byte prefix
            .big_endian()                   // also the default; explicit for clarity
            .max_frame_length(16 * 1024 * 1024)
            .new_codec();
        Ok(Self { framed: Framed::new(stream, codec) })
    }

    /// Send (header, jpeg_bytes) and await one JSON response frame from the
    /// Python worker, deserialized into the typed [`Response`].
    ///
    /// `Ok(Response)` is the client-facing reply (a `boxes`/`points`/
    /// `abstained` success OR a `Response::Error`); the WS handler serializes
    /// it straight to the client. `Err` is a transport / protocol failure that
    /// should close the WS — the framed UDS may have desynced, or the worker
    /// emitted a body that violates the A.2 schema.
    ///
    /// `prompt` is the COMPILED trained string and `prompt_task` is its
    /// wire-name (one of `prompt_validator::TemplateKind::wire_name`), both
    /// produced by `prompt_validator::compile` from the client's typed
    /// request. `generation_mode` is serialized to its snake_case wire form.
    /// The worker uses `prompt_task` to route off-shape model output per the
    /// trained task→shape contract (worker/inference.py::EXPECTED_SHAPE).
    pub async fn infer(
        &mut self,
        frame_id: &str,
        prompt: &str,
        prompt_task: &'static str,
        generation_mode: GenerationMode,
        jpeg_len: usize,
        jpeg: Bytes,
    ) -> Result<Response, ServerError> {
        // Serialize the generation mode to its wire string ("fast"/"hybrid"/
        // "slow") via serde, so the IPC string is the single source of truth.
        let mode_wire = serde_json::to_value(generation_mode).map_err(|e| {
            ServerError::Internal(format!(
                "could not serialize generation_mode for worker IPC: {e}"
            ))
        })?;
        let header_json = serde_json::to_string(&serde_json::json!({
            "kind":            "frame",
            "frame_id":        frame_id,
            "prompt":          prompt,
            "prompt_task":     prompt_task,
            "generation_mode": mode_wire,
            "jpeg_len":        jpeg_len,
        })).map_err(|e| {
            ServerError::Internal(format!(
                "could not serialize IPC frame header to JSON for worker: {e}"
            ))
        })?;
        self.framed.send(Bytes::from(header_json)).await.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "send(header) to Python worker over UDS failed: {e}"
            ))
        })?;
        self.framed.send(jpeg).await.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "send(jpeg) to Python worker over UDS failed: {e}"
            ))
        })?;
        let resp = self.framed.next().await.ok_or_else(|| {
            ServerError::WorkerUnavailable(
                "Python worker closed UDS before returning a response. \
                 This usually means the worker crashed mid-request — check \
                 `docker logs locate-anything` for a Python traceback.".into(),
            )
        })?.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "UDS read from Python worker failed: {e}"
            ))
        })?;

        // The body is `{ok:bool, <A.2 fields...>}`. Strip the internal `ok`
        // discriminator, then deserialize the rest into the typed Response.
        // `ok` is required (fail-loud if absent) and must AGREE with the
        // variant tag (`ok:false` ⇔ Error) — a contradiction is a worker bug.
        let mut v: serde_json::Value = serde_json::from_slice(&resp)?;
        let ok = match v.as_object_mut().and_then(|m| m.remove("ok")) {
            Some(serde_json::Value::Bool(b)) => b,
            Some(other) => {
                return Err(ServerError::WorkerProtocol(format!(
                    "worker response `ok` field is not a boolean (got {other}); \
                     it is the internal success discriminator and must be a bool"
                )));
            }
            None => {
                return Err(ServerError::WorkerProtocol(format!(
                    "worker response missing required `ok` boolean field; \
                     received keys: {:?}",
                    v.as_object().map(|m| m.keys().collect::<Vec<_>>())
                )));
            }
        };

        // Egress schema enforcement: the remaining body must be exactly one
        // A.2 variant (deny_unknown_fields + flatten reject drift here).
        let response: Response = serde_json::from_value(v).map_err(|e| {
            ServerError::WorkerProtocol(format!(
                "worker inference reply does not match the typed Response \
                 (A.2) schema: {e}. The Python worker emitted a body that \
                 drifted from the locked wire contract — almost certainly a \
                 worker bug; check the surrounding logs."
            ))
        })?;

        let is_error = matches!(response, Response::Error(_));
        if ok == is_error {
            return Err(ServerError::WorkerProtocol(format!(
                "worker response `ok`={ok} contradicts its `type` tag \
                 (variant is {}an error). The internal `ok` discriminator and \
                 the A.2 `type` tag must agree.",
                if is_error { "" } else { "not " }
            )));
        }

        Ok(response)
    }

    /// Control-plane request: capabilities, info, or other no-payload kinds.
    pub async fn control(
        &mut self,
        kind: &str,
    ) -> Result<serde_json::Value, ServerError> {
        let header = serde_json::json!({ "kind": kind });
        self.framed.send(Bytes::from(header.to_string())).await.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "send(control header kind={kind:?}) over UDS failed: {e}"
            ))
        })?;
        // Empty payload frame keeps the protocol uniform (header + payload).
        self.framed.send(Bytes::new()).await.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "send(control empty payload kind={kind:?}) over UDS failed: {e}"
            ))
        })?;
        let resp = self.framed.next().await.ok_or_else(|| {
            ServerError::WorkerUnavailable(format!(
                "Python worker closed UDS during control kind={kind:?} \
                 before returning a response"
            ))
        })?.map_err(|e| {
            ServerError::WorkerUnavailable(format!(
                "UDS read for control kind={kind:?} failed: {e}"
            ))
        })?;
        let v: serde_json::Value = serde_json::from_slice(&resp)?;
        Ok(v)
    }
}

pub async fn query_capabilities(socket_path: &Path) -> Result<serde_json::Value, ServerError> {
    let mut conn = WorkerConn::connect(socket_path)
        .await
        .map_err(|e| ServerError::WorkerUnavailable(e.to_string()))?;
    conn.control("capabilities").await
}

pub async fn query_info(socket_path: &Path) -> Result<serde_json::Value, ServerError> {
    let mut conn = WorkerConn::connect(socket_path)
        .await
        .map_err(|e| ServerError::WorkerUnavailable(e.to_string()))?;
    conn.control("info").await
}
