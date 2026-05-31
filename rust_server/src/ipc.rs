//! Rust → Python worker IPC over Unix domain socket.
//!
//! Wire format: `tokio_util::codec::LengthDelimitedCodec` with default
//! 4-byte big-endian u32 length prefix. The Rust side sends:
//!     [ length(4) ][ header JSON ]
//!     [ length(4) ][ JPEG bytes  ]
//! Python responds with:
//!     [ length(4) ][ response JSON ]
//!
//! Frame inference: the header JSON carries `kind:"frame"` plus the
//! fields from `InferHeader` (frame_id, prompt, generation_mode, jpeg_len).
//! Control queries (`{"kind":"capabilities"}` / `{"kind":"info"}`) carry
//! the kind only and follow with an empty-length JPEG frame so framing
//! stays uniform.
//!
//! Worker response envelope: every body carries a top-level `ok:bool`
//! discriminator the Rust side strips on the way out. `ok:true` → forward
//! as `type:"result"`; `ok:false` → forward as `type:"error"`. The
//! discriminator is internal to the UDS hop and never reaches the client.

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use tokio::net::UnixStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use crate::error::ServerError;
use crate::protocol::InferHeader;

pub struct WorkerConn {
    framed: Framed<UnixStream, LengthDelimitedCodec>,
}

/// Worker round-trip outcome. The `ok` discriminator has been stripped
/// before the body reaches the caller — both variants carry the body
/// the WS handler forwards verbatim to the client (after stamping
/// type + frame_id).
pub enum InferOutcome {
    Success(serde_json::Value),
    WorkerError(serde_json::Value),
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

    /// Send (header, jpeg_bytes) and await one JSON response frame from
    /// the Python worker.
    ///
    /// The `Ok` variants both carry a body the WS handler forwards verbatim
    /// (after stamping `type` + `frame_id`); only `Err` is a transport
    /// failure that should close the WS — the framed UDS may have desynced.
    ///
    /// `prompt_task` is the wire name of the canonical template the prompt
    /// was classified as (see `prompt_validator::TemplateKind::wire_name`).
    /// It is forwarded to the worker so the parser can drop off-shape
    /// output per the trained task→shape contract (e.g. a 4-coord box
    /// returned for a Point template is dropped before reaching the
    /// client). This is NOT exposed in the client-facing InferHeader
    /// schema — it is derived server-side from the validated prompt.
    pub async fn infer(
        &mut self,
        header: &InferHeader,
        jpeg: Bytes,
        prompt_task: &'static str,
    ) -> Result<InferOutcome, ServerError> {
        let header_json = serde_json::to_string(&serde_json::json!({
            "kind":            "frame",
            "frame_id":        &header.frame_id,
            "prompt":          &header.prompt,
            "prompt_task":     prompt_task,
            "generation_mode": &header.generation_mode,
            "jpeg_len":        header.jpeg_len,
        })).map_err(|e| {
            ServerError::Internal(format!(
                "could not serialize InferHeader to JSON for worker IPC: {e}"
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
        let mut v: serde_json::Value = serde_json::from_slice(&resp)?;
        match v.as_object_mut().and_then(|m| m.remove("ok")).and_then(|x| x.as_bool()) {
            Some(true)  => Ok(InferOutcome::Success(v)),
            Some(false) => Ok(InferOutcome::WorkerError(v)),
            None => Err(ServerError::WorkerProtocol(format!(
                "worker response missing required `ok` boolean field; \
                 received keys: {:?}",
                v.as_object().map(|m| m.keys().collect::<Vec<_>>())
            ))),
        }
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
