//! Rust → Python worker IPC over Unix domain socket.
//!
//! Wire format: `tokio_util::codec::LengthDelimitedCodec` with default
//! 4-byte big-endian u32 length prefix. The Rust side sends:
//!     [ length(4) ][ header JSON ]
//!     [ length(4) ][ JPEG bytes  ]
//! Python responds with:
//!     [ length(4) ][ response JSON ]
//!
//! Sending a capability/info query: header `{"kind":"capabilities"}` or
//! `{"kind":"info"}` followed by an empty-length payload frame. The Python
//! worker keys off the JSON `kind` field rather than payload presence —
//! see worker/la_worker.py.

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

    /// Send (header, jpeg_bytes) and await one JSON response frame.
    pub async fn infer(
        &mut self,
        header: &InferHeader,
        jpeg: Bytes,
    ) -> Result<serde_json::Value, ServerError> {
        let header_json = serde_json::to_string(header).map_err(|e| {
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
        let v: serde_json::Value = serde_json::from_slice(&resp)?;
        Ok(v)
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
