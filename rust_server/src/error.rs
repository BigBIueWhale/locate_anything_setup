use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Top-level Rust-side server error. Maps to a JSON body with a stable
/// `error_type` string and a numeric `code` matching the HTTP status.
/// The same shape is reused for the WebSocket error frame, so clients
/// have ONE schema to implement.
///
/// Note: errors REPORTED BY THE PYTHON WORKER do not go through this enum.
/// They arrive as `{ok: false, error_type, code, message, retriable}` from
/// the worker and are re-emitted verbatim by `ws::build_worker_error_body`,
/// preserving the worker's own taxonomy (e.g. `error_type: "worker_error"`
/// for tier-c failures like CUDA OOM). This enum covers only Rust-side
/// validation and transport errors.
#[derive(thiserror::Error, Debug)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("invalid image: {0}")]
    InvalidImage(String),

    #[error("worker unavailable: {0}")]
    WorkerUnavailable(String),

    #[error("worker protocol error: {0}")]
    WorkerProtocol(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl ServerError {
    pub fn status(&self) -> StatusCode {
        match self {
            ServerError::InvalidRequest(_)    => StatusCode::BAD_REQUEST,
            ServerError::InvalidImage(_)      => StatusCode::BAD_REQUEST,
            ServerError::WorkerUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ServerError::WorkerProtocol(_)    => StatusCode::BAD_GATEWAY,
            ServerError::Internal(_)          => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_type(&self) -> &'static str {
        match self {
            ServerError::InvalidRequest(_)    => "invalid_request",
            ServerError::InvalidImage(_)      => "invalid_image",
            ServerError::WorkerUnavailable(_) => "worker_unavailable",
            ServerError::WorkerProtocol(_)    => "worker_protocol",
            ServerError::Internal(_)          => "internal_error",
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = json!({
            "code":       status.as_u16(),
            "error_type": self.error_type(),
            "message":    self.to_string(),
            "retriable":  matches!(self, ServerError::WorkerUnavailable(_)),
        });
        (status, axum::Json(body)).into_response()
    }
}

impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        ServerError::WorkerUnavailable(format!(
            "Unix-socket I/O to Python worker failed: {} (os_kind={:?}). \
             The Python sidecar may have crashed; check `docker logs`.",
            e, e.kind()
        ))
    }
}

impl From<serde_json::Error> for ServerError {
    fn from(e: serde_json::Error) -> Self {
        ServerError::WorkerProtocol(format!(
            "JSON decode of worker response failed at line {}, column {}: {}. \
             This indicates the Python worker emitted malformed JSON — almost \
             certainly a bug; please open an issue with the surrounding logs.",
            e.line(), e.column(), e
        ))
    }
}
