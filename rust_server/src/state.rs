use crate::config::Args;
use std::sync::Arc;
use tokio::sync::Notify;

/// Shared application state. Wrap in Arc at the call site for cheap cloning.
pub struct AppState {
    pub args: Args,
    /// Notify signal published when the server begins shutting down.
    /// Every WebSocket handler watches it and, on notification, sends
    /// a graceful Close(1001 "going away") frame to the client before
    /// dropping the connection. Without this, axum's HTTP graceful
    /// shutdown does NOT propagate into already-upgraded WS handlers
    /// — they would be aborted abruptly when the tokio runtime drops.
    pub shutdown: Arc<Notify>,
}

impl AppState {
    pub fn new(args: Args) -> Self {
        Self { args, shutdown: Arc::new(Notify::new()) }
    }
}
