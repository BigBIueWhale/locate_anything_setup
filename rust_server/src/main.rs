//! la_server — HTTP + WebSocket frontend for the LocateAnything-3B worker.
//!
//! Responsibilities:
//!   • Validate all client input (JPEG header, frame dimensions, JSON schema).
//!   • Forward frames to the Python worker over a Unix domain socket using
//!     LengthDelimitedCodec framing.
//!   • Stream results back to the client over WebSocket.
//!   • Expose capability + health endpoints.
//!
//! No fallbacks anywhere. Every failure path is logged with a structured
//! error code and the client is notified with a typed JSON error frame.

mod config;
mod error;
mod ipc;
mod jpeg;        // header validator, used by ws::process_binary
mod protocol;
mod state;
mod ws;

use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, instrument};
use tracing_subscriber::EnvFilter;

use crate::config::Args;
use crate::state::AppState;

/// Worker thread count for tokio's multi-thread runtime. The default
/// (`num_cpus`) gives 24 on this host — wasteful for an I/O-bound
/// server whose CPU work is ~JPEG header parsing + JSON ser/de.
/// Four is plenty: listener + 3 spawned tasks per WS (reader,
/// processor, writer) × a small number of concurrent connections.
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_tracing(&args.log_format);
    install_panic_hook();

    info!(
        bind = %args.bind,
        socket = %args.worker_socket.display(),
        max_jpeg_bytes = args.max_jpeg_bytes,
        max_inflight = args.max_inflight,
        "la_server starting"
    );

    // Wait for the Python sidecar's Unix socket to appear before opening the
    // HTTP listener. We refuse to serve requests we cannot satisfy.
    wait_for_socket(&args.worker_socket, args.worker_socket_timeout_secs).await?;

    let state: Arc<AppState> = Arc::new(AppState::new(args.clone()));

    // One inference surface — the /v1/stream WebSocket — and three
    // *status* endpoints for orchestration / monitoring. Deliberately NO
    // single-shot HTTP inference path.
    //
    // Layering split: the status routes get a hard 10s timeout (per-
    // route, not per-process) so a wedged worker can't hang `/v1/info`
    // forever. The WS route deliberately gets NO TimeoutLayer — a
    // long-lived stream is the design.
    let status_router = axum::Router::new()
        .route("/v1/health",       axum::routing::get(http_health))
        .route("/v1/capabilities", axum::routing::get(http_capabilities))
        .route("/v1/info",         axum::routing::get(http_info))
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            std::time::Duration::from_secs(10),
        ))
        .layer(RequestBodyLimitLayer::new(64 * 1024));
    let ws_router = axum::Router::new()
        .route("/v1/stream",       axum::routing::get(ws::ws_route));
    let app = status_router
        .merge(ws_router)
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr: SocketAddr = args.bind.parse()?;
    let listener = TcpListener::bind(addr).await?;
    info!(local = %listener.local_addr()?, "listening");

    // Shutdown: wait for SIGINT/SIGTERM, then (a) notify every WS
    // handler so they can send a clean Close(1001 "going away"), and
    // (b) tell axum to stop accepting new connections and drain
    // in-flight HTTP. Give WS handlers a 5-second grace window after
    // notification — long enough to flush a final Close frame, short
    // enough that orchestrators don't think we're hung.
    let shutdown_state = state.clone();
    let shutdown = async move {
        wait_for_signal().await;
        info!("shutdown signal received — notifying WS handlers");
        shutdown_state.shutdown.notify_waiters();
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    // Brief drain window for WS handlers to send their Close frames
    // before the runtime drops. Empirically 5s is more than enough
    // for kernel-side TCP send buffers to flush a 30-byte Close frame
    // on loopback.
    info!("HTTP listener drained; allowing WS handlers 5s to flush Close frames");
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    info!("la_server stopped");
    Ok(())
}

/// Install a panic hook that logs via tracing AND flushes stderr before
/// the process aborts. With `panic = "abort"` in the release profile,
/// the OS-level stderr buffer (block-buffered when stderr is a pipe
/// to docker logs) would otherwise lose the last lines of the panic
/// message.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!(panic = %info, "panic in la_server");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        default(info);
    }));
}

async fn wait_for_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let term = async {
        let mut s = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
        .expect("install SIGTERM handler");
        s.recv().await;
    };
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

fn init_tracing(format: &config::LogFormat) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,la_server=debug"));
    let sub = tracing_subscriber::fmt().with_env_filter(filter);
    match format {
        config::LogFormat::Json   => sub.json().init(),
        config::LogFormat::Pretty => sub.init(),
    }
}

#[instrument(skip_all, fields(path = %path.display()))]
async fn wait_for_socket(path: &std::path::Path, timeout_secs: u64) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if tokio::fs::metadata(path).await.is_ok() {
            // Probe a real connection — file existing is not enough.
            if tokio::net::UnixStream::connect(path).await.is_ok() {
                info!("worker socket ready");
                return Ok(());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "Python worker did not expose {} within {}s. The model load \
                 probably failed; check the worker logs. la_server REFUSES \
                 to start without a live worker — no fallback.",
                path.display(), timeout_secs
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

// ---- Stub HTTP handlers — real impls in src/ws.rs and IPC layer ----------
//
// `/v1/health` is a DEEP health probe — it does a real round-trip through
// the Python worker (an `info` control request, which the worker handles
// inside its asyncio event loop and which calls into torch.cuda).
//
// What this CATCHES:
//   * Worker process crash         (UDS connect fails → 503)
//   * Asyncio event loop deadlock  (no response within 5s → 503)
//   * UDS socket gone              (connect fails → 503)
//   * CUDA driver gone             (torch.cuda.get_device_name throws)
//
// What this does NOT catch — known limitations, documented honestly:
//   * A CUDA wedge while another connection's inference is mid-flight.
//     The Python worker's `info` handler does not take the model lock,
//     so it remains responsive even while a stuck inference holds it.
//     Detecting that requires either a real inference round-trip (too
//     expensive for a 15-second poll cadence) or a watchdog thread that
//     periodically issues a tiny torch.cuda.synchronize() with timeout.
//     See docs/OPERATIONS.md for the recommended external probe pattern.
//
// Cost is one cheap IPC round-trip per healthcheck (every 15s).
async fn http_health(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    const DEEP_HEALTH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let probe = ipc::query_info(&state.args.worker_socket);
    let outcome = tokio::time::timeout(DEEP_HEALTH_TIMEOUT, probe).await;
    let (healthy, detail) = match outcome {
        Ok(Ok(v)) => {
            let ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
            let model_loaded = v.get("model_loaded").and_then(|x| x.as_bool()).unwrap_or(false);
            (ok && model_loaded, serde_json::json!({"info_ok": ok, "model_loaded": model_loaded}))
        }
        Ok(Err(e)) => (false, serde_json::json!({"error": e.to_string()})),
        Err(_) => (false, serde_json::json!({"error": format!(
            "deep health probe timed out after {:?}; worker is wedged or asyncio loop is deadlocked", DEEP_HEALTH_TIMEOUT)})),
    };
    let body = serde_json::json!({
        "status":  if healthy { "ok" } else { "degraded" },
        "worker":  if healthy { "up" } else { "down" },
        "detail":  detail,
    });
    let code = if healthy {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (code, axum::Json(body))
}

async fn http_capabilities(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>, error::ServerError> {
    // Capabilities come from the worker — single source of truth for the
    // model's spec (max image dims, model commit SHA, supported prompt
    // templates, supported generation modes). We do NOT cache here so a
    // model swap is reflected immediately.
    let raw = ipc::query_capabilities(&state.args.worker_socket).await?;
    Ok(axum::Json(raw))
}

async fn http_info(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>, error::ServerError> {
    let raw = ipc::query_info(&state.args.worker_socket).await?;
    Ok(axum::Json(raw))
}

