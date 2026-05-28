use clap::Parser;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum LogFormat {
    Json,
    Pretty,
}

/// `la_server` — HTTP + WebSocket frontend for LocateAnything-3B.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Bind address (host:port). REQUIRED — no default. The Docker
    /// entrypoint always passes this; outside Docker the operator is
    /// expected to declare an explicit bind. We will not guess.
    #[arg(long, env = "LA_BIND")]
    pub bind: String,

    /// Path of the Unix domain socket the Python worker listens on.
    /// REQUIRED — no default. The container's entrypoint passes this
    /// in via env; nothing else should be calling la_server directly.
    #[arg(long, env = "LA_IPC_SOCKET")]
    pub worker_socket: PathBuf,

    /// Max seconds to wait for the Python worker socket to appear at
    /// process startup. The Python worker performs validate_startup
    /// (file SHA-256 verification of 10 .py files plus weight size
    /// check, ~few s), model load (~30-60s for 3B bf16), and a 6-run
    /// boot calibration (~30-60s) before opening the UDS. 240s gives
    /// margin and matches the Docker healthcheck's start_period.
    #[arg(long, env = "LA_WORKER_BOOT_TIMEOUT", default_value = "240")]
    pub worker_socket_timeout_secs: u64,

    /// Maximum accepted JPEG payload size (bytes). Anything larger is
    /// rejected with a per-request error — no buffering, no chunking,
    /// no fallback. REQUIRED via env (entrypoint passes LA_MAX_JPEG_BYTES).
    #[arg(long, env = "LA_MAX_JPEG_BYTES")]
    pub max_jpeg_bytes: usize,

    /// Maximum in-flight frames per WebSocket connection. REQUIRED via
    /// env (entrypoint passes LA_MAX_INFLIGHT from versions.sh).
    #[arg(long, env = "LA_MAX_INFLIGHT")]
    pub max_inflight: usize,

    /// Hard upper bound on JPEG width or height in pixels. The model's
    /// native-resolution policy is governed by the 25600 ViT-patch cap;
    /// above ~2240 px square the preprocessor force-rescales (operating
    /// outside the trained policy). The Rust frontend rejects anything
    /// above this so the model is always given an input it can process
    /// without internal rescale. REQUIRED via env.
    #[arg(long, env = "LA_MAX_IMAGE_DIM")]
    pub max_image_dim: u16,

    /// Log output format.
    #[arg(long, env = "LA_LOG_FORMAT", default_value = "json")]
    pub log_format: LogFormat,
}
