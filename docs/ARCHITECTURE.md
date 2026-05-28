# Architecture

```
        ┌─────────────────── Docker container ───────────────────┐
        │                                                        │
client ─┼─→ TCP :8000 ─→ [ Rust la_server ] ──UDS /tmp/la.sock──→│ [ Python la_worker ] ─→ GPU
        │                axum + tokio                            │  asyncio + transformers
        │                                                        │  + flash-attn + custom .generate()
        └────────────────────────────────────────────────────────┘
host published: 127.0.0.1:8765 → container 8000
```

Two processes, two languages, one container. The split is deliberate:

* **Rust frontend** handles all network ingress: HTTP, WebSocket
  framing, request validation (JPEG magic / dimensions / payload
  size), structured error mapping, backpressure via bounded tokio
  channels.

* **Python sidecar** holds the model and runs inference. It is the
  authoritative source for capabilities (model SHA, max image dims,
  generation parameters, the pixel-token math, the calibration
  results). It exposes no HTTP — it speaks only the binary
  length-prefixed protocol described in
  [CLIENT_PROTOCOL.md](./CLIENT_PROTOCOL.md).

## Why this split (vs single-process Python)

* Memory-safe, compile-time-validated handling of untrusted client
  input. JPEG parsing and JSON parsing are the two boundary attack
  surfaces; both happen in Rust before a single byte reaches the
  model.

* Backpressure semantics are explicit in the Rust side. A bounded
  `tokio::sync::mpsc::channel` plus the
  `axum::extract::ws::SplitStream` reader pauses on
  `frame_tx.send().await` when the GPU is busy; that pause
  propagates back to TCP flow control and ultimately to the client's
  send buffer. The server never silently drops a frame.

* The model is Python-only (the upstream `modeling_locateanything.py`
  is loaded via `trust_remote_code=True`); there is no Rust path for
  the model itself. Isolating it as a sidecar keeps that hard
  dependency localized.

## Concurrency

* Multiple WebSocket connections are accepted concurrently by Rust.
  Each connection has its own bounded mpsc pair.
* The Python sidecar serializes all `model.generate()` calls behind
  an `asyncio.Lock`. PyTorch on a single GPU cannot run two
  generations in parallel — the lock makes that an explicit FIFO
  queue rather than an undefined-behavior race.
* Frame ordering inside one WebSocket connection is preserved by the
  Rust mpsc; cross-connection ordering is FIFO at the Python lock.

## Process supervision

`container/entrypoint.sh` is a small Bash script (PID 1 under tini)
that:

1. Starts the Python worker.
2. Starts the Rust frontend once `LA_IPC_SOCKET` exists (the frontend
   has its own `wait_for_socket()` loop with a 240 s deadline that
   matches the Docker healthcheck's `start_period`).
3. `wait -n` for either to exit, then SIGTERM the other.

If either child exits the container exits — Docker's `restart unless-stopped`
brings it back.

## Image layout

Multi-stage:

* **Stage 1 (`rust_builder`)**: `rust:1.95-bookworm`. `cargo build
  --release --locked`. Produces `la_server` (~10 MiB stripped binary).
* **Stage 2 (`runtime`)**: `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`.
  Installs Python 3.12, the model-mandated Python deps, builds
  `flash-attn==2.8.4` from source with `FLASH_ATTN_CUDA_ARCHS=120`,
  copies in the Rust binary, copies the Python worker code.

The image is **read-only at runtime** (`docker run --read-only`).
The only writable mount is `/tmp` (tmpfs, 512 MiB, `noexec,nosuid,nodev`)
for the Unix domain socket — no execution allowed even if someone
managed to write a binary into it. Weights are bind-mounted
read-only from the host.

## What's inside the container at runtime

* `/usr/local/bin/la_server` — Rust binary, PID assigned by entrypoint.
* `/opt/locate_anything/venv/bin/python` — Python 3.12 venv with all
  pinned deps.
* `/opt/locate_anything/worker/` — the Python sidecar code.
* `/opt/locate_anything/model/` — bind-mount of `./models/LocateAnything-3B/`
  on the host (RO).
* `/opt/locate_anything/hf_cache/` — HF download cache (RW, on the host).
* `/opt/locate_anything/test_data/calibration.jpg` — used by the boot
  self-test.
* `/tmp/la.sock` — Unix domain socket between Rust and Python.

## What is NOT in the container

* No SSH, no debugger daemons, no shell waiting for input. The
  container is a single-process tree.
* No host networking. The container has its own veth.
* No exposure of the Python sidecar to the network. The UDS is
  Unix-only, inside the container; the Rust frontend is the only
  process that opens a TCP port.
