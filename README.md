# LocateAnything-3B inference server

A reproducible, pinned-everything Docker setup for NVIDIA's
[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
(released 2026-05-26) on an RTX 5090 (Blackwell, sm_120). HTTP +
WebSocket frontend in Rust, model inference in Python; both run inside
one container and talk over a Unix domain socket.

This README only covers **how to run it**. For everything else — what
the model can and cannot do, the pixel-to-token math, why each pin is
the version it is, the wire protocol clients must speak, the
DMZ-exposure security model — see [`docs/`](./docs/).

---

## TL;DR

```bash
bash ./setup.sh
```

That's the whole thing. On a fresh box it takes ~25 minutes (~20
minutes is the `flash-attn` source build for sm_120 inside the Docker
image). On every subsequent run it's a no-op verify-and-skip.

After it finishes, the service is reachable at
**`http://127.0.0.1:8765`** (loopback only — see
[`docs/SECURITY.md`](./docs/SECURITY.md) for why).

To smoke-test the three status endpoints from another shell:

```bash
curl -s http://127.0.0.1:8765/v1/health        | jq .
curl -s http://127.0.0.1:8765/v1/capabilities  | jq .
curl -s http://127.0.0.1:8765/v1/info          | jq .
```

The actual inference surface is **WebSocket only**
(`ws://127.0.0.1:8765/v1/stream`). For an end-to-end client see
[`examples/reference_client.py`](./examples/reference_client.py)
and [`docs/CLIENT_PROTOCOL.md`](./docs/CLIENT_PROTOCOL.md). The
project-internal smoke test
([`scripts/04_smoke_test.sh`](./scripts/04_smoke_test.sh)) drives it
via a minimal Python WS client in a one-off Docker container.

To stop / restart:

```bash
docker stop locate-anything
docker start locate-anything
```

To uninstall (see [`docs/OPERATIONS.md`](./docs/OPERATIONS.md) for the full
lifecycle):

```bash
bash uninstall.sh                              # remove the running container only
bash uninstall.sh --remove-image               # also delete the built Docker image
bash uninstall.sh --remove-weights             # also delete the downloaded model (~7.66 GB)
bash uninstall.sh --purge --yes                # remove everything (image + weights + caches)
```

The default uninstall is conservative — instances only. The Docker image
and the model weights survive so that re-running `setup.sh` is fast (and
works offline, since neither needs re-downloading).

---

## What `setup.sh` actually does

`setup.sh` orchestrates four steps. Each step lives in its own script
under `scripts/`, each step is idempotent, and each step **fails loud
and refuses to fall back** on any condition the project doesn't
support.

| Step | Script                                          | What it does |
|---|---|---|
| 0 | [`scripts/00_validate_host.sh`](./scripts/00_validate_host.sh) | Verify OS = Ubuntu 24.04 LTS, driver ≥ 595.45.04, GPU is sm_120 (Blackwell), ≥ 24 GiB VRAM, Docker 29.x, the `nvidia` Docker runtime is registered and GPU passthrough works, free disk ≥ 30 GiB, host port 8765 is free, and that you are *not* running as root. |
| 1 | [`scripts/01_download_weights.sh`](./scripts/01_download_weights.sh) | Pull the pinned HF revision of `nvidia/LocateAnything-3B` into `./models/LocateAnything-3B/` using `huggingface_hub` in a one-off `python:3.12-slim-bookworm` container (no host-side Python). Generates a synthetic calibration image used by the boot self-test. Regenerates `rust_server/Cargo.lock` if missing. |
| 2 | [`scripts/02_build_image.sh`](./scripts/02_build_image.sh) | Builds the [Dockerfile](./Dockerfile). Multi-stage: Rust 1.95 builder → CUDA 13.0.3 runtime. Long step: ~20 min `flash-attn==2.8.4` source build with `FLASH_ATTN_CUDA_ARCHS=120`. Every package pin comes through as `--build-arg` from [`scripts/lib/versions.sh`](./scripts/lib/versions.sh). |
| 3 | [`scripts/03_start_service.sh`](./scripts/03_start_service.sh) | Starts the container with `--gpus all`, mounts the weights read-only, publishes to **`127.0.0.1:8765`** only, drops all Linux capabilities, runs read-only root with a 512 MiB tmpfs on `/tmp`, and waits up to 4 minutes for `/v1/health` to flip to `healthy`. |
| 4 | [`scripts/04_smoke_test.sh`](./scripts/04_smoke_test.sh) | Hits `/v1/health`, `/v1/capabilities`, `/v1/info` with `curl`, then drives one round-trip through `WS /v1/stream` via a minimal Python WebSocket client in a one-off `python:3.12-slim` container. Asserts the model loaded, calibration ran, and a JPEG → JSON round-trip works. |

The full pin table — every version of every package, every base image
tag, every model commit SHA — lives in
[`scripts/lib/versions.sh`](./scripts/lib/versions.sh). One file, one
truth. To upgrade a component, edit one line there and re-run
`setup.sh`.

---

## Host prerequisites

This setup runs on top of the personal-server stack documented at
[`/home/user/Desktop/personal_server/README.md`](../personal_server/README.md).
That covers driver, CUDA toolkit, Docker, NVIDIA Container Toolkit,
and developer tools. If you haven't run those scripts yet, run them
first.

What `00_validate_host.sh` strictly requires:

| Component       | Required pin                              | Where it's installed |
|-----------------|-------------------------------------------|----------------------|
| Ubuntu          | 24.04 LTS (`noble`)                       | OS install           |
| NVIDIA driver   | ≥ 595.45.04                               | personal_server §10  |
| GPU             | RTX 5090 / Blackwell **sm_120**           | hardware             |
| GPU VRAM        | ≥ 24 GiB                                  | hardware             |
| Docker          | 29.x                                      | personal_server §12  |
| nvidia ctk      | 1.19.0                                    | personal_server §13  |
| Disk free       | ≥ 30 GiB at the project directory         | OS                   |
| Host port 8765  | unbound                                   | OS                   |
| Rust (host)     | any (only to generate `Cargo.lock`)       | personal_server §16  |

If any check fails, the script aborts with the exact pin it expected
and the value it actually found. Do not modify the script to skip a
check — fix the underlying issue, then re-run.

---

## Where things live

```
locate_anything_setup/
├── README.md                       # this file
├── setup.sh                        # orchestrator
├── Dockerfile                      # multi-stage (Rust + CUDA Python)
├── docker-compose.yml              # optional alternative to scripts/03
├── container/
│   └── entrypoint.sh               # supervises both child processes
├── rust_server/                    # Rust HTTP + WebSocket frontend
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/
│       ├── main.rs                 # routes, signal handling
│       ├── config.rs               # CLI args
│       ├── state.rs
│       ├── error.rs                # structured error type → JSON body
│       ├── protocol.rs             # wire-protocol types (InferHeader + constants)
│       ├── jpeg.rs                 # JPEG header validation
│       ├── ipc.rs                  # Unix-socket client to Python
│       └── ws.rs                   # /v1/stream WebSocket handler
├── worker/                         # Python inference sidecar
│   ├── la_worker.py                # asyncio UDS server entrypoint
│   ├── inference.py                # LocateAnything model wrapper
│   ├── parsing.py                  # <box>…</box> regex parsers
│   ├── prompts.py                  # the 7 canonical prompt templates
│   ├── pixel_token_math.py         # resolution → token geometry
│   ├── tiling.py                   # external tiling for tiny objects
│   ├── calibration.py              # boot-time FPS measurement
│   └── validate_startup.py         # GPU, env, weights, flash_attn checks
├── scripts/
│   ├── lib/
│   │   ├── common.sh               # logging, traps, version loader
│   │   └── versions.sh             # the ONE source of pinned versions
│   ├── 00_validate_host.sh
│   ├── 01_download_weights.sh
│   ├── 02_build_image.sh
│   ├── 03_start_service.sh
│   └── 04_smoke_test.sh
├── test_data/
│   └── calibration.jpg             # generated by step 1
├── models/                         # weights live here (bind-mounted RO into the container)
└── docs/
    ├── ARCHITECTURE.md
    ├── CLIENT_PROTOCOL.md
    ├── DRONE_DETECTION.md
    ├── MODEL_CAPABILITIES.md
    ├── PIXEL_TO_TOKEN_MATH.md
    ├── PINNED_VERSIONS.md
    └── SECURITY.md
```

---

## Diagnosing failures

Every script's first line is `set -Eeuo pipefail` and traps on `ERR`.
A failure prints the exact line that aborted plus a `[FAIL]` message
explaining what was expected. Common categories:

* **Host validation** — fix the missing thing (driver, GPU, Docker
  GPU passthrough) and re-run.
* **Weight download** — usually a transient HF outage; re-run.
  `01_download_weights.sh` is idempotent.
* **Docker build** — flash-attn source compile failures are the
  most common. The image won't fall through to `sdpa` at runtime;
  the build must succeed. Capture full logs with
  `docker build --progress=plain ...` (which `02_build_image.sh`
  already does).
* **Container won't go healthy** — `docker logs locate-anything`.
  Most likely: weight directory empty, GPU not reachable from inside
  the container, or `flash_attn` import failure. Each prints a
  pointed message from `worker/validate_startup.py`.
* **Smoke test fails** — the service is running but the response is
  off. Check `docker logs locate-anything` and the structured error
  the JSON body returned.

There are no auto-fixes. Each error includes enough context to
diagnose by hand.

---

## What's next

For an actual drone-detection client, see
[`docs/CLIENT_PROTOCOL.md`](./docs/CLIENT_PROTOCOL.md) for the wire
spec and [`docs/DRONE_DETECTION.md`](./docs/DRONE_DETECTION.md) for the
honest assessment of what this model can and cannot do for FPV-drone
early-warning.
