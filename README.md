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
via `docker exec` against the live container — the Python WS client
script (`scripts/lib/smoke_ws_client.py`) runs in the container's
own venv, so no helper container is spun up.

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
| 2 | [`scripts/02_build_image.sh`](./scripts/02_build_image.sh) | Builds the [Dockerfile](./Dockerfile). Multi-stage: Rust 1.95 builder → CUDA 13.0.3 runtime. Long step: ~20 min `flash-attn==2.8.3` source build with `FLASH_ATTN_CUDA_ARCHS=120`. Every package pin comes through as `--build-arg` from [`scripts/lib/versions.sh`](./scripts/lib/versions.sh). |
| 3 | [`scripts/03_start_service.sh`](./scripts/03_start_service.sh) | Starts the container with `--gpus all`, mounts the weights read-only, publishes to **`127.0.0.1:8765`** only, drops all Linux capabilities, runs read-only root with a 512 MiB tmpfs on `/tmp`, and waits up to 4 minutes for `/v1/health` to flip to `healthy`. |
| 4 | [`scripts/04_smoke_test.sh`](./scripts/04_smoke_test.sh) | Hits `/v1/health`, `/v1/capabilities`, `/v1/info` with `curl`, then drives one round-trip through `WS /v1/stream` by `docker exec`-ing `scripts/lib/smoke_ws_client.py` inside the live container. Asserts the model loaded, calibration ran, and a JPEG → JSON round-trip works. |

The full pin table — every version of every package, every base image
tag, every model commit SHA — lives in
[`scripts/lib/versions.sh`](./scripts/lib/versions.sh). One file, one
truth. To upgrade a component, edit one line there and re-run
`setup.sh`.

---

## Host prerequisites

Driver, CUDA toolkit, Docker, NVIDIA Container Toolkit, and developer
tools must already be installed on the host — this project does NOT
touch host packages, only validates the pins below
(`scripts/00_validate_host.sh` enforces). The exact pins live in
[`scripts/lib/versions.sh`](./scripts/lib/versions.sh).

What `00_validate_host.sh` strictly requires:

| Component       | Required pin                              | Install via |
|-----------------|-------------------------------------------|----------------------|
| Ubuntu          | 24.04 LTS (`noble`)                       | OS install           |
| NVIDIA driver   | ≥ 595.45.04                               | host package manager (e.g. `nvidia-driver-595-open`) |
| GPU             | RTX 5090 / Blackwell **sm_120**           | hardware             |
| GPU VRAM        | ≥ 24 GiB                                  | hardware             |
| Docker          | 29.x                                      | host package manager (`docker-ce`)                   |
| nvidia ctk      | 1.19.0                                    | host package manager (`nvidia-container-toolkit`)    |
| Disk free       | ≥ 30 GiB at the project directory         | OS                   |
| Host port 8765  | unbound                                   | OS                   |
| Rust (host)     | any (only to generate `Cargo.lock`)       | host package manager / `rustup` |

If any check fails, the script aborts with the exact pin it expected
and the value it actually found. Do not modify the script to skip a
check — fix the underlying issue, then re-run.

---

## Where things live

```
locate_anything_setup/
├── README.md                       # this file
├── LICENSE                         # MIT (repo code) + NOTICE re: NVIDIA weights
├── setup.sh                        # orchestrator
├── uninstall.sh                    # tiered cleanup (see docs/OPERATIONS.md)
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
│       ├── prompt_validator.rs     # strict trained-correct prompt template gate
│       └── ws.rs                   # /v1/stream WebSocket handler
├── worker/                         # Python inference sidecar
│   ├── __init__.py
│   ├── la_worker.py                # asyncio UDS server entrypoint
│   ├── inference.py                # LocateAnything model wrapper
│   ├── parsing.py                  # <box>…</box> regex parsers
│   ├── prompts.py                  # the 7 canonical prompt templates (single source of truth)
│   ├── pixel_token_math.py         # resolution → token geometry
│   ├── tiling.py                   # external tiling helpers for tiny objects
│   ├── calibration.py              # boot-time FPS measurement on a real drone JPEG
│   └── validate_startup.py         # GPU, env, weights, drift checks
├── scripts/
│   ├── lib/
│   │   ├── common.sh                       # logging, traps, version loader
│   │   ├── versions.sh                     # the ONE source of pinned versions
│   │   ├── smoke_ws_client.py              # minimal WS client used by 04
│   │   └── concurrency_smoke_client.py     # FIFO-fairness probe used by 05
│   ├── 00_validate_host.sh
│   ├── 01_download_weights.sh
│   ├── 02_build_image.sh
│   ├── 03_start_service.sh
│   ├── 04_smoke_test.sh
│   └── 05_concurrency_smoke.sh
├── examples/
│   ├── README.md
│   └── reference_client.py         # documented client patterns
├── test_data/
│   ├── calibration.jpg             # generated by step 1; used by smoke tests
│   ├── drone_sirius.jpg            # default boot-calibration image (public domain)
│   ├── drone_byrobot.jpg           # additional drone test imagery
│   ├── drone_r18.jpg
│   └── cat_negative.jpg            # household-objects test image
├── models/                         # weights live here (bind-mounted RO into the container)
└── docs/
    ├── ARCHITECTURE.md             # multi-stage build + runtime layout
    ├── CLIENT_PROTOCOL.md          # HTTP endpoints + WebSocket /v1/stream wire format
    ├── DRONE_DETECTION.md          # honest assessment + measured throughput
    ├── MODEL_CAPABILITIES.md       # what the model does and does NOT do well
    ├── OPERATIONS.md               # lifecycle, restart, offline, uninstall
    ├── PIXEL_TO_TOKEN_MATH.md      # resolution geometry derivation
    ├── PINNED_VERSIONS.md          # every pin and its reason
    └── SECURITY.md                 # threat model + hardening checklist
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
