# syntax=docker/dockerfile:1.7
# ============================================================================
# LocateAnything-3B inference container.
#
# Two stages:
#   1) `rust_builder` — compiles the Rust HTTP/WebSocket frontend.
#   2) `runtime`      — CUDA + Python + the model worker. Copies the Rust
#                       binary in from stage 1.
#
# The Rust binary is the only network ingress. The Python sidecar is internal,
# reachable only over /tmp/la.sock (Unix domain socket).
#
# Build time: dominated by flash-attn 2.8.3 source build (~8–15 min on
# an 8-job 24-core/62-GiB host with FLASH_ATTN_CUDA_ARCHS=120).
# Image size: ~12 GiB (mostly torch+cu130 wheels + flash-attn).
# ============================================================================

# ----------------------------------------------------------------------------
# Global ARGs.
#
# Any ARG used in a FROM directive MUST be declared BEFORE the first
# FROM in the Dockerfile so it lives in the global scope. ARGs
# declared between FROM directives are stage-local to the preceding
# stage and are NOT visible to subsequent FROMs — BuildKit refuses
# the build with "base name should not be blank" in that case. We
# learned this the hard way the first time the build actually ran.
#
# If the operator forgets to pass either of these as --build-arg, the
# corresponding FROM below fails cleanly with an unambiguous error
# naming the ARG. We don't add a separate `RUN test -n "${ARG}"`
# guard because the FROM itself is already a strict-fail guard.
# ----------------------------------------------------------------------------
ARG LA_RUST_BUILDER_IMAGE
ARG LA_CUDA_BASE_IMAGE

# ----------------------------------------------------------------------------
# Stage 1 — Rust builder
# ----------------------------------------------------------------------------
FROM ${LA_RUST_BUILDER_IMAGE} AS rust_builder

WORKDIR /work
# Copy only the manifest first to leverage Docker's layer cache:
# `cargo fetch` will be re-run only if Cargo.toml / Cargo.lock change.
COPY rust_server/Cargo.toml rust_server/Cargo.lock ./
# Pre-fetch dependencies. If this fails (e.g., crates.io down), the build
# aborts here rather than mid-compile.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo fetch --locked

COPY rust_server/src ./src
# Release build. `--locked` refuses to update Cargo.lock — pinned build.
# `--frozen` would also refuse to touch the registry; locked is enough.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/target \
    cargo build --release --locked \
 && cp /work/target/release/la_server /work/la_server \
 && /work/la_server --version

# ----------------------------------------------------------------------------
# Stage 2 — Runtime
# (ARG LA_CUDA_BASE_IMAGE is declared globally at the top of this file.)
# ----------------------------------------------------------------------------
FROM ${LA_CUDA_BASE_IMAGE} AS runtime

# ---- Build args (forwarded from versions.sh through docker compose) ------
ARG LA_PYTHON_PKG
ARG LA_TORCH_VERSION
ARG LA_TORCHVISION_VERSION
ARG LA_TORCH_INDEX_URL
ARG LA_FLASH_ATTN_VERSION
ARG LA_FLASH_ATTN_ARCHS
ARG LA_TRANSFORMERS_VERSION
ARG LA_TOKENIZERS_VERSION
ARG LA_ACCELERATE_VERSION
ARG LA_PEFT_VERSION
ARG LA_SENTENCEPIECE_VERSION
ARG LA_NUMPY_VERSION
ARG LA_PILLOW_VERSION
ARG LA_OPENCV_VERSION
ARG LA_DECORD_VERSION
ARG LA_LMDB_VERSION
ARG LA_HFHUB_VERSION
ARG LA_HF_TRANSFER_VERSION
ARG LA_PSUTIL_VERSION
ARG LA_WEBSOCKETS_PY_VERSION
ARG LA_INTERNAL_PORT
ARG LA_MAX_IMAGE_DIM
ARG LA_MAX_JPEG_BYTES
ARG LA_MAX_INFLIGHT
ARG LA_UID=1000
ARG LA_GID=1000

# Fail-fast at build time if any required arg is empty.
RUN test -n "${LA_PYTHON_PKG}"          -a -n "${LA_TORCH_VERSION}" \
     -a -n "${LA_TORCHVISION_VERSION}"   -a -n "${LA_TORCH_INDEX_URL}" \
     -a -n "${LA_FLASH_ATTN_VERSION}"    -a -n "${LA_FLASH_ATTN_ARCHS}" \
     -a -n "${LA_TRANSFORMERS_VERSION}"  -a -n "${LA_TOKENIZERS_VERSION}" \
     -a -n "${LA_ACCELERATE_VERSION}"    -a -n "${LA_PEFT_VERSION}" \
     -a -n "${LA_SENTENCEPIECE_VERSION}" -a -n "${LA_NUMPY_VERSION}" \
     -a -n "${LA_PILLOW_VERSION}"        -a -n "${LA_OPENCV_VERSION}" \
     -a -n "${LA_DECORD_VERSION}"        -a -n "${LA_LMDB_VERSION}" \
     -a -n "${LA_HFHUB_VERSION}"         -a -n "${LA_HF_TRANSFER_VERSION}" \
     -a -n "${LA_PSUTIL_VERSION}"        -a -n "${LA_INTERNAL_PORT}" \
     -a -n "${LA_MAX_IMAGE_DIM}"         -a -n "${LA_MAX_JPEG_BYTES}" \
     -a -n "${LA_MAX_INFLIGHT}"           -a -n "${LA_WEBSOCKETS_PY_VERSION}" \
     || { echo "FAIL: missing build arg — every pin must be set"; exit 1; }

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/locate_anything/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    LA_MODEL_DTYPE=bfloat16 \
    LA_ATTN_IMPL=flash_attention_2 \
    LA_GEN_TEMPERATURE=0.7 \
    LA_GEN_TOP_P=0.9 \
    LA_GEN_DO_SAMPLE=1 \
    LA_GEN_REP_PEN=1.1 \
    LA_GEN_MAX_NEW_TOKENS=8192 \
    LA_GEN_MODE=hybrid \
    LA_GEN_N_FUTURE_TOKENS=6 \
    LA_INTERNAL_PORT=${LA_INTERNAL_PORT} \
    LA_IPC_SOCKET=/tmp/la.sock \
    LA_MAX_IMAGE_DIM=${LA_MAX_IMAGE_DIM} \
    LA_MAX_JPEG_BYTES=${LA_MAX_JPEG_BYTES} \
    LA_MAX_INFLIGHT=${LA_MAX_INFLIGHT}

# NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES: pin the GPU
#   contract into the image rather than relying on the base image's
#   defaults. compute = bf16 inference kernels (CUDA core compute);
#   utility = nvidia-smi, libnvidia-ml. We deliberately do NOT request
#   `video` (NVDEC/NVENC) or `graphics` (Vulkan/OpenGL) — they're not
#   used and excluding them shrinks the surface area injected by
#   nvidia-container-runtime.
# MALLOC_ARENA_MAX: glibc malloc arena fragmentation cap. Default is
#   8×CPU cores which can grow to multi-GB of resident-but-unused address
#   space over a long run. 2 arenas is enough for our single-process
#   worker and bounds fragmentation tightly.
# CUDA_CACHE_PATH: default $HOME/.nv/ComputeCache is on the read-only
#   rootfs; without redirecting here, CUDA JIT silently fails and the
#   autotune cost is paid on every restart. hf_cache is read-write and
#   persists across container lifetimes.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : the PyTorch CUDA
#   caching allocator grows existing segments rather than allocating new
#   ones on shape variation. With per-frame image-dim variation under a
#   live stream, this bounds fragmentation that would otherwise creep
#   into multi-GB over a 24h run. Supported on PyTorch 2.1+ on Linux;
#   we run 2.12.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MALLOC_ARENA_MAX=2 \
    CUDA_CACHE_PATH=/opt/locate_anything/hf_cache/.nv-cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- System packages -----------------------------------------------------
# tini : reliable PID 1, forwards SIGTERM correctly. We use it to supervise
#        the two child processes (Rust server + Python worker).
# ffmpeg, libgl1, libsm6, libxext6 : opencv-python-headless + decord runtime.
# build-essential, ninja-build, git : flash-attn source build only.
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
        ${LA_PYTHON_PKG} \
        ${LA_PYTHON_PKG}-venv \
        ${LA_PYTHON_PKG}-dev \
        python3-pip \
        build-essential \
        ninja-build \
        git \
        tini \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libsm6 \
        libxext6 \
 && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/${LA_PYTHON_PKG} 100 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/${LA_PYTHON_PKG} 100

ENV VENV=/opt/locate_anything/venv
RUN python -m venv "${VENV}"
ENV PATH="${VENV}/bin:${PATH}"

RUN python -m pip install --upgrade pip==25.2 setuptools==75.6.0 wheel==0.45.1 \
 && python -m pip install ninja==1.11.1.3 packaging==24.2

# ---- PyTorch (cu130, sm_120-capable) -------------------------------------
RUN python -m pip install \
        --index-url "${LA_TORCH_INDEX_URL}" \
        "torch==${LA_TORCH_VERSION}" \
        "torchvision==${LA_TORCHVISION_VERSION}" \
 && python -c "import torch, sys; v=torch.__version__; \
sys.exit(0 if '+cu130' in v else (sys.stderr.write(f'FAIL: torch wheel version {v} is not +cu130 — pin mismatch\\n'),1)[1])"

# ---- Model-mandated Python deps -----------------------------------------
RUN python -m pip install \
        "transformers==${LA_TRANSFORMERS_VERSION}" \
        "tokenizers==${LA_TOKENIZERS_VERSION}" \
        "accelerate==${LA_ACCELERATE_VERSION}" \
        "peft==${LA_PEFT_VERSION}" \
        "sentencepiece==${LA_SENTENCEPIECE_VERSION}" \
        "numpy==${LA_NUMPY_VERSION}" \
        "Pillow==${LA_PILLOW_VERSION}" \
        "opencv-python-headless==${LA_OPENCV_VERSION}" \
        "decord==${LA_DECORD_VERSION}" \
        "lmdb==${LA_LMDB_VERSION}" \
        "huggingface_hub==${LA_HFHUB_VERSION}" \
        "hf_transfer==${LA_HF_TRANSFER_VERSION}" \
        "psutil==${LA_PSUTIL_VERSION}" \
        "websockets==${LA_WEBSOCKETS_PY_VERSION}"

# ---- flash-attn 2.8.3 source build (sm_120 only) -------------------------
# This layer is the longest in the build (15–25 min). It is intentionally
# placed after stable model deps so that day-to-day server code edits
# (which only touch the COPY layers below) do NOT trigger a re-build.
#
# FLASH_ATTN_CUDA_ARCHS="120" — only RTX 5090 Blackwell kernels. Shortens
# build by ~5x vs. the default arch sweep.
# MAX_JOBS=8 — linker is the bottleneck; 8 jobs on 62 GiB RAM is safe.
# --no-build-isolation — the build script imports the installed torch
#   wheel to query CUDA version; with isolation it would install a stale
#   torch into the build env.
RUN FLASH_ATTN_CUDA_ARCHS="${LA_FLASH_ATTN_ARCHS}" \
    MAX_JOBS=8 \
    python -m pip install --no-build-isolation \
        "flash-attn==${LA_FLASH_ATTN_VERSION}" \
 && python -c "import flash_attn; \
import sys; \
sys.exit(0 if flash_attn.__version__ == '${LA_FLASH_ATTN_VERSION}' \
       else (sys.stderr.write(f'FAIL: flash_attn version {flash_attn.__version__}, expected ${LA_FLASH_ATTN_VERSION}\\n'),1)[1])"

# ---- Copy Rust binary from stage 1 ---------------------------------------
COPY --from=rust_builder /work/la_server /usr/local/bin/la_server
RUN chmod 0755 /usr/local/bin/la_server \
 && /usr/local/bin/la_server --version

# ---- Copy Python worker + helpers ---------------------------------------
WORKDIR /opt/locate_anything
COPY worker/        /opt/locate_anything/worker/
# `scripts/lib/` carries the smoke client; baking it in means the smoke
# test runs via `docker exec` against the running container — no helper
# container, no network. Other files in scripts/lib (common.sh,
# versions.sh) are also harmless to ship for diagnostics.
COPY scripts/lib/   /opt/locate_anything/scripts/lib/
# NOTE: test_data/ is intentionally NOT baked into the image. It is
# bind-mounted read-only at container start (see 03_start_service.sh).
# Reason: the calibration JPEG is synthesised by 01_download_weights.sh
# AFTER 02_build_image.sh runs (build is pure-from-source for
# reproducibility), so at build time the host's test_data/ is empty.
# A baked-in empty test_data/ would only ever shadow the bind mount
# anyway; cleaner to delete the source of confusion.
COPY container/entrypoint.sh /opt/locate_anything/entrypoint.sh
RUN chmod 0755 /opt/locate_anything/entrypoint.sh

# Bind-mount targets, created empty at build time.
#   model/     — model weights, populated by scripts/01_download_weights.sh
#   hf_cache/  — RW cache for hub state and CUDA JIT
#   test_data/ — host-generated calibration JPEG (see note above)
RUN mkdir -p /opt/locate_anything/model \
 && mkdir -p /opt/locate_anything/hf_cache \
 && mkdir -p /opt/locate_anything/test_data

# ---- Non-root user (maps to host UID for bind-mount writability) --------
# The Ubuntu 24.04 base image (which nvidia/cuda:...-ubuntu24.04
# inherits from) ships with a default 'ubuntu' user already at
# UID/GID 1000. A naive `groupadd -g 1000 la` collides with that
# preexisting group and fails with "GID '1000' already exists".
#
# We handle both cases robustly:
#   - If a user already exists at LA_UID, rename it to 'la' (and
#     rename its primary group to 'la' too), moving the home
#     directory to /home/la. This repurposes Ubuntu's default user.
#   - Otherwise, create 'la' fresh.
#
# Then chown the project tree to the resulting `la` user.
RUN set -eux; \
    OLD_USER=$(getent passwd ${LA_UID} 2>/dev/null | cut -d: -f1); \
    OLD_GROUP=$(getent group  ${LA_GID} 2>/dev/null | cut -d: -f1); \
    if [ -z "${OLD_USER}" ]; then \
        groupadd -g ${LA_GID} la; \
        useradd  -m -u ${LA_UID} -g ${LA_GID} -s /bin/bash la; \
    elif [ "${OLD_USER}" != "la" ]; then \
        groupmod -n la "${OLD_GROUP}"; \
        usermod  -l la -d /home/la -m -s /bin/bash "${OLD_USER}"; \
    fi; \
    chown -R la:la /opt/locate_anything
USER la

EXPOSE ${LA_INTERNAL_PORT}

# tini supervises the entrypoint script, ensuring zombie reaping and
# correct signal forwarding to PID 1's direct child (the shell). We pass
# `-g` so signals are sent to the child's entire process group, not just
# the shell PID. That matters because entrypoint.sh backgrounds the
# Rust binary and the Python worker and then `wait -n`s; when one of
# them dies and the shell exits, `-g` ensures the surviving child (and
# any of its descendants) is signalled too, instead of being orphaned
# and waited on by tini's grace timeout. Note: there is NO
# `--kill-on-parent-death` flag in tini; `-g` is the correct fix for
# our shell-supervised two-child topology.
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/locate_anything/entrypoint.sh"]
