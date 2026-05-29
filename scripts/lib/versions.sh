# shellcheck shell=bash
# Single source of truth for every pinned version in this project.
# All scripts source this file via `load_versions` in common.sh.
# To change a version: edit one line here, re-run setup.sh, never patch downstream.

# ===== Host prerequisites (validated, not installed by this project) =====
# Driver and CUDA on the host are installed by the personal-server scripts
# referenced in /home/user/Desktop/personal_server/README.md (sections §10, §11, §13).
LA_REQUIRE_DRIVER_BRANCH="595"           # §10 pin from personal_server
LA_REQUIRE_DRIVER_MIN="595.45.04"        # CUDA 13.0+ minimum
LA_REQUIRE_NVCTK_VERSION="1.19.0"        # §13 pin; supports driver 5xx through 595.x
LA_REQUIRE_DOCKER_MAJOR="29"             # §12 pin (29.x)
LA_REQUIRE_UBUNTU_CODENAME="noble"       # 24.04 LTS

# Required minimum GPU compute capability. RTX 5090 is sm_120 (Blackwell GB202).
# Any GPU below sm_120 will be REJECTED by the host validator: this project's
# torch wheels and flash-attn source build target sm_120 specifically. Older
# arches need a different pin table — do not silently downgrade.
LA_REQUIRE_GPU_COMPUTE_CAP="12.0"

# Minimum free VRAM at startup, in MiB. bf16 weights ~7 GiB + KV/activation
# headroom for 16K context. 24 GiB ensures comfortable operation.
LA_REQUIRE_GPU_MEM_MIN_MIB="24000"

# Minimum free disk space (host) for weights cache + Docker image layers.
LA_REQUIRE_DISK_FREE_GIB="30"

# ===== Model identity (pinned by HF revision SHA, not branch name) =====
LA_MODEL_HF_REPO="nvidia/LocateAnything-3B"
# Pinned full 40-char commit on HF as of 2026-05-28 (verified via
# https://huggingface.co/api/models/nvidia/LocateAnything-3B). Short SHAs
# are collision-prone and ambiguous after a force-push — use the full SHA.
LA_MODEL_HF_REVISION="7a81d810571dc5f244b2f0b6868128f24b1cbd85"
# Local mount path inside the container — also the directory that the host
# script `01_download_weights.sh` writes into.
LA_MODEL_LOCAL_DIR="/opt/locate_anything/model"

# Content SHA-256 of each Python file shipped with the HF revision above.
# Used at boot in validate_startup.py to refuse to load if any *.py file
# under the bind-mounted model directory differs from what we pinned.
# Defense against an attacker with write access to ./models/ swapping a
# same-size .py file (the manifest hash in worker/la_worker.py only
# fingerprints (name, size), not content). DO NOT regenerate these by
# rerunning hash — that defeats the pin.
LA_MODEL_PY_SHA256_configuration_locateanything="d2738cc180add2b77e88b8cf2bc87ff012f23bd417a99150a033f61b0a8eb857"
LA_MODEL_PY_SHA256_configuration_qwen2="1fda5efb735cae465debd414afc673389fe731afd95c934a469faa23d3d7fdf1"
LA_MODEL_PY_SHA256_generate_utils="863187051772549928bf103b58f6176263c9b786fb19ef83fd7b2e76169fa65e"
LA_MODEL_PY_SHA256_image_processing_locateanything="5109868add766c7e244487ecfff6a6f5a4aa1497b38b385a33a969f12e23b4ec"
LA_MODEL_PY_SHA256_mask_magi_utils="646b565e38b30d58cafe30aeecf0283aee83d198ce8e57936c3488c1dc7b9276"
LA_MODEL_PY_SHA256_mask_sdpa_utils="7e9d600eb25283963cc1696da060813066b9795da467cf9fb0ca68bfc8de1e1d"
LA_MODEL_PY_SHA256_modeling_locateanything="ffe736fb8ded5d597201704ccd85134d18a8e4dea43d309228644737234b7244"
LA_MODEL_PY_SHA256_modeling_qwen2="aadb676c0a587a16b7071977c159df16299fad22d88ee8ed9754881ab7f59575"
LA_MODEL_PY_SHA256_modeling_vit="96479eb121c840f009a32830c78740154171290419108caefcf8778580700373"
LA_MODEL_PY_SHA256_processing_locateanything="682145ed054b1e912e66273b476e51a25b2d666d4a37b26385af9300b66d40d8"

# ===== Container base image (multistage runtime stage) =====
# Pinned by sha256 manifest-list digest. Tags are mutable — digests are not.
# Obtained from registry-1.docker.io's Docker-Content-Digest header
# (the source of truth, not Hub's web UI).
LA_CUDA_BASE_IMAGE_TAG="nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04"
LA_CUDA_BASE_IMAGE_DIGEST="sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd"
LA_CUDA_BASE_IMAGE="${LA_CUDA_BASE_IMAGE_TAG}@${LA_CUDA_BASE_IMAGE_DIGEST}"

# ===== GPU smoke-test image (used by 00_validate_host.sh) =====
# A smaller CUDA base, just enough to run `nvidia-smi` for the GPU
# passthrough check. Same digest-pinning policy as the main base.
LA_GPU_SMOKE_IMAGE_TAG="nvidia/cuda:13.0.3-base-ubuntu24.04"
LA_GPU_SMOKE_IMAGE_DIGEST="sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1"
LA_GPU_SMOKE_IMAGE="${LA_GPU_SMOKE_IMAGE_TAG}@${LA_GPU_SMOKE_IMAGE_DIGEST}"

# ===== Python and PyTorch =====
# Ubuntu 24.04 ships python3.12 natively; we install python3.12 explicitly so
# the deb name pinning is reproducible. No PEP 668 issues inside the container.
LA_PYTHON_PKG="python3.12"

# PyTorch 2.12.0 cu130 wheels have native sm_120 (Blackwell) kernels.
# Verified via PyTorch 2.12 release notes + torch.cuda.get_arch_list().
LA_TORCH_VERSION="2.12.0"
LA_TORCHVISION_VERSION="0.27.0"
LA_TORCH_CUDA_TAG="cu130"
LA_TORCH_INDEX_URL="https://download.pytorch.org/whl/${LA_TORCH_CUDA_TAG}"

# ===== Flash-attention 2 (source-built for sm_120) =====
# 2.8.3 is the latest 2.x stable on PyPI (verified live against
# https://pypi.org/pypi/flash-attn/json — an earlier research subagent
# claimed 2.8.4 existed; it does not). FA4 does NOT run on RTX 5090
# — TMEM hardware missing on GB202. Source build (~25 min on this CPU).
LA_FLASH_ATTN_VERSION="2.8.3"
LA_FLASH_ATTN_ARCHS="120"     # only build sm_120 kernels — fast build, exact match

# ===== Model code's mandated dependencies (Embodied/pyproject.toml + model card) =====
LA_TRANSFORMERS_VERSION="4.57.1"
LA_TOKENIZERS_VERSION="0.22.0"
LA_ACCELERATE_VERSION="1.5.2"
LA_PEFT_VERSION="0.12.0"
LA_SENTENCEPIECE_VERSION="0.2.0"
# numpy: the upstream Embodied/pyproject.toml says >=1.25,<2. NVIDIA's
# model-card example pins exactly 1.25.0, but 1.25.0 has no cp312 wheel
# (Python 3.12 support landed in numpy 1.26.0, Sep 2023). pip's sdist
# fallback for 1.25.0 then fails on Python 3.12 because setuptools'
# pkg_resources references pkgutil.ImpImporter which 3.12 removed.
# 1.26.4 is the last 1.x release, ships cp312 wheels, and stays within
# the upstream <2 constraint.
LA_NUMPY_VERSION="1.26.4"
LA_PILLOW_VERSION="11.1.0"
LA_OPENCV_VERSION="4.11.0.86"
LA_DECORD_VERSION="0.6.0"
LA_LMDB_VERSION="1.7.5"

# ===== Python sidecar deps (orthogonal to model correctness) =====
# The model sidecar only talks Unix domain socket — no HTTP, no FastAPI.
# huggingface_hub. The earlier pin of 0.27.0 was incompatible with the
# transformers==4.57.1 pin: transformers 4.57.1 requires
# huggingface_hub>=0.34.0,<1.0. 0.36.2 is the latest stable in that
# range (Feb 2026). Verified by running pip install --dry-run with the
# full pin set inside python:3.12-slim-bookworm — resolves cleanly.
LA_HFHUB_VERSION="0.36.2"
LA_HF_TRANSFER_VERSION="0.1.8"  # fast downloads
LA_PSUTIL_VERSION="6.1.0"
# `websockets` is installed only because the smoke client (run via
# `docker exec` against the live container) needs a WS client library.
# Baking it into the image means the smoke test does not need network
# at all — important for offline-resumable setup.
LA_WEBSOCKETS_PY_VERSION="13.1"

# ===== Rust toolchain (HTTP/WebSocket frontend) =====
# 1.95-bookworm pinned by sha256 manifest-list digest (tag is mutable).
LA_RUST_BUILDER_IMAGE_TAG="rust:1.95-bookworm"
LA_RUST_BUILDER_IMAGE_DIGEST="sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1"
LA_RUST_BUILDER_IMAGE="${LA_RUST_BUILDER_IMAGE_TAG}@${LA_RUST_BUILDER_IMAGE_DIGEST}"

# Crate versions are pinned authoritatively in rust_server/Cargo.toml
# (with `=X.Y.Z` exact requirements) and frozen in rust_server/Cargo.lock.
# `cargo build --locked` enforces the lockfile. Do NOT duplicate the
# crate pins here — single source of truth.

# ===== Port the server binds to inside the container =====
LA_INTERNAL_PORT="8000"

# ===== Port the host publishes (loopback only — see docs/SECURITY.md) =====
LA_HOST_PORT="8765"
LA_HOST_BIND_IP="127.0.0.1"

# ===== Inference-input hard caps (forwarded to the Rust frontend) =====
# The model's preprocessor_config.json enforces a 25,600-patch budget
# (sqrt(25600) × patch_size=14 ≈ 2240 px per side at native resolution
# without forced downscale). Above this the preprocessor rescales — which
# changes pixel→token math and is documented as "operating outside the
# trained native-resolution policy". The frontend rejects anything above
# this cap so the client cannot accidentally trigger a rescale.
LA_MAX_IMAGE_DIM="2240"

# Hard JPEG payload byte cap. Sized for a quality-92 4K-ish JPEG plus
# slack — clients sending more are almost certainly misconfigured.
LA_MAX_JPEG_BYTES="4194304"

# Bounded mpsc capacity per WebSocket connection. Higher → more pipelining
# pressure on the GPU; lower → tighter backpressure (the GPU never has
# more than this many frames queued behind a connection). 2 keeps the
# GPU saturated without unbounded queueing.
LA_MAX_INFLIGHT="2"

# ===== Container resource limits =====
# Host RAM cap. The 3B+0.4B bf16 weights live in VRAM, not RAM; host RAM
# carries the Python interpreter, transformers/torch shared libs, CPU
# weight-load residue (~2 GiB after .to('cuda')), per-request transient
# buffers, and glibc malloc arenas. Verified peak over a 24h run at 2 FPS
# is ~6-7 GiB; 14 GiB is ~2x headroom and keeps the container's OOM
# deterministic (no swap thrash). DMZ-shared box rationale: a contained
# container can't drag the host into the kernel oom_killer.
LA_CONTAINER_MEM="14g"
LA_CONTAINER_CPUS="8"
LA_CONTAINER_PIDS="512"

# Docker json-file log rotation — without this, /var/lib/docker/containers/
# fills indefinitely on long-running services and eventually consumes the
# host root partition.
LA_LOG_MAX_SIZE="50m"
LA_LOG_MAX_FILES="5"

# ===== Container name =====
LA_CONTAINER_NAME="locate-anything"
LA_IMAGE_TAG="locate-anything:la3b-cu130-torch2.12-fa2.8.3"
