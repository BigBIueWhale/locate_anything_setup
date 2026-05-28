#!/usr/bin/env bash
# Build the Docker image with all version pins forwarded as --build-arg
# from scripts/lib/versions.sh. No silent defaults inside the Dockerfile.

set -Eeuo pipefail
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

PROJECT_ROOT="$(project_root)"
cd "${PROJECT_ROOT}"

# --- --rebuild flag ---
# Default behavior: if an image with the pinned tag is already present
# locally, skip the build. This is what makes setup.sh offline-resumable
# after a reboot when the build succeeded once. Pass `--rebuild` to
# force a fresh build (the BuildKit layer cache still helps — only
# changed layers are rebuilt).
LA_REBUILD=0
for arg in "$@"; do
    case "${arg}" in
        --rebuild) LA_REBUILD=1 ;;
        *) die "unknown arg ${arg@Q}; supported: --rebuild" ;;
    esac
done

if [[ "${LA_REBUILD}" -eq 0 ]] && docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
    LOCAL_ID=$(docker image inspect "${LA_IMAGE_TAG}" --format '{{.Id}}')
    log_ok "Image '${LA_IMAGE_TAG}' already built (id=${LOCAL_ID:7:12}); skipping docker build."
    log_info "Re-build with: bash scripts/02_build_image.sh --rebuild"
    exit 0
fi

log_section "Docker build: ${LA_IMAGE_TAG}"

# Use BuildKit for cache mounts in the Dockerfile.
export DOCKER_BUILDKIT=1

UID_NUM="$(id -u)"
GID_NUM="$(id -g)"

docker build \
    --progress=plain \
    -t "${LA_IMAGE_TAG}" \
    --build-arg LA_RUST_BUILDER_IMAGE="${LA_RUST_BUILDER_IMAGE}" \
    --build-arg LA_CUDA_BASE_IMAGE="${LA_CUDA_BASE_IMAGE}" \
    --build-arg LA_PYTHON_PKG="${LA_PYTHON_PKG}" \
    --build-arg LA_TORCH_VERSION="${LA_TORCH_VERSION}" \
    --build-arg LA_TORCHVISION_VERSION="${LA_TORCHVISION_VERSION}" \
    --build-arg LA_TORCH_INDEX_URL="${LA_TORCH_INDEX_URL}" \
    --build-arg LA_FLASH_ATTN_VERSION="${LA_FLASH_ATTN_VERSION}" \
    --build-arg LA_FLASH_ATTN_ARCHS="${LA_FLASH_ATTN_ARCHS}" \
    --build-arg LA_TRANSFORMERS_VERSION="${LA_TRANSFORMERS_VERSION}" \
    --build-arg LA_TOKENIZERS_VERSION="${LA_TOKENIZERS_VERSION}" \
    --build-arg LA_ACCELERATE_VERSION="${LA_ACCELERATE_VERSION}" \
    --build-arg LA_PEFT_VERSION="${LA_PEFT_VERSION}" \
    --build-arg LA_SENTENCEPIECE_VERSION="${LA_SENTENCEPIECE_VERSION}" \
    --build-arg LA_NUMPY_VERSION="${LA_NUMPY_VERSION}" \
    --build-arg LA_PILLOW_VERSION="${LA_PILLOW_VERSION}" \
    --build-arg LA_OPENCV_VERSION="${LA_OPENCV_VERSION}" \
    --build-arg LA_DECORD_VERSION="${LA_DECORD_VERSION}" \
    --build-arg LA_LMDB_VERSION="${LA_LMDB_VERSION}" \
    --build-arg LA_HFHUB_VERSION="${LA_HFHUB_VERSION}" \
    --build-arg LA_HF_TRANSFER_VERSION="${LA_HF_TRANSFER_VERSION}" \
    --build-arg LA_PSUTIL_VERSION="${LA_PSUTIL_VERSION}" \
    --build-arg LA_WEBSOCKETS_PY_VERSION="${LA_WEBSOCKETS_PY_VERSION}" \
    --build-arg LA_INTERNAL_PORT="${LA_INTERNAL_PORT}" \
    --build-arg LA_MAX_IMAGE_DIM="${LA_MAX_IMAGE_DIM}" \
    --build-arg LA_MAX_JPEG_BYTES="${LA_MAX_JPEG_BYTES}" \
    --build-arg LA_MAX_INFLIGHT="${LA_MAX_INFLIGHT}" \
    --build-arg LA_UID="${UID_NUM}" \
    --build-arg LA_GID="${GID_NUM}" \
    .

log_ok "Image built: ${LA_IMAGE_TAG}"
