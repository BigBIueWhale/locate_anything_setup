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

print_help() {
    cat <<EOF
02_build_image.sh — build the Docker image.

Usage:
    bash scripts/02_build_image.sh [-h|--help] [--rebuild]

Builds the multi-stage image at tag '${LA_IMAGE_TAG}' from
./Dockerfile. Forwards every version pin from scripts/lib/versions.sh
as a --build-arg.

First build: ~25 minutes wall time, dominated by the flash-attn
${LA_FLASH_ATTN_VERSION} source compile (FLASH_ATTN_CUDA_ARCHS=${LA_FLASH_ATTN_ARCHS},
MAX_JOBS=8). Subsequent runs reuse BuildKit's layer cache, so source
edits to worker/ or rust_server/ rebuild only the small COPY layers
near the end.

Flags:

    --rebuild   Force a fresh build even when an image with the
                pinned tag is already present locally. Without this
                flag, the script short-circuits with an "already
                built" message when 'docker image inspect ${LA_IMAGE_TAG}'
                succeeds — that's what makes setup.sh offline-resumable
                after a reboot.

Idempotent in the default mode (skip-if-built); not idempotent under
--rebuild (the build runs even when the image is current).

EOF
}

# --- arg parsing ---
LA_REBUILD=0
for arg in "$@"; do
    case "${arg}" in
        -h|--help) print_help; exit 0 ;;
        --rebuild) LA_REBUILD=1 ;;
        *) log_err "unknown argument: ${arg@Q}"
           log_err "Run 'bash scripts/02_build_image.sh --help' for usage."
           exit 2 ;;
    esac
done

if [[ "${LA_REBUILD}" -eq 0 ]] && docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
    LOCAL_ID=$(docker image inspect "${LA_IMAGE_TAG}" --format '{{.Id}}')
    log_ok "Image '${LA_IMAGE_TAG}' already built (id=${LOCAL_ID:7:12}); skipping docker build."
    log_info "Re-build with: bash scripts/02_build_image.sh --rebuild"
    exit 0
fi

# ---------------------------------------------------------------------
# RAM-capacity gate before the flash-attn compile.
#
# The Dockerfile invokes the flash-attn 2.8.3 source build with
# MAX_JOBS=8. Eight parallel nvcc processes compiling cutlass kernel
# templates can peak at roughly 30-50 GiB of host RAM in flight. If
# the host can't supply that, the kernel OOM-killer terminates one
# of the nvcc processes mid-compile and ninja aborts the build half
# done — wasting ~15 minutes of compute.
#
# Two independent checks:
#
#   (1) MemTotal >= 56 GiB. This is the STRUCTURAL check — does the
#       machine itself have enough RAM to ever run MAX_JOBS=8 safely?
#       A "64 GiB-class" workstation on Linux typically reports
#       58-62 GiB MemTotal after BIOS reservations; 56 is the floor
#       we accept. Below 56, the operator MUST change MAX_JOBS in
#       the Dockerfile — closing apps won't help on a too-small box.
#
#   (2) MemAvailable >= 35 GiB. This is the RUNTIME check — even on
#       a 64 GiB-class machine, if some other process is currently
#       hogging 30+ GiB the parallel compile will still get OOM-
#       killed. 35 GiB gives ~5 GiB headroom above the realistic
#       median nvcc peak (~30 GiB for sm_120-only with 8 jobs).
#       Below 35, the operator can either close those apps or drop
#       MAX_JOBS.
# ---------------------------------------------------------------------
LA_MIN_TOTAL_RAM_GIB=56
LA_MIN_AVAILABLE_RAM_GIB=35
MEMTOTAL_KIB=$(awk '/^MemTotal:/     {print $2}' /proc/meminfo)
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [[ -z "${MEMTOTAL_KIB}" || -z "${AVAILABLE_KIB}" ]]; then
    die "could not read MemTotal / MemAvailable from /proc/meminfo — host environment is non-standard."
fi
MEMTOTAL_GIB=$(( MEMTOTAL_KIB / 1024 / 1024 ))
AVAILABLE_GIB=$(( AVAILABLE_KIB / 1024 / 1024 ))

if (( MEMTOTAL_GIB < LA_MIN_TOTAL_RAM_GIB )); then
    die "Host machine has only ${MEMTOTAL_GIB} GiB MemTotal (< ${LA_MIN_TOTAL_RAM_GIB} GiB required for MAX_JOBS=8). Refusing to start the build because the flash-attn nvcc compile would OOM-kill itself.

This is a STRUCTURAL machine-capacity issue. Closing applications won't help — the machine itself is too small for parallel nvcc at MAX_JOBS=8.

Required: edit MAX_JOBS in Dockerfile (currently 8) to a lower value, then re-run:
    MAX_JOBS=4  → ~30-40 min build, peak ~15-25 GiB
    MAX_JOBS=2  → ~60 min build,    peak ~8-12 GiB
    MAX_JOBS=1  → ~90-120 min build, peak ~4-6 GiB (fully serial; works on any host)

If you'll be rebuilding on this hardware more than once, consider promoting
MAX_JOBS to scripts/lib/versions.sh as LA_FLASH_ATTN_MAX_JOBS so it's a one-line
edit instead of a Dockerfile change."
fi

if (( AVAILABLE_GIB < LA_MIN_AVAILABLE_RAM_GIB )); then
    die "Host has only ${AVAILABLE_GIB} GiB MemAvailable right now (of ${MEMTOTAL_GIB} GiB total); refusing because the flash-attn nvcc compile with MAX_JOBS=8 needs at least ${LA_MIN_AVAILABLE_RAM_GIB} GiB of free RAM.

This is a RUNTIME availability issue — the machine itself is big enough, but something is currently using too much RAM.

Options:
    (a) Inspect what's consuming RAM and close it:
            ps -eo pid,rss,comm --sort=-rss | head
        Then re-run.

    (b) Edit MAX_JOBS in Dockerfile to a lower value (see options above)."
fi
log_ok "Host RAM: ${MEMTOTAL_GIB} GiB total, ${AVAILABLE_GIB} GiB available (passes ${LA_MIN_TOTAL_RAM_GIB}/${LA_MIN_AVAILABLE_RAM_GIB} thresholds for MAX_JOBS=8)"

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
