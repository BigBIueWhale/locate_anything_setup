#!/usr/bin/env bash
# Validate that the host machine satisfies every precondition this project
# requires. Loud failure with a precise diagnostic on anything unexpected.
# No fallbacks, no auto-fixes — the user fixes the underlying issue.
#
# Run as the desktop user (NOT root). The script does not need elevated
# privileges; nvidia-smi, docker, and disk space are all readable by the
# user when membership in the `docker` group has been granted per §12 of
# the personal_server README.

set -Eeuo pipefail

_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

print_help() {
    cat <<EOF
00_validate_host.sh — strict host-precondition checks.

Usage:
    bash scripts/00_validate_host.sh [-h|--help]

Refuses to run as root. Verifies, in this order:

    OS                Ubuntu 24.04 LTS (noble)
    NVIDIA driver     ≥ ${LA_REQUIRE_DRIVER_MIN}
    GPU               compute capability == ${LA_REQUIRE_GPU_COMPUTE_CAP} (sm_120 / Blackwell)
    GPU VRAM          ≥ ${LA_REQUIRE_GPU_MEM_MIN_MIB} MiB
    Docker            major == ${LA_REQUIRE_DOCKER_MAJOR} and 'docker ps' works
    nvidia runtime    registered in 'docker info'
    nvidia-ctk        == ${LA_REQUIRE_NVCTK_VERSION}
    GPU passthrough   'docker run --gpus all nvidia/cuda nvidia-smi' succeeds
    Disk free         ≥ ${LA_REQUIRE_DISK_FREE_GIB} GiB at the project directory
    Host port         ${LA_HOST_PORT} is unbound

Read-only: this script does not modify the host. It only reads
system metadata (and pulls the GPU-smoke image if not cached, then
runs it once with --rm). All pins are defined in
scripts/lib/versions.sh.

Idempotent: re-running on a healthy host is a fast no-op.

EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) print_help; exit 0 ;;
        *) log_err "unknown argument: ${arg@Q}"
           log_err "Run 'bash scripts/00_validate_host.sh --help' for usage."
           exit 2 ;;
    esac
done

log_section "Host preflight"

# ---- root check (reverse) -------------------------------------------------
if [[ "$(id -u)" -eq 0 ]]; then
    die "Refusing to run as root. Run as the desktop user; the docker group provides the privileges this script needs."
fi

# ---- OS ----
if [[ ! -r /etc/os-release ]]; then
    die "/etc/os-release missing; cannot confirm OS."
fi
. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != "${LA_REQUIRE_UBUNTU_CODENAME}" ]]; then
    die "OS codename '${VERSION_CODENAME:-?}' != required '${LA_REQUIRE_UBUNTU_CODENAME}' (Ubuntu 24.04 LTS)."
fi
log_ok "OS: Ubuntu 24.04 (${VERSION_CODENAME})"

# ---- driver ----
require_cmd nvidia-smi
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
if ! version_ge "${DRIVER_VER}" "${LA_REQUIRE_DRIVER_MIN}"; then
    die "NVIDIA driver ${DRIVER_VER} < required minimum ${LA_REQUIRE_DRIVER_MIN}. Run §10 of personal_server."
fi
log_ok "NVIDIA driver: ${DRIVER_VER} (≥ ${LA_REQUIRE_DRIVER_MIN})"

# ---- GPU compute capability ----
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
if [[ "${GPU_CAP}" != "${LA_REQUIRE_GPU_COMPUTE_CAP}" ]]; then
    die "GPU compute capability ${GPU_CAP} != required ${LA_REQUIRE_GPU_COMPUTE_CAP} (RTX 5090 / Blackwell sm_120). \
This image's torch wheels and flash-attn build target sm_120 specifically — running on any other arch is unsupported."
fi
log_ok "GPU compute cap: ${GPU_CAP} (sm_120 Blackwell)"

# ---- GPU memory ----
GPU_MEM_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
if (( GPU_MEM_TOTAL_MIB < LA_REQUIRE_GPU_MEM_MIN_MIB )); then
    die "GPU memory ${GPU_MEM_TOTAL_MIB} MiB < required ${LA_REQUIRE_GPU_MEM_MIN_MIB} MiB."
fi
log_ok "GPU memory: ${GPU_MEM_TOTAL_MIB} MiB (≥ ${LA_REQUIRE_GPU_MEM_MIN_MIB})"

# ---- docker ----
require_cmd docker
DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
DOCKER_MAJ=${DOCKER_VER%%.*}
if [[ "${DOCKER_MAJ}" != "${LA_REQUIRE_DOCKER_MAJOR}" ]]; then
    die "Docker major version ${DOCKER_MAJ} != required ${LA_REQUIRE_DOCKER_MAJOR} (29.x). Run §12 of personal_server."
fi
if ! docker ps >/dev/null 2>&1; then
    die "docker ps failed — is the docker.service running and is your user in the docker group? (newgrp docker or re-login)."
fi
log_ok "Docker: ${DOCKER_VER} reachable"

# ---- nvidia-container-toolkit smoke ----
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    die "Docker reports no 'nvidia' runtime — run §13 of personal_server."
fi
# Pin enforcement: the daemon-registered runtime is one signal, but we also
# verify the toolkit binary version matches the pin in versions.sh. A
# silent drift on the host (e.g., apt upgrade installed a newer nvctk)
# could change container-side behavior without us noticing.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    die "nvidia-ctk binary not found on PATH — run §13 of personal_server."
fi
NVCTK_VER=$(nvidia-ctk --version 2>/dev/null | awk '/version/{print $NF; exit}' | tr -d 'v')
if [[ -z "${NVCTK_VER}" ]]; then
    die "could not parse 'nvidia-ctk --version' output."
fi
if [[ "${NVCTK_VER}" != "${LA_REQUIRE_NVCTK_VERSION}" ]]; then
    die "nvidia-container-toolkit ${NVCTK_VER} != required ${LA_REQUIRE_NVCTK_VERSION}. Run §13 of personal_server."
fi
log_ok "nvidia-container-toolkit: ${NVCTK_VER}"

# Verify GPU passthrough actually works.
#
# Two paths: if the smoke-test image is already cached locally, use
# `--pull never` so the check is fully offline. If it's NOT cached, do
# a `docker pull` explicitly first (so the failure message is precise:
# "couldn't pull" rather than "couldn't run").
LA_GPU_SMOKE_IMAGE="nvidia/cuda:13.0.3-base-ubuntu24.04"
if docker image inspect "${LA_GPU_SMOKE_IMAGE}" >/dev/null 2>&1; then
    log_ok "GPU smoke image '${LA_GPU_SMOKE_IMAGE}' present locally — running offline-safe smoke."
    GPU_SMOKE_PULL_FLAG="--pull=never"
else
    log_info "GPU smoke image not cached; pulling now (needs internet)…"
    if ! docker pull "${LA_GPU_SMOKE_IMAGE}" >/dev/null; then
        die "docker pull '${LA_GPU_SMOKE_IMAGE}' failed. If you are offline, pre-cache this image; otherwise check connectivity to docker.io."
    fi
    GPU_SMOKE_PULL_FLAG="--pull=never"
fi
if ! docker run --rm "${GPU_SMOKE_PULL_FLAG}" --gpus all "${LA_GPU_SMOKE_IMAGE}" \
        nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    die "Docker GPU passthrough failed — 'docker run --gpus all ${LA_GPU_SMOKE_IMAGE} nvidia-smi' did not work."
fi
log_ok "Docker GPU passthrough: smoke test passed"

# ---- host UID matches the Dockerfile's default LA_UID ----
# If the host user's UID isn't 1000, the image needs --build-arg
# LA_UID=$(id -u) LA_GID=$(id -g) so the in-container `la` user can
# write the bind-mounted hf_cache. 02_build_image.sh already passes
# these from id -u / id -g, but this assertion makes the dependency
# explicit if anyone runs docker build by hand.
if [[ "$(id -u)" -ne 1000 ]]; then
    log_warn "host UID is $(id -u), not the Dockerfile default 1000. \
The image build (scripts/02_build_image.sh) auto-forwards the host UID/GID, \
so this is informational — but anyone running docker build by hand must \
pass --build-arg LA_UID=$(id -u) LA_GID=$(id -g) to keep hf_cache writable."
fi

# ---- disk ----
PROJECT_ROOT="$(project_root)"
FREE_GIB=$(df -BG --output=avail "${PROJECT_ROOT}" | tail -n1 | tr -dc '0-9')
if (( FREE_GIB < LA_REQUIRE_DISK_FREE_GIB )); then
    die "Free disk under ${PROJECT_ROOT}: ${FREE_GIB} GiB < required ${LA_REQUIRE_DISK_FREE_GIB} GiB."
fi
log_ok "Disk free: ${FREE_GIB} GiB (≥ ${LA_REQUIRE_DISK_FREE_GIB})"

# ---- host Rust (for Cargo.lock regeneration) ----
if ! command -v cargo >/dev/null 2>&1; then
    log_warn "host cargo not on PATH — Cargo.lock will not be regenerated. If rust_server/Cargo.lock is absent the docker build will fail."
else
    HOST_RUST=$(rustc --version 2>/dev/null | awk '{print $2}')
    log_ok "Host Rust toolchain: ${HOST_RUST}"
fi

# ---- TCP port availability ----
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${LA_HOST_PORT}\$"; then
    die "Host port ${LA_HOST_PORT} is already bound — pick a different LA_HOST_PORT in scripts/lib/versions.sh."
fi
log_ok "Host port ${LA_HOST_PORT} free"

log_section "All preflight checks passed"
