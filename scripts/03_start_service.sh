#!/usr/bin/env bash
# Start the container as a long-running service. Bound to LOOPBACK only.
# Re-running is idempotent: existing container is removed first.

set -Eeuo pipefail
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

print_help() {
    cat <<EOF
03_start_service.sh — start the container and wait until healthy.

Usage:
    bash scripts/03_start_service.sh [-h|--help]

Concrete actions:

    1. Create ./cache/huggingface (host-side RW bind mount) if absent.
    2. If a previous '${LA_CONTAINER_NAME}' container exists, remove
       it (regardless of state).
    3. 'docker run -d' the image at '${LA_IMAGE_TAG}' with all the
       hardening + GPU flags (--gpus all, --read-only, --tmpfs /tmp,
       --cap-drop ALL, --security-opt no-new-privileges, AppArmor
       docker-default, --memory ${LA_CONTAINER_MEM},
       --cpus ${LA_CONTAINER_CPUS}, --pids-limit ${LA_CONTAINER_PIDS}, log
       rotation ${LA_LOG_MAX_SIZE}×${LA_LOG_MAX_FILES}, healthcheck
       deep-probe).
    4. Verify via 'ss -tlnH' that the kernel-side listener is exactly
       ${LA_HOST_BIND_IP}:${LA_HOST_PORT} (not 0.0.0.0, not [::], not [::1]).
    5. Poll 'docker inspect .State.Health.Status' until 'healthy' or
       fail loud (240 s start_period + 10 retries × 15 s interval).

Bind mounts:
    ./models/LocateAnything-3B  →  /opt/locate_anything/model     (RO)
    ./cache/huggingface         →  /opt/locate_anything/hf_cache  (RW)
    ./test_data                 →  /opt/locate_anything/test_data (RO)

The test_data/ mount carries the synthetic calibration JPEG generated
by 01_download_weights.sh — it is not baked into the image because
the image is built before that script runs (so test_data/ is empty
at build time).

Idempotent: re-running while the service is healthy stops the old
container, starts a fresh one off the same image, and re-verifies.

Prerequisites: the image at '${LA_IMAGE_TAG}' must exist locally
(scripts/02_build_image.sh) and the model snapshot at
./models/LocateAnything-3B/ must be present
(scripts/01_download_weights.sh).

EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) print_help; exit 0 ;;
        *) log_err "unknown argument: ${arg@Q}"
           log_err "Run 'bash scripts/03_start_service.sh --help' for usage."
           exit 2 ;;
    esac
done

PROJECT_ROOT="$(project_root)"
LOCAL_MODEL_DIR="${PROJECT_ROOT}/models/LocateAnything-3B"
HF_CACHE_DIR="${PROJECT_ROOT}/cache/huggingface"
TEST_DATA_DIR="${PROJECT_ROOT}/test_data"
CALIB_IMG="${TEST_DATA_DIR}/calibration.jpg"

mkdir -p "${HF_CACHE_DIR}"

log_section "Starting ${LA_CONTAINER_NAME}"

# Remove any prior container (idempotency).
if docker ps -aq -f name="^${LA_CONTAINER_NAME}\$" | grep -q .; then
    log_info "Removing previous container instance"
    docker rm -f "${LA_CONTAINER_NAME}" >/dev/null
fi

# Verify the weights are present (defensive — should already be done in 01).
if [[ ! -d "${LOCAL_MODEL_DIR}" ]]; then
    die "Model directory ${LOCAL_MODEL_DIR} missing — run scripts/01_download_weights.sh first."
fi

# Verify the calibration JPEG exists on the host before we bind-mount its
# parent directory into the container read-only. Without this check, an
# empty test_data/ would silently mount and the worker would die mid-boot
# with the file-not-found error reported by worker/calibration.py — clearer
# to fail here, before the container is even started, with a single-line
# fix-it pointer.
if [[ ! -f "${CALIB_IMG}" ]]; then
    die "Calibration image ${CALIB_IMG} missing on host — run scripts/01_download_weights.sh first.
The container reads it at /opt/locate_anything/test_data/calibration.jpg via the test_data bind mount."
fi

docker run -d \
    --name "${LA_CONTAINER_NAME}" \
    --gpus all \
    --restart unless-stopped \
    --read-only \
    --tmpfs /tmp:rw,size=512m,noexec,nosuid,nodev \
    -v "${LOCAL_MODEL_DIR}":/opt/locate_anything/model:ro \
    -v "${HF_CACHE_DIR}":/opt/locate_anything/hf_cache:rw \
    -v "${TEST_DATA_DIR}":/opt/locate_anything/test_data:ro \
    -p "${LA_HOST_BIND_IP}:${LA_HOST_PORT}:${LA_INTERNAL_PORT}/tcp" \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --security-opt=apparmor=docker-default \
    --shm-size=512m \
    --memory="${LA_CONTAINER_MEM}" \
    --memory-swap="${LA_CONTAINER_MEM}" \
    --cpus="${LA_CONTAINER_CPUS}" \
    --pids-limit="${LA_CONTAINER_PIDS}" \
    --log-driver=json-file \
    --log-opt "max-size=${LA_LOG_MAX_SIZE}" \
    --log-opt "max-file=${LA_LOG_MAX_FILES}" \
    --health-cmd 'curl -fsS http://127.0.0.1:'"${LA_INTERNAL_PORT}"'/v1/health || exit 1' \
    --health-interval=15s \
    --health-timeout=8s \
    --health-retries=10 \
    --health-start-period=240s \
    "${LA_IMAGE_TAG}"

log_ok "Container ${LA_CONTAINER_NAME} started (bound to ${LA_HOST_BIND_IP}:${LA_HOST_PORT})"

# ----------------------------------------------------------------------
# Runtime verification of the host-side listener.
#
# Docker's port-publish syntax `-p 127.0.0.1:PORT:PORT` is supposed to
# make the host listener IPv4-loopback-only, but we don't trust the
# claim — we VERIFY by querying the kernel via `ss`. The listener must
# bind to *exactly* one address: 127.0.0.1:${LA_HOST_PORT}. The script
# rejects any of:
#   * `0.0.0.0:8765`  — published on all IPv4 interfaces (DMZ-reachable)
#   * `[::]:8765`     — published on all IPv6 interfaces (DMZ-reachable)
#   * `[::1]:8765`    — IPv6 loopback (would surface only via `localhost`
#                       lookups returning ::1; doc rules ban localhost)
# Any of those = a misconfigured publish; refuse to declare success.
#
# The check uses `ss -tlnH` for a stable machine-parseable format. We
# look at the Local Address column (field 4) and assert exactly one
# entry matching :${LA_HOST_PORT}, and that entry is "127.0.0.1:PORT".
# ----------------------------------------------------------------------
require_cmd ss
log_info "Verifying host-side listener is exactly ${LA_HOST_BIND_IP}:${LA_HOST_PORT}…"
# docker-proxy starts asynchronously; brief poll so we don't race it.
PORT_LISTENERS=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
    PORT_LISTENERS=$(ss -tlnH "( sport = :${LA_HOST_PORT} )" 2>/dev/null | awk '{print $4}')
    [[ -n "${PORT_LISTENERS}" ]] && break
    sleep 0.5
done
if [[ -z "${PORT_LISTENERS}" ]]; then
    docker logs --tail 80 "${LA_CONTAINER_NAME}" >&2 || true
    die "no host listener on port ${LA_HOST_PORT} after 5s. The container is up but the published port did not bind. Check the docker proxy (is 'userland-proxy' disabled in /etc/docker/daemon.json? this script's verification assumes the proxy is enabled, which is Docker's default)."
fi
EXPECTED_LISTENER="${LA_HOST_BIND_IP}:${LA_HOST_PORT}"
while IFS= read -r addr; do
    if [[ "${addr}" != "${EXPECTED_LISTENER}" ]]; then
        die "host-side listener policy violation: kernel reports listener at '${addr}', expected EXACTLY '${EXPECTED_LISTENER}'. \
A listener on 0.0.0.0:${LA_HOST_PORT} or [::]:${LA_HOST_PORT} or [::1]:${LA_HOST_PORT} means the published port is reachable beyond IPv4 loopback. \
Refusing to declare success — stop the container and investigate the docker -p flag, daemon ipv6 config, or any conflicting host process."
    fi
done <<< "${PORT_LISTENERS}"
log_ok "Host listener verified: exactly ${EXPECTED_LISTENER} (no IPv6 listener, no INADDR_ANY listener)."

# Wait for the health check to flip to healthy (or fail).
log_info "Waiting for the service to become healthy (model load + calibration takes ~60s)…"
DEADLINE=$(( SECONDS + 240 ))
while (( SECONDS < DEADLINE )); do
    STATUS=$(docker inspect -f '{{.State.Health.Status}}' "${LA_CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    case "${STATUS}" in
        healthy)
            log_ok "Service is healthy"
            exit 0
            ;;
        unhealthy)
            log_err "Container is unhealthy. Recent logs:"
            docker logs --tail 80 "${LA_CONTAINER_NAME}" >&2 || true
            die "Service failed to become healthy."
            ;;
        starting|unknown)
            sleep 3
            ;;
        *)
            log_warn "Unexpected health status: ${STATUS}"
            sleep 3
            ;;
    esac
done

log_err "Service did not become healthy within timeout. Logs follow:"
docker logs --tail 200 "${LA_CONTAINER_NAME}" >&2 || true
die "Healthcheck timeout."
