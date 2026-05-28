#!/usr/bin/env bash
# End-to-end smoke test against the running service.
#
# Three status endpoints (/v1/health, /v1/capabilities, /v1/info) are
# reached from the host with curl. The inference path (/v1/stream) is
# WebSocket-only — we drive it via `docker exec` against the LIVE
# container, using the smoke client baked into the image. This way:
#
#   • No helper container is spun up.
#   • No `pip install` runs at smoke-test time.
#   • The smoke test is OFFLINE-SAFE — `setup.sh` works on a re-run
#     even with the internet down.

set -Eeuo pipefail
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

BASE_URL="http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}"
PROJECT_ROOT="$(project_root)"

require_cmd curl
require_cmd jq
require_cmd docker

log_section "Smoke test against ${BASE_URL}"

# ---- /v1/health ----
HEALTH=$(curl -fsS "${BASE_URL}/v1/health")
HEALTHY=$(echo "${HEALTH}" | jq -r '.status')
[[ "${HEALTHY}" == "ok" ]] || die "/v1/health returned status=${HEALTHY}: ${HEALTH}"
log_ok "/v1/health: ok"

# ---- /v1/capabilities ----
CAPS=$(curl -fsS "${BASE_URL}/v1/capabilities")
MODEL_NAME=$(echo "${CAPS}" | jq -r '.model')
[[ "${MODEL_NAME}" == "nvidia/LocateAnything-3B" ]] \
    || die "/v1/capabilities model=${MODEL_NAME}, expected nvidia/LocateAnything-3B"
MEDIAN_FPS=$(echo "${CAPS}" | jq -r '.calibration.median_fps')
log_ok "/v1/capabilities: model=${MODEL_NAME}, calibration FPS=${MEDIAN_FPS}"

# ---- /v1/info ----
INFO=$(curl -fsS "${BASE_URL}/v1/info")
GPU=$(echo "${INFO}" | jq -r '.gpu_name')
log_ok "/v1/info: gpu=${GPU}"

# ---- /v1/stream — one Frame round-trip via docker exec ----
# Confirm the container is up and the calibration image is in place.
if ! docker ps --format '{{.Names}}' | grep -qx "${LA_CONTAINER_NAME}"; then
    die "container ${LA_CONTAINER_NAME} is not running — start it with scripts/03_start_service.sh."
fi
if ! docker exec "${LA_CONTAINER_NAME}" test -f /opt/locate_anything/test_data/calibration.jpg; then
    die "calibration image /opt/locate_anything/test_data/calibration.jpg is missing INSIDE the container — rebuild the image."
fi
if ! docker exec "${LA_CONTAINER_NAME}" test -f /opt/locate_anything/scripts/lib/smoke_ws_client.py; then
    die "smoke client /opt/locate_anything/scripts/lib/smoke_ws_client.py is missing INSIDE the container — rebuild the image."
fi

LOCATE_PROMPT='Locate all the instances that matches the following description: bottle</c>book</c>cup</c>laptop.'

log_info "Running smoke WS client via docker exec…"
docker exec "${LA_CONTAINER_NAME}" \
    python /opt/locate_anything/scripts/lib/smoke_ws_client.py \
        --url "ws://127.0.0.1:${LA_INTERNAL_PORT}/v1/stream" \
        --image /opt/locate_anything/test_data/calibration.jpg \
        --prompt "${LOCATE_PROMPT}" \
        --mode hybrid \
        --timeout 120

log_section "Smoke test passed"
