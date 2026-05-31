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

print_help() {
    cat <<EOF
04_smoke_test.sh — end-to-end smoke test.

Usage:
    bash scripts/04_smoke_test.sh [-h|--help]

Concrete actions:

    1. curl GET http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}/v1/health         → assert status==ok
    2. curl GET http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}/v1/capabilities   → assert
                model=='nvidia/LocateAnything-3B'
    3. curl GET http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}/v1/info           → log GPU name
    4. docker exec '${LA_CONTAINER_NAME}' python /opt/locate_anything/
       scripts/lib/smoke_ws_client.py → one Frame round-trip through
       WS /v1/stream against the bundled calibration image. Asserts
       the model is loaded, calibration ran, JPEG → JSON works, and
       the parser produces a list shape.

Prerequisites:
    Container '${LA_CONTAINER_NAME}' must be running (scripts/03_start_service.sh)
    'curl' and 'jq' on the host.

Offline-safe: every step runs over loopback or via 'docker exec'.
No 'pip install' fires, no auxiliary container is launched.

Idempotent: read-only with respect to the running service.

EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) print_help; exit 0 ;;
        *) log_err "unknown argument: ${arg@Q}"
           log_err "Run 'bash scripts/04_smoke_test.sh --help' for usage."
           exit 2 ;;
    esac
done

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
    die "calibration image /opt/locate_anything/test_data/calibration.jpg is missing INSIDE the container.
The container reads it via a read-only bind mount of the host's ./test_data/.
Fix order:
    1. Confirm host file:     ls -l test_data/calibration.jpg
       (if missing, regenerate with: bash scripts/01_download_weights.sh)
    2. Confirm bind mount:    docker inspect ${LA_CONTAINER_NAME} --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'
       (test_data should appear; if not, restart with: bash scripts/03_start_service.sh)"
fi
if ! docker exec "${LA_CONTAINER_NAME}" test -f /opt/locate_anything/scripts/lib/smoke_ws_client.py; then
    die "smoke client /opt/locate_anything/scripts/lib/smoke_ws_client.py is missing INSIDE the container — rebuild the image."
fi

# Two probes against the synthetic calibration.jpg:
#   1. Template-1 closed-class detection. Verifies the multi-category
#      detection path end-to-end including per-category abstention
#      (`book` is structurally not present in the synthetic image, so
#      the response includes one <ref>book</ref><box>None</box> triple
#      that is correctly silently dropped by the parser; non-absent
#      categories return clean boxes). Covers prompt_task=detection.
#   2. Template-5 scene-text detection. Covers prompt_task=scene_text
#      AND the per-box <ref>text</ref><box>...</box> shape that is
#      otherwise only exercised on text-heavy imagery. The synthetic
#      polygon image has no real text — the model is observed to emit
#      a degenerate full-image labeled box on no-text input; that's
#      fine for CI structural coverage (the parser still consumes a
#      labeled box; the off-shape filter sees a "box" result for
#      prompt_task=scene_text and lets it through).
LOCATE_PROMPT='Locate all the instances that matches the following description: bottle</c>book</c>cup</c>laptop.'
SCENE_TEXT_PROMPT='Detect all the text in box format.'

log_info "Running smoke WS client (template 1 — detection) via docker exec…"
docker exec "${LA_CONTAINER_NAME}" \
    python /opt/locate_anything/scripts/lib/smoke_ws_client.py \
        --url "ws://127.0.0.1:${LA_INTERNAL_PORT}/v1/stream" \
        --image /opt/locate_anything/test_data/calibration.jpg \
        --prompt "${LOCATE_PROMPT}" \
        --mode hybrid \
        --expect-task detection \
        --timeout 120

log_info "Running smoke WS client (template 5 — scene-text) via docker exec…"
docker exec "${LA_CONTAINER_NAME}" \
    python /opt/locate_anything/scripts/lib/smoke_ws_client.py \
        --url "ws://127.0.0.1:${LA_INTERNAL_PORT}/v1/stream" \
        --image /opt/locate_anything/test_data/calibration.jpg \
        --prompt "${SCENE_TEXT_PROMPT}" \
        --mode hybrid \
        --expect-task scene_text \
        --timeout 120

log_section "Smoke test passed"
