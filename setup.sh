#!/usr/bin/env bash
# Top-level entrypoint for the LocateAnything-3B server setup.
#
# This script orchestrates four sub-scripts under scripts/ in order:
#   00_validate_host.sh        — strict host-precondition checks
#   01_download_weights.sh     — pull weights + assets to ./models/
#   02_build_image.sh          — build the Docker image
#   03_start_service.sh        — run the container, wait for health
#   04_smoke_test.sh           — hit the running service
#
# Each sub-script is idempotent. Re-running setup.sh on a healthy install
# is a no-op (validates everything, builds nothing).
#
# No fallbacks. Any failure aborts with a precise diagnostic and a hint
# at where to look. See README.md "Diagnosing failures" for the playbook.

set -Eeuo pipefail

_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=scripts/lib/common.sh
source "${_SCRIPT_DIR}/scripts/lib/common.sh"
load_versions

START_TS=$(date +%s)

log_section "LocateAnything-3B server setup"
log_info "Project root: ${_SCRIPT_DIR}"
log_info "Image tag:    ${LA_IMAGE_TAG}"
log_info "Bind:         ${LA_HOST_BIND_IP}:${LA_HOST_PORT} (loopback only)"

# --- step 0: host validation --------------------------------------------
bash "${_SCRIPT_DIR}/scripts/00_validate_host.sh"

# --- step 1: weights + Cargo.lock + calibration image -------------------
bash "${_SCRIPT_DIR}/scripts/01_download_weights.sh"

# --- step 2: docker build ------------------------------------------------
bash "${_SCRIPT_DIR}/scripts/02_build_image.sh"

# --- step 3: start container & wait healthy ------------------------------
bash "${_SCRIPT_DIR}/scripts/03_start_service.sh"

# --- step 4: smoke test --------------------------------------------------
bash "${_SCRIPT_DIR}/scripts/04_smoke_test.sh"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
log_section "Setup complete in ${ELAPSED}s"
log_info "Service is running at http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}"
log_info "Inspect logs: docker logs -f ${LA_CONTAINER_NAME}"
log_info "Stop service: docker stop ${LA_CONTAINER_NAME}"
log_info "See docs/ for capability matrix, pixel-token math, and the drone-detection caveat."
