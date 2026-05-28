#!/usr/bin/env bash
# Top-level entrypoint for the LocateAnything-3B server setup.
#
# This script orchestrates five sub-scripts under scripts/ in order:
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

# -------------------------- arg parsing --------------------------------
# Strict: only --help / -h is accepted. Anything else aborts BEFORE any
# action is taken — no silent-ignore of unknown flags. A destructive
# setup that runs because the user mistyped a flag is exactly the bug
# we're avoiding here.
print_help() {
    cat <<EOF
setup.sh — orchestrate the LocateAnything-3B server setup.

Usage:
    bash setup.sh [-h|--help]

Runs five sub-scripts under scripts/ in order. Each is idempotent;
re-running on a healthy install is a verify-and-skip no-op.

  scripts/00_validate_host.sh     host preflight (driver, GPU, Docker,
                                  nvidia-container-toolkit, free disk,
                                  free port). Refuses to run as root.
  scripts/02_build_image.sh       docker build. First run is ~25 min,
                                  dominated by the flash-attn 2.8.4
                                  source compile. Pass --rebuild to
                                  force a fresh build.
  scripts/01_download_weights.sh  pull nvidia/LocateAnything-3B at the
                                  pinned commit (~7.66 GiB on first run;
                                  no-op once present). Uses the main
                                  image (just built in step 02) as the
                                  download helper — no second base image,
                                  no pip install from PyPI.
  scripts/03_start_service.sh     docker run + wait until healthy. Also
                                  verifies the kernel-side listener is
                                  exactly ${LA_HOST_BIND_IP}:${LA_HOST_PORT}.
  scripts/04_smoke_test.sh        end-to-end JPEG → JSON round-trip
                                  through WS /v1/stream via docker exec
                                  (no network needed).

After install, the service is at http://${LA_HOST_BIND_IP}:${LA_HOST_PORT}
(loopback only — see docs/SECURITY.md). Tools:

    docker stop  ${LA_CONTAINER_NAME}       stop the service
    docker start ${LA_CONTAINER_NAME}       restart it
    docker logs -f --tail 200 ${LA_CONTAINER_NAME}
    bash uninstall.sh --help                inspect uninstall options

EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_err "unknown argument: ${arg@Q}"
            log_err "Run 'bash setup.sh --help' for usage."
            exit 2
            ;;
    esac
done

START_TS=$(date +%s)

log_section "LocateAnything-3B server setup"
log_info "Project root: ${_SCRIPT_DIR}"
log_info "Image tag:    ${LA_IMAGE_TAG}"
log_info "Bind:         ${LA_HOST_BIND_IP}:${LA_HOST_PORT} (loopback only)"

# --- step 0: host validation --------------------------------------------
bash "${_SCRIPT_DIR}/scripts/00_validate_host.sh"

# --- step 1: docker build ------------------------------------------------
# Build precedes the weight download because the weight-download helper
# uses the just-built main image (which carries huggingface_hub +
# hf_transfer + Pillow). Building first means we don't need to pull a
# second base image (python:3.12-slim-bookworm) and don't run any
# pip install from PyPI in the helper — both are offline-fragile.
bash "${_SCRIPT_DIR}/scripts/02_build_image.sh"

# --- step 2: weights + Cargo.lock + calibration image -------------------
bash "${_SCRIPT_DIR}/scripts/01_download_weights.sh"

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
