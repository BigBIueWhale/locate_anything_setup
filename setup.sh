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
_bind_descriptor() {
    # Render a human-readable suffix describing the current bind choice.
    if [[ "${LA_HOST_BIND_IP}" == "0.0.0.0" ]]; then
        printf "ALL INTERFACES — see docs/SECURITY.md"
    else
        printf "loopback only"
    fi
}

print_help() {
    cat <<EOF
setup.sh — orchestrate the LocateAnything-3B server setup.

Usage:
    bash setup.sh [--bind-all-interfaces | --bind-loopback] [-h|--help]

Runs five sub-scripts under scripts/ in order. Each is idempotent;
re-running on a healthy install is a verify-and-skip no-op.

Bind options (mutually exclusive, opt-in, persistent):

  --bind-all-interfaces  Publish the host-side HTTP / WebSocket endpoint
                         on 0.0.0.0:${LA_HOST_PORT} so it is reachable from
                         ANY network this machine sits on. Writes
                         install_state.env at the project root so the
                         choice persists across every re-run of setup.sh
                         and every container restart.

                         *** NO AUTHENTICATION. NO RATE-LIMITING. NO TLS. ***
                         Use ONLY on a network you trust to host an
                         unauthenticated inference service. The full
                         threat model is in docs/SECURITY.md §"What we
                         do NOT do".

  --bind-loopback        Restore the default loopback-only bind
                         (127.0.0.1:${LA_HOST_PORT}) by removing
                         install_state.env.

  (no flag)              Honor an existing install_state.env if any,
                         otherwise default to 127.0.0.1:${LA_HOST_PORT}.

  scripts/00_validate_host.sh     host preflight (driver, GPU, Docker,
                                  nvidia-container-toolkit, free disk,
                                  free port). Refuses to run as root.
  scripts/02_build_image.sh       docker build. First run is ~25 min,
                                  dominated by the flash-attn 2.8.3
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
($(_bind_descriptor)). Tools:

    docker stop  ${LA_CONTAINER_NAME}       stop the service
    docker start ${LA_CONTAINER_NAME}       restart it
    docker logs -f --tail 200 ${LA_CONTAINER_NAME}
    bash uninstall.sh --help                inspect uninstall options

EOF
}

# Strict arg parsing. Bind-flags are mutually exclusive; any unknown
# token aborts BEFORE any action — no silent-ignore. The bind state
# is materialised to install_state.env BEFORE the first sub-script runs,
# so every script (this one and the five under scripts/) sees the same
# overlay via scripts/lib/common.sh::load_versions.
BIND_FLAG=""
for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            print_help
            exit 0
            ;;
        --bind-all-interfaces|--bind-loopback)
            if [[ -n "${BIND_FLAG}" && "${BIND_FLAG}" != "${arg}" ]]; then
                log_err "conflicting bind flags: ${BIND_FLAG} and ${arg} are mutually exclusive."
                log_err "Run 'bash setup.sh --help' for usage."
                exit 2
            fi
            BIND_FLAG="${arg}"
            ;;
        *)
            log_err "unknown argument: ${arg@Q}"
            log_err "Run 'bash setup.sh --help' for usage."
            exit 2
            ;;
    esac
done

STATE_FILE="${_SCRIPT_DIR}/install_state.env"

case "${BIND_FLAG}" in
    --bind-all-interfaces)
        log_warn "Writing install_state.env: LA_HOST_BIND_IP=0.0.0.0 (opt-in)."
        log_warn ""
        log_warn "    *** PUBLIC NETWORK BIND ***"
        log_warn "    The HTTP / WebSocket endpoint will be reachable from any host"
        log_warn "    that can reach this machine on port ${LA_HOST_PORT}. There is"
        log_warn "    NO authentication, NO rate-limiting, NO TLS on this server."
        log_warn "    Use only on networks you trust. Full threat model in"
        log_warn "    docs/SECURITY.md. To revert: bash setup.sh --bind-loopback"
        log_warn ""
        cat > "${STATE_FILE}" <<'STATE'
# install_state.env — written by `setup.sh --bind-all-interfaces`.
# DO NOT HAND-EDIT. Re-run setup.sh with one of:
#   --bind-loopback         restore the default (127.0.0.1, loopback only)
#   --bind-all-interfaces   bind the host-side publish to 0.0.0.0
# This file is sourced by scripts/lib/common.sh::load_versions AFTER
# versions.sh, overlaying the LA_HOST_BIND_IP default for every script
# in the project. The allowlist in common.sh refuses any other key.
LA_HOST_BIND_IP="0.0.0.0"
STATE
        chmod 0644 "${STATE_FILE}"
        # The initial `load_versions` at the top of this script captured
        # the pre-flag values; re-load so the rest of setup.sh (logging,
        # the help block in error messages) reflects the new state.
        load_versions
        ;;
    --bind-loopback)
        if [[ -f "${STATE_FILE}" ]]; then
            log_info "--bind-loopback: removing install_state.env to restore default 127.0.0.1."
            rm -f -- "${STATE_FILE}"
            load_versions
        else
            log_info "--bind-loopback: no install_state.env present; already default 127.0.0.1."
        fi
        ;;
    "")
        # No flag — honor any existing state file (already loaded by
        # `load_versions` at the top of this script).
        ;;
esac

START_TS=$(date +%s)

log_section "LocateAnything-3B server setup"
log_info "Project root: ${_SCRIPT_DIR}"
log_info "Image tag:    ${LA_IMAGE_TAG}"
log_info "Bind:         ${LA_HOST_BIND_IP}:${LA_HOST_PORT} ($(_bind_descriptor))"

# Refuse to install on top of an existing instance. The project keeps
# ONE canonical teardown path — uninstall.sh — and setup.sh is not it.
# Doing a silent docker-stop+rm here would diverge from the documented
# `uninstall.sh && setup.sh` flow in corner cases (image rotation,
# weight cache state, etc.), so we make the operator run the canonical
# teardown explicitly. The bind state in install_state.env (if any) is
# preserved across the teardown so re-running setup.sh after uninstall
# still honors the bind intent.
if command -v docker >/dev/null 2>&1 \
        && docker ps -aq -f name="^${LA_CONTAINER_NAME}\$" 2>/dev/null | grep -q .; then
    log_err "A '${LA_CONTAINER_NAME}' container already exists. setup.sh installs into a clean state."
    log_err ""
    log_err "  Run the canonical teardown first, then re-run setup.sh:"
    log_err "    bash uninstall.sh --help          # see options"
    log_err "    bash uninstall.sh                 # container-only removal"
    log_err "    bash setup.sh                     # honors current install_state.env, no flag needed"
    log_err ""
    log_err "  If you supplied --bind-all-interfaces or --bind-loopback this run,"
    log_err "  install_state.env was already updated to reflect that intent and will be"
    log_err "  honored by the next setup.sh run."
    log_err "Refusing to proceed."
    exit 2
fi

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
if [[ "${LA_HOST_BIND_IP}" == "0.0.0.0" ]]; then
    LA_HOST_LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "${LA_HOST_LAN_IP:-}" ]]; then
        log_info "  Reachable from network as: http://${LA_HOST_LAN_IP}:${LA_HOST_PORT}"
    fi
    log_info "  (bound to all interfaces — see docs/SECURITY.md for the threat model)"
fi
log_info "Inspect logs: docker logs -f ${LA_CONTAINER_NAME}"
log_info "Stop service: docker stop ${LA_CONTAINER_NAME}"
log_info "See docs/ for capability matrix, pixel-token math, and the drone-detection caveat."
