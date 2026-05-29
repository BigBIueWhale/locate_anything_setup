#!/usr/bin/env bash
# Destructive teardown for the LocateAnything-3B server.
#
# Usage:
#   bash uninstall.sh                         # just remove the running container
#   bash uninstall.sh --remove-image          # also delete the built Docker image
#   bash uninstall.sh --remove-weights        # also delete ./models/ and calibration.jpg
#   bash uninstall.sh --remove-hf-cache       # also delete ./cache/huggingface
#   bash uninstall.sh --remove-rust-target    # also delete ./rust_server/target
#   bash uninstall.sh --purge                 # all of the above (image+weights+cache+target)
#   bash uninstall.sh --yes                   # skip the confirmation prompt
#   bash uninstall.sh --help
#
# The script never silently swallows errors. Each step:
#   • inspects whether the target exists,
#   • logs `present → removing` or `absent → skipping`,
#   • performs the removal,
#   • asserts the post-condition.
#
# The only way this script can fail is a logic bug or a system the OS is
# refusing operations on (e.g., the docker daemon is dead — in which case
# we report it and exit non-zero so you can investigate). It does NOT
# fall through to a "best-effort" cleanup.

set -Eeuo pipefail

_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=scripts/lib/common.sh
source "${_SCRIPT_DIR}/scripts/lib/common.sh"
load_versions

PROJECT_ROOT="$(project_root)"

REMOVE_IMAGE=0
REMOVE_WEIGHTS=0
REMOVE_HF_CACHE=0
REMOVE_RUST_TARGET=0
ASSUME_YES=0

print_help() {
    cat <<EOF
uninstall.sh — destructive teardown.

  --remove-image         Also delete the built Docker image (${LA_IMAGE_TAG}).
                         Re-running setup.sh will rebuild it (needs internet).
  --remove-weights       Also delete ./models/LocateAnything-3B and
                         ./test_data/calibration.jpg. Re-running setup.sh
                         will re-download (~7.66 GB; needs internet).
  --remove-hf-cache      Also delete ./cache/huggingface.
  --remove-rust-target   Also delete ./rust_server/target (build artifacts).
  --purge                Implies --remove-image, --remove-weights,
                         --remove-hf-cache, --remove-rust-target.
  --yes                  Skip the confirmation prompt.
  --help                 This message.

Default behavior (no flags): stop and remove the container instance only.
The Docker image, model weights, and HF cache are preserved so that
re-running setup.sh is fast and works offline.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --remove-image)       REMOVE_IMAGE=1 ;;
        --remove-weights)     REMOVE_WEIGHTS=1 ;;
        --remove-hf-cache)    REMOVE_HF_CACHE=1 ;;
        --remove-rust-target) REMOVE_RUST_TARGET=1 ;;
        --purge)
            REMOVE_IMAGE=1
            REMOVE_WEIGHTS=1
            REMOVE_HF_CACHE=1
            REMOVE_RUST_TARGET=1
            ;;
        --yes|-y) ASSUME_YES=1 ;;
        --help|-h) print_help; exit 0 ;;
        *) die "unknown arg ${arg@Q}. Run: bash uninstall.sh --help" ;;
    esac
done

# ----------------------------------------------------------------------
# Summarize what is about to happen and confirm with the operator.
# ----------------------------------------------------------------------
log_section "Uninstall plan"
log_info "Project:           ${PROJECT_ROOT}"
log_info "Container:         ${LA_CONTAINER_NAME} (stop + rm — always)"
log_info "Docker image:      ${LA_IMAGE_TAG} ($( [[ ${REMOVE_IMAGE} -eq 1 ]] && echo 'REMOVE' || echo 'keep' ))"
log_info "Model weights:     ${PROJECT_ROOT}/models/LocateAnything-3B \
($( [[ ${REMOVE_WEIGHTS} -eq 1 ]] && echo 'REMOVE — ~7.66 GB will need re-download' || echo 'keep' ))"
log_info "Calibration image: ${PROJECT_ROOT}/test_data/calibration.jpg \
($( [[ ${REMOVE_WEIGHTS} -eq 1 ]] && echo 'REMOVE' || echo 'keep' ))"
log_info "HF cache:          ${PROJECT_ROOT}/cache/huggingface \
($( [[ ${REMOVE_HF_CACHE} -eq 1 ]] && echo 'REMOVE' || echo 'keep' ))"
log_info "Rust target:       ${PROJECT_ROOT}/rust_server/target \
($( [[ ${REMOVE_RUST_TARGET} -eq 1 ]] && echo 'REMOVE' || echo 'keep' ))"

if [[ "${ASSUME_YES}" -ne 1 ]]; then
    printf '\nProceed? [y/N] '
    read -r REPLY
    case "${REPLY}" in
        y|Y|yes|YES) ;;
        *) die "Aborted by operator." ;;
    esac
fi

# ----------------------------------------------------------------------
# Pre-flight: docker daemon reachability. We need it for container/image
# steps; if it's down we report and refuse to silently skip those steps.
# ----------------------------------------------------------------------
DOCKER_REACHABLE=0
if docker info >/dev/null 2>&1; then
    DOCKER_REACHABLE=1
fi

# ----------------------------------------------------------------------
# Step 1 — Stop + remove the running container.
# ----------------------------------------------------------------------
log_section "Step 1: container '${LA_CONTAINER_NAME}'"
if [[ "${DOCKER_REACHABLE}" -eq 0 ]]; then
    log_warn "docker daemon not reachable — cannot operate on container; skipping container/image steps."
else
    CONTAINER_IDS=$(docker ps -aq -f "name=^${LA_CONTAINER_NAME}\$" || true)
    if [[ -n "${CONTAINER_IDS}" ]]; then
        log_info "Container present (id=${CONTAINER_IDS:0:12}); stopping…"
        docker stop "${LA_CONTAINER_NAME}" >/dev/null
        log_info "Container stopped; removing…"
        docker rm "${LA_CONTAINER_NAME}" >/dev/null
        # Verify post-condition.
        if docker ps -aq -f "name=^${LA_CONTAINER_NAME}\$" | grep -q .; then
            die "post-condition violated: container '${LA_CONTAINER_NAME}' still present after docker rm."
        fi
        log_ok "Container '${LA_CONTAINER_NAME}' removed."
    else
        log_ok "Container '${LA_CONTAINER_NAME}' not present; nothing to remove."
    fi
fi

# ----------------------------------------------------------------------
# Step 2 — Remove the Docker image (if --remove-image).
# ----------------------------------------------------------------------
log_section "Step 2: Docker image"
if [[ "${REMOVE_IMAGE}" -eq 1 ]]; then
    if [[ "${DOCKER_REACHABLE}" -eq 0 ]]; then
        die "--remove-image was requested but the docker daemon is not reachable."
    fi
    if docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
        IMG_ID=$(docker image inspect "${LA_IMAGE_TAG}" --format '{{.Id}}')
        log_info "Image present (id=${IMG_ID:7:12}); removing…"
        docker image rm "${LA_IMAGE_TAG}" >/dev/null
        if docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
            die "post-condition violated: image '${LA_IMAGE_TAG}' still present after docker image rm."
        fi
        log_ok "Image '${LA_IMAGE_TAG}' removed."
    else
        log_ok "Image '${LA_IMAGE_TAG}' not present; nothing to remove."
    fi
    # Note: we DO NOT prune the BuildKit cache here. That cache is shared
    # with other projects on the host and could speed up other unrelated
    # builds. If you want to free it, run `docker builder prune` manually.
    log_info "BuildKit layer cache preserved. Run \`docker builder prune\` manually if you want it gone."
else
    log_ok "--remove-image not set; image '${LA_IMAGE_TAG}' preserved."
fi

# ----------------------------------------------------------------------
# Step 3 — Remove model weights and calibration image.
# ----------------------------------------------------------------------
log_section "Step 3: model weights"
WEIGHT_DIR="${PROJECT_ROOT}/models/LocateAnything-3B"
CAL_IMG="${PROJECT_ROOT}/test_data/calibration.jpg"
if [[ "${REMOVE_WEIGHTS}" -eq 1 ]]; then
    if [[ -d "${WEIGHT_DIR}" ]]; then
        SZ=$(du -sb "${WEIGHT_DIR}" 2>/dev/null | awk '{print $1}')
        log_info "Weight dir present (${SZ} bytes); removing recursively…"
        rm -rf -- "${WEIGHT_DIR}"
        [[ -e "${WEIGHT_DIR}" ]] && die "post-condition violated: ${WEIGHT_DIR} still exists after rm -rf."
        log_ok "Weight directory removed."
    else
        log_ok "Weight directory not present; nothing to remove."
    fi
    if [[ -f "${CAL_IMG}" ]]; then
        log_info "Calibration image present; removing…"
        rm -f -- "${CAL_IMG}"
        [[ -e "${CAL_IMG}" ]] && die "post-condition violated: ${CAL_IMG} still exists after rm."
        log_ok "Calibration image removed."
    else
        log_ok "Calibration image not present; nothing to remove."
    fi
else
    log_ok "--remove-weights not set; ${WEIGHT_DIR} preserved."
fi

# ----------------------------------------------------------------------
# Step 4 — Remove HF cache.
# ----------------------------------------------------------------------
log_section "Step 4: HF download cache"
HF_CACHE_DIR="${PROJECT_ROOT}/cache/huggingface"
if [[ "${REMOVE_HF_CACHE}" -eq 1 ]]; then
    if [[ -d "${HF_CACHE_DIR}" ]]; then
        SZ=$(du -sb "${HF_CACHE_DIR}" 2>/dev/null | awk '{print $1}')
        log_info "HF cache present (${SZ} bytes); removing recursively…"
        rm -rf -- "${HF_CACHE_DIR}"
        [[ -e "${HF_CACHE_DIR}" ]] && die "post-condition violated: ${HF_CACHE_DIR} still exists after rm -rf."
        log_ok "HF cache removed."
    else
        log_ok "HF cache not present; nothing to remove."
    fi
else
    log_ok "--remove-hf-cache not set; ${HF_CACHE_DIR} preserved."
fi

# ----------------------------------------------------------------------
# Step 5 — Remove Rust build artifacts.
# ----------------------------------------------------------------------
log_section "Step 5: Rust target directory"
RUST_TARGET="${PROJECT_ROOT}/rust_server/target"
if [[ "${REMOVE_RUST_TARGET}" -eq 1 ]]; then
    if [[ -d "${RUST_TARGET}" ]]; then
        SZ=$(du -sb "${RUST_TARGET}" 2>/dev/null | awk '{print $1}')
        log_info "rust_server/target present (${SZ} bytes); removing recursively…"
        rm -rf -- "${RUST_TARGET}"
        [[ -e "${RUST_TARGET}" ]] && die "post-condition violated: ${RUST_TARGET} still exists after rm -rf."
        log_ok "rust_server/target removed."
    else
        log_ok "rust_server/target not present; nothing to remove."
    fi
else
    log_ok "--remove-rust-target not set; ${RUST_TARGET} preserved."
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
log_section "Uninstall complete"
# Step 1 always removed the container, so `docker start ${LA_CONTAINER_NAME}`
# would fail with "no such container" — never emit that hint. The correct
# fast-path back to a running service is to recreate the container from
# the preserved image (scripts/03_start_service.sh), but that only works
# when the image AND the calibration JPEG are still on disk. Otherwise
# the operator needs the full setup.sh.
log_info "To set up from scratch (rebuilds anything removed): bash setup.sh"
if [[ "${REMOVE_IMAGE}" -eq 0 && "${REMOVE_WEIGHTS}" -eq 0 ]]; then
    log_info "To re-create the container from the preserved image: bash scripts/03_start_service.sh"
fi
