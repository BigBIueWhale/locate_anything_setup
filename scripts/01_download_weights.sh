#!/usr/bin/env bash
# Download LocateAnything-3B weights and config to the local model directory.
# Idempotent: if the directory already contains a complete snapshot, the
# script verifies size and prints a no-op message.
#
# Uses the huggingface_hub CLI inside a one-off Docker container so we don't
# pollute the host's Python environment.

set -Eeuo pipefail
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

print_help() {
    cat <<EOF
01_download_weights.sh — pull the model snapshot and generate the
calibration image.

Usage:
    bash scripts/01_download_weights.sh [-h|--help]

Three concrete actions:

    1. Pull ${LA_MODEL_HF_REPO}
       at commit ${LA_MODEL_HF_REVISION}
       into ./models/LocateAnything-3B/ via huggingface_hub's
       snapshot_download (~7.66 GiB on first run).
    2. Synthesize ./test_data/calibration.jpg (1024x768 RGB with four
       polygons) for the in-container boot self-test.
    3. Regenerate rust_server/Cargo.lock if missing, using the host's
       cargo. (Already present on a normal checkout.)

Prerequisite: the main image at '${LA_IMAGE_TAG}' must exist
locally — this script uses it as the Python helper for steps 1 and 2
(it already carries huggingface_hub, hf_transfer, and Pillow at
pinned versions). Run scripts/02_build_image.sh first, or use
setup.sh which orders the steps correctly.

Idempotent: each step skips itself if its output already exists at
sufficient size. To force a re-download, delete ./models/LocateAnything-3B/.

Network: needed only when ./models/LocateAnything-3B/ is missing or
incomplete. The huggingface_hub download is checked against HF's LFS
SHA-256 per file during download. Boot-time content verification
(scripts/lib/versions.sh-pinned per-file SHA-256) catches any
corruption introduced after the download succeeded.

Runs under Docker as the host user (no sudo, no host pip pollution).

EOF
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help) print_help; exit 0 ;;
        *) log_err "unknown argument: ${arg@Q}"
           log_err "Run 'bash scripts/01_download_weights.sh --help' for usage."
           exit 2 ;;
    esac
done

PROJECT_ROOT="$(project_root)"
LOCAL_MODEL_DIR="${PROJECT_ROOT}/models/LocateAnything-3B"
TEST_DATA_DIR="${PROJECT_ROOT}/test_data"

log_section "Downloading ${LA_MODEL_HF_REPO} @ ${LA_MODEL_HF_REVISION}"

mkdir -p "${LOCAL_MODEL_DIR}"
mkdir -p "${TEST_DATA_DIR}"

# --- Skip if already populated ---
existing_safetensors=$(find "${LOCAL_MODEL_DIR}" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
required_min_bytes=$(( 7 * 1024 * 1024 * 1024 ))   # 7 GiB
if (( existing_safetensors >= required_min_bytes )); then
    log_ok "Local snapshot already complete (${existing_safetensors} bytes safetensors)."
    log_info "Re-download by deleting ${LOCAL_MODEL_DIR}."
else
    log_info "Local snapshot incomplete or absent — downloading."
    # The helper here runs INSIDE the main project image
    # (${LA_IMAGE_TAG}), which already carries pinned huggingface_hub,
    # hf_transfer, and Pillow from scripts/02_build_image.sh. We override
    # the image's ENTRYPOINT (tini + the worker supervisor) with an
    # explicit `bash -c "python ..."`. Three concrete benefits:
    #
    #   - No second base image to pull (python:3.12-slim-bookworm is
    #     gone). One less network dependency on every re-run.
    #   - No `pip install` from PyPI. The deps are already content-
    #     pinned in the main image's layers.
    #   - No /tmp / HOME / chown shenanigans: the image's `la` user
    #     UID maps to the host UID via --user, so files written into
    #     the bind-mount /dst are host-owned automatically.
    #
    # Precondition: ${LA_IMAGE_TAG} must exist locally — which is why
    # setup.sh runs scripts/02_build_image.sh BEFORE this script.
    if ! docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
        die "image '${LA_IMAGE_TAG}' not present locally. Run scripts/02_build_image.sh first, or use setup.sh which orders steps correctly."
    fi
    docker run --rm \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -e HF_HUB_DISABLE_TELEMETRY=1 \
        -v "${LOCAL_MODEL_DIR}":/dst \
        --user "$(id -u):$(id -g)" \
        --entrypoint "" \
        "${LA_IMAGE_TAG}" \
        bash -c "
            set -Eeuo pipefail
            python -c '
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=\"${LA_MODEL_HF_REPO}\",
    revision=\"${LA_MODEL_HF_REVISION}\",
    local_dir=\"/dst\",
    local_dir_use_symlinks=False,
    max_workers=8,
)
print(\"snapshot at\", path)
'
        "

    final_bytes=$(find "${LOCAL_MODEL_DIR}" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if (( final_bytes < required_min_bytes )); then
        die "Downloaded safetensors total ${final_bytes} bytes < required ${required_min_bytes} (~7 GiB). Retry."
    fi
    log_ok "Snapshot complete: ${final_bytes} bytes of safetensors."
fi

# --- Synthetic smoke-test image (required by scripts/04_smoke_test.sh
#     and scripts/05_concurrency_smoke.sh; the worker's boot calibration
#     uses test_data/drone_sirius.jpg by default, which is committed to
#     the repo rather than synthesized) ---
CAL_IMG="${TEST_DATA_DIR}/calibration.jpg"
if [[ -f "${CAL_IMG}" && -s "${CAL_IMG}" ]]; then
    log_ok "Smoke-test image already present: ${CAL_IMG}"
else
    log_info "Generating synthetic smoke-test image"
    # Synthesize a 1024x768 RGB image with three simple objects so the
    # smoke test ("we know what categories to expect") has a predictable
    # target. This avoids fetching a third-party image and keeps the
    # build hermetic.
    # Same main-image-as-helper pattern as the snapshot block above —
    # Pillow is already installed there.
    if ! docker image inspect "${LA_IMAGE_TAG}" >/dev/null 2>&1; then
        die "image '${LA_IMAGE_TAG}' not present locally. Run scripts/02_build_image.sh first, or use setup.sh which orders steps correctly."
    fi
    docker run --rm \
        -v "${TEST_DATA_DIR}":/dst \
        --user "$(id -u):$(id -g)" \
        --entrypoint "" \
        "${LA_IMAGE_TAG}" \
        bash -c "
            set -Eeuo pipefail
            python <<'PY'
from PIL import Image, ImageDraw
img = Image.new('RGB', (1024, 768), (230, 230, 235))
d = ImageDraw.Draw(img)
# A 'bottle' silhouette
d.rounded_rectangle((120, 220, 240, 580), radius=24, fill=(70, 110, 70))
d.rectangle((150, 180, 210, 230), fill=(70, 110, 70))
# A 'book' silhouette
d.rectangle((360, 340, 660, 500), fill=(150, 90, 70))
d.line((360, 340, 660, 340), fill=(110, 60, 40), width=4)
# A 'cup' silhouette
d.ellipse((760, 440, 920, 540), fill=(60, 60, 70))
d.rectangle((760, 350, 920, 460), fill=(70, 80, 100))
# A 'laptop' silhouette
d.polygon([(180, 660), (840, 660), (920, 720), (100, 720)], fill=(40, 40, 50))
d.rectangle((220, 540, 800, 660), fill=(80, 80, 100))
img.save('/dst/calibration.jpg', 'JPEG', quality=92)
print('wrote /dst/calibration.jpg')
PY
        "
    [[ -s "${CAL_IMG}" ]] || die "calibration image was not created"
    log_ok "Generated calibration image at ${CAL_IMG}"
fi

# --- Regenerate Cargo.lock if missing ---
if [[ ! -f "${PROJECT_ROOT}/rust_server/Cargo.lock" ]]; then
    if ! command -v cargo >/dev/null 2>&1; then
        die "rust_server/Cargo.lock is missing and host 'cargo' is not on PATH. Install Rust on the host (per §16 of personal_server) or commit Cargo.lock."
    fi
    log_info "Regenerating rust_server/Cargo.lock"
    ( cd "${PROJECT_ROOT}/rust_server" && cargo generate-lockfile )
    log_ok "Cargo.lock regenerated"
fi

log_section "Weights and assets ready"
