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
    docker run --rm \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -e HF_HUB_DISABLE_TELEMETRY=1 \
        -v "${LOCAL_MODEL_DIR}":/dst \
        --user "$(id -u):$(id -g)" \
        "python:3.12-slim-bookworm" \
        bash -c "
            set -Eeuo pipefail
            pip install --quiet --no-cache-dir huggingface_hub==${LA_HFHUB_VERSION} hf_transfer==${LA_HF_TRANSFER_VERSION}
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

# --- Calibration test image (required for boot self-test) ---
CAL_IMG="${TEST_DATA_DIR}/calibration.jpg"
if [[ -f "${CAL_IMG}" && -s "${CAL_IMG}" ]]; then
    log_ok "Calibration image already present: ${CAL_IMG}"
else
    log_info "Generating synthetic calibration image"
    # Synthesize a 1024x768 RGB image with three simple objects so the model
    # can produce a parseable output during boot calibration. This avoids
    # fetching a third-party image and keeps the build hermetic.
    docker run --rm \
        -v "${TEST_DATA_DIR}":/dst \
        --user "$(id -u):$(id -g)" \
        "python:3.12-slim-bookworm" \
        bash -c "
            set -Eeuo pipefail
            pip install --quiet --no-cache-dir Pillow==11.1.0
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
