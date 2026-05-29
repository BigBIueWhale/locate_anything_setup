#!/usr/bin/env bash
# Concurrency smoke test for the live container.
#
# Opens N (default 2) concurrent WebSocket clients against the running
# service, each sending K (default 4) frames sequentially at a fixed
# interval, with a per-client start offset so frames INTERLEAVE rather
# than arrive in lock-step. Records per-frame latency, verifies frame_id
# matching, and validates the per-client median-latency spread is within
# a tolerance — the practical proxy for "the asyncio.Lock in the worker
# is acquired in FIFO order".
#
# What this catches that 04_smoke_test.sh CANNOT:
#   - one user starving another (asyncio.Lock unfair across UDS conns)
#   - WS multiplexing bug: response delivered for the wrong frame_id
#   - cross-task state leak under back-to-back load
#   - silent worker hangs that pass the single-frame health probe
#
# NOT part of setup.sh — this script is the on-demand parallelism probe.
# Run it manually after the service is healthy.

set -Eeuo pipefail
_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=lib/common.sh
source "${_SCRIPT_DIR}/lib/common.sh"
load_versions

LA_DEFAULT_PROMPT='Locate all the instances that matches the following description: bottle</c>book</c>cup</c>laptop.'
LA_DEFAULT_NUM_CLIENTS=2
LA_DEFAULT_FRAMES_PER_CLIENT=4
LA_DEFAULT_SEND_INTERVAL=2.0
LA_DEFAULT_START_OFFSET=0.5
LA_DEFAULT_FRAME_TIMEOUT=120
LA_DEFAULT_FAIRNESS_RATIO=5.0

print_help() {
    cat <<EOF
05_concurrency_smoke.sh — multi-client concurrency probe against a healthy container.

Usage:
    bash scripts/05_concurrency_smoke.sh [-h|--help]
                                          [--num-clients N]
                                          [--frames-per-client K]
                                          [--send-interval S]
                                          [--start-offset O]
                                          [--frame-timeout T]
                                          [--fairness-ratio R]

Defaults:
    --num-clients        ${LA_DEFAULT_NUM_CLIENTS}
    --frames-per-client  ${LA_DEFAULT_FRAMES_PER_CLIENT}
    --send-interval      ${LA_DEFAULT_SEND_INTERVAL}s  (between consecutive sends within a client)
    --start-offset       ${LA_DEFAULT_START_OFFSET}s  (between client K and K+1's first send)
    --frame-timeout      ${LA_DEFAULT_FRAME_TIMEOUT}s  (per-frame response deadline)
    --fairness-ratio     ${LA_DEFAULT_FAIRNESS_RATIO}× (max permitted spread of per-client median latencies)

Prerequisites:
    - The container '${LA_CONTAINER_NAME}' is running (start with scripts/03_start_service.sh).
    - calibration.jpg is bind-mounted at /opt/locate_anything/test_data/.
    - The concurrency smoke client is baked into the image at
      /opt/locate_anything/scripts/lib/concurrency_smoke_client.py.

Idempotent. Side-effect-free (no container restart, no file writes).

Exit codes:
    0   all checks passed
    1   one or more concurrency violations (timeouts, errors, fairness)
    2   prerequisite missing or invalid CLI argument

EOF
}

# ---- arg parsing ----
LA_NUM_CLIENTS="${LA_DEFAULT_NUM_CLIENTS}"
LA_FRAMES_PER_CLIENT="${LA_DEFAULT_FRAMES_PER_CLIENT}"
LA_SEND_INTERVAL="${LA_DEFAULT_SEND_INTERVAL}"
LA_START_OFFSET="${LA_DEFAULT_START_OFFSET}"
LA_FRAME_TIMEOUT="${LA_DEFAULT_FRAME_TIMEOUT}"
LA_FAIRNESS_RATIO="${LA_DEFAULT_FAIRNESS_RATIO}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        --num-clients)        LA_NUM_CLIENTS="$2";        shift 2 ;;
        --frames-per-client)  LA_FRAMES_PER_CLIENT="$2";  shift 2 ;;
        --send-interval)      LA_SEND_INTERVAL="$2";      shift 2 ;;
        --start-offset)       LA_START_OFFSET="$2";       shift 2 ;;
        --frame-timeout)      LA_FRAME_TIMEOUT="$2";      shift 2 ;;
        --fairness-ratio)     LA_FAIRNESS_RATIO="$2";     shift 2 ;;
        *)
            log_err "unknown argument: ${1@Q}"
            log_err "Run 'bash scripts/05_concurrency_smoke.sh --help' for usage."
            exit 2
            ;;
    esac
done

# ---- prerequisite checks ----
log_section "Concurrency smoke against ${LA_CONTAINER_NAME}"

if ! docker ps --format '{{.Names}}' | grep -qx "${LA_CONTAINER_NAME}"; then
    die "container '${LA_CONTAINER_NAME}' is not running.
Bring it up first: bash scripts/03_start_service.sh
Then re-run this script."
fi

# Verify the container's health is "healthy" before stressing it — otherwise
# the test's failure messages will conflate "container not ready yet" with
# "worker has a real concurrency bug", making diagnosis ambiguous.
HEALTH=$(docker inspect -f '{{.State.Health.Status}}' "${LA_CONTAINER_NAME}" 2>/dev/null || echo "unknown")
if [[ "${HEALTH}" != "healthy" ]]; then
    die "container '${LA_CONTAINER_NAME}' health status is '${HEALTH}' (expected 'healthy').
Wait for the service to finish booting (model load + calibration) before
running concurrency tests. Tail the logs:
    docker logs -f ${LA_CONTAINER_NAME}"
fi

if ! docker exec "${LA_CONTAINER_NAME}" \
        test -f /opt/locate_anything/test_data/calibration.jpg; then
    die "calibration.jpg is missing INSIDE the container at /opt/locate_anything/test_data/.
The container reads it via a read-only bind mount of the host's ./test_data/.
Re-run scripts/03_start_service.sh after confirming ./test_data/calibration.jpg
exists on the host."
fi

if ! docker exec "${LA_CONTAINER_NAME}" \
        test -f /opt/locate_anything/scripts/lib/concurrency_smoke_client.py; then
    die "concurrency_smoke_client.py is missing INSIDE the container at
/opt/locate_anything/scripts/lib/. The image was built before this file
was added. Re-build: bash scripts/02_build_image.sh --rebuild
then restart: bash scripts/03_start_service.sh"
fi

# ---- run the probe ----
log_info "running ${LA_NUM_CLIENTS} concurrent WS clients, "\
"${LA_FRAMES_PER_CLIENT} frames each, "\
"send-interval=${LA_SEND_INTERVAL}s, start-offset=${LA_START_OFFSET}s, "\
"fairness-ratio=${LA_FAIRNESS_RATIO}×"

# `docker exec` propagates the helper's exit code, which the helper sets
# from its run() return value: 0 = pass, 1 = concurrency violation,
# 2 = prerequisite/usage issue. `set -e` above will let any non-zero
# code abort with the standard error trap.
docker exec "${LA_CONTAINER_NAME}" \
    python /opt/locate_anything/scripts/lib/concurrency_smoke_client.py \
        --url "ws://127.0.0.1:${LA_INTERNAL_PORT}/v1/stream" \
        --image "/opt/locate_anything/test_data/calibration.jpg" \
        --prompt "${LA_DEFAULT_PROMPT}" \
        --mode "hybrid" \
        --num-clients "${LA_NUM_CLIENTS}" \
        --frames-per-client "${LA_FRAMES_PER_CLIENT}" \
        --send-interval "${LA_SEND_INTERVAL}" \
        --start-offset "${LA_START_OFFSET}" \
        --frame-timeout "${LA_FRAME_TIMEOUT}" \
        --fairness-ratio "${LA_FAIRNESS_RATIO}"

log_ok "concurrency smoke passed"
