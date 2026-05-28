#!/usr/bin/env bash
# Container entrypoint: supervise the Python worker and Rust server.
#
# - Start the Python worker first; it must publish its Unix socket before
#   the Rust frontend opens its TCP listener.
# - If either process exits, kill the other and propagate the exit code.
# - Forward SIGTERM/SIGINT to both children cleanly.
#
# NO FALLBACKS. If the Python worker fails to start, the container exits
# non-zero. The Docker healthcheck and orchestrator decide what to do next.

set -Eeuo pipefail

if [[ -z "${LA_IPC_SOCKET:-}" ]]; then
    echo "[entrypoint] FAIL: LA_IPC_SOCKET env var is unset" >&2
    exit 64
fi
if [[ -z "${LA_INTERNAL_PORT:-}" ]]; then
    echo "[entrypoint] FAIL: LA_INTERNAL_PORT env var is unset" >&2
    exit 64
fi

# Remove stale socket from a previous container instance (bind-mount path).
rm -f "${LA_IPC_SOCKET}" 2>/dev/null || true

echo "[entrypoint] starting python worker"
python -u -m worker.la_worker --socket "${LA_IPC_SOCKET}" &
PY_PID=$!

echo "[entrypoint] python pid=${PY_PID}; starting rust frontend"
# All caps come through env vars set in the Dockerfile (LA_MAX_IMAGE_DIM,
# LA_MAX_JPEG_BYTES, LA_MAX_INFLIGHT). LA_BIND is computed here from
# LA_INTERNAL_PORT — the Rust binary refuses to start without it.
LA_BIND="0.0.0.0:${LA_INTERNAL_PORT}" \
    /usr/local/bin/la_server \
    --log-format json &
RUST_PID=$!

echo "[entrypoint] rust pid=${RUST_PID}"

# Forward signals.
forward_signal() {
    local sig="$1"
    echo "[entrypoint] forwarding ${sig} to children"
    kill -"${sig}" "${PY_PID}" 2>/dev/null || true
    kill -"${sig}" "${RUST_PID}" 2>/dev/null || true
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT'  INT

# Wait for ANY child to exit, then kill the other and exit with its code.
# `wait -n` returns the exit code of the first child that exits.
set +e
wait -n "${PY_PID}" "${RUST_PID}"
FIRST_EXIT=$?
set -e

echo "[entrypoint] one child exited with code=${FIRST_EXIT}; shutting down peer"
kill -TERM "${PY_PID}" 2>/dev/null || true
kill -TERM "${RUST_PID}" 2>/dev/null || true

# Give them 5 seconds to exit gracefully, then SIGKILL.
for _ in $(seq 1 10); do
    if ! kill -0 "${PY_PID}" 2>/dev/null && ! kill -0 "${RUST_PID}" 2>/dev/null; then
        break
    fi
    sleep 0.5
done
kill -KILL "${PY_PID}" 2>/dev/null || true
kill -KILL "${RUST_PID}" 2>/dev/null || true

echo "[entrypoint] exit=${FIRST_EXIT}"
exit "${FIRST_EXIT}"
