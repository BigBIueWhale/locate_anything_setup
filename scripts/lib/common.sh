# shellcheck shell=bash
# Shared helpers for every script. Source this, then call `load_versions`.

set -Eeuo pipefail

# Disable shell traps that would let a non-zero exit silently propagate.
trap 'on_err $? $LINENO $BASH_COMMAND' ERR

_SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
_PROJECT_ROOT="$( cd "${_SELF_DIR}/../.." && pwd )"

# ANSI colors (only when stdout is a tty).
if [[ -t 1 ]]; then
    _C_RED=$'\033[31m'
    _C_GREEN=$'\033[32m'
    _C_YELLOW=$'\033[33m'
    _C_BLUE=$'\033[34m'
    _C_BOLD=$'\033[1m'
    _C_RESET=$'\033[0m'
else
    _C_RED='' _C_GREEN='' _C_YELLOW='' _C_BLUE='' _C_BOLD='' _C_RESET=''
fi

log_info()    { printf "%s[INFO]%s %s\n"  "${_C_BLUE}"   "${_C_RESET}" "$*"; }
log_ok()      { printf "%s[OK]%s   %s\n"  "${_C_GREEN}"  "${_C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN]%s %s\n"  "${_C_YELLOW}" "${_C_RESET}" "$*" >&2; }
log_err()     { printf "%s[FAIL]%s %s\n"  "${_C_RED}"    "${_C_RESET}" "$*" >&2; }
log_section() {
    printf "\n%s==== %s ====%s\n" "${_C_BOLD}" "$*" "${_C_RESET}"
}

on_err() {
    local exit_code="$1" line_no="$2" cmd="$3"
    log_err "Script aborted at line ${line_no} (exit=${exit_code}): ${cmd}"
    log_err "No fallback was attempted. The error above is the root cause."
    log_err "See README.md \"Diagnosing failures\" for the next step."
}

die() {
    log_err "$*"
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found on PATH: '${cmd}'."
}

# Exactly-equal version compare. Use for strict pinning; do not call for ranges.
require_version_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" != "$actual" ]]; then
        die "Version mismatch for ${label}: expected '${expected}', got '${actual}'."
    fi
}

# Numeric ≥ compare (dot-separated). Returns 0 if $1 ≥ $2, else 1. No fallbacks.
version_ge() {
    local a="$1" b="$2"
    [[ "$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -n1)" = "$b" ]]
}

load_versions() {
    # shellcheck disable=SC1091
    source "${_SELF_DIR}/versions.sh"
}

project_root() {
    printf "%s" "${_PROJECT_ROOT}"
}
