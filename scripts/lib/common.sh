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

    # Optional install-time overlay. `setup.sh --bind-all-interfaces` /
    # `--bind-loopback` write `${PROJECT_ROOT}/install_state.env` so a
    # bind-IP choice persists across re-runs of setup.sh and across
    # container restarts. Strict allowlist: refuses any unknown key
    # and any unknown value, so a typo in the file cannot quietly
    # change behavior every script downstream depends on.
    local state_file="${_PROJECT_ROOT}/install_state.env"
    [[ -f "${state_file}" ]] || return 0

    local line key
    while IFS= read -r line || [[ -n "${line}" ]]; do
        # Trim leading + trailing whitespace.
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        # Skip blank lines and comment lines.
        [[ -z "${line}" || "${line:0:1}" == "#" ]] && continue
        if [[ ! "${line}" =~ ^([A-Z_][A-Z0-9_]*)= ]]; then
            die "install_state.env line ${line@Q} is not in KEY=value form. Refusing to apply — delete the file or fix the line."
        fi
        key="${BASH_REMATCH[1]}"
        case "${key}" in
            LA_HOST_BIND_IP) ;;
            *)
                die "install_state.env contains key '${key}'. Only LA_HOST_BIND_IP is permitted by the allowlist in scripts/lib/common.sh::load_versions. Refusing to apply."
                ;;
        esac
    done < "${state_file}"

    # Every line had a permitted key — safe to source the file as bash.
    # shellcheck disable=SC1090
    source "${state_file}"

    case "${LA_HOST_BIND_IP}" in
        127.0.0.1|0.0.0.0) ;;
        *)
            die "install_state.env LA_HOST_BIND_IP=${LA_HOST_BIND_IP@Q} is not one of the supported values (127.0.0.1, 0.0.0.0). Refusing to apply."
            ;;
    esac
}

project_root() {
    printf "%s" "${_PROJECT_ROOT}"
}
