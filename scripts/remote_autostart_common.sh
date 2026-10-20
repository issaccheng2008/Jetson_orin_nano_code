#!/usr/bin/env bash
# Configuration/preflight only; sourcing this file never starts a policy.
set -euo pipefail
REMOTE_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REMOTE_CONFIG="${REMOTE_CONFIG:-$REMOTE_REPO_DIR/config/remote_start.env}"

load_remote_config() {
    [[ -f "$REMOTE_CONFIG" ]] || { printf 'Missing config: %s. Run scripts/install_remote_autostart.sh first.\n' "$REMOTE_CONFIG" >&2; return 1; }
    # Private, user-owned file is trusted shell syntax, not EnvironmentFile syntax.
    if [[ "$(uname -s)" = Linux ]]; then
        [[ "$(stat -c '%a' -- "$REMOTE_CONFIG")" = 600 && -O "$REMOTE_CONFIG" ]] || {
            printf 'Remote config must be owned by the current user with mode 0600: %s\n' "$REMOTE_CONFIG" >&2; return 1;
        }
    fi
    REPO_DIR="$REMOTE_REPO_DIR"
    POLICY_PYTHON="$HOME/venvs/humanoid_policy/bin/python"
    WALKING_MODEL="$REPO_DIR/humanoid_jetson_deploy/policy(13).onnx"
    STM32_PORT=/dev/ttyACM0
    REMOTE_BIND=0.0.0.0
    REMOTE_PORT=8080
    POLICY_PORT=5007
    POLICY_STARTUP_TIMEOUT=20
    MAX_VX=0.4
    MAX_WZ=0.5
    POLICY_STEP_CM=''
    POLICY_KP_SCALE=1
    POLICY_KD_SCALE=1
    source "$REMOTE_CONFIG"
    local name
    for name in REPO_DIR POLICY_PYTHON WALKING_MODEL STM32_PORT REMOTE_BIND REMOTE_PORT POLICY_PORT POLICY_STARTUP_TIMEOUT MAX_VX MAX_WZ POLICY_KP_SCALE POLICY_KD_SCALE ROBOT_REMOTE_TOKEN; do
        [[ -n "${!name:-}" ]] || { printf 'Missing or empty setting %s in %s\n' "$name" "$REMOTE_CONFIG" >&2; return 1; }
    done
    [[ "$REPO_DIR" = /* && "$POLICY_PYTHON" = /* && "$WALKING_MODEL" = /* ]] || {
        printf 'REPO_DIR, POLICY_PYTHON and WALKING_MODEL must be absolute paths.\n' >&2; return 1;
    }
    export ROBOT_REMOTE_TOKEN PYTHONUNBUFFERED=1
}

print_remote_command() {
    printf '%q ' "$@"
    printf '\n'
}

preflight_remote_config() {
    local file
    for file in "$REPO_DIR/remote_control/__main__.py" "$REPO_DIR/humanoid_jetson_deploy/main.py" "$WALKING_MODEL"; do
        [[ -f "$file" ]] || { printf 'Missing required file: %s\n' "$file" >&2; return 1; }
    done
    [[ -x "$POLICY_PYTHON" ]] || { printf 'Policy Python is not executable: %s\n' "$POLICY_PYTHON" >&2; return 1; }
    (
        cd -- "$REPO_DIR"
        "$POLICY_PYTHON" -c 'import numpy, onnxruntime, serial; import remote_control.server; print("Remote/policy dependencies OK")'
        "$POLICY_PYTHON" -c '
import math, os, sys
web, policy = map(int, sys.argv[1:3])
if not (1 <= web <= 65535 and 1 <= policy <= 65535 and web != policy):
    raise ValueError("Invalid/conflicting ports")
# Match ControlState and PolicyProcess without constructing a process,
# temporary ready directory, network listener, or serial connection.
def number(text, name, lower, upper, inclusive_lower=False):
    value = float(text)
    lower_ok = value >= lower if inclusive_lower else value > lower
    if not (math.isfinite(value) and lower_ok and value <= upper):
        raise ValueError(name + " is outside the supported range")
number(sys.argv[3], "POLICY_STARTUP_TIMEOUT", 0, math.inf)
number(sys.argv[4], "MAX_VX", 0, 1)
number(sys.argv[5], "MAX_WZ", 0, .5)
number(sys.argv[6], "POLICY_KP_SCALE", 0, 1, True)
number(sys.argv[7], "POLICY_KD_SCALE", 0, 1, True)
if len(sys.argv) > 8:
    number(sys.argv[8], "POLICY_STEP_CM", 0, math.inf)
token = os.environ.get("ROBOT_REMOTE_TOKEN", "")
if not token or not token.isascii() or any(c.isspace() for c in token):
    raise ValueError("ROBOT_REMOTE_TOKEN must be nonempty ASCII without whitespace")
' "$REMOTE_PORT" "$POLICY_PORT" "$POLICY_STARTUP_TIMEOUT" "$MAX_VX" "$MAX_WZ" "$POLICY_KP_SCALE" "$POLICY_KD_SCALE" ${POLICY_STEP_CM:+"$POLICY_STEP_CM"}
    )
}
