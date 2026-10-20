#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/remote_autostart_common.sh"
[[ $# -eq 0 || ( $# -eq 1 && "$1" = --dry-run ) ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
load_remote_config
command=("$POLICY_PYTHON" -m remote_control --launch-policy
    --policy-python "$POLICY_PYTHON" --model "$WALKING_MODEL" --serial-port "$STM32_PORT"
    --policy-port "$POLICY_PORT" --port "$REMOTE_PORT" --bind "$REMOTE_BIND"
    --policy-startup-timeout "$POLICY_STARTUP_TIMEOUT" --max-vx "$MAX_VX" --max-wz "$MAX_WZ"
    --policy-kp-scale "$POLICY_KP_SCALE" --policy-kd-scale "$POLICY_KD_SCALE")
[[ -z "$POLICY_STEP_CM" ]] || command+=(--policy-step-cm "$POLICY_STEP_CM")
if [[ "${1:-}" = --dry-run ]]; then
    print_remote_command "${command[@]}"
    exit 0
fi
preflight_remote_config
cd -- "$REPO_DIR"
# The server starts the policy only when the page's Enable action is accepted.
exec "${command[@]}"
