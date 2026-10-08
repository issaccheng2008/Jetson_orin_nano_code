#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/button_autostart_common.sh"
load_button_config
stamp="$(date +%Y%m%d_%H%M%S)_$$"
read -r -a left_wz <<< "$HEADING_LEFT_WZ"
command=("$VISION_PYTHON" -u "$REPO_DIR/new_vision/jetson/run_policy_vision.py"
    --camera "$CAMERA" --headless --start-gate button --start-policy-on-gate
    --start-policy-python "$POLICY_PYTHON" --start-policy-port "$STM32_PORT"
    --start-policy-model "$WALKING_MODEL" --start-policy-one-foot-model "$ONE_FOOT_MODEL"
    --start-policy-max-seconds "$POLICY_MAX_SECONDS"
    --wz-mode heading --vx "$VX" --max-wz "$MAX_WZ" --wz-step "$WZ_STEP"
    --heading-lookahead-cm "$HEADING_LOOKAHEAD_CM" --heading-corridor-cm "$HEADING_CORRIDOR_CM"
    --heading-right-tolerance-deg "$HEADING_RIGHT_TOLERANCE_DEG"
    --heading-left-tolerance-deg "$HEADING_LEFT_TOLERANCE_DEG"
    --heading-full-scale-deg "$HEADING_FULL_SCALE_DEG" --heading-left-wz "${left_wz[@]}"
    --card-trigger-dist-cm "$CARD_TRIGGER_DIST_CM" --shape-every "$SHAPE_EVERY"
    --line-log-dir "$RECORDS_DIR/line_telemetry"
    --dump-on-loss "$RECORDS_DIR/loss_$stamp" --shape-dump "$RECORDS_DIR/shape_$stamp")
if [[ "${1:-}" = --dry-run ]]; then
    print_command "${command[@]}"
    exit 0
fi
[[ $# = 0 ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
# Recheck both environments before opening hardware; no motor process is started here.
preflight_button_config
mkdir -p -- "$RECORDS_DIR"
cd -- "$REPO_DIR"
exec "${command[@]}"
