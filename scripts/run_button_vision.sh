#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/button_autostart_common.sh"
load_button_config
read -r -a left_wz <<< "$HEADING_LEFT_WZ"
case "$STEERING_SEGMENT_FALLBACK" in
    1) segment_fallback_flag=--steering-segment-fallback ;;
    0) segment_fallback_flag=--no-steering-segment-fallback ;;
    *) printf 'STEERING_SEGMENT_FALLBACK must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$RECORD_VIDEO" in
    1) video_flag=--record-video ;;
    0) video_flag=--no-record-video ;;
    *) printf 'RECORD_VIDEO must be 0 or 1\n' >&2; exit 2 ;;
esac
command=("$VISION_PYTHON" -u "$REPO_DIR/new_vision/jetson/run_policy_vision.py"
    --camera "$CAMERA" --headless --start-gate button --start-policy-on-gate
    --start-policy-python "$POLICY_PYTHON" --start-policy-port "$STM32_PORT"
    --start-policy-model "$WALKING_MODEL" --start-policy-one-foot-model "$ONE_FOOT_MODEL"
    --start-policy-max-seconds "$POLICY_MAX_SECONDS"
    --command-min-hold-s "$COMMAND_MIN_HOLD_S"
    --steering-command-window-s "$STEERING_COMMAND_WINDOW_S"
    --steering-command-median "$STEERING_COMMAND_MEDIAN"
    --steering-loss-fallback-wz "$STEERING_LOSS_FALLBACK_WZ"
    --startup-first-walk-s "$STARTUP_FIRST_WALK_S"
    --startup-sequence "$STARTUP_SEQUENCE"
    --steering-filter-mode "$STEERING_FILTER_MODE" --steering-filter-algorithm "$STEERING_FILTER_ALGORITHM"
    --steering-filter-min-hz "$STEERING_FILTER_MIN_HZ" --steering-filter-max-hz "$STEERING_FILTER_MAX_HZ"
    --steering-filter-beta "$STEERING_FILTER_BETA" --steering-filter-derivative-hz "$STEERING_FILTER_DERIVATIVE_HZ"
    --steering-filter-robust-tau-s "$STEERING_FILTER_ROBUST_TAU_S"
    --steering-filter-robust-window-s "$STEERING_FILTER_ROBUST_WINDOW_S"
    --steering-filter-robust-slew-deg-s "$STEERING_FILTER_ROBUST_SLEW_DEG_S"
    --steering-loss-mode "$STEERING_LOSS_MODE" --steering-loss-max-s "$STEERING_LOSS_MAX_S"
    --steering-loss-history-s "$STEERING_LOSS_HISTORY_S" "$segment_fallback_flag"
    --steering-filter-position-tau-s "$STEERING_FILTER_POSITION_TAU_S"
    --steering-hysteresis-deg "$STEERING_HYSTERESIS_DEG"
    --steering-enter-deg "$STEERING_ENTER_DEG" --steering-exit-deg "$STEERING_EXIT_DEG"
    --wz-mode "$WZ_MODE" --line-preprocess "$LINE_PREPROCESS"
    --line-adaptive-c "$LINE_ADAPTIVE_C"
    --shape-preprocess "$SHAPE_PREPROCESS" --photometric-mode "$PHOTOMETRIC_MODE"
    --vx "$VX" --max-wz "$MAX_WZ" --wz-step "$WZ_STEP"
    --lost-hold-s "$LOST_HOLD_S"
    --heading-lookahead-cm "$HEADING_LOOKAHEAD_CM" --heading-corridor-cm "$HEADING_CORRIDOR_CM"
    --heading-right-tolerance-deg "$HEADING_RIGHT_TOLERANCE_DEG"
    --heading-left-tolerance-deg "$HEADING_LEFT_TOLERANCE_DEG"
    --heading-full-scale-deg "$HEADING_FULL_SCALE_DEG" --heading-left-wz "${left_wz[@]}"
    --card-trigger-dist-cm "$CARD_TRIGGER_DIST_CM" --shape-every "$SHAPE_EVERY"
    --recording-root "$RECORDS_DIR/tests" "$video_flag"
    --video-fps "$VIDEO_FPS" --video-width "$VIDEO_WIDTH")
for setting in HEADING_REGIONS_CM HEADING_NEAR_CM HEADING_FAR_CM STEERING_ANGLE_WZ_TABLE SEGMENT_REGIONS_CM; do
    if [[ -n "${!setting:-}" ]]; then
        flag="${setting,,}"
        command+=("--${flag//_/-}" "${!setting}")
    fi
done
position_names=(POSITION_GAIN POSITION_DEAD_CM POSITION_LOOKAHEAD_CM POSITION_MAX_DEG
                POSITION_RECOVERY_CM POSITION_RECOVERY_FULL_SCALE_CM POSITION_CONFIRM_FRAMES)
for setting in "${position_names[@]}"; do
    flag="${setting,,}"
    command+=("--${flag//_/-}" "${!setting}")
done
command+=(--camera-exposure-mode "$CAMERA_EXPOSURE_MODE"
          --camera-white-balance-mode "$CAMERA_WHITE_BALANCE_MODE")
camera_names=(CAMERA_EXPOSURE_MS CAMERA_BRIGHTNESS CAMERA_CONTRAST CAMERA_SATURATION
              CAMERA_SHARPNESS CAMERA_WHITE_BALANCE_K CAMERA_POWER_LINE_HZ)
camera_flags=(--camera-exposure-ms --camera-brightness --camera-contrast --camera-saturation
              --camera-sharpness --camera-white-balance-k --camera-power-line-hz)
for i in "${!camera_names[@]}"; do
    setting="${camera_names[$i]}"
    if [[ -n "${!setting}" ]]; then
        command+=("${camera_flags[$i]}" "${!setting}")
    fi
done
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
