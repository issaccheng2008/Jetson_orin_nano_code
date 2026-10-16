#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/button_autostart_common.sh"
load_button_config
read -r -a left_wz <<< "$HEADING_LEFT_WZ"
case "$SINGLE_BOUNDARY_TRACKING" in
    1) boundary_tracking_flag=--single-boundary-tracking ;;
    0) boundary_tracking_flag=--no-single-boundary-tracking ;;
esac
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
case "${LANE_GAP_FILL:-1}" in
    1) gap_flag=--lane-gap-fill ;;
    0) gap_flag=--no-lane-gap-fill ;;
    *) printf 'LANE_GAP_FILL must be 0 or 1\n' >&2; exit 2 ;;
esac
command=("$VISION_PYTHON" -u "$REPO_DIR/new_vision/jetson/run_policy_vision.py"
    --lane-observation-mode "${LANE_OBSERVATION_MODE:-continuous}" "$gap_flag"
    --lane-strictness-pct "${LANE_STRICTNESS_PCT:-50}"
    --lane-gap-max-cm "${LANE_GAP_MAX_CM:-12}"
    --camera "$CAMERA" --headless --start-gate button --start-policy-on-gate
    --start-policy-python "$POLICY_PYTHON" --start-policy-port "$STM32_PORT"
    --start-policy-model "$WALKING_MODEL"
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
    --wz-mode "$WZ_MODE" "$boundary_tracking_flag" --line-preprocess "$LINE_PREPROCESS"
    --line-adaptive-c "$LINE_ADAPTIVE_C"
    --shape-preprocess "$SHAPE_PREPROCESS" --photometric-mode "$PHOTOMETRIC_MODE"
    --vx "$VX" --max-wz "$MAX_WZ" --wz-step "$WZ_STEP"
    --lost-hold-s "$LOST_HOLD_S"
    --heading-lookahead-cm "$HEADING_LOOKAHEAD_CM" --heading-corridor-cm "$HEADING_CORRIDOR_CM"
    --heading-right-tolerance-deg "$HEADING_RIGHT_TOLERANCE_DEG"
    --heading-left-tolerance-deg "$HEADING_LEFT_TOLERANCE_DEG"
    --heading-full-scale-deg "$HEADING_FULL_SCALE_DEG" --heading-left-wz "${left_wz[@]}"
    --card-trigger-dist-cm "$CARD_TRIGGER_DIST_CM"
    --stop-cool-down "${STOP_COOL_DOWN:-5}"
    --card-failure-cooldown-s "${CARD_FAILURE_COOLDOWN_S:-0.5}" --shape-every "$SHAPE_EVERY"
    --recording-root "$RECORDS_DIR/hz_test" "$video_flag"
    --video-source "$VIDEO_SOURCE" --video-fps "$VIDEO_FPS" --video-width "$VIDEO_WIDTH")
# An explicit LOG line is the master selection, including video. Missing LOG
# keeps older installed configs and RECORD_VIDEO behavior unchanged.
if [[ ${LOG+x} ]]; then
    command+=(--log-items "$LOG")
fi
case "${LANE_VISIBLE_DIRECTION_FALLBACK:-1}" in
    1) command+=(--lane-visible-direction-fallback) ;;
    0) command+=(--no-lane-visible-direction-fallback) ;;
    *) printf 'LANE_VISIBLE_DIRECTION_FALLBACK must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${STRAIGHT_ROAD_GUARD:-1}" in
    1) command+=(--straight-road-guard) ;;
    0) command+=(--no-straight-road-guard) ;;
    *) printf 'STRAIGHT_ROAD_GUARD must be 0 or 1\n' >&2; exit 2 ;;
esac
command+=(--straight-near-cm "${STRAIGHT_NEAR_CM:-10}"
          --straight-far-cm "${STRAIGHT_FAR_CM:-40}"
          --straight-confirm-s "${STRAIGHT_CONFIRM_S:-1}"
          --straight-exit-s "${STRAIGHT_EXIT_S:-0.6}"
          --straight-hold-s "${STRAIGHT_HOLD_S:-3}")
# Keep the saved secondary camera settings, but require an explicit opt-in.
case "$CARD_CAMERA_ENABLED" in
    0) card_camera=-1 ;;
    1) card_camera="${CARD_CAMERA:-1}" ;;
    *) printf 'CARD_CAMERA_ENABLED must be 0 or 1\n' >&2; exit 2 ;;
esac
command+=(--card-camera "$card_camera")
# Installed configs predating dual-camera support also use the fixed names.
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/robot_main_camera}"
if [[ "$card_camera" != -1 ]]; then
    CARD_CAMERA_DEVICE="${CARD_CAMERA_DEVICE:-/dev/robot_card_camera}"
else
    CARD_CAMERA_DEVICE=''
fi
for setting in CAMERA_DEVICE CARD_CAMERA_DEVICE CARD_CAMERA_WIDTH CARD_CAMERA_HEIGHT CARD_CAMERA_HEIGHT_CM CARD_CAMERA_PITCH_DEG CARD_CAMERA_VFOV_DEG CARD_CAMERA_PHOTOMETRIC_MODE; do
    if [[ -n "${!setting:-}" ]]; then
        flag="${setting,,}"
        command+=("--${flag//_/-}" "${!setting}")
    fi
done
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
