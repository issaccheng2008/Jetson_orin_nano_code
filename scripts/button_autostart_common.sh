#!/usr/bin/env bash
# Shared configuration only; never launch a policy here.
set -euo pipefail
BUTTON_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUTTON_CONFIG="${BUTTON_CONFIG:-$BUTTON_REPO_DIR/config/button_start.env}"

load_button_config() {
    if [[ ! -f "$BUTTON_CONFIG" ]]; then
        printf 'Missing config: %s. Run bash scripts/install_button_autostart.sh first.\n' "$BUTTON_CONFIG" >&2
        return 1
    fi
    # User-owned configuration is trusted shell syntax, not a systemd EnvironmentFile.
    # Older installed configs lack this setting; preserve their heading behavior.
    WZ_MODE=heading
    LOST_HOLD_S=0.2
    WZ_BIAS=0
    COMMAND_MIN_HOLD_S=0
    STARTUP_FIRST_WALK_S=0.5
    STARTUP_SEQUENCE=''
    RECORD_VIDEO=1
    VIDEO_FPS=10
    VIDEO_WIDTH=960
    CAMERA_EXPOSURE_MODE=keep
    CAMERA_EXPOSURE_MS=''
    CAMERA_BRIGHTNESS=''
    CAMERA_CONTRAST=''
    CAMERA_SATURATION=''
    CAMERA_SHARPNESS=''
    CAMERA_WHITE_BALANCE_MODE=keep
    CAMERA_WHITE_BALANCE_K=''
    CAMERA_POWER_LINE_HZ=''
    STEERING_FILTER_MODE=legacy
    STEERING_FILTER_ALGORITHM=robust
    STEERING_FILTER_ROBUST_TAU_S=0.45
    STEERING_FILTER_ROBUST_WINDOW_S=0.6
    STEERING_FILTER_ROBUST_SLEW_DEG_S=45
    STEERING_LOSS_MODE=history-turn
    STEERING_LOSS_MAX_S=0.8
    STEERING_LOSS_HISTORY_S=0.8
    STEERING_SEGMENT_FALLBACK=1
    STEERING_FILTER_MIN_HZ=1.5
    STEERING_FILTER_MAX_HZ=4
    STEERING_FILTER_BETA=0.03
    STEERING_FILTER_DERIVATIVE_HZ=1
    STEERING_FILTER_POSITION_TAU_S=0.1
    STEERING_HYSTERESIS_DEG=1
    STEERING_ENTER_DEG=2
    STEERING_EXIT_DEG=1
    source "$BUTTON_CONFIG"
    case "$WZ_MODE" in
        heading|segments|continuous|discrete) ;;
        *) printf 'Invalid WZ_MODE=%s; use heading, segments, continuous or discrete.\n' "$WZ_MODE" >&2; return 1 ;;
    esac
    for name in REPO_DIR VISION_PYTHON POLICY_PYTHON WALKING_MODEL ONE_FOOT_MODEL STM32_PORT CAMERA VX MAX_WZ WZ_STEP HEADING_LOOKAHEAD_CM HEADING_CORRIDOR_CM HEADING_RIGHT_TOLERANCE_DEG HEADING_LEFT_TOLERANCE_DEG HEADING_FULL_SCALE_DEG HEADING_LEFT_WZ CARD_TRIGGER_DIST_CM SHAPE_EVERY POLICY_MAX_SECONDS RECORDS_DIR; do
        if [[ -z "${!name:-}" ]]; then
            printf 'Missing or empty setting %s in %s\n' "$name" "$BUTTON_CONFIG" >&2
            return 1
        fi
    done
    [[ "$REPO_DIR" = /* && "$VISION_PYTHON" = /* && "$POLICY_PYTHON" = /* && "$RECORDS_DIR" = /* ]] || {
        printf 'REPO_DIR, both Python paths, and RECORDS_DIR must be absolute paths.\n' >&2
        return 1
    }
    export PYTHONUNBUFFERED=1
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

preflight_button_config() {
    local file
    for file in "$REPO_DIR/connector.py" "$REPO_DIR/new_vision/jetson/run_policy_vision.py" "$REPO_DIR/humanoid_jetson_deploy/main.py" "$WALKING_MODEL" "$ONE_FOOT_MODEL"; do
        [[ -f "$file" ]] || { printf 'Missing required file: %s\n' "$file" >&2; return 1; }
    done
    [[ -x "$VISION_PYTHON" ]] || { printf 'Vision Python is not executable: %s\n' "$VISION_PYTHON" >&2; return 1; }
    [[ -x "$POLICY_PYTHON" ]] || { printf 'Policy Python is not executable: %s\n' "$POLICY_PYTHON" >&2; return 1; }
    "$VISION_PYTHON" -c 'import numpy, cv2, serial; print("Vision dependencies OK: numpy, cv2, serial")'
    "$POLICY_PYTHON" -c 'import onnxruntime, serial; print("Policy dependencies OK: onnxruntime, serial")'
}
