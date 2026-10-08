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
