#!/usr/bin/env bash
# Standalone vision only. Does not start connector, policy or button services.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_DIR"
SHAPE_PREPROCESS=selective
PHOTOMETRIC_MODE=legacy
CAMERA_EXPOSURE_MODE=keep
CAMERA_WHITE_BALANCE_MODE=keep
config="${BUTTON_CONFIG:-$REPO_DIR/config/button_start.env}"
if [[ -f "$config" ]]; then source "$config"; fi
python_bin="${CARD_DEBUG_PYTHON:-${VISION_PYTHON:-$REPO_DIR/.venv/bin/python}}"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
args=(--camera "${CAMERA_DEVICE:-/dev/robot_main_camera}"
      --shape-preprocess "$SHAPE_PREPROCESS" --photometric-mode "$PHOTOMETRIC_MODE"
      --camera-exposure-mode "$CAMERA_EXPOSURE_MODE"
      --camera-white-balance-mode "$CAMERA_WHITE_BALANCE_MODE")
for setting in CAMERA_EXPOSURE_MS CAMERA_BRIGHTNESS CAMERA_CONTRAST CAMERA_SATURATION CAMERA_SHARPNESS CAMERA_WHITE_BALANCE_K CAMERA_POWER_LINE_HZ PHOTOMETRIC_LUT_DIR; do
    if [[ -n "${!setting:-}" ]]; then
        flag="${setting,,}"
        args+=("--${flag//_/-}" "${!setting}")
    fi
done
exec "$python_bin" "$REPO_DIR/new_vision/jetson/card_debug.py" "${args[@]}" "$@"
