#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/button_autostart_common.sh"
load_button_config
command=("$VISION_PYTHON" -u "$REPO_DIR/connector.py"
    --vision-port 5006 --policy-port 5005 --max-vx-accel 1 --max-wz-accel 0
    --wz-bias "$WZ_BIAS")
if [[ "${1:-}" = --dry-run ]]; then
    print_command "${command[@]}"
    exit 0
fi
[[ $# = 0 ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
cd -- "$REPO_DIR"
exec "${command[@]}"
