#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/button_autostart_common.sh"
mode="${1:-install}"
case "$mode" in
    install|--dry-run|--check) ;;
    *) printf 'Usage: bash %s [--dry-run|--check]\n' "$0" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { printf 'Too many arguments.\n' >&2; exit 2; }
target_user="$(id -un)"
if ! target_group="$(id -gn 2>/dev/null)"; then
    target_group="$(id -g)"
fi

render_config() {
    printf 'REPO_DIR=%q\nVISION_PYTHON=%q\nPOLICY_PYTHON=%q\n' \
        "$BUTTON_REPO_DIR" "$BUTTON_REPO_DIR/.venv/bin/python" "$HOME/venvs/humanoid_policy/bin/python"
    tail -n +4 "$BUTTON_REPO_DIR/config/button_start.env.example"
}

# systemd quoted values need literal percent signs doubled (specifier escaping).
unit_quote() {
    local value="$1"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || { printf 'Newlines are not supported in unit paths.\n' >&2; return 1; }
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

render_unit() {
    local component="$1" dependencies
    dependencies='After=network.target'
    if [[ "$component" = vision ]]; then
        dependencies=$'Requires=humanoid-button-connector.service\nAfter=network.target humanoid-button-connector.service'
    fi
    cat <<EOF
[Unit]
Description=Humanoid button startup $component
$dependencies
StartLimitIntervalSec=0

[Service]
Type=exec
User=$target_user
Group=$target_group
# Both wrappers change to REPO_DIR before executing Python.
Environment=PYTHONUNBUFFERED=1
Environment=$(unit_quote "BUTTON_CONFIG=$BUTTON_CONFIG")
ExecStart=/bin/bash $(unit_quote "$BUTTON_REPO_DIR/scripts/run_button_$component.sh")
Restart=on-failure
RestartSec=5
KillMode=control-group
KillSignal=SIGINT
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF
}

if [[ "$mode" = --dry-run ]]; then
    if [[ -f "$BUTTON_CONFIG" ]]; then
        printf 'Existing config preserved: %s\n' "$BUTTON_CONFIG"
    else
        printf 'Config to create at %s:\n' "$BUTTON_CONFIG"
        render_config
    fi
    for component in connector vision; do
        printf '\n/etc/systemd/system/humanoid-button-%s.service:\n' "$component"
        render_unit "$component"
    done
    printf '\nPreflight on install/check: required scripts and both models; vision imports numpy/cv2/serial; policy imports onnxruntime/serial.\n'
    printf 'Verify units, install, then daemon-reload and enable both services. No service is started during installation.\n'
    exit 0
fi

[[ "$(uname -s)" = Linux ]] || { printf 'Install/check must run on the Jetson Linux host. Use --dry-run elsewhere.\n' >&2; exit 1; }
[[ "$EUID" -ne 0 ]] || { printf 'Run as your normal Jetson account, without sudo. The installer uses sudo only for system files/systemctl.\n' >&2; exit 1; }
if [[ ! -f "$BUTTON_CONFIG" ]]; then
    if [[ "$mode" = --check ]]; then
        printf 'Missing config: %s. Run the installer to create it.\n' "$BUTTON_CONFIG" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "$BUTTON_CONFIG")"
    (umask 077; render_config > "$BUTTON_CONFIG")
    printf 'Created user config: %s\n' "$BUTTON_CONFIG"
fi
load_button_config
preflight_button_config
if [[ "$mode" = --check ]]; then
    printf 'Preflight passed. Camera, STM32, button wiring and motor behavior still need a hardware test.\n'
    exit 0
fi
command -v systemctl >/dev/null || { printf 'systemctl is required.\n' >&2; exit 1; }
command -v systemd-analyze >/dev/null || { printf 'systemd-analyze is required to verify service files.\n' >&2; exit 1; }
unit_dir="$(mktemp -d)"
trap 'rm -f -- "$unit_dir/humanoid-button-connector.service" "$unit_dir/humanoid-button-vision.service"; rmdir -- "$unit_dir"' EXIT
for component in connector vision; do
    render_unit "$component" > "$unit_dir/humanoid-button-$component.service"
done
systemd-analyze verify "$unit_dir/humanoid-button-connector.service" "$unit_dir/humanoid-button-vision.service"
for component in connector vision; do
    sudo install -m 0644 "$unit_dir/humanoid-button-$component.service" "/etc/systemd/system/humanoid-button-$component.service"
done
sudo systemctl daemon-reload
sudo systemctl enable humanoid-button-connector.service humanoid-button-vision.service
printf 'Installed and enabled for next boot. To start now: sudo systemctl start humanoid-button-vision.service\n'
