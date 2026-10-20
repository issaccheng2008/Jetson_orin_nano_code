#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/remote_autostart_common.sh"
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

unit_quote() {
    local value="$1"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || { printf 'Newlines are not supported in unit values.\n' >&2; return 1; }
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

render_unit() {
    cat <<EOF
[Unit]
Description=Humanoid remote webpage (policy starts on Enable)
After=network.target
Conflicts=humanoid-button-vision.service humanoid-button-connector.service
StartLimitIntervalSec=0

[Service]
Type=exec
User=$target_user
Group=$target_group
Environment=PYTHONUNBUFFERED=1
Environment=$(unit_quote "HOME=$HOME")
Environment=$(unit_quote "REMOTE_CONFIG=$REMOTE_CONFIG")
ExecStart=/bin/bash $(unit_quote "$REMOTE_REPO_DIR/scripts/run_remote_control.sh")
Restart=on-failure
RestartSec=5
KillMode=control-group
KillSignal=SIGINT
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF
}

render_config() {
    printf 'REPO_DIR=%q\nPOLICY_PYTHON=%q\n' "$REMOTE_REPO_DIR" "$HOME/venvs/humanoid_policy/bin/python"
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        case "$line" in REPO_DIR=*|POLICY_PYTHON=*|ROBOT_REMOTE_TOKEN=*) continue ;; esac
        printf '%s\n' "$line"
    done < "$REMOTE_REPO_DIR/config/remote_start.env.example"
    # No token is generated or printed by --dry-run.
    printf 'ROBOT_REMOTE_TOKEN=%q\n' "$1"
}

if [[ "$mode" = --dry-run ]]; then
    if [[ -f "$REMOTE_CONFIG" ]]; then
        printf 'Existing private config preserved (contents hidden): %s\n' "$REMOTE_CONFIG"
    else
        printf 'Create private config (0600): %s; generate a random token once.\n' "$REMOTE_CONFIG"
        render_config '<generated during install; hidden>'
    fi
    printf '\n/etc/systemd/system/humanoid-remote.service:\n'
    render_unit
    printf '\nPreflight: files/model, Python imports numpy/onnxruntime/serial/remote_control.server and numeric settings; no hardware or policy is opened.\n'
    printf 'Then systemd-analyze verify; disable/stop only humanoid-button-vision.service and humanoid-button-connector.service.\n'
    printf 'Move existing regular legacy units into /etc/systemd/system/humanoid-remote-legacy-backup/<unique timestamp>/; preserve previous backup directories; persistently mask both units.\n'
    printf 'Install and enable humanoid-remote.service for next boot only. No service starts during install.\n'
    exit 0
fi

[[ "$(uname -s)" = Linux ]] || { printf 'Install/check must run on the Jetson Linux host. Use --dry-run elsewhere.\n' >&2; exit 1; }
[[ "$EUID" -ne 0 ]] || { printf 'Run as your normal Jetson account, without sudo. System operations use sudo internally.\n' >&2; exit 1; }
[[ "$REMOTE_CONFIG" = /* ]] || { printf 'REMOTE_CONFIG must be an absolute path.\n' >&2; exit 1; }
[[ ! -L "$REMOTE_CONFIG" ]] || { printf 'Remote config must not be a symlink.\n' >&2; exit 1; }
if [[ ! -f "$REMOTE_CONFIG" ]]; then
    [[ "$mode" != --check ]] || { printf 'Missing config: %s. Run installer first.\n' "$REMOTE_CONFIG" >&2; exit 1; }
    mkdir -p -- "$(dirname -- "$REMOTE_CONFIG")"
    token="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
    [[ ${#token} -eq 64 ]] || { printf 'Could not generate token.\n' >&2; exit 1; }
    (umask 077; set -o noclobber; render_config "$token" > "$REMOTE_CONFIG")
    unset token
    printf 'Created private config: %s\n' "$REMOTE_CONFIG"
elif [[ "$mode" = install ]]; then
    # Preserve all values and the existing token; restrict permissions on reinstall.
    [[ -O "$REMOTE_CONFIG" ]] || { printf 'Config must be owned by the current user.\n' >&2; exit 1; }
    chmod 0600 -- "$REMOTE_CONFIG"
fi
load_remote_config
preflight_remote_config
if [[ "$mode" = --check ]]; then
    printf 'Preflight passed; no service, policy, serial port or camera was started.\n'
    exit 0
fi
command -v systemctl >/dev/null || { printf 'systemctl is required.\n' >&2; exit 1; }
command -v systemd-analyze >/dev/null || { printf 'systemd-analyze is required.\n' >&2; exit 1; }
unit_dir="$(mktemp -d)"
trap 'rm -f -- "$unit_dir/humanoid-remote.service"; rmdir -- "$unit_dir"' EXIT
render_unit > "$unit_dir/humanoid-remote.service"
# All preflight/unit checks complete before touching the old running services.
systemd-analyze verify "$unit_dir/humanoid-remote.service"
backup_dir=''
for legacy in humanoid-button-vision.service humanoid-button-connector.service; do
    if [[ -e "/etc/systemd/system/$legacy" ]] || systemctl cat "$legacy" >/dev/null 2>&1; then
        sudo systemctl disable --now "$legacy"
    fi
    if [[ -f "/etc/systemd/system/$legacy" && ! -L "/etc/systemd/system/$legacy" ]]; then
        if [[ -z "$backup_dir" ]]; then
            sudo mkdir -p -- /etc/systemd/system/humanoid-remote-legacy-backup
            backup_dir="$(sudo mktemp -d "/etc/systemd/system/humanoid-remote-legacy-backup/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
        fi
        sudo mv -- "/etc/systemd/system/$legacy" "$backup_dir/$legacy"
    fi
    sudo systemctl mask "$legacy"
done
sudo install -m 0644 "$unit_dir/humanoid-remote.service" /etc/systemd/system/humanoid-remote.service
sudo systemctl daemon-reload
sudo systemctl enable humanoid-remote.service
printf 'Installed/enabled for next boot. To start now: sudo systemctl start humanoid-remote.service\n'
[[ -z "$backup_dir" ]] || printf 'Legacy unit backup: %s\n' "$backup_dir"
