"""Explicit camera names for one verified USB connection, without port rules.

Identical USB serials cannot provide persistent physical identity. A registration
uses a bus/device instance ID and expires across reboot or USB re-enumeration.
Only metadata is read here; no capture, camera controls or motor IO.
"""
import json
import os
from pathlib import Path
import subprocess

REGISTRY = Path('/etc/robot/camera_roles.json')
ROLE_NAMES = ('robot_main_camera', 'robot_card_camera')


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def camera_identity(node):
    node = Path(node).resolve()
    entry = Path('/sys/class/video4linux') / node.name
    if not entry.exists() or (entry / 'index').read_text().strip() != '0':
        raise ValueError(f'Not a camera capture node: {node}')
    props = subprocess.run(['udevadm', 'info', '--query=property', '--name='+str(node)],
                           check=True, capture_output=True, text=True)
    properties = dict(line.split('=', 1) for line in props.stdout.splitlines() if '=' in line)
    if ':capture:' not in properties.get('ID_V4L_CAPABILITIES', ''):
        raise ValueError(f'Not a video capture device: {node}')
    device = (entry / 'device').resolve()
    for parent in (device, *device.parents):
        if (parent / 'idVendor').exists():
            def value(name):
                p = parent / name
                return p.read_text().strip() if p.exists() else ''
            return dict(node=str(node), name=(entry/'name').read_text().strip(),
                        device_id=f"usb:{int(value('busnum')):03d}:{int(value('devnum')):03d}",
                        vendor=value('idVendor'), product=value('idProduct'),
                        serial=value('serial'),
                        initialized=properties.get('USEC_INITIALIZED', ''))
    raise ValueError(f'Camera is not a USB device: {node}')


def list_cameras():
    result = []
    for entry in sorted(Path('/sys/class/video4linux').glob('video*')):
        if (entry/'index').read_text().strip() == '0':
            try:
                result.append(camera_identity('/dev/'+entry.name))
            except ValueError:
                continue
    return result


def register_roles(main_id, card_id, *, registry=REGISTRY, device_dir=Path('/dev')):
    cameras = list_cameras()
    selected = {}
    for role, wanted in zip(ROLE_NAMES, (main_id, card_id)):
        matches = [item for item in cameras if item['device_id'] == wanted]
        if len(matches) != 1:
            raise ValueError(f'{role}: current device ID {wanted} is missing or ambiguous')
        selected[role] = matches[0]
    if selected[ROLE_NAMES[0]]['node'] == selected[ROLE_NAMES[1]]['node']:
        raise ValueError('Main and card camera must be different physical devices')
    registry, device_dir = Path(registry), Path(device_dir)
    for role in ROLE_NAMES:
        alias = device_dir / role
        if alias.exists() and not alias.is_symlink():
            raise ValueError(f'Refusing to replace non-symlink device: {alias}')
    registration = dict(schema_version=1, boot_id=boot_id(), cameras=selected)
    registry.parent.mkdir(parents=True, exist_ok=True)
    temp = registry.with_name(registry.name+'.new')
    temp.write_text(json.dumps(registration, ensure_ascii=False, indent=2)+'\n')
    temp.chmod(0o644)
    for role, identity in selected.items():
        alias = device_dir/role
        staging = device_dir/(role+'.new')
        if staging.is_symlink():
            staging.unlink()
        staging.symlink_to(identity['node'])
        os.replace(staging, alias)
    os.replace(temp, registry)
    return registration


def verify_named_camera(path, *, registry=REGISTRY):
    role = Path(path).name
    if role not in ROLE_NAMES:
        return  # Explicit legacy paths remain supported; no role is guessed.
    try:
        saved = json.loads(Path(registry).read_text())
        expected = saved['cameras'][role]
        if saved['schema_version'] != 1 or saved['boot_id'] != boot_id():
            raise ValueError('camera registration expired after reboot')
        current = camera_identity(path)
        if current != expected:
            raise ValueError('camera link/connection identity changed')
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(f'{role}: {exc}; re-register camera names with '
                         'scripts/name_robot_cameras.py before starting vision') from exc
