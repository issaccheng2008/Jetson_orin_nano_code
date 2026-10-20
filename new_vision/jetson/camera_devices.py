"""Resolve explicit stable Linux device links without guessing camera roles."""
import os
from pathlib import Path

MAIN_LINK = '/dev/robot_main_camera'
CARD_LINK = '/dev/robot_card_camera'
MAIN_PORT_LINKS = tuple(
    f'/dev/v4l/by-path/platform-3610000.usb-usb-0:{port}:1.0-video-index0'
    for port in ('2.3', '2.4')
)


def verify_camera_roles(main, card):
    """Require one camera on the allowed main ports and one outside them."""
    if main != MAIN_LINK and card != CARD_LINK:
        return  # Legacy explicit device paths/indices remain compatible.
    main_nodes = {os.path.realpath(p) for p in MAIN_PORT_LINKS if Path(p).exists()}
    if not main_nodes:
        raise ValueError('Main camera missing on USB 2.3/2.4; refusing to swap roles')
    if len(main_nodes) != 1:
        raise ValueError('Multiple cameras on main USB 2.3/2.4 ports; ambiguous main role')
    expected_main = next(iter(main_nodes))
    actual_main = os.path.realpath(main if isinstance(main, str) else f'/dev/video{main}')
    if actual_main != expected_main:
        raise ValueError('Main camera name does not resolve to the camera on USB 2.3/2.4')
    if card is None:
        return
    nodes = {os.path.realpath(p) for p in Path('/dev/v4l/by-path').glob(
        'platform-3610000.usb-usb-0:*-video-index0') if p.exists()}
    others = nodes - {expected_main}
    if len(others) != 1:
        raise ValueError('Expected exactly one secondary capture camera outside USB 2.3/2.4; '
                         'missing or ambiguous camera roles')
    actual_card = os.path.realpath(card if isinstance(card, str) else f'/dev/video{card}')
    if actual_card not in others:
        raise ValueError('Card camera name does not resolve to the other capture device')

def resolve_devices(main_index, card_index, main_path=None, card_path=None):
    def resolve(path,index):
        if path:
            p=Path(path)
            if not str(p).startswith('/dev/') or not p.exists():
                raise ValueError(f'Camera device path missing or not under /dev: {path}')
            return str(p)
        return index if index >= 0 else None
    main=resolve(main_path,main_index)
    card=resolve(card_path,card_index)
    def physical(device):
        if device is None:return None
        path=device if isinstance(device,str) else f'/dev/video{device}'
        return os.path.realpath(path)
    if card is not None and physical(main)==physical(card):
        raise ValueError('Main and stopped-card camera resolve to the same device')
    verify_camera_roles(main, card)
    return main,card
