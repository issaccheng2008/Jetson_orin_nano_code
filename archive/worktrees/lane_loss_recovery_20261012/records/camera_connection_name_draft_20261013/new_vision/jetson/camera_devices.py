"""Resolve explicit stable Linux device links without guessing camera roles."""
import os
from pathlib import Path
from camera_identity import verify_named_camera

def resolve_devices(main_index, card_index, main_path=None, card_path=None):
    def resolve(path,index):
        if path:
            p=Path(path)
            if not str(p).startswith('/dev/') or not p.exists():
                raise ValueError(f'Camera device path missing or not under /dev: {path}')
            verify_named_camera(p)
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
    return main,card
