#!/usr/bin/env python3
"""List or name current camera connection IDs; never open capture or motors."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from camera_identity import list_cameras, register_roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--main-id', help='Current connection ID, e.g. usb:001:005')
    parser.add_argument('--card-id', help='Current connection ID, e.g. usb:001:010')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list_cameras(), ensure_ascii=False, indent=2))
        return 0
    if not args.main_id or not args.card_id:
        parser.error('use --list, or specify both --main-id and --card-id')
    try:
        result = register_roles(args.main_id, args.card_id)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc)+'\n')
    for role, identity in result['cameras'].items():
        print(f"{role} = {identity['device_id']} ({identity['name']})")
    print('Names are verified for this connection only; re-register after reboot/replug.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
