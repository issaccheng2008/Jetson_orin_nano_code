#!/usr/bin/env python3
"""Join the field WiFi through NetworkManager.

NetworkManager remembers a network once it has connected, so this is mainly for
the first join after a flash, or after the profile was lost. The password is not
in this file - this repository is public. It lives in
``~/.config/robocup_wifi`` (mode 600) on the robot, or comes from ``--password``
or ``ROBOCUP_WIFI_PASSWORD``. Set it up once:

    python new_vision/jetson/wifi_connect.py --password luoboluobo --save
    python new_vision/jetson/wifi_connect.py

Re-running is safe. Use ``--forget`` to drop the stored profile and start clean.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys

DEFAULT_SSID = "Robocup"
CREDENTIALS_PATH = "~/.config/robocup_wifi"
CONNECT_TIMEOUT_S = 30


def nmcli(*args, check=True):
    if shutil.which("nmcli") is None:
        sys.exit("nmcli not found - this needs NetworkManager, i.e. the Jetson")
    result = subprocess.run(["nmcli", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        sys.exit(f"nmcli {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def load_credentials(path):
    if not os.path.exists(path):
        return {}
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def save_credentials(path, ssid, password):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Open with the mode already set: a plain write would briefly be world readable.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"ssid={ssid}\npassword={password}\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def wifi_device():
    for line in nmcli("-t", "-f", "DEVICE,TYPE,STATE", "device", "status").splitlines():
        fields = line.split(":")
        if len(fields) >= 2 and fields[1] == "wifi":
            return fields[0]
    sys.exit("no wifi device found")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssid", default=None, help=f"default {DEFAULT_SSID}")
    parser.add_argument("--password", default=None,
                        help="prefer the credentials file; this is visible in ps")
    parser.add_argument("--credentials", default=CREDENTIALS_PATH,
                        help=f"default {CREDENTIALS_PATH}")
    parser.add_argument("--save", action="store_true",
                        help="write ssid/password to the credentials file and exit")
    parser.add_argument("--forget", action="store_true",
                        help="delete the stored profile first, then reconnect")
    args = parser.parse_args()

    path = os.path.expanduser(args.credentials)
    stored = load_credentials(path)
    ssid = args.ssid or stored.get("ssid") or DEFAULT_SSID
    password = (args.password or os.environ.get("ROBOCUP_WIFI_PASSWORD")
                or stored.get("password"))

    if args.save:
        if not password:
            sys.exit("--save needs --password or ROBOCUP_WIFI_PASSWORD")
        save_credentials(path, ssid, password)
        print(f"saved {ssid} to {path} (mode 600)")
        return 0

    if not password:
        sys.exit(f"no password for {ssid}. Either\n"
                 f"  python {sys.argv[0]} --password <pw> --save\n"
                 f"or write {path}:\n  ssid={ssid}\n  password=<pw>")

    nmcli("radio", "wifi", "on")
    if args.forget:
        nmcli("connection", "delete", ssid, check=False)
        print(f"dropped the stored {ssid} profile")

    print(f"connecting to {ssid} ...")
    nmcli("--wait", str(CONNECT_TIMEOUT_S), "device", "wifi", "connect",
          ssid, "password", password)
    # Reconnect on its own from now on: this is what makes it automatic.
    nmcli("connection", "modify", ssid, "connection.autoconnect", "yes")

    device = wifi_device()
    state = nmcli("-t", "-f", "GENERAL.STATE", "device", "show", device)
    address = nmcli("-t", "-f", "IP4.ADDRESS", "device", "show", device)
    print(f"{device}: {state}")
    print(f"ip: {address or '(none)'}")
    return 0 if "connected" in state else 1


if __name__ == "__main__":
    raise SystemExit(main())
