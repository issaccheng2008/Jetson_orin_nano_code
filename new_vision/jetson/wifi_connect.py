#!/usr/bin/env python3
"""Join the field WiFi through NetworkManager.

NetworkManager remembers a network once it has connected and reconnects on its
own, so this is mainly for the first join after a flash, or after the profile
was lost. Re-running is safe; --forget drops the stored profile first.

    python new_vision/jetson/wifi_connect.py

The field AP and its password are the defaults below. Override them with
--ssid/--password or ROBOCUP_WIFI_SSID/ROBOCUP_WIFI_PASSWORD for another site.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_SSID = "Robocup"
DEFAULT_PASSWORD = "luoboluobo"
CONNECT_TIMEOUT_S = 30


def nmcli(*args, check=True):
    if shutil.which("nmcli") is None:
        sys.exit("nmcli not found - this needs NetworkManager, i.e. the Jetson")
    result = subprocess.run(["nmcli", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        sys.exit(f"nmcli {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def wifi_device():
    for line in nmcli("-t", "-f", "DEVICE,TYPE,STATE", "device", "status").splitlines():
        fields = line.split(":")
        if len(fields) >= 2 and fields[1] == "wifi":
            return fields[0]
    sys.exit("no wifi device found")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssid", default=os.environ.get("ROBOCUP_WIFI_SSID", DEFAULT_SSID))
    parser.add_argument("--password",
                        default=os.environ.get("ROBOCUP_WIFI_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--forget", action="store_true",
                        help="delete the stored profile first, then reconnect")
    args = parser.parse_args()

    nmcli("radio", "wifi", "on")
    if args.forget:
        nmcli("connection", "delete", args.ssid, check=False)
        print(f"dropped the stored {args.ssid} profile")

    print(f"connecting to {args.ssid} ...")
    nmcli("--wait", str(CONNECT_TIMEOUT_S), "device", "wifi", "connect",
          args.ssid, "password", args.password)
    # Reconnect on its own from now on: this is what makes it automatic.
    nmcli("connection", "modify", args.ssid, "connection.autoconnect", "yes")

    device = wifi_device()
    state = nmcli("-t", "-f", "GENERAL.STATE", "device", "show", device)
    address = nmcli("-t", "-f", "IP4.ADDRESS", "device", "show", device)
    print(f"{device}: {state}")
    print(f"ip: {address or '(none)'}")
    return 0 if "connected" in state else 1


if __name__ == "__main__":
    raise SystemExit(main())
