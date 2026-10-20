#!/usr/bin/env python3
"""Make the Jetson join the field WiFi on its own, every boot.

Run this **once**. It writes a NetworkManager profile for the field AP with
autoconnect enabled, then tries to bring it up. From then on NetworkManager
connects by itself at startup whenever that SSID is in range - nothing else has
to run, and this script does not need to be on any startup list.

    python new_vision/jetson/wifi_connect.py

Running it out of range is fine: the profile is still written, so the robot
picks the network up the next time it boots near it. --forget deletes the
profile; --ssid/--password (or ROBOCUP_WIFI_SSID/ROBOCUP_WIFI_PASSWORD) point it
at a different site.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_SSID = "Robocup"
DEFAULT_PASSWORD = "luoboluobo"


def run_nmcli(*args):
    if shutil.which("nmcli") is None:
        sys.exit("nmcli not found - this needs NetworkManager, i.e. the Jetson")
    return subprocess.run(["nmcli", *args], capture_output=True, text=True)


def nmcli(*args, check=True):
    result = run_nmcli(*args)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        sys.exit(f"nmcli {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def profile_names():
    return nmcli("-t", "-f", "NAME", "connection", "show").splitlines()


def ensure_profile(ssid, password):
    """Write the profile whether or not the AP is reachable right now."""
    settings = ["connection.autoconnect", "yes",
                "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password]
    if ssid in profile_names():
        nmcli("connection", "modify", ssid, *settings)
        print(f"updated the {ssid} profile")
    else:
        nmcli("connection", "add", "type", "wifi", "con-name", ssid,
              "ifname", "*", "ssid", ssid, *settings)
        print(f"created the {ssid} profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssid", default=os.environ.get("ROBOCUP_WIFI_SSID", DEFAULT_SSID))
    parser.add_argument("--password",
                        default=os.environ.get("ROBOCUP_WIFI_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--forget", action="store_true",
                        help="delete the stored profile and exit")
    args = parser.parse_args()

    nmcli("radio", "wifi", "on")

    if args.forget:
        nmcli("connection", "delete", args.ssid, check=False)
        print(f"deleted the {args.ssid} profile; it will not be joined automatically")
        return 0

    ensure_profile(args.ssid, args.password)

    attempt = run_nmcli("connection", "up", args.ssid)
    if attempt.returncode != 0:
        reason = (attempt.stderr or attempt.stdout).strip() or "out of range"
        print(f"not connected yet ({reason})")
        print(f"the {args.ssid} profile is saved and autoconnect is on, so the "
              f"robot will join by itself the next time it boots near it")
        return 0

    device = next((line.split(":")[0]
                   for line in nmcli("-t", "-f", "DEVICE,TYPE", "device", "status").splitlines()
                   if line.split(":")[1:2] == ["wifi"]), None)
    if device:
        print(f"{device}: {nmcli('-t', '-f', 'GENERAL.STATE', 'device', 'show', device)}")
        print(f"ip: {nmcli('-t', '-f', 'IP4.ADDRESS', 'device', 'show', device) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
