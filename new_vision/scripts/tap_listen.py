#!/usr/bin/env python3
"""Record the connector's 50 Hz output, so a restart transient is visible.

`connector.py --tap-port PORT` mirrors every packet it publishes, byte for byte.
This listens on that port and writes one row per packet with the arrival time,
because the packets carry no host clock of their own.

    python -u new_vision/scripts/tap_listen.py --port 5010 --out logs/tap.csv

Run it from the repo root on the same machine as the connector, so its clock
matches the motor-position CSV's. Then the question the vision log cannot answer
- did the robot turn the way it was told - becomes two columns:

    wz from this file        what the vision asked for, at 50 Hz
    d(rpy[2])/dt from        what the robot did
    logs/motor_positions/    (column 2 is host time, last column is yaw)

The vision log samples at 2 Hz, and the vx ramp after a card stop lasts about
0.4 s, so it never appears there at all.

Columns: t_host, host_time_iso, vx, vy, wz, qr, event_id, event_action
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5010,
                        help="must match connector.py --tap-port")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--out", default="logs/tap.csv")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    print(f"listening on {args.bind}:{args.port} -> {args.out}; Ctrl+C to stop")

    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_host", "host_time_iso", "vx", "vy", "wz", "qr",
                         "event_id", "event_action"])
        try:
            while True:
                payload, _ = sock.recvfrom(4096)
                now = time.time()
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                writer.writerow([
                    f"{now:.6f}", time.strftime("%H:%M:%S", time.localtime(now)),
                    message.get("vx", ""), message.get("vy", ""),
                    message.get("wz", ""), message.get("qr", ""),
                    message.get("event_id", ""), message.get("event_action", ""),
                ])
                rows += 1
                if rows % 50 == 0:      # once a second at 50 Hz
                    handle.flush()
        except KeyboardInterrupt:
            print(f"\n{rows} packets recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
