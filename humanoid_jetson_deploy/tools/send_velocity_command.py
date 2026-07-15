#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5005)
parser.add_argument("--vx", type=float, required=True)
parser.add_argument("--wz", type=float, required=True)
args = parser.parse_args()

message = json.dumps({"vx": args.vx, "vy": 0.0, "wz": args.wz}).encode("utf-8")
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.sendto(message, ("127.0.0.1", args.port))
print(message.decode("utf-8"))
