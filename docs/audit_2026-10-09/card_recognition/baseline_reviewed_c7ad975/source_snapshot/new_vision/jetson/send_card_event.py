#!/usr/bin/env python3
"""Publish one shape-card event straight to the policy, for testing the STM32.

No camera, no line detector, no shape detector. The robot is told to stand
still (vx=wz=0) and one event is republished at --hz for --hold seconds, which
is what run_policy_vision does after a real recognition. Everything downstream -
connector, policy, serial link, STM32 - sees the same message it would then.

    # stop the connector and vision first, or they will overwrite these packets
    python new_vision/jetson/send_card_event.py --card 2

Then the policy should print

    [shape] event=<id> card=2 accepted

and the arm or head should move. Nothing here means the camera works - it only
proves the path from the UDP port to the servos.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

CARDS = {1: "circle", 2: "pentagon", 3: "square",
         4: "diamond", 5: "cross", 6: "triangle"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", type=int, required=True, choices=sorted(CARDS),
                        help="1 circle, 2 pentagon, 3 square, 4 diamond, 5 cross, 6 triangle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005,
                        help="policy command port; 5006 goes through the connector instead")
    parser.add_argument("--hold", type=float, default=5.0,
                        help="seconds to keep the event asserted; must exceed the "
                             "receiver's 0.25 s command timeout")
    parser.add_argument("--hz", type=float, default=20.0, help="publish rate")
    args = parser.parse_args()
    if args.hold <= 0.0 or args.hz <= 0.0:
        parser.error("hold and hz must be positive")

    # Same shape as run_policy_vision: a millisecond stamp, truncated to 32 bits.
    event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
    event = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": args.card,
             "event_id": event_id, "event_action": args.card}
    stop = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (args.host, args.port)
    print(f"card {args.card} ({CARDS[args.card]}) event_id={event_id} -> "
          f"{args.host}:{args.port} for {args.hold:.0f}s")

    period = 1.0 / args.hz
    deadline = time.monotonic() + args.hold
    next_tick = time.monotonic()
    sent = 0
    try:
        while time.monotonic() < deadline:
            sock.sendto(json.dumps(event, separators=(",", ":")).encode(), address)
            sent += 1
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        for _ in range(3):
            sock.sendto(json.dumps(stop, separators=(",", ":")).encode(), address)
    except KeyboardInterrupt:
        sock.sendto(json.dumps(stop, separators=(",", ":")).encode(), address)
    finally:
        sock.close()
    print(f"sent {sent} packets, then zeros. Look for "
          f"'[shape] event={event_id} card={args.card} accepted' on the policy.")


if __name__ == "__main__":
    raise SystemExit(main())
