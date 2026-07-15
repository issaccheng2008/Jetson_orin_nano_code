#!/usr/bin/env python3
"""Bridge vision navigation output to the humanoid policy command receiver.

The vision process publishes JSON over UDP to ``VISION_INPUT_PORT``.  This
process validates and processes that message, then republishes the policy
command at a fixed rate to ``POLICY_OUTPUT_PORT``.  Keeping the connector in a
separate process prevents camera processing delays from blocking the 50 Hz
policy loop.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def process_vision_output(message: dict[str, Any]) -> dict[str, float | int]:
    """Example hook for converting vision output into a policy command.

    Replace or extend this function later for QR-specific behavior, obstacle
    state machines, speed scheduling, or command smoothing.  For now it:

    * validates finite numeric inputs;
    * clamps commands to the ranges used during policy training;
    * always forces target lateral velocity ``vy`` to zero; and
    * forwards the currently visible QR value (or ``-1``).
    """

    vx = float(message["vx"])
    wz = float(message["wz"])
    qr = int(message.get("qr", -1))
    if not math.isfinite(vx) or not math.isfinite(wz):
        raise ValueError("vx and wz must be finite")
    if qr not in (-1, 1, 2, 3, 4, 5, 6):
        qr = -1

    return {
        "vx": clamp(vx, 0.0, 1.0),
        "vy": 0.0,
        "wz": clamp(wz, -0.5, 0.5),
        "qr": qr,
    }


def select_output(
    latest: dict[str, float | int],
    last_vision_update: float,
    now: float,
    timeout_s: float,
) -> tuple[dict[str, float | int], bool, float]:
    """Hold the latest vision command until it becomes stale.

    Vision may run near 10 Hz while this connector publishes at 50 Hz.  This
    zero-order hold returns the same latest command on every connector tick, so
    the policy still receives a target on every inference step.
    """

    age_s = max(0.0, now - last_vision_update) if last_vision_update > 0.0 else math.inf
    fresh = age_s <= timeout_s
    if fresh:
        return latest, True, age_s
    return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}, False, age_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-bind", default="127.0.0.1")
    parser.add_argument("--vision-port", type=int, default=5006)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5005)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument(
        "--vision-timeout",
        type=float,
        default=0.25,
        help="Publish a zero command when vision is stale for this many seconds",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print the current connector-to-policy target every N 50 Hz publications",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publish_hz <= 0.0 or args.vision_timeout <= 0.0:
        raise SystemExit("publish-hz and vision-timeout must be positive")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setblocking(False)
    receiver.bind((args.vision_bind, args.vision_port))
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    latest = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
    last_vision_update = 0.0
    period = 1.0 / args.publish_hz
    next_tick = time.monotonic()
    step = 0
    vision_update = 0

    print(
        f"Connector: vision udp://{args.vision_bind}:{args.vision_port} -> "
        f"policy udp://{args.policy_host}:{args.policy_port} at {args.publish_hz:.1f} Hz"
    )

    try:
        while True:
            while True:
                try:
                    payload, _address = receiver.recvfrom(4096)
                except BlockingIOError:
                    break

                try:
                    decoded = json.loads(payload.decode("utf-8"))
                    if not isinstance(decoded, dict):
                        raise ValueError("message must be a JSON object")
                    latest = process_vision_output(decoded)
                    last_vision_update = time.monotonic()
                    vision_update += 1
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    print(f"[connector] ignored invalid vision message: {exc}")

            now = time.monotonic()
            output, vision_fresh, vision_age_s = select_output(
                latest,
                last_vision_update,
                now,
                args.vision_timeout,
            )
            publisher.sendto(
                json.dumps(output, separators=(",", ":")).encode("utf-8"),
                (args.policy_host, args.policy_port),
            )

            if step % max(1, args.log_every) == 0:
                age_text = f"{vision_age_s * 1000.0:6.1f}ms" if math.isfinite(vision_age_s) else " never"
                print(
                    f"[connector -> policy] publish={step:7d} "
                    f"vision_update={vision_update:7d} fresh={vision_fresh} "
                    f"age={age_text} qr={output['qr']} "
                    f"target_velocity=[vx={output['vx']:+.3f} m/s, "
                    f"vy={output['vy']:+.3f} m/s, wz={output['wz']:+.3f} rad/s]"
                )

            step += 1
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stop = b'{"vx":0.0,"vy":0.0,"wz":0.0,"qr":-1}'
        for _ in range(3):
            publisher.sendto(stop, (args.policy_host, args.policy_port))
        receiver.close()
        publisher.close()

    print("Connector stopped; zero command sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
