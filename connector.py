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


def slew_toward(current: float, target: float, max_step: float) -> float:
    """Move current toward target by at most max_step, landing exactly on it."""
    delta = target - current
    if abs(delta) <= max_step:
        return target  # the target itself, not an accumulation: a zero stays float 0.0
    return current + math.copysign(max_step, delta)


def process_vision_output(message: dict[str, Any]) -> dict[str, float | int]:
    """Example hook for converting vision output into a policy command.

    Replace or extend this function later for QR-specific behavior, obstacle
    state machines, or speed scheduling (command smoothing is done downstream
    by ``CommandSmoother``).  For now it:

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


class CommandSmoother:
    """Slew-rate limit the policy command so it never steps.

    The target jumps whenever vision is lost or reacquired, the watchdog
    expires, or the first packet arrives.  Moving toward it by at most
    ``accel * dt`` per tick keeps a bipedal gait from being handed a velocity
    step it cannot absorb.  ``qr`` is discrete and passes through untouched.
    """

    def __init__(self, max_vx_accel: float, max_wz_accel: float) -> None:
        for value in (max_vx_accel, max_wz_accel):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("acceleration limits must be finite and positive")
        self.max_vx_accel = max_vx_accel
        self.max_wz_accel = max_wz_accel
        self.vx = 0.0
        self.wz = 0.0

    def update(
        self, target: dict[str, float | int], dt: float
    ) -> dict[str, float | int]:
        self.vx = slew_toward(self.vx, float(target["vx"]), self.max_vx_accel * dt)
        self.wz = slew_toward(self.wz, float(target["wz"]), self.max_wz_accel * dt)
        return {"vx": self.vx, "vy": 0.0, "wz": self.wz, "qr": int(target["qr"])}


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
        "--max-vx-accel",
        type=float,
        default=1.0,
        help="Slew limit on the forwarded forward speed, m/s^2 (0.4 m/s in 0.40 s)",
    )
    parser.add_argument(
        "--max-wz-accel",
        type=float,
        default=2.0,
        help="Slew limit on the forwarded yaw rate, rad/s^2 (0.5 rad/s in 0.25 s)",
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
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (args.max_vx_accel, args.max_wz_accel)
    ):
        raise SystemExit("max-vx-accel and max-wz-accel must be finite and positive")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setblocking(False)
    receiver.bind((args.vision_bind, args.vision_port))
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    latest = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
    last_vision_update = 0.0
    period = 1.0 / args.publish_hz
    next_tick = time.monotonic()
    last_tick = next_tick
    smoother = CommandSmoother(args.max_vx_accel, args.max_wz_accel)
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
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
                    print(f"[connector] ignored invalid vision message: {exc}")

            now = time.monotonic()
            target, vision_fresh, vision_age_s = select_output(
                latest,
                last_vision_update,
                now,
                args.vision_timeout,
            )
            # Measured dt, not the nominal period, so a stalled loop still ramps
            # in wall-clock time; capped at two ticks so one long stall cannot
            # buy a bigger jump than the configured rate allows.
            dt = min(max(now - last_tick, 0.0), 2.0 * period)
            last_tick = now
            output = smoother.update(target, dt)
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
                    f"cmd_velocity=[vx={output['vx']:+.3f} m/s, "
                    f"vy={output['vy']:+.3f} m/s, wz={output['wz']:+.3f} rad/s] "
                    f"target_velocity=[vx={target['vx']:+.3f} m/s, "
                    f"wz={target['wz']:+.3f} rad/s]"
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
        # Ctrl+C here is the documented first step of the stop sequence, so the
        # last packet the policy sees should ramp down rather than step.
        zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
        deadline = time.monotonic() + 2.0
        while (smoother.vx != 0.0 or smoother.wz != 0.0) and time.monotonic() < deadline:
            publisher.sendto(
                json.dumps(smoother.update(zero, period), separators=(",", ":")).encode("utf-8"),
                (args.policy_host, args.policy_port),
            )
            time.sleep(period)
        stop = b'{"vx":0.0,"vy":0.0,"wz":0.0,"qr":-1}'
        for _ in range(3):
            publisher.sendto(stop, (args.policy_host, args.policy_port))
        receiver.close()
        publisher.close()

    print("Connector stopped; zero command sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
