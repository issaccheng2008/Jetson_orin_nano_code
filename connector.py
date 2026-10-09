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

STRAIGHT_WZ = 0.1  # Straight-walking yaw compensation, rad/s.


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def slew_toward(current: float, target: float, max_step: float) -> float:
    """Move current toward target by at most max_step, landing exactly on it."""
    delta = target - current
    if abs(delta) <= max_step:
        return target  # the target itself, not an accumulation: a zero stays float 0.0
    return current + math.copysign(max_step, delta)


def _command_mode(message: dict[str, Any]) -> str | None:
    """Validate the optional transmission mode; explicit null is invalid."""
    if "command_mode" not in message:
        return None
    mode = message["command_mode"]
    if not isinstance(mode, str) or mode not in ("held", "continuous"):
        raise ValueError("command_mode must be 'held' or 'continuous'")
    return mode


def process_vision_output(message: dict[str, Any], wz_bias: float = 0.0) -> dict[str, Any]:
    """Example hook for converting vision output into a policy command.

    Replace or extend this function later for QR-specific behavior, obstacle
    state machines, or speed scheduling (command smoothing is done downstream
    by ``CommandSmoother``).  For now it:

    * validates finite numeric inputs;
    * offsets moving yaw commands before clamping (stops/posture are exempt);
    * clamps commands to the ranges used during policy training;
    * always forces target lateral velocity ``vy`` to zero; and
    * forwards the currently visible QR value (or ``-1``); and
    * validates and forwards optional ``command_mode`` held/continuous tags.
    """

    mode = _command_mode(message)
    vx = float(message["vx"])
    wz = float(message["wz"])
    qr = int(message.get("qr", -1))
    if not math.isfinite(vx) or not math.isfinite(wz):
        raise ValueError("vx and wz must be finite")
    if not math.isfinite(wz_bias):
        raise ValueError("wz bias must be finite")
    if qr not in (-1, 1, 2, 3, 4, 5, 6):
        qr = -1

    result = {
        "vx": clamp(vx, 0.0, 1.0),
        "vy": 0.0,
        "wz": clamp(wz, -0.5, 0.5),
        "qr": qr,
        # Vision is standing the robot still for a card and wants the body held
        # upright while it reads it. Absent means no, so an older vision is unchanged.
        "hold_upright": bool(message.get("hold_upright", False)),
        # Same window, different mechanism: the STM32 re-poses the body and reports a
        # frozen attitude instead of the Nano holding a pose. The two drive the same
        # joints, so a rig uses one or the other, never both.
        "card_tilt": bool(message.get("card_tilt", False)),
    }
    # Apply once per received packet, not on each 50 Hz publication. Walking
    # straight (vx > 0, wz == 0) gets compensation; a full stop stays zero.
    # Posture requests are stopped downstream by CommandSmoother.
    if (not result["hold_upright"] and not result["card_tilt"]
            and (result["vx"] != 0.0 or wz != 0.0)):
        if result["vx"] > 0.0 and wz == 0.0:
            wz = STRAIGHT_WZ
        result["wz"] = clamp(wz + wz_bias, -0.5, 0.5)
    if mode is not None:
        result["command_mode"] = mode
    if "event_id" in message or "event_action" in message:
        event_id = int(message["event_id"])
        event_action = int(message["event_action"])
        if not 0 < event_id <= 0xFFFFFFFF or event_action not in (1, 2, 3, 4, 5, 6):
            raise ValueError("invalid shape event")
        result["event_id"] = event_id
        result["event_action"] = event_action
    return result


def select_output(
    latest: dict[str, Any],
    last_vision_update: float,
    now: float,
    timeout_s: float,
) -> tuple[dict[str, Any], bool, float]:
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
    """Slew-limit legacy/continuous commands; transmit held commands unchanged.

    Stops, stale-vision zeroes and card posture requests stop immediately and
    reset the smoother. Held commands synchronize its state, so changing mode
    cannot resume an older ramp. The connector still republishes at 50 Hz;
    held describes command semantics, not a slower heartbeat.
    """

    def __init__(self, max_vx_accel: float, max_wz_accel: float) -> None:
        if not math.isfinite(max_vx_accel) or max_vx_accel <= 0.0:
            raise ValueError("vx acceleration limit must be finite and positive")
        # 0 is not a limit of zero, it is no limit at all: --wz-mode discrete 要的就是
        # 方波传递。这个斜率是为连续指令准备的（0.5 rad/s 要爬 0.25 s），短脉冲
        # 经它一削就只剩个三角形，峰值也到不了。
        if not math.isfinite(max_wz_accel) or max_wz_accel < 0.0:
            raise ValueError("wz acceleration limit must be finite and nonnegative")
        self.max_vx_accel = max_vx_accel
        self.max_wz_accel = max_wz_accel
        self.vx = 0.0
        self.wz = 0.0

    def update(
        self, target: dict[str, Any], dt: float
    ) -> dict[str, Any]:
        mode = _command_mode(target)
        vx, wz = float(target["vx"]), float(target["wz"])
        stop = (bool(target.get("hold_upright", False))
                or bool(target.get("card_tilt", False))
                or (vx == 0.0 and wz == 0.0))
        if stop:
            self.vx = self.wz = 0.0
        elif mode == "held":
            self.vx, self.wz = vx, wz
        else:
            self.vx = slew_toward(self.vx, vx, self.max_vx_accel * dt)
            if self.max_wz_accel > 0.0:
                self.wz = slew_toward(self.wz, wz, self.max_wz_accel * dt)
            else:
                self.wz = wz
        output = {"vx": self.vx, "vy": 0.0, "wz": self.wz, "qr": int(target["qr"]),
                  "hold_upright": bool(target.get("hold_upright", False)),
                  "card_tilt": bool(target.get("card_tilt", False))}
        if mode is not None:
            output["command_mode"] = mode
        if "event_id" in target:
            output["event_id"] = int(target["event_id"])
            output["event_action"] = int(target["event_action"])
        return output


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
        help="Slew limit on the forwarded yaw rate, rad/s^2 (0.5 rad/s in 0.25 s). "
             "0 = no limit at all, passed through on the same tick - that is what "
             "--wz-mode discrete needs, because a short square pulse comes out of "
             "this limiter as a triangle that never reaches its target",
    )
    parser.add_argument(
        "--wz-bias",
        type=float,
        default=0.0,
        help="Signed yaw-rate offset in rad/s, applied before +/-0.5 clamping. "
             "Full stops, posture requests and stale-vision zeroes stay stopped.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print the current connector-to-policy target every N 50 Hz publications",
    )
    parser.add_argument(
        "--tap-port",
        type=int,
        default=0,
        help="Mirror every published packet verbatim to this port on the same host "
             "(0 = off). A listener there sees the true 50 Hz command ramp, which "
             "the vision log's 2 Hz sampling hides; being on the same clock as the "
             "policy's observation dump is the point.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publish_hz <= 0.0 or args.vision_timeout <= 0.0:
        raise SystemExit("publish-hz and vision-timeout must be positive")
    if not math.isfinite(args.max_vx_accel) or args.max_vx_accel <= 0.0:
        raise SystemExit("max-vx-accel must be finite and positive")
    if not math.isfinite(args.max_wz_accel) or args.max_wz_accel < 0.0:
        raise SystemExit("max-wz-accel must be finite and nonnegative (0 = no limit)")
    if not math.isfinite(args.wz_bias):
        raise SystemExit("wz-bias must be finite")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setblocking(False)
    receiver.bind((args.vision_bind, args.vision_port))
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.tap_port else None

    latest = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
    last_vision_update = 0.0
    period = 1.0 / args.publish_hz
    next_tick = time.monotonic()
    last_tick = next_tick
    smoother = CommandSmoother(args.max_vx_accel, args.max_wz_accel)
    step = 0
    vision_update = 0
    # Rate counters for the log line, reset on every print so the numbers say
    # what is happening now instead of growing forever.
    log_step = log_vision = 0
    log_time = time.monotonic()

    print(
        f"Connector: vision udp://{args.vision_bind}:{args.vision_port} -> "
        f"policy udp://{args.policy_host}:{args.policy_port} at {args.publish_hz:.1f} Hz; "
        f"wz_bias={args.wz_bias:+g} rad/s"
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
                    latest = process_vision_output(decoded, wz_bias=args.wz_bias)
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
            wire = json.dumps(output, separators=(",", ":")).encode("utf-8")
            publisher.sendto(wire, (args.policy_host, args.policy_port))
            if tap is not None:
                # Same bytes, same tick: the tap has to be the wire, not a re-encode.
                # The default 50 Hz stream carries no host time, so a listener that
                # needs to align it with policy_runner's POLICY_OBS_CSV should stamp
                # arrival time itself - both live on this machine's clock.
                tap.sendto(wire, (args.policy_host, args.tap_port))

            if step % max(1, args.log_every) == 0:
                now = time.monotonic()
                span = max(now - log_time, 1e-6)
                publish_hz = (step - log_step) / span
                vision_hz = (vision_update - log_vision) / span
                log_time, log_step, log_vision = now, step, vision_update
                # out = what the policy is actually given, in = how fast vision
                # feeds it. The target only matters while it differs, i.e. during
                # a slew or after vision went stale, so it is shown only then.
                age_text = (f"{vision_age_s * 1000.0:5.1f}ms" if math.isfinite(vision_age_s)
                            else "  never")
                line = (f"[connector] {publish_hz:4.0f}Hz out  {vision_hz:4.0f}Hz in  "
                        f"age={age_text}  vx={output['vx']:+.3f} wz={output['wz']:+.3f}")
                if (abs(output["vx"] - target["vx"]) > 1e-3
                        or abs(output["wz"] - target["wz"]) > 1e-3):
                    line += f" -> {target['vx']:+.3f} {target['wz']:+.3f}"
                if output["qr"] != -1 or "event_id" in output:
                    line += f"  qr={output['qr']} event={output.get('event_id', 0)}"
                if not vision_fresh:
                    line += "  STALE"
                print(line)

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
        # Clear held/ramped state and send an immediate stop on shutdown too.
        zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
        smoother.update(zero, 0.0)
        stop = b'{"vx":0.0,"vy":0.0,"wz":0.0,"qr":-1}'
        for _ in range(3):
            publisher.sendto(stop, (args.policy_host, args.policy_port))
        receiver.close()
        publisher.close()
        if tap is not None:
            tap.close()

    print("Connector stopped; zero command sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
