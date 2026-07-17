#!/usr/bin/env python3
"""Move each selected joint a few degrees in both command directions."""

from __future__ import annotations

import argparse
import math
import signal

from motor_test_common import (
    MotorTestLoop,
    STANDING_COMMAND,
    STM32_BRIDGE_JOINT_NAMES,
    parse_joint_selection,
    require_confirmation,
    validate_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--kp-scale", type=float, default=0.12)
    parser.add_argument("--kd-scale", type=float, default=0.20)
    parser.add_argument("--ramp-seconds", type=float, default=4.0)
    parser.add_argument("--move-seconds", type=float, default=1.2)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--amplitude-deg", type=float, default=3.0)
    parser.add_argument(
        "--joints",
        default="all",
        help="Comma-separated STM32 channels (0..11) or exact joint names; default: all",
    )
    parser.add_argument(
        "--max-start-error",
        type=float,
        default=0.75,
        help="Abort if any standing target is farther than this many radians from feedback",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the typed safety confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_gains(args.kp_scale, args.kd_scale)
    if args.ramp_seconds < 1.0 or args.move_seconds < 0.5:
        raise SystemExit("Use --ramp-seconds >= 1.0 and --move-seconds >= 0.5")
    if args.pause_seconds < 0.0:
        raise SystemExit("--pause-seconds cannot be negative")
    if not 0.2 <= args.amplitude_deg <= 5.0:
        raise SystemExit("--amplitude-deg must be between 0.2 and 5.0 degrees")
    try:
        selected = parse_joint_selection(args.joints)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("\nCurrent STM32 protocol channel mapping:")
    for index in selected:
        print(f"  channel {index:2d}: {STM32_BRIDGE_JOINT_NAMES[index]}")
    print("Each channel will move +offset, return, -offset, then return to standing.")
    require_confirmation(args.yes, "DIRECTION TEST")

    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    amplitude = math.radians(args.amplitude_deg)
    loop = MotorTestLoop(
        args.port,
        args.baud,
        args.kp_scale,
        args.kd_scale,
        lambda: stopping,
    )
    try:
        loop.initialize(args.max_start_error)
        print(f"Ramping to standing over {args.ramp_seconds:.1f} s")
        if not loop.transition(STANDING_COMMAND, args.ramp_seconds, "ramping to standing"):
            return 0
        if args.pause_seconds and not loop.hold(STANDING_COMMAND, args.pause_seconds, "standing"):
            return 0

        for channel in selected:
            name = STM32_BRIDGE_JOINT_NAMES[channel]
            print(
                f"\nTesting channel {channel}: {name} "
                f"with +/-{args.amplitude_deg:.1f} deg commands"
            )
            for sign, direction in ((+1.0, "POSITIVE"), (-1.0, "NEGATIVE")):
                offset_target = STANDING_COMMAND.copy()
                offset_target[channel] += sign * amplitude
                if not loop.transition(
                    offset_target,
                    args.move_seconds,
                    f"{name} {direction}",
                ):
                    return 0
                if args.pause_seconds and not loop.hold(
                    offset_target,
                    args.pause_seconds,
                    f"{name} {direction} hold",
                ):
                    return 0
                if not loop.transition(
                    STANDING_COMMAND,
                    args.move_seconds,
                    f"{name} return",
                ):
                    return 0
                if args.pause_seconds and not loop.hold(
                    STANDING_COMMAND,
                    args.pause_seconds,
                    "standing",
                ):
                    return 0
    except Exception as exc:
        print(f"FAULT: {exc}")
        return 1
    finally:
        print("Sending e-stop and disable packets...")
        loop.emergency_stop_and_close()

    print("Motors disabled; direction test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

