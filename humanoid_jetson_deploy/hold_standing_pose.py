#!/usr/bin/env python3
"""Gently move to the firmware's standing pose and command it until Ctrl+C."""

from __future__ import annotations

import argparse
import signal

from motor_test_common import (
    MotorTestLoop,
    STANDING_COMMAND,
    require_confirmation,
    validate_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--kp-scale", type=float, default=0.15)
    parser.add_argument("--kd-scale", type=float, default=0.25)
    parser.add_argument("--ramp-seconds", type=float, default=4.0)
    parser.add_argument(
        "--max-start-error",
        type=float,
        default=0.75,
        help="Abort if any standing target is farther than this many radians from feedback",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Seconds to hold after the ramp; 0 means until Ctrl+C",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the typed safety confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_gains(args.kp_scale, args.kd_scale)
    if args.ramp_seconds < 1.0:
        raise SystemExit("--ramp-seconds must be at least 1.0")
    if args.max_start_error <= 0.0:
        raise SystemExit("--max-start-error must be positive")
    if args.max_seconds < 0.0:
        raise SystemExit("--max-seconds cannot be negative")

    require_confirmation(args.yes, "STANDING TEST")

    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"Opening {args.port}; motor commands will be sent at 50 Hz")
    loop = MotorTestLoop(
        args.port,
        args.baud,
        args.kp_scale,
        args.kd_scale,
        lambda: stopping,
    )
    try:
        loop.initialize(args.max_start_error)
        print(f"Ramping to the predefined standing command over {args.ramp_seconds:.1f} s")
        if not loop.transition(STANDING_COMMAND, args.ramp_seconds, "ramping to standing"):
            return 0
        duration = None if args.max_seconds == 0.0 else args.max_seconds
        print("Holding standing pose. Press Ctrl+C to stop.")
        loop.hold(STANDING_COMMAND, duration, "standing hold")
    except Exception as exc:
        print(f"FAULT: {exc}")
        return 1
    finally:
        print("Sending e-stop and disable packets...")
        loop.emergency_stop_and_close()

    print("Motors disabled; standing test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

