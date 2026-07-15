#!/usr/bin/env python3
"""Test bidirectional FK723 USB-CDC communication without ONNX or motor enable."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import STATE_COMMAND_FRESH, STATE_ENCODERS_VALID, STATE_IMU_VALID
from serial_link import SerialLink


def monotonic_us() -> int:
    return (time.monotonic_ns() // 1000) & 0xFFFFFFFF


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--seconds", type=float, default=5.0)
args = parser.parse_args()

received = 0
sequence_drops = 0
last_sequence = None
command_seen = False
start = time.monotonic()
next_command = start

with SerialLink(args.port, 921600) as link:
    first = link.wait_for_state(5.0)
    print("First packet:")
    print("  sequence:", first.sequence)
    print("  q[0:3]:", first.joint_position[:3])
    print("  accel:", first.accel_m_s2)
    print("  gyro:", first.gyro_rad_s)
    print(f"  flags: 0x{first.status_flags:08X}")

    while time.monotonic() - start < args.seconds:
        now = time.monotonic()
        state = link.get_latest_state(0.1)
        if last_sequence != state.sequence:
            if last_sequence is not None:
                expected = (last_sequence + 1) & 0xFFFF
                sequence_drops += (state.sequence - expected) & 0xFFFF
            last_sequence = state.sequence
            received += 1
            command_seen |= bool(state.status_flags & STATE_COMMAND_FRESH)

        if now >= next_command:
            # Echo the current pose with enable=0 and zero gain scales.
            link.send_command(monotonic_us(), state.joint_position, 0.0, 0.0, 0)
            next_command += 0.02
        time.sleep(0.001)

    elapsed = time.monotonic() - start
    required = STATE_IMU_VALID | STATE_ENCODERS_VALID
    if (state.status_flags & required) != required:
        raise SystemExit(f"FAIL: IMU/encoder valid flags missing: 0x{state.status_flags:08X}")
    if not np.isfinite(state.joint_position).all() or not np.isfinite(state.accel_m_s2).all():
        raise SystemExit("FAIL: state contains NaN or Inf")
    if not command_seen:
        raise SystemExit("FAIL: STM32 never reported STATE_COMMAND_FRESH")

    print("PASS")
    print(f"  observed state rate: {received / elapsed:.1f} Hz")
    print(f"  sequence drops: {sequence_drops}")
    print(f"  CRC errors: {link.decoder.crc_errors}")
    print("  STM32 received disabled Jetson commands: yes")
