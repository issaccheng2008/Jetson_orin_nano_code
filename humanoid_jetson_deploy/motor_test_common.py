"""Shared safety and timing helpers for motor-enabled STM32 bench tests.

These tests intentionally use the command/state coordinate order implemented by
the current STM32 ``jetson_robot_bridge.c``.  That bridge maps protocol channels
0..5 to the physical left leg and 6..11 to the physical right leg.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Callable, TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import COMMAND_ENABLE, COMMAND_ESTOP, STATE_ENCODERS_VALID, STATE_FAULT

if TYPE_CHECKING:
    from serial_link import SerialLink


CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Protocol channel -> physical joint mapping in the current STM32 bridge.
STM32_BRIDGE_JOINT_NAMES = (
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
)

# Exact standing command expected by the current STM32 bridge.  After the
# bridge's four sign conversions, this produces the same physical motor targets
# as the firmware's existing Action_Goto standing pose:
#   right raw motor targets: [+0.15, 0, 0, -0.30, -0.15, 0]
#   left  raw motor targets: [-0.15, 0, 0, +0.30, +0.15, 0]
STANDING_COMMAND = np.array(
    [
        +0.15,
        0.0,
        0.0,
        +0.30,
        -0.15,
        0.0,
        -0.15,
        0.0,
        0.0,
        -0.30,
        +0.15,
        0.0,
    ],
    dtype=np.float32,
)


def monotonic_us() -> int:
    return (time.monotonic_ns() // 1000) & 0xFFFFFFFF


def require_confirmation(skip_confirmation: bool, test_name: str) -> None:
    if skip_confirmation:
        return
    print("\nSAFETY CHECK")
    print("  - Suspend the robot so its feet cannot catch the floor.")
    print("  - Keep a physical emergency stop within reach.")
    print("  - Keep people, cables, and tools outside the leg workspace.")
    print("  - Be ready to press Ctrl+C; the program then sends e-stop/disable frames.")
    try:
        answer = input(f'Type "{test_name}" to enable the motors: ').strip()
    except EOFError as exc:
        raise SystemExit("No terminal confirmation received; motors remain disabled.") from exc
    if answer != test_name:
        raise SystemExit("Confirmation did not match; motors remain disabled.")


def validate_gains(kp_scale: float, kd_scale: float) -> None:
    if not 0.0 < kp_scale <= 0.5:
        raise SystemExit("--kp-scale must be greater than 0 and no more than 0.5")
    if not 0.0 < kd_scale <= 0.5:
        raise SystemExit("--kd-scale must be greater than 0 and no more than 0.5")


def parse_joint_selection(value: str) -> list[int]:
    """Parse ``all``, indices, or exact joint names into unique channel indices."""
    if value.strip().lower() == "all":
        return list(range(len(STM32_BRIDGE_JOINT_NAMES)))

    name_to_index = {name: index for index, name in enumerate(STM32_BRIDGE_JOINT_NAMES)}
    selected: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if token in name_to_index:
            index = name_to_index[token]
        else:
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(f"Unknown joint {token!r}") from exc
            if not 0 <= index < len(STM32_BRIDGE_JOINT_NAMES):
                raise ValueError(f"Joint index {index} is outside 0..11")
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("No joints selected")
    return selected


class MotorTestLoop:
    """50 Hz command loop with state checks and repeated emergency shutdown."""

    def __init__(
        self,
        port: str,
        baud: int,
        kp_scale: float,
        kd_scale: float,
        stop_requested: Callable[[], bool],
    ) -> None:
        # Keep offline tests and ``--help`` usable without pyserial. The real
        # hardware dependency is required as soon as a serial device is opened.
        from serial_link import SerialLink

        self.link = SerialLink(port, baud)
        self.kp_scale = kp_scale
        self.kd_scale = kd_scale
        self.stop_requested = stop_requested
        self.last_target = np.zeros(12, dtype=np.float32)
        self.next_tick = time.monotonic()
        self.last_log = 0.0

    def initialize(self, maximum_start_error: float) -> np.ndarray:
        state = self.link.wait_for_state(timeout_s=5.0)
        self._validate_state(state)
        self.last_target = np.asarray(state.joint_position, dtype=np.float32).copy()
        largest_delta = float(np.max(np.abs(STANDING_COMMAND - self.last_target)))
        print(f"Received STM32 state sequence {state.sequence}; flags=0x{state.status_flags:08X}")
        print(f"Largest initial move needed to reach standing: {math.degrees(largest_delta):.1f} deg")
        if largest_delta > maximum_start_error:
            raise RuntimeError(
                "Standing target is too far from the measured pose "
                f"({largest_delta:.3f} rad > {maximum_start_error:.3f} rad). "
                "Check motor zero positions before enabling this test."
            )
        return self.last_target.copy()

    @staticmethod
    def _validate_state(state) -> None:
        if state.status_flags & STATE_FAULT:
            raise RuntimeError(f"STM32 reports a motor fault: flags=0x{state.status_flags:08X}")
        if not state.status_flags & STATE_ENCODERS_VALID:
            raise RuntimeError(f"STM32 encoder-valid flag is missing: 0x{state.status_flags:08X}")
        if not np.isfinite(state.joint_position).all():
            raise RuntimeError("STM32 joint state contains NaN or Inf")

    def _send_tick(self, target: np.ndarray, label: str = "") -> bool:
        if self.stop_requested():
            return False
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise RuntimeError("Refusing to send an invalid 12-joint target")

        state = self.link.get_latest_state(max_age_s=0.10)
        self._validate_state(state)
        self.link.send_command(
            monotonic_us(),
            target,
            self.kp_scale,
            self.kd_scale,
            COMMAND_ENABLE,
        )
        self.last_target = target.copy()

        now = time.monotonic()
        if now - self.last_log >= 1.0:
            error = float(np.max(np.abs(target - state.joint_position)))
            print(
                f"{label or 'holding'}: state_seq={state.sequence:5d} "
                f"max_tracking_error={math.degrees(error):5.1f} deg "
                f"crc_errors={self.link.decoder.crc_errors}"
            )
            self.last_log = now

        self.next_tick += CONTROL_DT
        delay = self.next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            self.next_tick = time.monotonic()
        return True

    def transition(self, destination: np.ndarray, duration_s: float, label: str) -> bool:
        """Move with a cosine profile so velocity is zero at both endpoints."""
        if duration_s <= 0.0:
            raise ValueError("Transition duration must be positive")
        start = self.last_target.copy()
        destination = np.asarray(destination, dtype=np.float32)
        steps = max(2, int(math.ceil(duration_s * CONTROL_HZ)))
        for step in range(1, steps + 1):
            phase = step / steps
            blend = 0.5 - 0.5 * math.cos(math.pi * phase)
            target = start + np.float32(blend) * (destination - start)
            if not self._send_tick(target, label):
                return False
        return True

    def hold(self, target: np.ndarray, duration_s: float | None, label: str) -> bool:
        start = time.monotonic()
        while duration_s is None or time.monotonic() - start < duration_s:
            if not self._send_tick(target, label):
                return False
        return True

    def emergency_stop_and_close(self) -> None:
        # Send e-stop first, then keep the enable bit clear in several additional
        # packets.  Repetition makes shutdown robust to one dropped USB packet.
        for index in range(10):
            try:
                flags = COMMAND_ESTOP if index < 5 else 0
                self.link.send_command(monotonic_us(), self.last_target, 0.0, 0.0, flags)
            except Exception:
                break
            time.sleep(0.01)
        self.link.close()
