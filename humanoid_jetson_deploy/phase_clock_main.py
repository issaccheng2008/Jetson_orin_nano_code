#!/usr/bin/env python3
"""One-shot phase-clock crossing on the existing 50 Hz STM32 motor link.

The operator restores the crouched initial pose and positions the leading toe
8 cm from the near edge of the stick before sending {"start": true} to the
loopback UDP port. Completion disables motor commands as requested by the
operator; it is not a physical success or balance detector.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
from pathlib import Path
import socket
import time

import numpy as np

import config
from imu_filter import projected_gravity_from_quaternion, validate_stationary_imu_sample
from main import monotonic_us, send_disable
from phase_clock_adapter import DEFAULT_BUNDLE, load_crossing_controller
from protocol import COMMAND_ENABLE, STATE_ENCODERS_VALID, STATE_FAULT, STATE_IMU_VALID
from serial_link import SerialLink
from target_safety import TargetSafety, add_target_safety_arguments


class StartCueSocket:
    """Poll a loopback-only, one-shot JSON UDP start request without a thread."""

    def __init__(self, bind: str, port: int):
        address = ipaddress.ip_address(bind)
        if not address.is_loopback or address.version != 4:
            raise ValueError("start-bind must be an IPv4 loopback address")
        if not 1 <= port <= 65535:
            raise ValueError("start-port must be between 1 and 65535")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.sock.setblocking(False)

    def poll(self) -> bool:
        requested = False
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            try:
                requested |= json.loads(data.decode("utf-8")) == {"start": True}
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        return requested

    def close(self) -> None:
        self.sock.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--start-bind", default="127.0.0.1")
    p.add_argument("--start-port", type=int, default=5008)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--enable-motors", action="store_true")
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--kd-scale", type=float, default=1.0)
    p.add_argument("--pose-tolerance-deg", type=float, default=10.0)
    p.add_argument("--max-joint-speed", type=float, default=0.2,
                   help="Maximum absolute encoder speed at the start cue, rad/s")
    p.add_argument("--log", type=Path, default=Path("logs/phase_clock_run.csv"))
    add_target_safety_arguments(p)
    args = p.parse_args()
    if not all(np.isfinite(v) for v in (args.kp_scale, args.kd_scale,
                                         args.pose_tolerance_deg, args.max_joint_speed)):
        p.error("gain and initial-pose limits must be finite")
    if not (0 <= args.kp_scale <= 1 and 0 <= args.kd_scale <= 1
            and args.pose_tolerance_deg > 0 and args.max_joint_speed > 0):
        p.error("gain or initial-pose limits out of range")
    try:
        TargetSafety.from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def initial_pose_ready(q_policy: np.ndarray, qd_policy: np.ndarray,
                       tolerance_deg: float, max_speed: float) -> bool:
    return bool(np.max(np.abs(q_policy - config.Q_DEFAULT))
                <= np.deg2rad(tolerance_deg)
                and np.max(np.abs(qd_policy)) <= max_speed)


def main() -> int:
    args = parse_args()
    try:
        target_safety = TargetSafety.from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(target_safety.describe(), flush=True)
    config.validate_imu_configuration()
    if args.enable_motors and not (config.CALIBRATION_CONFIRMED
                                   and config.IMU_CALIBRATION_CONFIRMED):
        raise SystemExit("Motor and IMU calibration must be confirmed in config.py")
    controller = load_crossing_controller(args.bundle)
    cue = StartCueSocket(args.start_bind, args.start_port)
    link = None
    last_q_motor = np.zeros(config.NUM_JOINTS, dtype=np.float32)
    started = False
    finished = False
    faulted = False
    try:
        link = SerialLink(args.port, args.baud)
        first = link.wait_for_state(timeout_s=5.0)
        required = STATE_IMU_VALID | STATE_ENCODERS_VALID
        if (first.status_flags & required) != required:
            raise RuntimeError("Initial IMU or encoder data invalid")
        gravity = projected_gravity_from_quaternion(
            first.orientation_wxyz, config.IMU_TO_POLICY,
            sensor_to_world=config.IMU_QUATERNION_IS_SENSOR_TO_WORLD)
        validate_stationary_imu_sample(config.IMU_TO_POLICY @ first.accel_m_s2,
                                       config.IMU_TO_POLICY @ first.gyro_rad_s,
                                       gravity)
        last_q_motor = first.joint_position.copy()
        last_q_target = config.motor_to_policy_position(last_q_motor)
        controller.reset()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="", encoding="utf-8") as handle:
            log = csv.writer(handle)
            log.writerow(("state_timestamp_us", "state_sequence", "read_monotonic_s",
                          "phase", "tick", "elapsed_s", "infer_done_monotonic_s",
                          "send_done_monotonic_s", "infer_ms", "motors_enabled")
                         + tuple(f"q_{name}" for name in config.JOINT_NAMES)
                         + tuple(f"qd_{name}" for name in config.JOINT_NAMES)
                         + ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                            "gravity_x", "gravity_y", "gravity_z")
                         + tuple(f"action_{name}" for name in config.JOINT_NAMES))
            next_tick = time.monotonic()
            print(f"Waiting for UDP {{\"start\": true}} at {args.start_bind}:{args.start_port}; "
                  f"motors={'enabled' if args.enable_motors else 'disabled'}", flush=True)
            while True:
                now = time.monotonic()
                state = link.get_latest_state(max_age_s=0.05)
                if state.status_flags & STATE_FAULT:
                    raise RuntimeError(f"STM32 fault flags=0x{state.status_flags:08X}")
                if (state.status_flags & required) != required:
                    raise RuntimeError("IMU or encoder data invalid")
                q = config.motor_to_policy_position(state.joint_position)
                qd = config.motor_to_policy_velocity(state.joint_velocity)
                acc = config.IMU_TO_POLICY @ state.accel_m_s2
                gyro = config.IMU_TO_POLICY @ state.gyro_rad_s
                gravity = projected_gravity_from_quaternion(
                    state.orientation_wxyz, config.IMU_TO_POLICY,
                    sensor_to_world=config.IMU_QUATERNION_IS_SENSOR_TO_WORLD)
                if cue.poll() and not started:
                    if not initial_pose_ready(q, qd, args.pose_tolerance_deg,
                                              args.max_joint_speed):
                        print("Start ignored: initial crouch or joint speed out of range",
                              flush=True)
                    else:
                        started = controller.start(now)
                        if started:
                            print("Phase clock started; external 8 cm toe-to-stick placement "
                                  "is assumed", flush=True)
                if started:
                    infer_start = time.monotonic()
                    sample = controller.tick(acc, gyro, gravity, q, qd, now=now)
                    infer_done = time.monotonic()
                    if sample.sequence_finished:
                        finished = True
                        log.writerow((state.timestamp_us, state.sequence, now, sample.phase,
                                      controller.clock.read(now).tick, sample.elapsed_s,
                                      infer_done, "", (infer_done - infer_start) * 1000.0,
                                      int(args.enable_motors))
                                     + tuple(float(v) for v in q)
                                     + tuple(float(v) for v in qd)
                                     + tuple(float(v) for v in acc)
                                     + tuple(float(v) for v in gyro)
                                     + tuple(float(v) for v in gravity)
                                     + ("",) * config.ACTION_DIM)
                        handle.flush()
                        print("sequence_finished: clock ended; disabling motors "
                              "(physical crossing success unknown)", flush=True)
                        break
                    if not sample.active:
                        raise RuntimeError("Started phase clock returned no target")
                    target = sample.joint_targets
                    phase = sample.phase
                    elapsed = sample.elapsed_s
                    tick = controller.clock.read(now).tick
                    action = sample.action
                else:
                    # Hold the pose already prepared by the operator; do not drive
                    # toward a new pose before the explicit start cue.
                    infer_start = infer_done = time.monotonic()
                    target = last_q_target
                    phase, elapsed, tick = -1, 0.0, -1
                    action = None
                target = target_safety.apply(target, last_q_target, q, config.POLICY_DT)
                last_q_target = target
                last_q_motor = config.policy_to_motor_position(target)
                link.send_command(monotonic_us(), last_q_motor, args.kp_scale,
                                  args.kd_scale, COMMAND_ENABLE if args.enable_motors else 0)
                sent = time.monotonic()
                log.writerow((state.timestamp_us, state.sequence, now, phase, tick,
                              elapsed, infer_done, sent,
                              (infer_done - infer_start) * 1000.0,
                              int(args.enable_motors))
                             + tuple(float(v) for v in q)
                             + tuple(float(v) for v in qd)
                             + tuple(float(v) for v in acc)
                             + tuple(float(v) for v in gyro)
                             + tuple(float(v) for v in gravity)
                             + (tuple(float(v) for v in action)
                                if action is not None else ("",) * config.ACTION_DIM))
                if started:
                    handle.flush()
                next_tick += config.POLICY_DT
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    print(f"WARNING: control deadline missed by {-delay * 1000:.1f} ms",
                          flush=True)
                    next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("Interrupted; disabling motors", flush=True)
    except Exception as exc:
        faulted = True
        print(f"FAULT: {exc}", flush=True)
        return 1
    finally:
        if link is not None:
            send_disable(link, last_q_motor, estop=faulted)
            link.close()
        cue.close()
    return 0 if finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
