#!/usr/bin/env python3
"""Run the current 48-input humanoid ONNX policy and exchange data with STM32."""

from __future__ import annotations

import argparse
import signal
import time

import numpy as np

import config
from command_source import FixedCommandSource, UdpCommandSource
from imu_filter import ProjectedGravityFilter
from policy_runner import HumanoidPolicy
from protocol import (
    COMMAND_ENABLE,
    COMMAND_ESTOP,
    STATE_ENCODERS_VALID,
    STATE_FAULT,
    STATE_IMU_VALID,
)
from serial_link import SerialLink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to policy.onnx")
    parser.add_argument("--port", default="/dev/ttyACM0", help="STM32 serial device")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward command in m/s")
    parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw-rate command in rad/s")
    parser.add_argument("--udp-command-port", type=int, default=0, help="Use local UDP JSON commands")
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--log-every", type=int, default=25, help="Print every N policy steps")
    return parser.parse_args()


def monotonic_us() -> int:
    return (time.monotonic_ns() // 1000) & 0xFFFFFFFF


def slew_limit(target: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    maximum_change = config.MAX_TARGET_SPEED_RAD_S * dt
    return previous + np.clip(target - previous, -maximum_change, maximum_change)


def send_disable(link: SerialLink, q_motor: np.ndarray, estop: bool = False) -> None:
    flags = COMMAND_ESTOP if estop else 0
    for _ in range(3):
        try:
            link.send_command(monotonic_us(), q_motor, 0.0, 0.0, flags)
        except Exception:
            break
        time.sleep(0.005)


def main() -> int:
    args = parse_args()
    if args.enable_motors and not config.CALIBRATION_CONFIRMED:
        raise SystemExit(
            "Refusing to enable motors: calibrate MOTOR_SIGN and MOTOR_ZERO_RAD in "
            "config.py, then set CALIBRATION_CONFIRMED=True."
        )
    if not 0.0 <= args.kp_scale <= 1.0 or not 0.0 <= args.kd_scale <= 1.0:
        raise SystemExit("kp-scale and kd-scale must be between 0 and 1")

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    policy = HumanoidPolicy(args.model)
    gravity_filter = ProjectedGravityFilter()
    command_source = (
        UdpCommandSource(args.udp_command_port)
        if args.udp_command_port
        else FixedCommandSource(args.vx, args.wz)
    )

    print(f"ONNX input={policy.input_name!r}, output={policy.output_name!r}")
    print(f"Opening {args.port} (line coding {args.baud}; native USB CDC ignores physical baud)")
    print("MOTORS ENABLED" if args.enable_motors else "DRY RUN: command enable flag is OFF")

    link = SerialLink(args.port, args.baud)
    last_q_motor = np.zeros(config.NUM_JOINTS, dtype=np.float32)
    try:
        first_state = link.wait_for_state(timeout_s=5.0)
        last_q_motor = first_state.joint_position.copy()
        last_q_policy_target = config.motor_to_policy_position(last_q_motor)
        print(f"Received STM32 state packet, sequence={first_state.sequence}")

        next_tick = time.monotonic()
        previous_tick = next_tick
        start_time = next_tick
        step = 0

        while not stop_requested:
            now = time.monotonic()
            if args.max_seconds > 0.0 and now - start_time >= args.max_seconds:
                break

            state = link.get_latest_state(max_age_s=0.05)
            if state.status_flags & STATE_FAULT:
                raise RuntimeError(f"STM32 reports a fault: flags=0x{state.status_flags:08X}")
            required = STATE_IMU_VALID | STATE_ENCODERS_VALID
            if (state.status_flags & required) != required:
                raise RuntimeError(f"IMU/encoder data invalid: flags=0x{state.status_flags:08X}")

            dt = float(np.clip(now - previous_tick, 0.005, 0.05))
            previous_tick = now

            q_policy = config.motor_to_policy_position(state.joint_position)
            qd_policy = config.motor_to_policy_velocity(state.joint_velocity)
            accel_policy = config.IMU_TO_POLICY @ state.accel_m_s2
            gyro_policy = config.IMU_TO_POLICY @ state.gyro_rad_s
            projected_gravity = gravity_filter.update(accel_policy, gyro_policy, dt)
            velocity_command = command_source.get()

            q_policy_target, action, obs, latency_ms = policy.step(
                accel_m_s2=accel_policy,
                gyro_rad_s=gyro_policy,
                projected_gravity=projected_gravity,
                velocity_command=velocity_command,
                joint_position_policy=q_policy,
                joint_velocity_policy=qd_policy,
            )
            q_policy_target = config.clamp_policy_target(q_policy_target)
            q_policy_target = slew_limit(q_policy_target, last_q_policy_target, dt)
            last_q_policy_target = q_policy_target
            last_q_motor = config.policy_to_motor_position(q_policy_target)

            flags = COMMAND_ENABLE if args.enable_motors else 0
            link.send_command(
                monotonic_us(),
                last_q_motor,
                args.kp_scale,
                args.kd_scale,
                flags,
            )

            if step % max(1, args.log_every) == 0:
                print(
                    f"step={step:6d} state_seq={state.sequence:5d} "
                    f"policy_target_velocity=[vx={velocity_command[0]:+.3f} m/s, "
                    f"vy={velocity_command[1]:+.3f} m/s, "
                    f"wz={velocity_command[2]:+.3f} rad/s] "
                    f"infer={latency_ms:.3f}ms |obs|max={np.max(np.abs(obs)):.3f} "
                    f"|action|max={np.max(np.abs(action)):.3f} "
                    f"crc_errors={link.decoder.crc_errors}"
                )

            step += 1
            next_tick += config.POLICY_DT
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                print(f"WARNING: policy deadline missed by {-sleep_s * 1000.0:.2f} ms")
                next_tick = time.monotonic()
    except Exception as exc:
        print(f"FAULT: {exc}")
        send_disable(link, last_q_motor, estop=True)
        return 1
    finally:
        send_disable(link, last_q_motor)
        command_source.close()
        link.close()

    print("Policy stopped; disable packets sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
