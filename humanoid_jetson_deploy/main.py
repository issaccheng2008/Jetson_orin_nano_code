#!/usr/bin/env python3
"""Run a walking or one-foot standing ONNX policy and exchange data with STM32."""

from __future__ import annotations

import argparse
import signal
import time

import numpy as np

import config
from command_source import FixedCommandSource, UdpCommandSource
from imu_filter import (
    projected_gravity_from_quaternion,
    roll_pitch_yaw_from_quaternion,
    validate_stationary_imu_sample,
)
from policy_runner import HumanoidPolicy
from one_foot_policy import OneFootCommand, OneFootPolicy
from position_monitor import LivePositionPlot, PositionCsvLogger
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
    parser.add_argument(
        "--policy", choices=("walking", "one-foot"), default="walking",
        help="Observation/action interface: walking=49 inputs, one-foot=46 inputs",
    )
    parser.add_argument(
        "--support-foot", choices=("right", "left"), default="right",
        help="One-foot mode: supporting foot (the opposite foot lifts)",
    )
    parser.add_argument(
        "--stand-seconds", type=float, default=1.0,
        help="One-foot mode: initial command-zero duration (seconds)",
    )
    parser.add_argument(
        "--lift-seconds", type=float, default=4.0,
        help="One-foot mode: command-one duration, then command zero until exit",
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="STM32 serial device")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--command-source",
        choices=("fixed", "vision"),
        default="fixed",
        help="Walking: fixed test command or new_vision UDP commands",
    )
    parser.add_argument("--udp-command-bind", default="127.0.0.1")
    parser.add_argument("--udp-command-port", type=int, default=5005)
    parser.add_argument("--command-timeout", type=float, default=0.25,
                        help="Zero velocity after this many seconds without valid UDP data")
    parser.add_argument("--walk-seconds", type=float, default=5.0,
                        help="Fixed mode only: command zero after this duration; 0 disables timer")
    parser.add_argument(
        "--vx", type=float, default=config.DEFAULT_FORWARD_VELOCITY,
        help="Fixed mode only: forward command in m/s (positive, at most 1.0)",
    )
    parser.add_argument(
        "--wz", type=float, default=0.0,
        help="Fixed mode only: yaw-rate command in rad/s, within [-0.5, 0.5]",
    )
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--log-every", type=int, default=25, help="Print every N policy steps")
    parser.add_argument(
        "--position-log-dir",
        default="logs/motor_positions",
        help="Directory for per-run target/actual motor-position CSV logs",
    )
    parser.add_argument(
        "--plot-history-seconds",
        type=float,
        default=10.0,
        help="Seconds of motor-position history visible in the live plot",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=5,
        help="Refresh the motor-position plot every N policy steps",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable only the live window for headless runs; CSV logging remains enabled",
    )
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
    config.validate_imu_configuration()
    if args.enable_motors and (
        not config.CALIBRATION_CONFIRMED or not config.IMU_CALIBRATION_CONFIRMED
    ):
        raise SystemExit(
            "Refusing to enable motors: confirm the motor and IMU mounting "
            "calibrations in config.py first."
        )
    if not 0.0 <= args.kp_scale <= 1.0 or not 0.0 <= args.kd_scale <= 1.0:
        raise SystemExit("kp-scale and kd-scale must be between 0 and 1")
    if args.plot_every < 1:
        raise SystemExit("plot-every must be at least 1")
    if args.plot_history_seconds <= 0.0:
        raise SystemExit("plot-history-seconds must be positive")
    if args.policy == "walking":
        if args.command_source == "vision":
            if not 1 <= args.udp_command_port <= 65535:
                raise SystemExit("udp-command-port must be between 1 and 65535")
            if not np.isfinite(args.command_timeout) or args.command_timeout <= 0:
                raise SystemExit("command-timeout must be finite and positive")
        else:
            if not np.isfinite(args.vx) or not 0.0 < args.vx <= 1.0:
                raise SystemExit("vx must be finite and in (0, 1] for constant forward walking")
            if not np.isfinite(args.wz) or not -0.5 <= args.wz <= 0.5:
                raise SystemExit("wz must be finite and in [-0.5, 0.5]")
            if not np.isfinite(args.walk_seconds) or args.walk_seconds < 0:
                raise SystemExit("walk-seconds must be finite and nonnegative")

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.policy == "one-foot":
        try:
            command_source = OneFootCommand(args.stand_seconds, args.lift_seconds)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        policy = OneFootPolicy(args.model, support_foot=args.support_foot)
        command_source_description = (
            f"one-foot standing, support={args.support_foot}; "
            f"stand {args.stand_seconds:g}s -> lift {args.lift_seconds:g}s -> lower/stand; "
            "vision disconnected; no velocity, step-distance, or crossing observations"
        )
    else:
        policy = HumanoidPolicy(args.model)
        if args.command_source == "vision":
            command_source = UdpCommandSource(
                args.udp_command_port, timeout_s=args.command_timeout,
                bind=args.udp_command_bind,
            )
            command_source_description = (
                f"new_vision UDP on udp://{args.udp_command_bind}:"
                f"{args.udp_command_port}; timeout={args.command_timeout:.3f}s; "
                "live vx/wz, vy=0; fixed-command timer disabled"
            )
        else:
            command_source = FixedCommandSource(args.vx, args.wz)
            command_source_description = (
                f"fixed walking vx={args.vx:+.3f} m/s, vy=0, wz={args.wz:+.3f}; "
                f"walk_seconds={args.walk_seconds:g} (0=continuous); vision disconnected"
            )
    position_logger = PositionCsvLogger(args.position_log_dir, config.JOINT_NAMES)
    position_plot = None
    if not args.no_plot:
        try:
            position_plot = LivePositionPlot(config.JOINT_NAMES, args.plot_history_seconds)
        except Exception as exc:
            position_logger.close()
            raise SystemExit(
                f"Could not open the motor-position window: {exc}. "
                "Run with --no-plot on a headless system; CSV logging will remain enabled."
            ) from exc

    print(f"ONNX input={policy.input_name!r}, output={policy.output_name!r}")
    print(f"Policy command source: {command_source_description}")
    print(f"Opening {args.port} (line coding {args.baud}; native USB CDC ignores physical baud)")
    print("MOTORS ENABLED" if args.enable_motors else "DRY RUN: command enable flag is OFF")
    print(f"Motor-position and IMU log: {position_logger.path}")
    if position_plot is not None:
        print("Motor/IMU window opened (knee motors and IMU data selected by default)")

    link = SerialLink(args.port, args.baud)
    last_q_motor = np.zeros(config.NUM_JOINTS, dtype=np.float32)
    timed_run_completed = False
    try:
        first_state = link.wait_for_state(timeout_s=5.0)
        required = STATE_IMU_VALID | STATE_ENCODERS_VALID
        if (first_state.status_flags & required) != required:
            raise RuntimeError(
                f"Initial IMU/encoder data invalid: flags=0x{first_state.status_flags:08X}"
            )
        initial_accel_policy = config.IMU_TO_POLICY @ first_state.accel_m_s2
        initial_gyro_policy = config.IMU_TO_POLICY @ first_state.gyro_rad_s
        initial_projected_gravity = projected_gravity_from_quaternion(
            first_state.orientation_wxyz,
            config.IMU_TO_POLICY,
            sensor_to_world=config.IMU_QUATERNION_IS_SENSOR_TO_WORLD,
        )
        validate_stationary_imu_sample(
            initial_accel_policy,
            initial_gyro_policy,
            initial_projected_gravity,
        )
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
                timed_run_completed = True
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
            projected_gravity = projected_gravity_from_quaternion(
                state.orientation_wxyz,
                config.IMU_TO_POLICY,
                sensor_to_world=config.IMU_QUATERNION_IS_SENSOR_TO_WORLD,
            )
            orientation_rpy = roll_pitch_yaw_from_quaternion(
                state.orientation_wxyz,
                config.IMU_TO_POLICY,
                sensor_to_world=config.IMU_QUATERNION_IS_SENSOR_TO_WORLD,
            )
            if args.policy == "one-foot":
                lift_command = command_source.get(now - start_time)
                command_values = {"lift_command": lift_command}
                command_status = f"lift_command={int(lift_command)} support={args.support_foot} "
            else:
                velocity_command = command_source.get()
                # A fixed-test timer must never override live vision commands.
                if (args.command_source == "fixed" and args.walk_seconds > 0
                        and now - start_time >= args.walk_seconds):
                    velocity_command[:] = 0.0  # vx=0, vy=0, wz=0

                command_values = {"velocity_command": velocity_command}
                command_status = (
                    f"policy_target_velocity=[vx={velocity_command[0]:+.3f} m/s, "
                    f"vy={velocity_command[1]:+.3f} m/s, "
                    f"wz={velocity_command[2]:+.3f} rad/s] "
                )
                if args.command_source == "vision":
                    udp_status = command_source.status()
                    age_text = (
                        f"{udp_status['age_s'] * 1000.0:.1f}ms"
                        if np.isfinite(udp_status["age_s"]) else "never"
                    )
                    command_status += (
                        f"udp_fresh={udp_status['fresh']} udp_age={age_text} "
                        f"udp_valid={udp_status['valid_packets']} "
                        f"udp_invalid={udp_status['invalid_packets']} "
                    )

            q_policy_target, action, obs, latency_ms = policy.step(
                accel_m_s2=accel_policy,
                gyro_rad_s=gyro_policy,
                projected_gravity=projected_gravity,
                **command_values,
                joint_position_policy=q_policy,
                joint_velocity_policy=qd_policy,
            )
            q_policy_target = config.clamp_policy_target(q_policy_target)
            q_policy_target = slew_limit(q_policy_target, last_q_policy_target, dt)
            q_policy_target = config.clamp_policy_target_to_current(q_policy_target, q_policy)
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

            elapsed_s = now - start_time
            position_logger.write(
                elapsed_s,
                step,
                state.sequence,
                last_q_motor,
                state.joint_position,
                accel_policy,
                orientation_rpy,
            )
            if position_plot is not None and step % args.plot_every == 0:
                position_plot.update(
                    elapsed_s,
                    last_q_motor,
                    state.joint_position,
                    accel_policy,
                    orientation_rpy,
                )

            if step % max(1, args.log_every) == 0:
                print(
                    f"step={step:6d} state_seq={state.sequence:5d} "
                    f"{command_status}"
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
        position_logger.close()
        if position_plot is not None and not timed_run_completed:
            position_plot.close()
        if args.policy == "walking":
            command_source.close()
        link.close()

    print("Policy stopped; disable packets sent")
    if position_plot is not None and timed_run_completed:
        if position_plot.is_open():
            print("Timed run completed; close the motor-position window to exit")
            while position_plot.is_open() and not stop_requested:
                time.sleep(0.1)
        position_plot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
