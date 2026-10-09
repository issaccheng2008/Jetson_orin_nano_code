#!/usr/bin/env python3
"""Run a walking or one-foot standing ONNX policy and exchange data with STM32."""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

import config
from attitude_broadcast import AttitudeBroadcaster
from command_source import (MAX_WZ, FixedCommandSource, ScriptedCommandSource,
                            UdpCommandSource, curve_legs, parse_legs)
from fixed_joint_policy import FixedJointPolicy
from imu_filter import (
    projected_gravity_from_quaternion,
    roll_pitch_yaw_from_quaternion,
    validate_stationary_imu_sample,
)
from policy_runner import HumanoidPolicy
from one_foot_policy import OneFootCommand, OneFootPolicy
from shape_actions import ShapeActionController, UPPER_CARDS
from position_monitor import LivePositionPlot, PositionCsvLogger
from protocol import (
    ACTION_BUSY,
    ACTION_CARD_RESTORE,
    ACTION_CARD_TILT,
    ACTION_DONE,
    COMMAND_ENABLE,
    COMMAND_ESTOP,
    STATE_ENCODERS_VALID,
    STATE_COMMAND_FRESH,
    STATE_FAULT,
    STATE_IMU_VALID,
    STATE_MOTORS_ENABLED,
)
from serial_link import SerialLink
from target_safety import TargetSafety, add_target_safety_arguments, limit_target_slew
from control_diagnostics import ControlDiagnostics, add_diagnostic_arguments, receive_metadata
from walking_command_hold import WalkingCommandHold


def write_startup_ready_if_confirmed(path: str | None, state_flags: int,
                                     command_written: bool, step: int) -> bool:
    """Signal vision only after STM32 reports a fresh enabled command."""
    required = STATE_COMMAND_FRESH | STATE_MOTORS_ENABLED
    if (not path or step < 1 or not command_written
            or (state_flags & required) != required):
        return False
    marker = Path(path)
    temporary = marker.with_name(marker.name + ".tmp")
    temporary.write_text("ready\n", encoding="ascii")
    os.replace(temporary, marker)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Path to policy.onnx")
    source.add_argument("--fixed-policy", help="Path to joint_frames_v1 JSON")
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
    parser.add_argument("--one-foot-model", help="One-foot ONNX model for vision cards 3/4")
    parser.add_argument("--shape-lift-seconds", type=float, default=3.0,
                        help="Vision card one-foot model duration (default 3 seconds)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="STM32 serial device")
    parser.add_argument("--startup-ready-file", default=None,
                        help="Optional one-shot ready marker for the vision start gate")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--command-source",
        choices=("fixed", "scripted", "curve", "vision"),
        default="fixed",
        help="Walking: fixed test command, a --scripted-legs timeline, the "
             "--curve-* open-loop bulge, or new_vision/connector UDP commands",
    )
    parser.add_argument(
        "--curve-straight-s", type=float, default=3.16,
        help="curve mode: straight leg before and after the turn, seconds "
             "(default 3.16 = 0.632 m of track at 0.2 m/s)",
    )
    parser.add_argument(
        "--curve-turn-s", type=float, default=12.19,
        help="curve mode: how long to hold --curve-turn-wz, seconds "
             "(default 12.19 = 180 deg at 0.258 rad/s). Scale this if the robot "
             "does not achieve the commanded yaw rate",
    )
    parser.add_argument(
        "--curve-turn-wz", type=float, default=0.258,
        help="curve mode: yaw rate held through the turn, rad/s, positive = left "
             "(default 0.258 = v/R, the rate that follows a 0.776 m arc)",
    )
    parser.add_argument(
        "--curve-vx", type=float, default=0.2,
        help="curve mode: forward speed for the straight legs, m/s (the turn "
             "uses --curve-turn-vx, unset = this)",
    )
    parser.add_argument(
        "--curve-turn-vx", type=float, default=None,
        help="curve mode: forward speed held through the turn, m/s; unset keeps "
             "--curve-vx, so the whole bulge walks at one speed",
    )
    parser.add_argument(
        "--curve-once", action="store_true",
        help="curve mode: walk ONE bulge (straight, turn, straight) and then stop. "
             "Default is to alternate straight/turn forever, which needs "
             "--max-seconds -- the stop condition is yours to give",
    )
    parser.add_argument(
        "--scripted-legs",
        help="scripted mode only: 'seconds:vx:wz' legs separated by ; or , -- "
             "e.g. '3.2:0.2:0; 12.2:0.2:0.258; 3.2:0.2:0'. Open loop: no vision, "
             "no feedback. The clock starts on the first walking tick, not at "
             "process start",
    )
    parser.add_argument("--udp-command-bind", default="127.0.0.1")
    parser.add_argument("--udp-command-port", type=int, default=5005)
    parser.add_argument("--attitude-bind", default="127.0.0.1",
                        help="Where run_policy_vision listens for the body attitude")
    parser.add_argument("--attitude-port", type=int, default=5007,
                        help="Broadcast the policy-frame world-down vector here at 10 Hz "
                             "(0 disables it); vision needs it because its camera pitch "
                             "is otherwise a static config value while the body moves")
    parser.add_argument("--command-timeout", type=float, default=0.25,
                        help="Zero velocity after this many seconds without connector data")
    parser.add_argument("--command-min-hold-s", type=float, default=0.0,
                        help="Minimum duration of actual walking (vx,vy,wz) at model input; "
                             "0 disables, 0.5 restores the original duration; stops/takeovers preempt")
    parser.add_argument(
        "--walk-seconds", type=float, default=5.0,
        help="Fixed mode only: how long to hold (--vx, --wz). With "
             "--pause-seconds it is the walking leg of a walk/pause loop; without "
             "it the command goes zero after this long, once (0 disables the timer)",
    )
    parser.add_argument(
        "--pause-seconds", type=float, default=0.0,
        help="Fixed mode only: with walk-seconds > 0, alternate forever -- "
             "walk-seconds at (--vx, --wz), then this long at zero. 0 keeps the "
             "one-shot walk-then-stop",
    )
    parser.add_argument(
        "--vx", type=float, default=config.MAX_COMMAND_VX,
        help="Fixed mode only: forward command in m/s (positive, at most 1.0)",
    )
    parser.add_argument(
        "--wz", type=float, default=0.0,
        help="Fixed mode only: yaw-rate command in rad/s, within [-0.5, 0.5]",
    )
    parser.add_argument(
        "--step-cm", type=float, default=config.STEP_LENGTH_CM,
        help="Step length in cm, the ONE step number. Used as is whatever vx is, "
             "so it does not move when the speed moves. NOTE: the cadence is the "
             "policy's own and is NOT vx/step -- it moves with vx and step_distance "
             "(measured 2026-10-03: lowering vx made the stepping visibly faster). "
             "So this sets the stride the policy is asked for, not the resulting "
             "speed",
    )
    parser.add_argument(
        "--forward-ankle-bias-rad", type=float, default=0.0,
        help="Ankle pitch bias while the walking policy receives vx > 0: "
             "protocol joint 4 += this value, joint 10 -= this value. "
             "Zero speed, one-foot actions, and upright hold use no bias; 0 disables it",
    )
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--log-every", type=int, default=25, help="Print every N policy steps")
    parser.add_argument(
        "--position-log-dir",
        default="records/motor_positions",
        help="Directory for per-run target/actual motor-position CSV logs. "
             "Everything a board run produces goes under records/ (see the "
             "operating manual section 0) -- do not scatter it in the repo or "
             "home root. Created if missing",
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
    add_target_safety_arguments(parser)
    add_diagnostic_arguments(parser)
    args = parser.parse_args()
    try:
        TargetSafety.from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.forward_ankle_bias_rad) or args.forward_ankle_bias_rad < 0.0:
        parser.error("--forward-ankle-bias-rad must be finite and nonnegative")
    if not np.isfinite(args.command_min_hold_s) or args.command_min_hold_s < 0:
        parser.error("--command-min-hold-s must be finite and nonnegative")
    return args


def monotonic_us() -> int:
    return (time.monotonic_ns() // 1000) & 0xFFFFFFFF


# How long the policy keeps being fed the pre-untilt observation after the untilt
# (action 8) goes out.
#
# The STM32 clears its feedback freeze in the SAME instant it starts ramping the lean
# back, so from that tick on it reports a body that is still tilted and moving - a
# pose the model never commanded. On the 2026-10-02 run that is exactly where the
# 87-degree hip split came from. So hold the snapshot on this side until the ramp
# has had time to finish.
#
# The ramp is a firmware constant: LEAN_ANGLE_RAD 0.3491 / LEAN_RAMP_RATE 0.6 rad/s
# = 0.582 s. Keep CARD_UNTILT_HOLD_S in step with it; a bit of margin costs nothing
# because the model is standing still either way.
CARD_UNTILT_HOLD_S = 0.65


def slew_limit(target: np.ndarray, previous: np.ndarray, dt: float) -> np.ndarray:
    return limit_target_slew(target, previous, dt, config.MAX_TARGET_SPEED_RAD_S)


def apply_forward_ankle_bias(target: np.ndarray, velocity_command: np.ndarray | None,
                             walking_policy_active: bool, bias_rad: float) -> np.ndarray:
    """Move the former STM32 motor-side ankle trim into the Nano target path.

    STM32 reverses the signs of protocol joints 4 and 10 before motor output.
    Thus +bias at 4 and -bias at 10 reproduce motor targets -bias and +bias.
    This runs before the existing absolute, slew, and feedback-window limits.
    """
    biased = np.asarray(target, dtype=np.float32).copy()
    if (walking_policy_active and velocity_command is not None
            and float(velocity_command[0]) > 0.0 and bias_rad > 0.0):
        biased[4] += bias_rad
        biased[10] -= bias_rad
    return biased


def reconnect_link(link: SerialLink, port: str, baud: int,
                   timeout_s: float = 10.0) -> SerialLink:
    """STM32 复位/掉线之后重连：等 USB 枚举回来、重新打开、等到第一帧状态。

    固件有几条路径会**主动复位**（100ms 没收到新命令、ESTOP、目标非法），复位时
    USB 要消失 0.2~0.5s 再枚举回来。主机原来直接 FAULT 退出，一次复位就毁掉整趟；
    这里给它一个窗口，超时才把异常抛回去走原来的 FAULT。

    回来的名字**不一定是原来的**：本进程还攥着已经死掉的 ttyACM0 时，新的设备
    会枚举成 ttyACM1（2026-10-04 实车两次，等固定路径就 10s 超时）。按 ttyACM*
    全扫，第一个能回状态包的端口就接。进程退出后再上电通常又回 ttyACM0。

    旧句柄必须先放干净（`reader_alive()` 就是干这个的）：reader 线程还卡在
    `read()` 上时重开，新端口会复用同一个 fd，旧线程会把新连接的字节偷走。
    """
    try:
        link.close()
    except Exception:
        pass
    pattern = os.path.join(os.path.dirname(port) or "/dev", "ttyACM*")
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        for candidate in sorted(glob.glob(pattern)):
            fresh = None
            try:
                fresh = SerialLink(candidate, baud)
                fresh.wait_for_state(timeout_s=1.0)
                return fresh
            except (TimeoutError, OSError) as exc:
                last_error = exc
                if fresh is not None:
                    try:
                        fresh.close()
                    except Exception:
                        pass
        time.sleep(0.05)
    raise TimeoutError(
        f"STM32 did not come back on {pattern} within {timeout_s:.0f}s ({last_error})")


def fixed_walk_active(elapsed_s: float, walk_s: float, pause_s: float) -> bool:
    """Fixed-command duty cycle: walking leg vs pause leg at this elapsed time.

    pause_s = 0 keeps the original one-shot behaviour (walk once, then zero
    forever); pause_s > 0 alternates walk/pause for as long as the process runs.
    walk_s = 0 disables the timer and the command holds forever.
    """
    if walk_s <= 0.0:
        return True
    if pause_s <= 0.0:
        return elapsed_s < walk_s
    return (elapsed_s % (walk_s + pause_s)) < walk_s


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
    try:
        target_safety = TargetSafety.from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.one_foot_model and (args.fixed_policy or args.policy != "walking"
                                or args.command_source != "vision"):
        raise SystemExit("--one-foot-model requires walking policy with --command-source vision")
    if args.one_foot_model and (not np.isfinite(args.shape_lift_seconds)
                                or args.shape_lift_seconds < 3.0):
        raise SystemExit("--shape-lift-seconds must be at least 3 seconds")
    if args.startup_ready_file and (args.policy != "walking"
                                    or args.command_source != "vision"
                                    or not args.enable_motors):
        raise SystemExit("--startup-ready-file requires enabled walking vision mode")
    config.validate_imu_configuration()
    if args.enable_motors and (
        not config.CALIBRATION_CONFIRMED or not config.IMU_CALIBRATION_CONFIRMED
    ):
        raise SystemExit(
            "Refusing to enable motors: confirm the motor and IMU mounting "
            "calibrations in config.py first."
        )
    if (not np.isfinite([args.kp_scale, args.kd_scale]).all()
            or args.kp_scale < 0.0 or args.kd_scale < 0.0):
        raise SystemExit("kp-scale and kd-scale must be finite and non-negative")
    if args.plot_every < 1:
        raise SystemExit("plot-every must be at least 1")
    if args.plot_history_seconds <= 0.0:
        raise SystemExit("plot-history-seconds must be positive")
    if not np.isfinite(args.step_cm) or args.step_cm <= 0.0:
        raise SystemExit("step-cm must be finite and positive")
    step_distance_m = args.step_cm / 100.0
    if not args.fixed_policy and args.policy == "walking":
        if args.command_source == "vision":
            if not 1 <= args.udp_command_port <= 65535:
                raise SystemExit("udp-command-port must be between 1 and 65535")
            if not np.isfinite(args.command_timeout) or args.command_timeout <= 0:
                raise SystemExit("command-timeout must be finite and positive")
        elif args.command_source == "scripted":
            if not args.scripted_legs:
                raise SystemExit("scripted mode needs --scripted-legs")
            try:
                legs = parse_legs(args.scripted_legs)
                ScriptedCommandSource(legs)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            for duration_s, vx, wz in legs:
                if not 0.0 <= vx <= 1.0:
                    raise SystemExit(f"leg vx must be in [0, 1], got {vx}")
                if not -MAX_WZ <= wz <= MAX_WZ:
                    raise SystemExit(
                        f"leg wz must be in [{-MAX_WZ}, {MAX_WZ}], got {wz}")
        elif args.command_source == "curve":
            try:
                legs = curve_legs(args.curve_straight_s, args.curve_turn_s,
                                  args.curve_vx, args.curve_turn_wz,
                                  turn_vx=args.curve_turn_vx,
                                  loop=not args.curve_once)
                ScriptedCommandSource(legs, loop=not args.curve_once)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if not 0.0 <= args.curve_vx <= 1.0:
                raise SystemExit("curve-vx must be in [0, 1]")
            if args.curve_turn_vx is not None and not 0.0 <= args.curve_turn_vx <= 1.0:
                raise SystemExit("curve-turn-vx must be in [0, 1]")
            if not -MAX_WZ <= args.curve_turn_wz <= MAX_WZ:
                raise SystemExit(f"curve-turn-wz must be in [{-MAX_WZ}, {MAX_WZ}]")
        else:
            # vx=0 is allowed and means "let the policy stand": the robot holds its
            # own stopped pose instead of the all-zero joint frame --fixed-policy
            # plays, which is the only way to watch that pose without vision in the
            # loop. It is also the polite way to check a policy before driving it.
            if not np.isfinite(args.vx) or not 0.0 <= args.vx <= 1.0:
                raise SystemExit("vx must be finite and in [0, 1]")
            if not np.isfinite(args.wz) or not -MAX_WZ <= args.wz <= MAX_WZ:
                raise SystemExit(f"wz must be finite and in [{-MAX_WZ}, {MAX_WZ}]")
            if not np.isfinite(args.walk_seconds) or args.walk_seconds < 0:
                raise SystemExit("walk-seconds must be finite and nonnegative")
            if not np.isfinite(args.pause_seconds) or args.pause_seconds < 0:
                raise SystemExit("pause-seconds must be finite and nonnegative")
            if args.pause_seconds > 0 and args.walk_seconds <= 0:
                raise SystemExit("pause-seconds needs walk-seconds > 0 to alternate")

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    shape_controller = None
    card_policy = None

    if args.fixed_policy:
        if args.policy != "walking" or args.command_source != "fixed":
            raise SystemExit("--fixed-policy cannot be combined with one-foot or vision commands")
        try:
            policy = FixedJointPolicy(args.fixed_policy, target_safety=target_safety)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        command_source = None
        command_source_description = f"fixed joint frames from {args.fixed_policy} at 50 Hz"
    elif args.policy == "one-foot":
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
        policy = HumanoidPolicy(args.model, step_distance_m)
        if args.command_source == "vision":
            if args.one_foot_model:
                card_policy = OneFootPolicy(args.one_foot_model)
                shape_controller = ShapeActionController(lift_seconds=args.shape_lift_seconds)
            command_source = UdpCommandSource(
                args.udp_command_port, timeout_s=args.command_timeout,
                bind=args.udp_command_bind,
            )
            command_source_description = (
                f"new_vision/connector on udp://{args.udp_command_bind}:"
                f"{args.udp_command_port}; timeout={args.command_timeout:.3f}s; "
                "live vx/wz, vy=0; fixed-command timer disabled"
            )
        elif args.command_source == "scripted":
            command_source = ScriptedCommandSource(parse_legs(args.scripted_legs))
            command_source_description = (
                f"SCRIPTED open loop, {len(parse_legs(args.scripted_legs))} legs, "
                f"{command_source.total_s:g}s total: {args.scripted_legs}; "
                "vision disconnected, no feedback"
            )
        elif args.command_source == "curve":
            loop = not args.curve_once
            command_source = ScriptedCommandSource(curve_legs(
                args.curve_straight_s, args.curve_turn_s,
                args.curve_vx, args.curve_turn_wz,
                turn_vx=args.curve_turn_vx, loop=loop), loop=loop)
            turn_deg = math.degrees(args.curve_turn_wz * args.curve_turn_s)
            turn_vx = (args.curve_vx if args.curve_turn_vx is None
                       else args.curve_turn_vx)
            shape = ("straight, turn, straight - one bulge" if not loop
                     else f"straight/turn alternating every "
                          f"{command_source.total_s:g}s, until the process stops")
            command_source_description = (
                f"CURVE open loop, {'ONCE' if not loop else 'LOOPING'}"
                f" ({shape}): straight {args.curve_straight_s:g}s / turn "
                f"{args.curve_turn_s:g}s at wz={args.curve_turn_wz:+.3f} rad/s, "
                f"straight vx={args.curve_vx:g} / turn vx={turn_vx:g} m/s "
                f"({turn_deg:+.0f} deg per turn if "
                "the yaw rate is achieved); vision disconnected, no feedback"
            )
        else:
            command_source = FixedCommandSource(args.vx, args.wz)
            if args.walk_seconds > 0 and args.pause_seconds > 0:
                duty = (f"walk {args.walk_seconds:g}s / pause "
                        f"{args.pause_seconds:g}s alternating")
            else:
                duty = f"walk_seconds={args.walk_seconds:g} (0=continuous)"
            command_source_description = (
                f"fixed walking vx={args.vx:+.3f} m/s, vy=0, wz={args.wz:+.3f}; "
                f"{duty}; vision disconnected"
            )
    # 只有视觉在指挥时才广播：fixed 模式是"视觉断开"的意思，不该凭空开一个
    # UDP 口，那边也没人在听。
    attitude = (
        AttitudeBroadcaster(args.attitude_bind, args.attitude_port)
        if args.attitude_port > 0
        and args.policy == "walking" and args.command_source == "vision"
        else None
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

    if not args.fixed_policy:
        print(f"ONNX input={policy.input_name!r}, output={policy.output_name!r}")
        if card_policy is not None:
            print(f"Shape one-foot ONNX input={card_policy.input_name!r}, "
                  f"output={card_policy.output_name!r}")
    print(f"Policy command source: {command_source_description}")
    # 步长就是一个数，不随 vx 变。步频不在这里算 —— 它是策略自己的，会随 vx
    # 和步长变（实测：vx 调小反而步频变快），不是 vx/步长。
    print(f"Step: {step_distance_m * 100:.2f} cm (--step-cm), independent of vx; "
          f"the cadence is the policy's own")
    print(f"Forward ankle pitch bias: {args.forward_ankle_bias_rad:g} rad "
          "(--forward-ankle-bias-rad; applied only to walking vx > 0)")
    print(target_safety.describe())
    print(f"Body attitude broadcast: "
          + (f"udp://{args.attitude_bind}:{args.attitude_port} at 10 Hz"
             if attitude is not None else "off"))
    print(f"Opening {args.port} (line coding {args.baud}; native USB CDC ignores physical baud)")
    print("MOTORS ENABLED" if args.enable_motors else "DRY RUN: command enable flag is OFF")
    if not args.fixed_policy and args.policy == "walking":
        print(f"Walking command minimum hold: {args.command_min_hold_s:g} s at model input; "
              "stops and takeovers preempt")
    print(f"Motor-position and IMU log: {position_logger.path}")
    if position_plot is not None:
        print("Motor/IMU window opened (knee motors and IMU data selected by default)")

    link = SerialLink(args.port, args.baud)
    last_q_motor = np.zeros(config.NUM_JOINTS, dtype=np.float32)
    timed_run_completed = False
    diagnostics = None
    diagnostic_error = None
    walking_command_hold = WalkingCommandHold(min_hold_s=args.command_min_hold_s)
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
        if getattr(first_state, "command_rx_count", None) is None:
            print("WARNING: STM32 diagnostic counters unavailable in STATE frames; "
                  "flash firmware with the extended 152-byte payload before a counter run",
                  flush=True)
        else:
            print(f"STM32 diagnostic counters: received_commands={first_state.command_rx_count} "
                  f"control_cycles={first_state.system_control_cycle}", flush=True)
        if not getattr(args, "no_control_diagnostics", False):
            sources = ({"fixed_joint_frames": args.fixed_policy} if args.fixed_policy else
                       {"onefoot46" if args.policy == "one-foot" else "walking49": args.model})
            if args.one_foot_model:
                sources["card_onefoot46"] = args.one_foot_model
            directory = (getattr(args, "diagnostic_log_dir", None)
                         or Path(args.position_log_dir) / "control_diagnostics")
            diagnostics = ControlDiagnostics(directory, args, target_safety, sources,
                                             first_state, "main.py", position_logger.path)
        else:
            print("Control diagnostics explicitly disabled")

        next_tick = time.monotonic()
        previous_tick = next_tick
        start_time = next_tick
        step = 0
        card_policy_active = False
        upright_active = False
        card_tilt_active = False
        card_tilt_event = 0
        tilt_release_at = 0.0
        held_observation = None
        last_action_tx = -float("inf")
        startup_ready_sent = False

        def link_loss_recovery(exc: Exception, where: str) -> None:
            """掉线后的统一恢复：重连，并把节拍和命令保持重新起表。"""
            nonlocal link, previous_tick, next_tick
            print(f"[link] STM32 掉线（{where}: {exc}）—— 等 USB 回来重连", flush=True)
            link = reconnect_link(link, args.port, args.baud)
            print("[link] 重连成功，继续", flush=True)
            previous_tick = time.monotonic()
            next_tick = previous_tick
            walking_command_hold.clear()

        while not stop_requested:
            now = time.monotonic()
            if args.fixed_policy and policy.index >= len(policy.frames):
                timed_run_completed = True
                break
            if args.max_seconds > 0.0 and now - start_time >= args.max_seconds:
                timed_run_completed = True
                break

            try:
                state = link.get_latest_state(max_age_s=0.05)
            except (TimeoutError, OSError) as exc:
                link_loss_recovery(exc, "state")
                continue
            diagnostic_read_time = time.monotonic_ns() * 1e-9
            state_receive_info = receive_metadata(link, state, diagnostic_read_time) if diagnostics is not None else None
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
            step_policy = policy
            upright_hold = False
            held_reference = False
            diagnostic_velocity = None
            diagnostic_lift = None
            diagnostic_mode = "walking49"
            diagnostic_source = "onnx"
            requested_velocity = None
            command_hold_remaining = 0.0
            command_hold_reason = "not_walking"
            if args.fixed_policy:
                diagnostic_mode, diagnostic_source = "fixed", "fixed_joint_frames"
                q_policy_target = policy.next_target()
                if q_policy_target is None:
                    timed_run_completed = True
                    break
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
                obs = np.zeros(1, dtype=np.float32)
                latency_ms = 0.0
                command_status = f"fixed_frame={policy.index}/{len(policy.frames)} "
            elif args.policy == "one-foot":
                diagnostic_mode = "onefoot46"
                lift_command = command_source.get(now - start_time)
                diagnostic_lift = lift_command
                command_values = {"lift_command": lift_command}
                command_status = f"lift_command={int(lift_command)} support={args.support_foot} "
            else:
                # Snapshot whenever the source has one, not only when a card policy is
                # loaded: the upright hold reads it too and must work without
                # --one-foot-model.
                snapshot = (command_source.get_snapshot()
                            if args.command_source == "vision" else None)
                velocity_command = snapshot.velocity if snapshot is not None else command_source.get()
                requested_velocity = velocity_command.copy()
                decision = None
                # Both the card action and the upright hold need it, and it is the same
                # question either way: has the robot actually settled?
                stopped = (np.max(np.abs(velocity_command)) <= 0.02
                           and np.max(np.abs(qd_policy)) <= 0.2)
                # Two edges, and they do NOT both fire on the vision's flag.
                #
                # RISING (send 7) waits for `stopped`. The STM32 latches its frozen
                # snapshot AND starts the lean ramp the instant 7 lands, so 7 has to
                # land while the robot is already standing. Sent on the stop-trigger
                # frame the snapshot is a mid-stride pose - on the 2026-10-02 run, body
                # level at -2.85 deg with one knee straight and the other bent 27 deg.
                # The lean is there to undo a ~20 deg back-tilt, so on a level body it
                # just pushed the robot 20 deg past level, and the model - which never
                # saw any of it - was handed that pose the moment the freeze lifted.
                #
                # FALLING (send 8) does not wait: the vision dropped the flag, the body
                # should start going back now.
                #
                # Both edges go out BEFORE the shape action. The STM32 runs one action
                # at a time and anything arriving while it is busy comes back
                # ACTION_BUSY and is dropped - silently, because nothing polls this
                # one's status. Sent second, the untilt is the one that gets dropped and
                # the body stays pitched through the whole arm/head action.
                #
                # The STM32 re-poses the body and freezes the attitude it reports, so
                # the vision's geometry never sees the stop at all. Each edge carries a
                # fresh event id - the STM32 reads a repeated event id as a
                # retransmission, not as a new command.
                card_tilt = snapshot is not None and snapshot.card_tilt
                if not card_tilt_active and card_tilt and stopped:
                    card_tilt_active = True
                    card_tilt_event += 1
                    link.send_action(card_tilt_event, ACTION_CARD_TILT)
                    # 停车这一秒多里唯一发出去的两条命令，和视觉那边的
                    # "停车读卡 / 识别到图卡"两条 banner 对成一对。
                    print("\n" + "=" * 68)
                    print(f"  ●●●  姿态事件 action={ACTION_CARD_TILT} "
                          f"event={card_tilt_event}"
                          f"    重摆：机身扳回安装姿态（等到了站定才发）")
                    print("=" * 68 + "\n")
                elif card_tilt_active and not card_tilt:
                    card_tilt_active = False
                    card_tilt_event += 1
                    link.send_action(card_tilt_event, ACTION_CARD_RESTORE)
                    # Snapshot the frozen observation and hold it for the ramp back -
                    # see CARD_UNTILT_HOLD_S. Taken here, before 8 is processed, so
                    # these are still the STM32's frozen values.
                    tilt_release_at = now + CARD_UNTILT_HOLD_S
                    held_observation = (q_policy, qd_policy, accel_policy,
                                        gyro_policy, projected_gravity)
                    print("\n" + "=" * 68)
                    print(f"  ●●●  姿态事件 action={ACTION_CARD_RESTORE} "
                          f"event={card_tilt_event}"
                          f"    恢复：机身回站姿；策略继续看旧状态 "
                          f"{CARD_UNTILT_HOLD_S * 1000:.0f}ms")
                    print("=" * 68 + "\n")
                if shape_controller is not None:
                    if snapshot.event_id:
                        if shape_controller.accept(snapshot.event_id, snapshot.event_action, now):
                            print(f"[shape] event={snapshot.event_id} card={snapshot.event_action} accepted")
                    status = (link.get_action_status(shape_controller.event_id)
                              if shape_controller.action_id in UPPER_CARDS else 0)
                    if not args.enable_motors and status == ACTION_BUSY:
                        print(f"[shape] dry run: STM32 did not execute card={shape_controller.action_id}")
                        status = ACTION_DONE
                    was_busy = shape_controller.phase != "idle"
                    # The card re-pose (8) still owns the body for its return ramp.
                    # Leg policy targets must wait for that ramp; the first card at
                    # the start has no re-pose and can enter the model immediately.
                    decision = shape_controller.advance(
                        now, stopped, status, ready_to_lift=now >= tilt_release_at)
                    if was_busy and not decision.busy:
                        print(f"[shape] event={shape_controller.event_id} card={shape_controller.action_id} complete")
                    if decision.busy:
                        velocity_command[:] = 0.0
                    if decision.send_upper and now - last_action_tx >= 0.1:
                        link.send_action(decision.event_id, decision.action_id)
                        last_action_tx = now
                    if decision.policy == "one-foot":
                        if not card_policy_active:
                            card_policy.select_support_foot(decision.support_foot)
                            card_policy_active = True
                        step_policy = card_policy
                    elif card_policy_active:
                        policy.reset()
                        card_policy_active = False
                # Vision is reading a card and wants the legs held straight. The policy's
                # own stopped pose is pitched back about 20 degrees, and the card
                # geometry is calibrated at the mount angle, so that pose is what pushes
                # the box outside the square gate.
                #
                # `stopped`, not just the flag: the pose must change once the robot has
                # settled, not while the gait is still ending. Switching mid-stride is
                # the untested case, and it is also the one where the policy's residual
                # motion and the new static target would be fighting. Waiting costs the
                # first fraction of a second of a three-to-five second window.
                #
                # A running leg action wins: it is a different pose, commanded by the
                # same joints, and only one of the two may drive them.
                #
                # `stopped` gates the way IN only, not the way through. The legs move
                # while they straighten, so re-testing it every frame would drop the
                # hold the moment it started working, hand the legs back to the policy,
                # let them settle, and engage again - the two poses alternating. Once
                # engaged, only the vision dropping the flag or a leg action ends it.
                requested = (snapshot is not None and snapshot.hold_upright
                             and not card_policy_active)
                upright_hold = requested and (upright_active or stopped)
                if upright_active and not upright_hold:
                    policy.reset()
                upright_active = upright_hold
                # A fixed-test timer must never override live vision commands.
                if (args.command_source == "fixed"
                        and not fixed_walk_active(now - start_time,
                                                  args.walk_seconds,
                                                  args.pause_seconds)):
                    velocity_command[:] = 0.0  # vx=0, vy=0, wz=0

                if step_policy is card_policy:
                    command_values = {"lift_command": decision.lift_command}
                    command_status = (f"shape={decision.action_id} one-foot "
                                      f"support={decision.support_foot} "
                                      f"lift={int(decision.lift_command)} ")
                elif upright_hold:
                    command_values = {}
                    command_status = "hold_upright "
                else:
                    command_values = {"velocity_command": velocity_command}
                    command_status = (
                        f"policy_target_velocity=[vx={velocity_command[0]:+.3f} m/s, "
                        f"vy={velocity_command[1]:+.3f} m/s, "
                        f"wz={velocity_command[2]:+.3f} rad/s] "
                    )

            # The untilt ramp. The STM32 clears its feedback freeze in the same
            # instant it starts ramping the lean back, so it immediately starts
            # reporting a tilted, moving body - one the model never commanded. That
            # is the pose that produced the 87-degree hip split on the 2026-10-02
            # run. Keep replaying the snapshot taken when 8 went out until the ramp
            # has had time to finish (CARD_UNTILT_HOLD_S), then hand the model the
            # live state again - by which point the body is back on its stand pose.
            if tilt_release_at > 0.0:
                if now >= tilt_release_at:
                    tilt_release_at = 0.0
                    held_observation = None
                    print("[shape] untilt ramp done; policy sees live state again")
                elif held_observation is not None:
                    held_reference = True
                    (q_policy, qd_policy, accel_policy, gyro_policy,
                     projected_gravity) = held_observation

            if not args.fixed_policy:
                if upright_hold:
                    walking_command_hold.clear()
                    command_hold_reason = "upright_takeover"
                    # Same all-zero frame examples/stand_upright_hold.json plays: knees
                    # straight, no policy in the loop. The slew limiter and the
                    # deviation window below still rate-limit the way in and out, so
                    # this is not a step.
                    q_policy_target = np.zeros(config.NUM_JOINTS, dtype=np.float32)
                    action = np.zeros(config.ACTION_DIM, dtype=np.float32)
                    obs = np.zeros(1, dtype=np.float32)
                    latency_ms = 0.0
                else:
                    if args.policy == "walking" and step_policy is not card_policy:
                        # Timestamp the actual first use at the model boundary,
                        # rather than consuming time spent receiving/processing.
                        command_now = time.monotonic_ns() * 1e-9
                        velocity_command = walking_command_hold.apply(velocity_command, command_now)
                        command_values = {"velocity_command": velocity_command}
                        command_hold_remaining = walking_command_hold.remaining(command_now)
                        command_hold_reason = ("stop" if np.all(velocity_command == 0.) else "normal_walking")
                        command_status = (f"policy_target_velocity=[vx={velocity_command[0]:+.3f} m/s, "
                                          f"vy={velocity_command[1]:+.3f} m/s, "
                                          f"wz={velocity_command[2]:+.3f} rad/s] "
                                          f"hold_remaining={command_hold_remaining:.3f}s ")
                    else:
                        walking_command_hold.clear()
                        command_hold_reason = "onefoot_takeover"
                    q_policy_target, action, obs, latency_ms = step_policy.step(
                        accel_m_s2=accel_policy,
                        gyro_rad_s=gyro_policy,
                        projected_gravity=projected_gravity,
                        **command_values,
                        joint_position_policy=q_policy,
                        joint_velocity_policy=qd_policy,
                    )
            if not args.fixed_policy and args.policy == "walking":
                diagnostic_velocity = velocity_command.copy()
            if upright_hold:
                diagnostic_mode, diagnostic_source = "upright_hold", "hold_upright_zero_target"
            elif not args.fixed_policy and args.policy == "walking":
                if step_policy is card_policy:
                    diagnostic_mode = "onefoot46"
                    diagnostic_lift = command_values["lift_command"]
                else:
                    diagnostic_velocity = velocity_command.copy()
            q_policy_target = apply_forward_ankle_bias(
                q_policy_target,
                velocity_command if not args.fixed_policy and args.policy == "walking" else None,
                not args.fixed_policy and args.policy == "walking"
                and not upright_hold and step_policy is policy,
                args.forward_ankle_bias_rad,
            )
            q_policy_target, target_trace = target_safety.apply_with_trace(
                q_policy_target, last_q_policy_target, q_policy, dt)
            last_q_policy_target = q_policy_target
            last_q_motor = config.policy_to_motor_position(q_policy_target)

            # Use the held/overridden command actually used by the walking model.
            # Other target sources have no walking velocity and use standing PD.
            gain_velocity = (velocity_command if not args.fixed_policy and args.policy == "walking"
                             and step_policy is policy and not upright_hold else None)
            motion_kp, motion_kd = config.motion_gain_scales(gain_velocity)
            kp_scale, kd_scale = args.kp_scale * motion_kp, args.kd_scale * motion_kd
            flags = COMMAND_ENABLE if args.enable_motors else 0
            command_timestamp = monotonic_us()
            diagnostic_values = dict(state=state, trace=target_trace, motor_target=last_q_motor,
                action=action if diagnostic_source == "onnx" else None, reference_qd=qd_policy,
                reference_accel=accel_policy, reference_gyro=gyro_policy, reference_gravity=projected_gravity,
                observation=obs if diagnostic_source == "onnx" else None, receive_info=state_receive_info,
                velocity_command=diagnostic_velocity, lift_command=diagnostic_lift,
                step=step, read_monotonic_s=diagnostic_read_time, elapsed_s=now - start_time, policy_mode=diagnostic_mode,
                target_source=diagnostic_source, phase=shape_controller.phase if shape_controller is not None else "",
                held_reference=int(held_reference), card_tilt_active=int(card_tilt_active), upright_hold=int(upright_hold),
                command_hold_remaining_s=command_hold_remaining, command_hold_reason=command_hold_reason,
                command_flags=flags, command_timestamp_us=command_timestamp, infer_ms=latency_ms)
            if requested_velocity is not None:
                diagnostic_values.update(zip(("requested_cmd_vx", "requested_cmd_vy", "requested_cmd_wz"),
                                             requested_velocity.tolist()))
            if diagnostic_source == "onnx" and diagnostic_mode == "walking49":
                diagnostic_values.update(command_step_distance_m=float(obs[11]), command_crossing=float(obs[12]))
            if diagnostic_mode == "onefoot46":
                diagnostic_values["support_foot"] = (args.support_foot if args.policy == "one-foot"
                                                      else decision.support_foot)
                if args.policy == "one-foot":
                    diagnostic_values["phase"] = "lift" if diagnostic_lift else "standing"
            if args.fixed_policy:
                diagnostic_values.update(phase="fixed_frame", phase_tick=policy.index - 1)
            try:
                send_result = link.send_command(command_timestamp, last_q_motor,
                                                kp_scale, kd_scale, flags)
            except OSError as exc:
                if diagnostics is not None:
                    diagnostics.write(**diagnostic_values, send_result="link_lost",
                                      send_error=str(exc),
                                      send_done_monotonic_s=time.monotonic_ns() * 1e-9)
                link_loss_recovery(exc, "write")
                continue
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.write(**diagnostic_values, send_result="error", send_error=str(exc),
                                      send_done_monotonic_s=time.monotonic_ns() * 1e-9)
                raise
            if diagnostics is not None:
                diagnostics.write(**diagnostic_values,
                    send_result="written" if send_result is True else "not_written" if send_result is False else "unknown",
                    send_done_monotonic_s=time.monotonic_ns() * 1e-9)
            # The vision process is still publishing zero speed while this process
            # loads the model and opens USB. Release its 0.5 s first step only after
            # the STM32 has acknowledged a fresh enabled command in a state frame.
            if (not startup_ready_sent and write_startup_ready_if_confirmed(
                    args.startup_ready_file, state.status_flags,
                    send_result is True, step)):
                startup_ready_sent = True
                print("[start-gate] policy ready; enabled COMMAND acknowledged by STM32", flush=True)
            if args.fixed_policy:
                response_deadline = time.monotonic() + 0.05
                response = link.get_latest_state(max_age_s=0.05)
                while response.sequence == state.sequence:
                    if time.monotonic() >= response_deadline:
                        raise TimeoutError("No new STM32 state after a fixed frame")
                    time.sleep(0.001)
                    response = link.get_latest_state(max_age_s=0.05)
                if response.status_flags & STATE_FAULT:
                    raise RuntimeError(f"STM32 reports a fault: flags=0x{response.status_flags:08X}")
                if (response.status_flags & required) != required:
                    raise RuntimeError(f"IMU/encoder data invalid: flags=0x{response.status_flags:08X}")

            elapsed_s = now - start_time
            position_logger.write(
                elapsed_s,
                step,
                state.sequence,
                last_q_motor,
                state.joint_position,
                accel_policy,
                orientation_rpy,
                gyro_policy,
            )
            if position_plot is not None and step % args.plot_every == 0:
                position_plot.update(
                    elapsed_s,
                    last_q_motor,
                    state.joint_position,
                    accel_policy,
                    orientation_rpy,
                )

            # 10 Hz 就够：视觉那边用长时间常数低通，滤掉的正是步态摆动。
            if attitude is not None and step % 5 == 0:
                # Only the current lean event may open the next voting window.
                # A previous card's DONE must not be mistaken for this card's DONE.
                tilt_event_id = card_tilt_event if card_tilt_active else 0
                attitude.publish(
                    projected_gravity, elapsed_s,
                    card_tilt_event_id=tilt_event_id,
                    card_tilt_done=(tilt_event_id != 0 and
                                    link.get_action_status(tilt_event_id) == ACTION_DONE),
                )

            if step % max(1, args.log_every) == 0:
                print(
                    f"step={step:6d} state_seq={state.sequence:5d} "
                    f"{command_status}"
                    f"KP×{kp_scale:.3g} KD×{kd_scale:.3g} "
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
        walking_command_hold.clear()
        diagnostic_error = str(exc)
        print(f"FAULT: {exc}")
        send_disable(link, last_q_motor, estop=True)
        return 1
    finally:
        walking_command_hold.clear()
        send_disable(link, last_q_motor)
        if attitude is not None:
            attitude.close()
        position_logger.close()
        if position_plot is not None and not timed_run_completed:
            position_plot.close()
        if command_source is not None and args.policy == "walking":
            command_source.close()
        link.close()
        if diagnostics is not None:
            diagnostics.close(status="fault" if diagnostic_error else "completed", error=diagnostic_error)

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
