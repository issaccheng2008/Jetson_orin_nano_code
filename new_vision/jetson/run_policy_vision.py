#!/usr/bin/env python3
"""New line detector -> steering PID -> UDP connector -> walking ONNX policy.

Run this entry point for policy walking, not run_robot.py's V2 serial output.
Only the policy process owns the STM32 serial device.

A confirmed shape card is reported once as ``event_id``/``event_action`` and
retained for ``--card-hold-ms`` so the policy receiver can deduplicate it.
``qr`` describes the current recognition and returns to -1 when absent.
Bar crossing is still not signalled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import time

import numpy as np

from camera_config import load as load_camera
from policy_bridge import ConnectorClient, SteeringController

# 图卡的中文名，只给日志用 —— 操作员看日志时认的是图形，不是 "pentagon"。
CARD_NAMES_ZH = {
    "circle": "圆形", "pentagon": "五角星", "square": "正方形",
    "diamond": "菱形", "cross": "十字形", "triangle": "三角形",
}

# 两个 dump 各自写出来的文件名。清理时只认这两种，不认就不动 —— 目录被指到
# 别处时（比如 home）整清会连别人的东西一起删。序号每次跑都从 1 重来，所以
# 上一次跑的文件不清掉就会和新的一次混在同一层，同名还会互相覆盖。
SHAPE_DUMP_RE = re.compile(r"^\d{4}_.+_cy[^_]*_g[01]\.(jpg|json)$")
LOSS_DUMP_RE = re.compile(r"^\d{3}_\d{2}_pair.+_conf.+"
                          r"(_frame\.jpg|_vis\.jpg|\.json)$")


def _clear_dump_dir(path, pattern):
    """删掉上一次跑留下的 dump 文件，返回删了几个。"""
    removed = 0
    for name in os.listdir(path):
        if not pattern.match(name):
            continue
        try:
            os.remove(os.path.join(path, name))
            removed += 1
        except OSError:
            pass
    return removed


def parse_args():
    camera = load_camera()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAM_IDX", camera["index"])))
    parser.add_argument("--width", type=int, default=camera["width"])
    parser.add_argument("--height", type=int, default=camera["height"])
    parser.add_argument("--camera-height-cm", type=float,
                        default=float(os.getenv("CAM_HEIGHT_CM", camera["mount_height_cm"])))
    parser.add_argument("--camera-pitch-deg", type=float,
                        default=float(os.getenv("CAM_PITCH_DEG", camera["pitch_deg"])))
    parser.add_argument("--camera-vfov-deg", type=float,
                        default=float(os.getenv("CAM_VFOV_DEG", camera["vfov_deg"])))
    parser.add_argument("--connector-host", default="127.0.0.1")
    parser.add_argument("--connector-port", type=int, default=5006)
    parser.add_argument("--attitude-bind", default="127.0.0.1")
    parser.add_argument("--attitude-port", type=int, default=5007,
                        help="Listen here for the body attitude broadcast by "
                             "humanoid_jetson_deploy/main.py (0 disables it). Without "
                             "it the camera pitch stays the static config value while "
                             "the body swings 30-40 deg under it")
    parser.add_argument("--attitude-tau-s", type=float, default=1.2,
                        help="Low-pass on the incoming pitch; the gait swing is a "
                             "zero-mean 1.7 Hz oscillation and only the slow lean is "
                             "wanted. 0.4 s leaves 23%% of it (+/-8 deg), 1.2 s 7.8%%")
    parser.add_argument("--vx", type=float, default=0.3,
                        help="Forward speed with valid detection, m/s")
    parser.add_argument("--max-wz", type=float, default=0.5,
                        help="Yaw-rate limit, rad/s (0..0.5)")
    parser.add_argument("--steer-full-scale-cm", type=float, default=10.0,
                        help="Cross-track error (cm) producing max-wz; smaller means stronger steering")
    parser.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1,
                        help="Image steering to policy yaw; flip it if the robot turns the wrong way")
    parser.add_argument("--step-len-cm", type=float, default=float(os.getenv("STEP_LEN_CM", "8")))
    parser.add_argument("--preview-gain", type=float,
                        default=float(os.getenv("PREVIEW_GAIN", "0")),
                        help="Heading feedforward: steps of predicted drift to steer out. "
                             "0 (default) leaves heading feedback to the detector's angle term, "
                             "which is 7x weaker but carries far less of the angle bias")
    parser.add_argument("--lost-hold-s", type=float, default=0.2,
                        help="Hold the last command this long after line loss before stopping")
    parser.add_argument("--deriv-pole", type=float,
                        default=float(os.getenv("JETSON_PID_D_FILTER", "0.78")),
                        help="IIR pole on the D term; higher is smoother, 0 disables the filter")
    parser.add_argument("--bias-cm", type=float,
                        default=float(os.getenv("STEER_BIAS_CM", "3")),
                        help="Standing trim added to fused_err_cm on curves, shifting where "
                             "the loop settles to cancel a one-sided lateral offset. Set it "
                             "to the err the log shows standing in the curve. 0 disables it")
    parser.add_argument("--bias-gate-px", type=float,
                        default=float(os.getenv("STEER_BIAS_GATE_PX", "12")),
                        help="Smoothed abs(curve_px) at which --bias-cm is fully applied")
    parser.add_argument("--bias-dead-px", type=float,
                        default=float(os.getenv("STEER_BIAS_DEAD_PX", "6")),
                        help="Smoothed abs(curve_px) under which the trim sits at "
                             "--bias-straight-cm instead of part way up the ramp")
    parser.add_argument("--bias-straight-cm", type=float,
                        default=float(os.getenv("STEER_BIAS_STRAIGHT_CM", "1")),
                        help="Trim on a straight; it fades to --bias-cm by --bias-gate-px")
    parser.add_argument("--max-lateral-cm", type=float,
                        default=float(os.getenv("MAX_LATERAL_CM", "0")),
                        help="0 disables. Otherwise, half the lane width: the near "
                             "band's lateral offset past this puts the robot off the "
                             "track, which cannot be true while it follows the line, "
                             "so the frame counts as loss (hold, then stop) instead "
                             "of steering on it. Off by default because the bound has "
                             "not been measured against a normal lap yet")
    parser.add_argument("--no-shape-detect", action="store_true",
                        help="Skip geometric card detection entirely; qr stays -1")
    parser.add_argument("--dump-on-loss", default="",
                        help="Directory to write the frames around a bottom-lock "
                             "drop-out or a confidence collapse into. Empty is off. "
                             "Writes both the raw frame and the detector's own "
                             "overlay for the last few frames before the trip and "
                             "the first one after, split by time")
    parser.add_argument("--dump-on-loss-ring", type=int, default=6,
                        help="How many recent frames to hold for --dump-on-loss")
    parser.add_argument("--dump-on-loss-cooldown", type=float, default=3.0,
                        help="Seconds before --dump-on-loss may write again, so one "
                             "bad stretch leaves one dump rather than hundreds")
    parser.add_argument("--shape-dump", default="",
                        help="Directory to write a frame and its detection dict into, "
                             "on every shape call where a card is in view. Empty is off. "
                             "Needs a card to be present, so a lap writes tens of pairs, "
                             "not thousands")
    parser.add_argument("--no-red-detect", action="store_true",
                        help="Ignore red completely: no red bar, no narrow gate, and "
                             "a red row no longer blocks the band scan or the bottom "
                             "lock. Temporary, for isolating red's effect on line "
                             "following")
    parser.add_argument("--card-hold-ms", type=float,
                        default=float(os.getenv("CARD_HOLD_MS", "5000")),
                        help="Once the shape is identified: keep the event available and stay "
                             "stopped this long before resuming speed. 5000 is the rules' "
                             "action window plus margin")
    parser.add_argument("--card-stop-ms", type=float,
                        default=float(os.getenv("CARD_STOP_MS", "3000")),
                        help="Stand still at most this long waiting for the shape to "
                             "settle, counted from the moment the box came close enough. "
                             "Bounds the wait when no shape is ever identified. "
                             "0 disables stopping")
    parser.add_argument("--card-trigger-frac", type=float, default=None,
                        help="Explicit override: box centroid height in the frame "
                             "(0=top, 1=bottom) at which the robot stops. Prefer "
                             "--card-trigger-dist-cm; this is the same thing in the "
                             "frame's own units, for tests that drive a fake cy")
    parser.add_argument("--card-trigger-dist-cm", type=float,
                        default=float(os.getenv("CARD_TRIGGER_DIST_CM", "43.0")),
                        help="Ground distance, in cm, at which the robot stops and "
                             "identifies the shape. 43.0 is the old "
                             "--card-trigger-frac 0.4 written in cm, so it is the "
                             "behaviour that has been on the robot; lower it to stop "
                             "closer. Converted to a frame height by the camera "
                             "geometry (mount height, pitch, vfov); seeing a card "
                             "earlier only slows it down")
    parser.add_argument("--card-slow-vx", type=float,
                        default=float(os.getenv("CARD_SLOW_VX", "0.2")),
                        help="Forward speed while a card is in view but not yet close "
                             "enough to act on")
    parser.add_argument("--card-resume-ms", type=float,
                        default=float(os.getenv("CARD_RESUME_MS", "500")),
                        help="On the stopped-to-walking edge, hold --card-slow-vx this "
                             "long before releasing to full speed, so the start mirrors "
                             "the stop's 0.4 -> 0.2 -> 0 shape. 0 releases on the first "
                             "walking frame, leaving only the connector's slew")
    parser.add_argument("--card-tilt-ms", type=float,
                        default=float(os.getenv("CARD_TILT_MS", "1000")),
                        help="After the stop trigger, spend this long not looking at "
                             "the card at all while the robot re-poses the body. The "
                             "STM32 is asked for the tilt on the trigger and this is "
                             "the window it gets; identifying during it reads a body "
                             "that is still moving. 0 identifies immediately")
    parser.add_argument("--card-cold-start", action="store_true",
                        help="On the stopped-to-walking edge, wipe the controller's loop "
                             "state and the detector's memory instead of keeping them. "
                             "Off by default: the robot stops the moment it has "
                             "recognised a card, which is a point where the detector was "
                             "tracking the line well, and keeping its estimate of where "
                             "the lane is beats restarting from the image middle. Turn "
                             "it on to break a lock that had already gone wrong")
    parser.add_argument("--card-clear-calls", type=int,
                        default=max(1, int(os.getenv("CARD_CLEAR_CALLS", "4"))),
                        help="Consecutive detection calls with no card before the flag "
                             "drops and the same card may trigger again. One absent frame "
                             "used to be enough, which let a flickering cue re-trigger")
    parser.add_argument("--card-stable-frames", type=int,
                        default=max(1, int(os.getenv("CARD_STABLE_FRAMES", "2"))),
                        help="Detection calls that must return the same shape name before "
                             "the card is acted on. run_robot.py used 3; 2 trades a little "
                             "precision for firing on cards whose classification flickers")
    parser.add_argument("--shape-every", type=int,
                        default=max(1, int(os.getenv("SHAPE_EVERY", "6"))),
                        help="Run card detection every N frames. Measured at 1280x720: "
                             "31 ms with no card in view, 95-122 ms with one, against "
                             "29 ms for the line detector alone. 6 keeps the loop near "
                             "29 Hz clear and 23 Hz while a card is visible")
    parser.add_argument("--card-every-stopped", type=int,
                        default=max(1, int(os.getenv("CARD_EVERY_STOPPED", "2"))),
                        help="Card detection period while stopped, instead of running "
                             "the whole detector on every frame. Every frame costs "
                             "94-122 ms, which drops the loop to 6-8 Hz and coarsens "
                             "the steering the policy sees (a held packet lasts 6-8 "
                             "of its 50 Hz ticks instead of 3). 2 roughly doubles the "
                             "loop rate and still gives ~6 attempts a second")
    parser.add_argument("--center-dead-cm", type=float,
                        default=float(os.getenv("CENTER_DEAD_CM", "4.0")),
                        help="Inside this lateral error the yaw authority fades linearly "
                             "toward zero at the centre, so a reading of a centimetre or "
                             "two on a 35 cm lane does not command a steady turn. Applied "
                             "to the output after the caps, so it limits how hard the loop "
                             "steers on a small error without moving where it settles. "
                             "0 disables it")
    parser.add_argument("--max-wz-right", type=float, default=None,
                        help="Yaw-rate limit for right turns, as opposed to --max-wz "
                             "for left. Negative wz is a right turn in the published "
                             "log. Unset runs symmetric with --max-wz; setting it lower "
                             "caps right turns only, and a right curve needing more "
                             "than that runs wide")
    parser.add_argument("--single-line-gain", type=float,
                        default=float(os.getenv("SINGLE_LINE_GAIN", "1")),
                        help="Loop-gain multiplier applied only while one track "
                             "boundary is visible on a curve. 1.0 (default) leaves "
                             "the gains exactly as they are")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="0 runs until Ctrl+C")
    args = parser.parse_args()
    if not 1 <= args.connector_port <= 65535:
        parser.error("connector-port must be between 1 and 65535")
    if args.width <= 0 or args.height <= 0:
        parser.error("camera dimensions must be positive")
    if (not all(math.isfinite(v) for v in (args.camera_height_cm, args.camera_pitch_deg,
                                         args.camera_vfov_deg, args.max_seconds))
            or args.camera_height_cm <= 0 or not 0 < args.camera_vfov_deg < 180
            or not 0 < args.camera_pitch_deg < 90 or args.max_seconds < 0):
        parser.error("invalid camera geometry or max-seconds")
    if not math.isfinite(args.card_hold_ms) or args.card_hold_ms < 0:
        parser.error("card-hold-ms must be finite and nonnegative")
    if not math.isfinite(args.card_stop_ms) or args.card_stop_ms < 0:
        parser.error("card-stop-ms must be finite and nonnegative")
    if not math.isfinite(args.card_slow_vx) or not 0 <= args.card_slow_vx <= 1:
        parser.error("card-slow-vx must be in [0, 1]")
    if not math.isfinite(args.center_dead_cm) or args.center_dead_cm < 0:
        parser.error("center-dead-cm must be finite and nonnegative")
    if args.max_wz_right is not None and not (
            math.isfinite(args.max_wz_right)
            and 0 < args.max_wz_right <= args.max_wz):
        parser.error("max-wz-right must be in (0, max-wz] when set")
    if not math.isfinite(args.card_resume_ms) or args.card_resume_ms < 0:
        parser.error("card-resume-ms must be finite and nonnegative")
    if args.card_trigger_frac is not None and not (
            math.isfinite(args.card_trigger_frac) and 0 < args.card_trigger_frac <= 1):
        parser.error("card-trigger-frac must be in (0, 1]")
    if not math.isfinite(args.card_trigger_dist_cm) or args.card_trigger_dist_cm <= 0:
        parser.error("card-trigger-dist-cm must be positive")
    if args.card_clear_calls < 1:
        parser.error("card-clear-calls must be at least 1")
    if args.card_stable_frames < 1:
        parser.error("card-stable-frames must be at least 1")
    if args.card_every_stopped < 1:
        parser.error("card-every-stopped must be at least 1")
    if args.shape_every < 1:
        parser.error("shape-every must be at least 1")
    if args.dump_on_loss_ring < 1:
        parser.error("dump-on-loss-ring must be at least 1")
    if not math.isfinite(args.dump_on_loss_cooldown) or args.dump_on_loss_cooldown < 0:
        parser.error("dump-on-loss-cooldown must be finite and nonnegative")
    return args


def fmt(value, spec):
    """Format one detector debug value; '-' when the detector did not report it."""
    return "-" if value is None else format(value, spec)


def main():
    args = parse_args()
    # Reuse the dual-mode PID defaults/environment overrides of run_robot.py.
    def gains(mode, defaults):
        return tuple(float(os.getenv(f"JETSON_PID_{mode}_{name}", str(value)))
                     for name, value in zip(("KP", "KI", "KD"), defaults))

    controller = SteeringController(
        vx=args.vx, max_wz=args.max_wz, steer_full_scale_cm=args.steer_full_scale_cm,
        yaw_sign=args.yaw_sign, step_len_cm=args.step_len_cm, preview_gain=args.preview_gain,
        straight_gains=gains("STRAIGHT", (0.83, 0.004, 0.095)),
        curve_gains=gains("CURVE", (0.83, 0.006, 0.16)),
        integral_limit=float(os.getenv("JETSON_PID_I_CLAMP", "60")),
        lost_hold_s=args.lost_hold_s, deriv_pole=args.deriv_pole,
        bias_cm=args.bias_cm, bias_gate_px=args.bias_gate_px,
        bias_dead_px=args.bias_dead_px, bias_straight_cm=args.bias_straight_cm,
        max_lateral_cm=args.max_lateral_cm,
        max_wz_right=(args.max_wz if args.max_wz_right is None
                      else args.max_wz_right),
        single_line_gain=args.single_line_gain,
        center_dead_cm=args.center_dead_cm,
    )
    # Lazy imports keep --help and controller tests usable without a camera stack.
    import cv2
    from line_detector_v1_warp import LineDetector
    from utils import open_camera, show_debug_windows

    from shape_detector import trigger_frac_at_dist, trigger_width_at_dist
    # 两个阈值同源：同一个 --card-trigger-dist-cm，一个换成 cy、一个换成框宽。
    # 有框时用后者（对机身俯仰不敏感），没框时只能退回前者。
    card_trigger_frac = (args.card_trigger_frac if args.card_trigger_frac is not None
                         else trigger_frac_at_dist(args.card_trigger_dist_cm))
    card_trigger_width = trigger_width_at_dist(args.card_trigger_dist_cm)

    shape = shape_names = None
    if not args.no_shape_detect:
        from shape_detector import ShapeDetector
        # run_robot.py's cooldown, so a card cannot re-fire while it is still in view.
        shape = ShapeDetector(stable_frames=args.card_stable_frames,
                              cooldown_ms=3200, debug=False)
        shape_names = {number: name for name, number in shape.action_map.items()}

    # 机身姿态只喂给图卡，不喂巡线。走路时俯仰以 1.7Hz 摆 30~40°，低通过的
    # 滞后值描述不了当前这一帧，喂进 IPM 反而更糟；巡线那边靠车道宽锚定解决，
    # 那个只用"赛道多宽"这个物理事实，不依赖姿态。图卡是停稳之后才认的，
    # 那时姿态本来就稳，低通几个时间常数就跟上了 —— 滞后不是问题。
    attitude = None
    if args.attitude_port > 0:
        try:
            from attitude_input import AttitudeInput
            attitude = AttitudeInput(args.attitude_port, args.camera_pitch_deg,
                                     bind=args.attitude_bind, tau_s=args.attitude_tau_s)
        except OSError as exc:
            print(f"[attitude] 端口 {args.attitude_port} 收不了：{exc}；"
                  "相机俯角退回静态安装角")

    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    client = ConnectorClient(args.connector_host, args.connector_port)
    cap = None
    try:
        cap = open_camera(args.camera, args.width, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        detector = LineDetector(width, height, cam_height_cm=args.camera_height_cm,
                                cam_pitch_deg=args.camera_pitch_deg,
                                cam_vfov_deg=args.camera_vfov_deg)
        if args.no_red_detect:
            detector.red_detect_enable = False
        print(f"Camera {args.camera}: {width}x{height}; UDP -> "
              f"{args.connector_host}:{args.connector_port}; vx={args.vx} m/s; "
              f"max_wz={args.max_wz} rad/s "
              f"(right {args.max_wz if args.max_wz_right is None else args.max_wz_right}); "
              f"yaw_sign={args.yaw_sign}; "
              f"red={'off' if args.no_red_detect else 'on'}; "
              f"sl_gain={args.single_line_gain}; "
              f"center_dead={args.center_dead_cm}cm; "
              f"bias={args.bias_straight_cm}->{args.bias_cm}"
              f"(dead {args.bias_dead_px}, full {args.bias_gate_px})", flush=True)
        if attitude is not None:
            print(f"Body attitude: udp://{args.attitude_bind}:{args.attitude_port}"
                  f" tau={args.attitude_tau_s}s; 安装角 {args.camera_pitch_deg:.1f}°"
                  f" 会被机身俯仰实时修正", flush=True)
        start = previous = time.monotonic()
        last_log = -math.inf
        last_shape_log = -math.inf
        last_log_at = start
        log_frames = 0
        frames = 0
        card_action = -1
        card_event_id = 0
        card_until = 0.0
        stop_until = 0.0
        resume_until = 0.0
        tilt_until = 0.0
        card_flag = False        # a card is in view on this approach
        card_absent = 0          # consecutive detection calls without one
        card_triggered = False   # this card has already been acted on
        card_armed = True        # and it has since been seen far enough to trigger
        card_action_triggered = False
        card_dbg = {}
        lateral_warned = None    # None until the lateral bound first trips
        card_window_open = False
        dumped = 0
        if args.shape_dump:
            os.makedirs(args.shape_dump, exist_ok=True)
            print(f"[shape] {args.shape_dump}: cleared "
                  f"{_clear_dump_dir(args.shape_dump, SHAPE_DUMP_RE)} files "
                  f"from the last run", flush=True)
        # The frames around a lock drop-out, kept short so the last good frame before
        # the drop is still in the ring when it trips -- that is the one that shows
        # what the detector was looking at while it still agreed with itself.
        loss_ring = []
        loss_dumped = 0
        loss_next_ok = 0.0
        prev_pair = 0.0
        prev_conf = 0.0
        if args.dump_on_loss:
            os.makedirs(args.dump_on_loss, exist_ok=True)
            print(f"[vision] {args.dump_on_loss}: cleared "
                  f"{_clear_dump_dir(args.dump_on_loss, LOSS_DUMP_RE)} files "
                  f"from the last run", flush=True)
        while not stopped:
            now = time.monotonic()
            if args.max_seconds > 0 and now - start >= args.max_seconds:
                break
            ok, frame = cap.read()
            if not ok:
                controller.reset()
                event = ({"event_id": card_event_id, "event_action": card_action}
                         if card_event_id and now < card_until else {})
                client.publish(0.0, 0.0, -1, **event)
                if now - last_log >= 0.5:
                    print("[vision -> connector] camera read failed; vx=0 wz=0", flush=True)
                    last_log = now
                time.sleep(0.02)
                continue
            # Stamp the frame on arrival, before the detector runs. stop_until and
            # card_until are set from these stamps, so the stopped-to-walking edge
            # below has to be judged with the same clock as the window that set it -
            # judging it from the top of the loop ran one frame late.
            processed = time.monotonic()
            window_open = processed < stop_until or processed < card_until
            if card_window_open and not window_open:
                # Handover on the stopped-to-walking edge, BEFORE this frame is
                # processed. Doing anything to the state after detector.process() left
                # the first walking frame computed from the old state, so only the
                # second frame was clean - and the first decides where it goes.
                #
                # Default is to keep both states: stopping is a pause, and the loop
                # state and the detector's memory are what the walking process had
                # built. --card-cold-start wipes them instead, which is what breaks a
                # lock that had already gone wrong - but it also throws away a good
                # estimate of where the lane is and re-arms the 25-frame relaxed
                # startup window.
                controller.drop_held_command()
                if args.card_cold_start:
                    controller.reset(clear_hold=True)
                    detector.reset_state()
                # The start mirrors the stop, which is 0.4 -> 0.2 -> 0: hold the slow
                # speed one more stage before releasing, so the robot builds speed in
                # two steps instead of the connector's single 0.4 s slew.
                resume_until = processed + args.card_resume_ms / 1000.0
                print(f"[vision] card window closed; "
                      f"{'cold start' if args.card_cold_start else 'resuming frozen'} "
                      f"(hold={controller.hold[0]:+.2f},{controller.hold[1]:+.2f} "
                      f"lost_s={controller.lost_s:.2f}) "
                      f"vx<={args.card_slow_vx:+.2f} for {args.card_resume_ms:.0f}ms",
                      flush=True)
            card_window_open = window_open
            if attitude is not None:
                attitude.poll()
                if shape is not None:
                    shape.set_camera_pitch_deg(attitude.value)
            _, _, confidence, visualization, debug = detector.process(frame)
            frames += 1
            log_frames += 1
            recognized_this_frame = False
            if args.dump_on_loss:
                pair_now = float(debug.get("bottom_pair_ratio", 0.0))
                loss_ring.append((frame.copy(), visualization.copy(),
                                  dict(debug), float(confidence)))
                del loss_ring[:-args.dump_on_loss_ring]
                # A lock that was pairing every row and now pairs none, or a
                # confidence that falls off a cliff. Either is the moment the near
                # band stopped agreeing with itself.
                tripped = ((prev_pair > 0.5 and pair_now <= 0.0)
                           or (prev_conf > 0.5 and confidence < 0.2))
                if tripped and processed >= loss_next_ok:
                    loss_dumped += 1
                    loss_next_ok = processed + args.dump_on_loss_cooldown
                    print(f"[vision] lock lost (pair {prev_pair:.2f}->{pair_now:.2f}, "
                          f"conf {prev_conf:.2f}->{confidence:.2f}) -> "
                          f"{args.dump_on_loss} #{loss_dumped}", flush=True)
                    for index, (raw, vis, info, conf) in enumerate(loss_ring):
                        stem = os.path.join(
                            args.dump_on_loss,
                            f"{loss_dumped:03d}_{index:02d}_pair"
                            f"{float(info.get('bottom_pair_ratio', 0.0)):.2f}"
                            f"_conf{conf:.2f}")
                        cv2.imwrite(stem + "_frame.jpg", raw)
                        cv2.imwrite(stem + "_vis.jpg", vis)
                        with open(stem + ".json", "w", encoding="utf-8") as handle:
                            json.dump({key: value for key, value in info.items()
                                       if isinstance(value, (int, float, str, bool))
                                       or value is None},
                                      handle, indent=1, default=str)
                    loss_ring.clear()
                prev_pair, prev_conf = pair_now, float(confidence)
            # A card in view gets the stopped rate, not just the stop. The stop fires
            # on the first cy at or above the trigger line, so --shape-every 6 samples
            # that line about every 0.3 s - most of a 12 cm step at 0.4 m/s - and the
            # robot sailed past 43 cm to ~22 cm, close enough that the card's bottom
            # left the frame and no quad could close. The fine reading has to exist
            # before the decision, not after it.
            #
            # min(), not a plain switch: --shape-every 1 is a request for every frame,
            # and a card in view is no reason to slow that down. The stop branch keeps
            # card_every_stopped outright - standing still, cy barely moves and the
            # loop can have the cycles back.
            if processed < stop_until or processed < card_until:
                shape_period = args.card_every_stopped
            elif card_flag:
                shape_period = min(args.shape_every, args.card_every_stopped)
            else:
                shape_period = args.shape_every
            # The robot is re-posing the body for the first --card-tilt-ms of the stop.
            # Not looking at all, rather than looking and rejecting: a frame taken
            # mid-tilt is a frame of a body in motion, and feeding those to the
            # classifier is how a card gets read as the wrong shape. The card is
            # stationary and the window is seconds long, so the wait is free.
            tilting = processed < tilt_until
            if shape is not None and not tilting and (frames % shape_period == 0):
                action, card_dbg = shape.update(
                    frame, lane_offset_cm=float(debug.get("base_err_cm", 0.0)))
                # "多近了"这一个量，动作闸和停车闸共用。有真框就量框宽（对机身俯仰
                # 不敏感），没框才退回 cy。理由见下面停车那段。
                _quad = card_dbg.get("quad_work")
                card_width_px = None
                if _quad is not None:
                    _qx = np.asarray(_quad, dtype=float)[:, 0]
                    card_width_px = float(_qx.max() - _qx.min())
                if card_width_px is not None:
                    card_reach, card_reach_line = card_width_px, card_trigger_width
                else:
                    card_reach = card_dbg.get("presence_cy_frac")
                    card_reach_line = card_trigger_frac
                if args.shape_dump and (card_dbg.get("card_found")
                                        or card_dbg.get("presence")):
                    dumped += 1
                    stem = os.path.join(
                        args.shape_dump,
                        f"{dumped:04d}_{card_dbg.get('shape')}"
                        f"_cy{card_dbg.get('presence_cy_frac')}"
                        f"_g{card_dbg.get('card_found') and 1 or 0}")
                    cv2.imwrite(stem + ".jpg", frame)
                    with open(stem + ".json", "w", encoding="utf-8") as handle:
                        json.dump({"t": processed, **card_dbg}, handle,
                                  indent=1, default=str)
                # Phase two waits for phase one. The classifier is only reliable on a
                # card that is close, and the trigger line is what says it is. Acting
                # as soon as a shape appears classified a card 50 cm away - top=186,
                # cy=0.40 - on a small warp. The box only reaches that far down the
                # frame after card_trigger_frac, so nothing may act before it.
                # (hu was cited here as the reading that was right. It is not: on the
                # 2026-09-26 laps every frame of every card read hu=circle at 0.005
                # to 0.07, including the pentagram and the cross. It is degenerate,
                # not corroborating.)
                reached = (card_reach is not None
                           and card_reach >= card_reach_line)
                if action is not None and reached and not card_action_triggered and card_event_id == 0:
                    card_action = action
                    card_event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                    card_until = processed + args.card_hold_ms / 1000.0
                    card_action_triggered = True
                    recognized_this_frame = True
                    # Meant to be impossible to miss in a scrolling log: this is the
                    # one line that says the robot knew what it was looking at.
                    quad = card_dbg.get("quad_work")
                    width = ""
                    if quad is not None:
                        qx = np.asarray(quad, dtype=float)[:, 0]
                        width = f"框宽={float(qx.max() - qx.min()):.0f}px "
                    print("\n" + "=" * 68, flush=True)
                    print(f"  ★★★  识别到图卡：{CARD_NAMES_ZH.get(shape_names.get(action), '?')}"
                          f"（{shape_names.get(action, '?')}）  qr={action}", flush=True)
                    print(f"        距离画面 {fmt(card_dbg.get('presence_cy_frac'), '.2f')}  "
                          f"{width}保持 {args.card_hold_ms:.0f} ms", flush=True)
                    print("=" * 68 + "\n", flush=True)
                # The cue flickers while the robot walks, so the flag needs several
                # consecutive misses before it drops. A single absent frame used to
                # re-arm the stop, and the robot crept forward and stopped again.
                if bool(card_dbg.get("presence")):
                    card_absent = 0
                    card_flag = True
                else:
                    card_absent += 1
                    if card_absent >= args.card_clear_calls:
                        card_flag = False
                        card_triggered = False
                        card_action_triggered = False
                # Seeing a card only slows the robot down. Stopping waits until the card
                # is close, on the same card_reach card_reach_line the action gate uses.
                #
                # A trigger has to be earned again by seeing the card well short of the
                # line. presence is armed-gated, so the action firing makes it read
                # false and card_absent clears card_triggered on its own; the card the
                # robot just drove past is still in frame past the line, and stopped the
                # robot a second time mid-curve. A card already past the line has not
                # been approached, so it cannot fire.
                if card_reach is not None and card_reach < card_reach_line:
                    card_armed = True
                if (card_flag and not card_triggered and card_armed
                        and card_reach is not None
                        and card_reach >= card_reach_line):
                    card_triggered = True
                    card_armed = False
                    stop_until = processed + args.card_stop_ms / 1000.0
                    tilt_until = processed + args.card_tilt_ms / 1000.0
                    if card_width_px is not None:
                        print(f"[shape] card {card_reach:.0f}px of "
                              f"{card_reach_line:.0f}px "
                              f"(cy={fmt(card_dbg.get('presence_cy_frac'), '.2f')}) "
                              f"-> stand still {args.card_stop_ms:.0f} ms", flush=True)
                    else:
                        print(f"[shape] cue cy {card_reach:.2f} of "
                              f"{card_reach_line:.2f}, no box "
                              f"-> stand still {args.card_stop_ms:.0f} ms", flush=True)
                # Why a card that is plainly in view did not become an action: the
                # quad gates (found/closure), the classifier (shape/rules), or the
                # consecutive-frame latch. Only while a card is around, at 4 Hz.
                if ((card_dbg.get("card_found") or card_dbg.get("presence") or card_flag)
                        and processed - last_shape_log >= 0.25):
                    last_shape_log = processed
                    # One compact line per detection call while a card is around. The
                    # three numbers that matter are the stages a frame can die at:
                    # quad=N生成的候选, g=过了几何闸的, v=进了打分表的。g0 就是
                    # 一个候选都没过闸 —— 2026-09-30 实车每帧都是 g0，卡根本没找到。
                    rej = sorted((card_dbg.get("geom_rejects") or {}).items(),
                                 key=lambda kv: -kv[1])[:2]
                    print(
                        f"[shape] cy={fmt(card_dbg.get('presence_cy_frac'), '.2f')} "
                        f"cue={fmt(card_dbg.get('presence_cue'), '.2f')} "
                        f"found={int(bool(card_dbg.get('card_found')))} "
                        f"shape={card_dbg.get('shape') or '-'} "
                        f"quad={card_dbg.get('quad_total', '?')}"
                        f"g{card_dbg.get('quad_geom', '?')}"
                        f"v{len(card_dbg.get('scores') or [])} "
                        f"rej={' '.join(f'{k}:{n}' for k, n in rej) or '-'} "
                        f"armed={int(bool(getattr(shape, 'armed', True)))} "
                        f"cand={getattr(shape, 'candidate', None)}"
                        f"x{getattr(shape, 'candidate_count', 0)}", flush=True)
            if card_action != -1 and processed >= card_until:
                card_action = -1
                card_event_id = 0
            # Two windows, whichever ends later: --card-stop-ms caps how long we wait
            # for a shape that may never settle, --card-hold-ms is the rules' action
            # window once we do know it.
            if processed < stop_until or processed < card_until:
                # Not running the controller here is the whole point: it would be fed
                # the card-corrupted err for the length of the stop, and its integral
                # and filtered derivative would carry that corruption into the first
                # real frame. Idle it, and leave the state it brought in untouched - the
                # stop is a pause in the walking state, not the start of a new one.
                # Nothing else to do: the controller is not called here, so nothing it
                # holds can reach the wheels, and the handover below drops the stored
                # command before the first frame that does call it.
                vx, wz = 0.0, 0.0
            else:
                vx, wz = controller.command(debug, confidence, processed - previous)
                # Printed on the transition, not every frame: one line per time the
                # near band hands over an offset the lane cannot produce. How often
                # this fires on a real lap is the measurement.
                if controller.rejected_lateral is not None:
                    if lateral_warned is None:
                        print(f"[vision] near band says "
                              f"{controller.rejected_lateral:+.1f}cm, past the "
                              f"{args.max_lateral_cm:.1f}cm lane half-width; fading "
                              f"out then stopping instead of steering on it", flush=True)
                    lateral_warned = True
                else:
                    lateral_warned = False
                # card_flag alone, not "and not card_triggered": a card that never got
                # classified lets --card-stop-ms expire, and the robot used to resume at
                # full speed straight past it - the one place the near band is fully
                # covered by the card. Creep instead, so the classifier still has frames
                # to work with before the card is behind the robot.
                if card_flag or processed < resume_until:
                    vx = min(vx, args.card_slow_vx)
            previous = processed
            visible_qr = card_action if recognized_this_frame else -1
            event = ({"event_id": card_event_id, "event_action": card_action}
                     if card_event_id else {})
            # Re-read here rather than reusing the top of the loop: the window can be
            # opened by this very frame's trigger, and the standstill it asks for
            # starts now, not on the next one.
            client.publish(vx, wz, visible_qr,
                           hold_upright=(processed < stop_until
                                         or processed < card_until),
                           card_tilt=(processed < stop_until
                                      or processed < card_until),
                           **event)
            if processed - last_log >= 0.5:
                # Left of the bar is what the robot is doing; right of it is why.
                # Read only the left if it is behaving.
                print(
                    f"[vision] {log_frames / max(processed - last_log_at, 1e-6):4.0f}Hz "
                    f"vx={vx:+.3f} wz={wz:+.3f} "
                    f"err={debug.get('fused_err_cm', 0.0):+.1f}cm "
                    f"conf={confidence:.2f} qr={visible_qr}"
                    # Same id the connector and policy log, so the three can be
                    # lined up by hand when an event goes missing in the middle.
                    + (f"/ev{card_action}#{card_event_id}" if card_event_id else "")
                    + " | "
                    f"steer={controller.last_steer:+.2f} eff={controller.last_err_eff:+.1f} "
                    f"ang={debug.get('angle_err_deg', 0.0):+.1f} "
                    f"curve={int(bool(debug.get('curve_mode', False)))}"
                    f"/{debug.get('curve_px', 0.0):+.0f} "
                    f"far={debug.get('far_err_px', 0.0):+.0f} "
                    # Which track boundaries the near band actually saw: lr=11 both,
                    # lr=01 only the right one (and the centre is then inferred from
                    # it), lr=00 neither.
                    f"lr={int(bool(debug.get('left_seen', False)))}"
                    f"{int(bool(debug.get('right_seen', False)))} "
                    # Why a frame produced nothing: the bottom lock (the pairing
                    # check that keeps the near band on the right line) and how many
                    # rows it paired. conf=0 with pair=0 means no band at all; conf=0
                    # with lock=0 means the near band was rejected as asymmetric.
                    f"lock={int(bool(debug.get('bottom_lock_valid', False)))}"
                    f"pair={debug.get('bottom_pair_ratio', 0.0):.2f} "
                    f"lost={debug.get('lost_frames', '?')} "
                    # 车道宽锚和实时俯角：前者是横向比例尺的绝对缩放，后者只在
                    # 有姿态广播时才会从静态安装角上动起来。
                    f"lscale={debug.get('lateral_scale', 1.0):.3f} "
                    f"pitch={(attitude.value if attitude is not None else args.camera_pitch_deg):.1f}",
                    flush=True)
                last_log = processed
                last_log_at = processed
                log_frames = 0
            if not args.headless:
                cv2.putText(frame, f"vx={vx:+.3f} wz={wz:+.3f} Q=quit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Policy vision", frame)
                show_debug_windows(debug, visualization)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        client.close()
        if attitude is not None:
            attitude.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
