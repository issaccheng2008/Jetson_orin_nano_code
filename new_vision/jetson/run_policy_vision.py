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

# 赛道中线半径（m），文档 §3。只用来在启动横幅里打一行参考："跟住一个弯需要
# 多少角速度"（ω = vx / R）。**脉冲幅度不按它推** —— 0.4/0.5 是实车量出来的好值。
LANE_RADIUS_M = 0.776


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
    parser.add_argument("--hold-still", action="store_true",
                        help="Publish zero vx/wz whatever the controller decides. "
                             "Detection, logging and the card logic all keep running "
                             "and the robot just stands, which is what a bench test of "
                             "the body-pitch effect needs: it lets the full stack "
                             "run (the attitude comes from main.py) without driving")
    parser.add_argument("--line-pitch", action="store_true",
                        help="Also feed the body pitch to the LINE detector, but only "
                             "across the card window and --line-pitch-hold-s after it "
                             "(the stretch where the stop pose leaves the body off its "
                             "install angle). While walking the static mount angle is "
                             "kept, because the 1.7 Hz gait swing is not something a "
                             "low-passed pitch describes. Off by default")
    parser.add_argument("--line-pitch-hold-s", type=float,
                        default=float(os.getenv("LINE_PITCH_HOLD_S", "2.5")),
                        help="How long after the card window closes the live pitch "
                             "keeps going to the line detector. The body takes about "
                             "2 s to walk out of the leaning stand pose")
    parser.add_argument("--start-gate", choices=("off", "qr", "shape", "both"),
                        default="off",
                        help="Hold the robot at vx=wz=0 until the start valves pass. "
                             "'qr' = a QR code decoded to --start-gate-qr-payload; "
                             "'shape' = the first card classified to a shape on "
                             "--start-gate-shape-frames consecutive detection calls; "
                             "'both' = the competition setting. While held the body is "
                             "also asked to stand upright: the policy's own stopped "
                             "pose leans back about 20 deg and the card geometry is "
                             "calibrated at the install angle, so without that the "
                             "classifier rejects every card with rej=ground and the "
                             "gate never opens. Off by default, and a run without it "
                             "behaves exactly as before")
    parser.add_argument("--start-gate-qr-payload", default="1",
                        help="The QR payload that opens the first valve")
    parser.add_argument("--start-gate-shape-frames", type=int, default=2,
                        help="Consecutive detection calls naming the same shape "
                             "before the second valve latches")
    parser.add_argument("--qr-every", type=int, default=5,
                        help="Decode a QR every N frames while the first valve is "
                             "still waiting. The decoder is CPU-only and costs tens "
                             "of milliseconds; this is the knob to turn if the loop "
                             "feels slow while the robot waits at the start")
    parser.add_argument("--qr-max-side", type=int, default=1280,
                        help="Shrink the frame so its long side is at most this "
                             "before decoding; 0 disables. Stops a >720p camera from "
                             "paying for a 2x upscale of an already large frame")
    parser.add_argument("--qr-upscale", type=float, default=2.0,
                        help="Retry the decode on a LANCZOS4-upscaled frame at this "
                             "factor when the raw pass fails; 1 disables. Small or "
                             "distant codes need it")
    parser.add_argument("--qr-min-edge-px", type=float, default=15.0,
                        help="Reject a decoded code whose mean side is shorter than "
                             "this, in original-frame pixels")
    parser.add_argument("--qr-max-edge-px", type=float, default=450.0,
                        help="Reject a decoded code whose mean side is longer than "
                             "this, in original-frame pixels")
    parser.add_argument("--start-gate-log-s", type=float, default=1.0,
                        help="How often to print the start-gate status line while "
                             "the gate is still closed")
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
                        default=float(os.getenv("CARD_SLOW_VX", "0")),
                        help="Forward speed while a card is in view but not yet close "
                             "enough to act on, and the speed held for "
                             "--card-resume-ms after the stop. 0 (default) disables "
                             "both: the robot no longer creeps at a card and comes "
                             "back up to --vx in one step, which the connector's slew "
                             "still ramps. It used to be 0.2; at --vx 0.2 that was "
                             "already a no-op, and it only did anything at higher "
                             "speeds")
    parser.add_argument("--card-slow-wz", type=float,
                        default=float(os.getenv("CARD_SLOW_WZ", "0")),
                        help="Fixed yaw rate, rad/s, held for as long as a card is in "
                             "view and the robot is creeping toward it - the same "
                             "stretch --card-slow-vx covers. It replaces the line "
                             "controller's steering there; negative turns right. The "
                             "card is a fixed target rather than a lane, so lining up "
                             "on it beats following the line underneath, and lining up "
                             "is what keeps the stop square to the card. 0 (default) "
                             "leaves the controller alone; it used to be -0.2")
    parser.add_argument("--card-resume-ms", type=float,
                        default=float(os.getenv("CARD_RESUME_MS", "500")),
                        help="On the stopped-to-walking edge, hold --card-slow-vx this "
                             "long before releasing to full speed, so the start mirrors "
                             "the stop's 0.4 -> 0.2 -> 0 shape. 0 releases on the first "
                             "walking frame, leaving only the connector's slew. Inert "
                             "while --card-slow-vx is 0")
    parser.add_argument("--hold-upright", action="store_true",
                        help="Superseded by --card-tilt-ms and NOT used: this asks the "
                             "Nano to hold the legs straight through the stop, which "
                             "drives the same joints the STM32 re-pose drives. Turn on "
                             "exactly one of the two")
    parser.add_argument("--card-tilt-ms", type=float,
                        default=float(os.getenv("CARD_TILT_MS", "1800")),
                        help="After the stop trigger, spend this long not looking at "
                             "the card at all. The classifier does not run until it "
                             "expires, because a frame taken mid-tilt is a frame of a "
                             "body in motion. 0 identifies immediately. "
                             "It has to outlast two things: how long the robot takes "
                             "to settle - main.py only asks for the tilt once it reads "
                             "`stopped`, and asking earlier snapshots a mid-stride pose "
                             "- plus the STM32's lean ramp, which is a firmware "
                             "constant (20 deg at 0.6 rad/s = 0.58s). ~0.9 + 0.58 is "
                             "where 1800 comes from. "
                             "Too early and the classifier looks at a body that is "
                             "still leaning, while the geometry assumes the install "
                             "pose - every card gets rejected. Check --card-stop-ms "
                             "(default 3000) still covers this plus identification.")
    parser.add_argument("--card-cold-start", action="store_true",
                        help="On the stopped-to-walking edge, wipe the controller's loop "
                             "state and the detector's memory instead of keeping them. "
                             "Off by default: the robot stops the moment it has "
                             "recognised a card, which is a point where the detector was "
                             "tracking the line well, and keeping its estimate of where "
                             "the lane is beats restarting from the image middle. Turn "
                             "it on to break a lock that had already gone wrong")
    parser.add_argument("--card-vote-frames", type=int,
                        default=max(1, int(os.getenv("CARD_VOTE_FRAMES", "20"))),
                        help="Stopped, count every per-frame classification and let "
                             "the plurality decide the card. One frame is never "
                             "trustworthy - blurred, mid-gait, half out of frame - but "
                             "which shape wins out of 20 is. The result goes out as "
                             "soon as this many votes are in, and every stop gets an "
                             "answer: if --card-stop-ms runs out first the plurality "
                             "so far is used anyway. Needs the classifier to actually "
                             "get that many looks - it runs at ~10 Hz, so this wants "
                             "roughly 2s between --card-tilt-ms and --card-stop-ms")
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
                        default=max(1, int(os.getenv("SHAPE_EVERY", "2"))),
                        help="Run card detection every N frames. Measured at 1280x720: "
                             "31 ms with no card in view, 95-122 ms with one, against "
                             "29 ms for the line detector alone. The stop trigger is a "
                             "line on cy, so the period is an overshoot: at 3 Hz the "
                             "robot drives ~10 cm between samples and 2026-10-02 landed "
                             "at 17.5 cm instead of the 30 cm it was set for - close "
                             "enough that the card's bottom left the frame")
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
                        help="Loop-gain multiplier applied on a curve whenever the "
                             "near band's reading cannot be trusted: only one "
                             "boundary visible, OR the bottom lock invalid (off "
                             "centre past the symmetry tolerance / edges never "
                             "paired). 1.0 (default) leaves the gains exactly as "
                             "they are")
    parser.add_argument("--wz-mode", choices=("continuous", "discrete"),
                        default="continuous",
                        help="'continuous' (default) publishes the PID's yaw rate as "
                             "it always has. 'discrete' replaces only the published "
                             "wz with a short pulse from {0, +-step-lo, +-step-hi}: "
                             "zero until |err| crosses --wz-fire-cm, then one "
                             "--wz-pulse-s burst, then zero again. Straights come out "
                             "almost perfectly straight and curves become a polygon. "
                             "It only touches the output - SteeringController itself is "
                             "untouched, and a run without this flag is bit-for-bit "
                             "the old behaviour. Use connector --max-wz-accel 0 with "
                             "it, or the slew limiter turns each pulse into a triangle")
    parser.add_argument("--wz-fire-cm", type=float, default=5.0,
                        help="Dead band, cm: inside it the published wz is exactly "
                             "0 - no scaling, no half authority, straight. Below "
                             "this nothing happens at all. It is the one knob that "
                             "decides how straight a straight is, and the reason it "
                             "is 5 and not 3: the measured curve steady state is "
                             "+5~6 cm, so 3 had the robot pulsing almost "
                             "continuously even while it was basically on the line")
    parser.add_argument("--wz-fire-strong-cm", type=float, default=8.0,
                        help="|eff err| that upgrades the pulse to --wz-step-hi")
    # 0.4 / 0.5 是**实车跑出来的好值**。不要拿去跟 vx/R 之类的算术比然后"修正"
    # 它 —— 2026-10-03 试过一版按 vx/0.776 推的（--vx 0.2 下推成 0.258），
    # 推出来的数在车上是错的。这两个就是常数。
    parser.add_argument("--wz-step-lo", type=float, default=0.4,
                        help="Small pulse amplitude, rad/s. Measured good on the "
                             "robot; a fixed constant, not derived from --vx")
    parser.add_argument("--wz-step-hi", type=float, default=0.5,
                        help="Large pulse amplitude, rad/s; at most --max-wz")
    parser.add_argument("--wz-pulse-s", type=float, default=0.15,
                        help="How long one pulse lasts, seconds. Counted in vision "
                             "frames (16~29 Hz), so the real width is a whole number "
                             "of frames - 0.15 s is 2~4 of them")
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
    if not math.isfinite(args.card_slow_wz) or abs(args.card_slow_wz) > args.max_wz:
        parser.error("card-slow-wz must be finite and within +/-max-wz")
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
    if args.start_gate in ("shape", "both") and args.no_shape_detect:
        parser.error("start-gate shape/both needs shape detection")
    if args.start_gate in ("qr", "both") and not args.start_gate_qr_payload.strip():
        parser.error("start-gate-qr-payload must not be empty")
    if args.start_gate_shape_frames < 1:
        parser.error("start-gate-shape-frames must be at least 1")
    if args.qr_every < 1:
        parser.error("qr-every must be at least 1")
    if not math.isfinite(args.qr_upscale) or args.qr_upscale < 1:
        parser.error("qr-upscale must be finite and at least 1")
    if args.qr_max_side < 0:
        parser.error("qr-max-side must be nonnegative")
    if not (math.isfinite(args.qr_min_edge_px) and math.isfinite(args.qr_max_edge_px)
            and 0 < args.qr_min_edge_px < args.qr_max_edge_px):
        parser.error("need 0 < qr-min-edge-px < qr-max-edge-px")
    if not math.isfinite(args.start_gate_log_s) or args.start_gate_log_s <= 0:
        parser.error("start-gate-log-s must be positive")
    if not (all(math.isfinite(v) for v in (args.wz_fire_cm, args.wz_fire_strong_cm,
                                           args.wz_step_lo, args.wz_step_hi,
                                           args.wz_pulse_s))
            and 0 < args.wz_fire_cm < args.wz_fire_strong_cm
            and 0 < args.wz_step_lo <= args.wz_step_hi <= args.max_wz
            and args.wz_pulse_s > 0):
        parser.error("need 0 < wz-fire-cm < wz-fire-strong-cm, "
                     "0 < wz-step-lo <= wz-step-hi <= max-wz, and wz-pulse-s > 0")
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
    # 离散模式只在外面包一层，只换发出去的 wz —— SteeringController 一个字不改，
    # 关着这个开关时走的就是上面构造出来的那个对象本身。
    if args.wz_mode == "discrete":
        from discrete_steering import DiscreteSteeringController
        controller = DiscreteSteeringController(
            controller, fire_cm=args.wz_fire_cm, strong_cm=args.wz_fire_strong_cm,
            step_lo=args.wz_step_lo, step_hi=args.wz_step_hi,
            pulse_s=args.wz_pulse_s)
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
        shape_numbers = dict(shape.action_map)

    # 起跑门控。--help 依旧不碰 cv2：QrReader 只在真的要用二维码时才 import。
    start_gate = qr_reader = None
    if args.start_gate != "off":
        from start_gate import StartGate
        start_gate = StartGate(mode=args.start_gate,
                               expected_qr=args.start_gate_qr_payload,
                               shape_confirm=args.start_gate_shape_frames)
        if start_gate.require_qr:
            from qr_reader import QrReader
            qr_reader = QrReader(min_edge_px=args.qr_min_edge_px,
                                 max_edge_px=args.qr_max_edge_px,
                                 upscale=args.qr_upscale,
                                 max_side=args.qr_max_side)

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
    if args.line_pitch and attitude is None:
        # 这个组合会静默失效：effective_pitch 恒等于静态安装角，而
        # set_camera_pitch_deg(安装角) 是逐位 no-op —— 开关开着，什么也没发生。
        # 2026-10-02 台架就是这么白跑了一趟（日志里 pitch=45.0 一直不变）。
        print("[vision] ⚠️ --line-pitch 开着，但姿态收不到（见上一行）—— "
              "巡线会一直用静态安装角，这个开关等于没开", flush=True)

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
              f"line_pitch={'on' if args.line_pitch else 'off'}; "
              f"hold-still={'on' if args.hold_still else 'off'}; "
              f"gate={args.start_gate}; "
              f"wz_mode={args.wz_mode}; "
              f"center_dead={args.center_dead_cm}cm; "
              f"bias={args.bias_straight_cm}->{args.bias_cm}"
              f"(dead {args.bias_dead_px}, full {args.bias_gate_px})", flush=True)
        if attitude is not None:
            print(f"Body attitude: udp://{args.attitude_bind}:{args.attitude_port}"
                  f" tau={args.attitude_tau_s}s; 安装角 {args.camera_pitch_deg:.1f}°"
                  f" 会被机身俯仰实时修正", flush=True)
        if args.wz_mode == "discrete":
            print(f"[wz] 离散模式：wz 只会是 "
                  f"{{0, ±{args.wz_step_lo}, ±{args.wz_step_hi}}}，"
                  f"|eff| ≥ {args.wz_fire_cm}cm 打 {args.wz_step_lo}"
                  f"、≥ {args.wz_fire_strong_cm}cm 打 {args.wz_step_hi}，"
                  f"每次 {args.wz_pulse_s:.2f}s（约 {args.wz_pulse_s * 20:.0f} 帧）。"
                  f"触发看日志里的 eff=；--center-dead-cm / "
                  f"--steer-full-scale-cm / PID 增益在这个模式下不再影响输出。",
                  flush=True)
            print(f"[wz] ⚠️ A 那边要用 --max-wz-accel 0，否则脉冲会被它的斜率"
                  f"限制削成三角形（默认 2.0 时 0→{args.wz_step_lo} 要爬 "
                  f"{args.wz_step_lo / 2.0:.2f}s）", flush=True)
            print(f"[wz] 幅度写死 {args.wz_step_lo}/{args.wz_step_hi}（实车量出来的，"
                  f"不跟着 --vx 走）。参考：--vx {args.vx} 下跟住一个弯要 "
                  f"{args.vx / LANE_RADIUS_M:.3f} rad/s。触发看原始 err=；"
                  f"--bias-cm / --center-dead-cm / --steer-full-scale-cm / "
                  f"--preview-gain 在这个模式下都不参与", flush=True)
        if start_gate is not None:
            print(f"[start-gate] {args.start_gate}：站住不动，直到两个阀都过；"
                  f"期间机身按住直立（否则后仰 20°，几何闸会把每张卡都 "
                  f"rej=ground）。", flush=True)
            if start_gate.require_qr:
                print(f"             阀1 qr_every={args.qr_every} "
                      f"payload={args.start_gate_qr_payload!r} "
                      f"upscale={args.qr_upscale} max_side={args.qr_max_side}",
                      flush=True)
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
        line_pitch_until = 0.0       # 窗口关掉之后，还要继续喂巡线俯角到什么时候
        last_cmd_vx = 0.0            # 上一帧发出去的速度，判"车在不在走"
        tilt_until = 0.0
        card_flag = False        # a card is in view on this approach
        card_absent = 0          # consecutive detection calls without one
        card_triggered = False   # this card has already been acted on
        card_armed = True        # and it has since been seen far enough to trigger
        card_action_triggered = False
        card_dbg = {}
        # 这次停车的形状票：{形状名: 票数}。停车触发时清空，见下面。
        card_votes = {}
        card_vote_total = 0
        lateral_warned = None    # None until the lateral bound first trips
        card_window_open = False
        # 门控的"上一帧"状态。闸门关着时 window_open 为真，于是窗口关闭那套交接
        # （drop_held_command / resume_until / line_pitch_until）被原样复用，
        # 释放的那一帧走的就是停车窗口关闭走过的同一条路。
        gate_window_open = start_gate is not None
        gate_released = False
        gate_last_log = -math.inf
        qr_odd_seen = set()
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
                client.publish(0.0, 0.0, -1, hold_upright=gate_window_open, **event)
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
            # 起跑门控关着 = 车还不许走，它算进 window_open —— 于是释放后的交接
            # 自动走下面这段"窗口关闭"，一行都不用另写。
            stop_window = processed < stop_until or processed < card_until
            window_open = stop_window or gate_window_open
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
                # 机身从重摆姿态走回站姿要 ~2s，这段巡线也得跟着俯角走，
                # 否则恢复以后那几秒的几何是错的（--line-pitch）。
                line_pitch_until = processed + args.line_pitch_hold_s
                print(f"[vision] {'start gate released' if gate_released else 'card window closed'}; "
                      f"{'cold start' if args.card_cold_start else 'resuming frozen'} "
                      f"(hold={controller.hold[0]:+.2f},{controller.hold[1]:+.2f} "
                      f"lost_s={controller.lost_s:.2f}) "
                      + (f"vx<={args.card_slow_vx:+.2f} for "
                         f"{args.card_resume_ms:.0f}ms"
                         if args.card_slow_vx > 0.0 else
                         "vx 不压（--card-slow-vx 0），直接回 --vx"),
                      flush=True)
                gate_released = False   # 一次性：用掉了就清，别让后面的卡窗口顶着这句话
            card_window_open = window_open
            effective_pitch = args.camera_pitch_deg
            if attitude is not None:
                attitude.poll()
                # The STM32 freezes the attitude it reports while it re-poses the body,
                # so the policy does not see the offset being applied. That frozen value
                # is not where the camera is: the re-pose puts the body back on its
                # install pose, which the static config already describes. Feeding the
                # frozen value to the card geometry reintroduces exactly the failure the
                # re-pose exists to fix -- measured at the 43 cm trigger distance, the
                # ground-square gate accepts an assumed pitch of [38.6, 59.0] deg and
                # the frozen 25 deg is far outside it, so the card is rejected either
                # way. Gated on the window rather than the frame: the classifier does
                # not run at all until --card-tilt-ms expires, and by then the low-pass
                # has long since left the window.
                #
                # stop_window 而不是 window_open：冻结只在 card_tilt 重摆期间发生，
                # 而重摆只属于停车窗口。起跑门控期间机身是被 hold_upright 扳直的、
                # 姿态广播是活的，那几秒必须喂实时值 —— 机身还在从后仰走回直立的
                # 路上，拿静态 45° 去算几何就是拿错假设分类。
                if not (stop_window and card_event_id == 0):
                    effective_pitch = attitude.value
            if shape is not None:
                shape.set_camera_pitch_deg(effective_pitch)
            if args.line_pitch:
                # 巡线也吃俯角，条件是"车没在走"：停车窗口、窗口关掉之后的
                # --line-pitch-hold-s（机身要走回站姿）、以及任何发出去的
                # vx<=0 的帧（台架 --hold-still 就落在这一档）。
                # 走路时仍旧回静态安装角 —— 步态以 1.7Hz 摆 30~40°，低通的值
                # 描述不了当前这一帧，这是原注释，针对的正是走路那一段。
                #
                # 一开始只写了"窗口 + hold"，结果台架测不到：台架上没有卡，
                # 窗口从没开过，于是每帧都喂静态安装角 = no-op。
                detector.set_camera_pitch_deg(
                    effective_pitch
                    if (window_open or processed < line_pitch_until
                        or last_cmd_vx <= 0.0)
                    else args.camera_pitch_deg)
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
            # on the first cy at or above the trigger line, so the period is the whole
            # overshoot: at 6 frames the line was sampled every ~0.3 s - most of a 12 cm
            # step at 0.4 m/s - and the robot sailed past 43 cm to ~22 cm, close enough
            # that the card's bottom left the frame and no quad could close. The fine
            # reading has to exist before the decision, not after it.
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
            # 每帧清空：下面投票那段在检测块外面，读到上一帧的 card_dbg 就会拿同一次
            # 检测投两次票（--card-every-stopped 2 时票数正好翻倍）。
            card_dbg = None
            # ── 阀1：二维码 ──
            # 只在还没扫到时跑，而且每 --qr-every 帧才解一次：detectAndDecode 是纯
            # CPU 的，1280x720 raw 十几毫秒、再放大一倍几十毫秒，逐帧跑会把主循环
            # 拖垮。扫到就锁存，之后一次都不再进来 —— 行进段一分钱都不付。
            if (qr_reader is not None and not start_gate.qr_passed
                    and frames % args.qr_every == 0):
                reading = qr_reader.decode(frame)
                if reading is not None:
                    if start_gate.observe_qr(reading.payload):
                        print("\n" + "=" * 68, flush=True)
                        print(f"  ◆◆◆  阀1 通过：二维码 payload={reading.payload!r}"
                              f"    {reading.strategy} 边长 {reading.edge_px:.0f}px"
                              f"    解码 {reading.cost_ms:.0f}ms", flush=True)
                        print("=" * 68 + "\n", flush=True)
                    elif reading.payload not in qr_odd_seen:
                        # 扫到了但不是要的那个：让它可见。场上还有别的码、或者
                        # 规则换了 payload，两种情况看到这一行就知道该改什么。
                        qr_odd_seen.add(reading.payload)
                        print(f"[start-gate] ⚠️ 扫到 payload={reading.payload!r}，"
                              f"不是 {start_gate.expected_qr!r}，忽略", flush=True)
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
                # 动作不在这里出 —— 单帧确认就发是旧路子，会在一张卡很远的时候就
                # 开火（50cm、top=186、cy=0.40 那次）。现在停车之后投票，见下面。
                # update() 返回的 action 受 cooldown_ms 限制、一次停车最多给一次，
                # 投票用不上它（票从 card_dbg["shape"] 来）。
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
                        # 卡走了，票也跟着作废。不清的话：窗口关闭后
                        # card_event_id 归零、这里又把 card_action_triggered 归零，
                        # 兜底分支就会拿着上一批旧票再投一次 —— 实机是刚起步又停下。
                        card_votes = {}
                        card_vote_total = 0
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
                # 门控期间不许开火。card_armed 的初值是 True，而起点那张卡本来就
                # 在触发线以内（--card-trigger-dist-cm 43，卡在 40cm 内），不按住
                # 的话机器人还没起步就会停车、重摆、出 event。只动 card_armed ——
                # 下面那段判据一个字不改，卡的清理路径也没变。
                if gate_window_open:
                    card_armed = False
                if (card_flag and not card_triggered and card_armed
                        and card_reach is not None
                        and card_reach >= card_reach_line):
                    card_triggered = True
                    card_armed = False
                    stop_until = processed + args.card_stop_ms / 1000.0
                    tilt_until = processed + args.card_tilt_ms / 1000.0
                    card_votes = {}
                    card_vote_total = 0
                    # 和下面"识别到图卡"那条配成一对：这两个时刻是整趟里唯一需要
                    # 肉眼确认的，中间重摆那一秒多什么都不会打印，所以它们要能
                    # 从刷屏里一眼捞出来。
                    if card_width_px is not None:
                        how = (f"框宽 {card_reach:.0f}px / 阈值 "
                               f"{card_reach_line:.0f}px")
                    else:
                        how = (f"没框，退回 cy {card_reach:.2f} / 阈值 "
                               f"{card_reach_line:.2f}")
                    print("\n" + "=" * 68, flush=True)
                    print(f"  ●●●  停车读卡 —— stand still {args.card_stop_ms:.0f}ms"
                          f"    {how}"
                          f"    cy={fmt(card_dbg.get('presence_cy_frac'), '.2f')}",
                          flush=True)
                    print(f"        先重摆 {args.card_tilt_ms:.0f}ms（这期间不认形状），"
                          f"之后才开始识别", flush=True)
                    print("=" * 68 + "\n", flush=True)
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
            # ── 阀2：认出第一张图卡的形状 ──
            # 放在停车触发之后是故意的：释放那一帧仍然算在门控里（card_armed 这一帧
            # 刚被按住），下一帧起 card_armed 才是真的可以开火 —— 避免"开门与开火
            # 同帧"这种时序上说不清的状态。
            if start_gate is not None:
                if (start_gate.require_shape and not start_gate.shape_passed
                        and card_dbg and card_dbg.get("shape")):
                    if start_gate.observe_shape(card_dbg["shape"]):
                        print("\n" + "=" * 68, flush=True)
                        print(f"  ◆◆◆  阀2 通过：图卡 "
                              f"{CARD_NAMES_ZH.get(start_gate.last_shape, '?')}"
                              f"（{start_gate.last_shape}）"
                              f"    连续 {start_gate.shape_streak} 帧", flush=True)
                        print("=" * 68 + "\n", flush=True)
                if start_gate.passed and gate_window_open:
                    gate_released = True
                    card_armed = True
                    if card_width_px is not None:
                        print(f"[start-gate] 起步：第一张卡实测框宽 {card_width_px:.0f}px "
                              f"/ 停车线 {card_reach_line:.0f}px —— "
                              f"{'已经在线内，会立刻进停车读卡' if card_width_px >= card_reach_line else '还在线外，会走一段再停'}",
                              flush=True)
                    else:
                        print("[start-gate] 起步：这一帧没有框，停车线由 cy 判",
                              flush=True)
            # ── 停车投票 ──
            # 停车窗口里，每一帧的分类结果投一票。用 card_dbg["shape"]（每帧都写），
            # 不是 update() 返回的 action —— 那个受 cooldown_ms 限制，一次停车最多
            # 给一次，投不了票。计数放在停车触发之后：触发那一帧会清票，先投会被
            # 它抹掉。
            #
            # 已经判过的卡不再进票箱。这一条是让"不重复"在**本层自己成立**：出票
            # 分支靠 card_action_triggered / card_event_id 挡，清票靠卡离开那一段，
            # 三处只要有一处没跟上就会拿着残留的票再定一次案。判过的卡连票都不该
            # 收，就不需要依赖那三处同步。
            if (processed < stop_until and not card_action_triggered
                    and card_dbg and card_dbg.get("shape")):
                _name = card_dbg["shape"]
                card_votes[_name] = card_votes.get(_name, 0) + 1
                card_vote_total += 1

            # 票够了就出；停车的预算（--card-stop-ms）用完还没够，也按手上的多数
            # 票出 —— **每次停车必须给出一个**。出完 card_until 会把窗口接上，
            # 机器人继续停着等动作。一票都没有才什么都不出（那次确实没看到卡）。
            # 并列时按名字排序取第一个，结果可复现。
            if (card_votes and not card_action_triggered and card_event_id == 0
                    and (card_vote_total >= args.card_vote_frames
                         or processed >= stop_until)):
                winner = max(sorted(card_votes), key=card_votes.get)
                card_action = shape_numbers[winner]
                card_event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                card_until = processed + args.card_hold_ms / 1000.0
                card_action_triggered = True
                recognized_this_frame = True
                tally = " ".join(f"{n}:{card_votes[n]}" for n in
                                 sorted(card_votes, key=card_votes.get, reverse=True))
                votes_cast = card_vote_total
                # 出完就清票：同一批票只能定一次案。
                card_votes = {}
                card_vote_total = 0
                print("\n" + "=" * 68, flush=True)
                print(f"  ★★★  识别到图卡（投票）："
                      f"{CARD_NAMES_ZH.get(winner, '?')}（{winner}）  "
                      f"qr={card_action}", flush=True)
                print(f"        票 {votes_cast} 张 / 要求 {args.card_vote_frames}"
                      f"    {tally}"
                      f"    保持 {args.card_hold_ms:.0f} ms", flush=True)
                print("=" * 68 + "\n", flush=True)

            if card_action != -1 and processed >= card_until:
                card_action = -1
                card_event_id = 0
            # Two windows, whichever ends later: --card-stop-ms caps how long we wait
            # for a shape that may never settle, --card-hold-ms is the rules' action
            # window once we do know it. 就地重读，不用顶上那个 window_open：窗口
            # 可能就是这一帧的触发开的，它要的"站住"从这一帧算起。
            # 起跑门控同一档 —— 它也不该把 controller 叫起来：站着等十分钟，
            # PID 的积分会拿一段看不见的误差把自己喂饱，起步第一帧就用它去转。
            if (processed < stop_until or processed < card_until
                    or gate_window_open):
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
                # --card-slow-vx 0 = 整档关掉（看见卡不减速、停车后也不留缓冲段）。
                # 默认就是 0：在 --vx 0.2 下减速本来等于没减，留着只会在以后提速时
                # 突然生效。
                if args.card_slow_vx > 0.0 and (card_flag or processed < resume_until):
                    vx = min(vx, args.card_slow_vx)
                # 往卡走的那一段，转向不再跟线：卡是个固定目标，对着它对准比跟着底下
                # 的线走更能停正。**只在还没为这张卡停下之前压** —— 停下之后（以及
                # 起步那段，resume_until）卡往往还在画面里，那时候再压就是刚起步往
                # 一边偏了。card_triggered 落地时清零，所以换一张卡会重新压上。
                if card_flag and not card_triggered and args.card_slow_wz != 0.0:
                    wz = args.card_slow_wz
            previous = processed
            visible_qr = card_action if recognized_this_frame else -1
            event = ({"event_id": card_event_id, "event_action": card_action}
                     if card_event_id else {})
            # Re-read here rather than reusing the top of the loop: the window can be
            # opened by this very frame's trigger, and the standstill it asks for
            # starts now, not on the next one.
            # `card_tilt` is the mechanism in use: the STM32 re-poses the body and
            # freezes the attitude it reports. `hold_upright` is the superseded one and
            # is off unless asked for - both drive the same joints.
            #
            # The re-pose is only for READING the card. Once the shape is named the
            # body goes back and the action runs on truthful attitude again, so the
            # fall comes off at the event, not at the end of the window:
            #   tilt -> identify -> untilt -> act
            # card_event_id, not card_action_triggered: the latter is re-armed when
            # the card finally leaves the frame, which would tilt a second time.
            in_card_window = processed < stop_until or processed < card_until
            # --hold-still：只观测不驱动。台架测"机身姿态对读数的影响"时要开
            # C（姿态是 C 从 STM32 读了广播的），但 C 一使能电机、B 这边一发
            # vx=0.3 车就走了。这里把命令压成 0 —— 检测、日志、图卡那套逻辑
            # 全都照跑，只有"发出去的 vx/wz"是零，于是策略原地站着（也就是
            # 那个后仰的站姿），读数还能正常观察。
            if args.hold_still:
                vx, wz = 0.0, 0.0
            last_cmd_vx = vx
            # 门控期间按住直立：策略自己的站姿后仰约 20°，而相机 45° 是在直立时
            # 标定的，几何闸只认 38.6~59° —— 不扳直，阀2 会把每一张卡都拒掉，机器人
            # 永远不走。释放那帧仍然按着（gate_window_open 是上一帧的值），下一帧
            # 才落，card_tilt 那条线上不会和它撞在同一帧。
            client.publish(vx, wz, visible_qr,
                           hold_upright=((args.hold_upright and in_card_window)
                                         or (gate_window_open
                                             and start_gate is not None
                                             and start_gate.require_shape)),
                           card_tilt=(in_card_window and card_event_id == 0),
                           **event)
            gate_window_open = start_gate is not None and not start_gate.passed
            if start_gate is not None and processed - gate_last_log >= args.start_gate_log_s:
                gate_last_log = processed
                scans = qr_reader.scans if qr_reader is not None else 0
                rej = qr_reader.geom_rejects if qr_reader is not None else 0
                cost = qr_reader.last_cost_ms if qr_reader is not None else 0.0
                print(f"[start-gate] {start_gate.status()} | 扫 {scans} 次/"
                      f"{rej} 拒 单次 {cost:.0f}ms | 车=站着不动", flush=True)
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
                    f"pitch={effective_pitch:.1f}"
                    # 重摆期间这两个不一样，而 (body …) 就是固件冻的那个值 ——
                    # 它会一直不变，所以这一行也是"锁存到底什么时候发生"唯一
                    # 看得见的地方（调 CARD_TILT_SETTLE_MAX_MS 靠它）。
                    + (f"(body {attitude.value:.1f})"
                       if attitude is not None and effective_pitch != attitude.value
                       else ""),
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
