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
import hashlib
import json
import math
import os
import signal
import sys
import time
import zlib

import numpy as np

from camera_config import load as load_camera
from policy_bridge import ConnectorClient, SteeringController
from line_telemetry import LineTelemetry, execution_log_fields
from command_video import line_lost, recording_frame
from steering_command_window import SteeringCommandWindow, effective_policy_hold
from start_gate import resolve_card_shapes
from card_flow import CardFlow
from recording_options import LOG_ITEMS, parse_log_items
from performance_telemetry import RuntimeCounters, environment_metadata

# 图卡的中文名，只给日志用 —— 操作员看日志时认的是图形，不是 "pentagon"。
CARD_NAMES_ZH = {
    "circle": "圆形", "pentagon": "五角星", "square": "正方形",
    "diamond": "菱形", "cross": "十字形", "triangle": "三角形",
}

# 赛道中线半径（m），文档 §3。只用来在启动横幅里打一行参考："跟住一个弯需要
# 多少角速度"（ω = vx / R）。**脉冲幅度不按它推** —— 0.4/0.5 是实车量出来的好值。
LANE_RADIUS_M = 0.776
STARTUP_CARD_RETRIGGER_GAP_S = 3.0

# 固定曝光→自动曝光标定结果（fit_photometric_mapping.py 的导出目录）。
_DEFAULT_LUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "photometric_mapping"))


def card_action_stop_s(action, hold_ms):
    """Allow untilt plus the 5.3-second STM32 leg action before walking."""
    return max(6.0 if action in (3, 4) else 3.0, hold_ms / 1000.0)


def ordinary_card_feedback(attitude, now):
    """Use fresh policy acknowledgments; old pose-only publishers still work."""
    if attitude is None:
        return {}
    supported = getattr(attitude, 'card_tilt_status_seen', False) is True
    result = dict(pose_supported=supported)
    stamp = getattr(attitude, 'last_packet_s', None)
    if type(stamp) in (int, float) and -.01 <= now-stamp <= .5:
        result.update(tilt_id=attitude.card_tilt_event_id,
                      tilt_done=attitude.card_tilt_done)
    task = attitude.card_task(now)
    if isinstance(task, dict):
        result.update(task)
    return result


def _new_dump_run(path, run_id, metadata):
    """Keep each run and its actual command together without removing older data."""
    directory = os.path.join(path, run_id)
    os.makedirs(directory, exist_ok=True)
    manifest = os.path.join(directory, "run_manifest.json")
    # Both dump flags may point to the same root. Their file names differ, and
    # their shared manifest is written once; existing evidence is never replaced.
    try:
        with open(manifest, "x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except FileExistsError:
        pass
    return directory


def parse_args():
    camera = load_camera()
    parser = argparse.ArgumentParser(description=__doc__)
    from camera_controls import add_arguments as add_camera_arguments, validate_args as validate_camera_args
    add_camera_arguments(parser)
    parser.add_argument('--recording-root', default='',
                        help='Group all test recordings under ROOT/number using persistent test_counter.txt; no clock wait. Overrides individual dump paths; '
                             'button mode creates recordings only after the STM32 start button is received.')
    parser.add_argument('--log-items', type=parse_log_items, default=None,
                        help='LOG selection: space/comma separated vision control video motor policy shape loss; '
                             'all or none. When set, this also controls video instead of --record-video.')
    parser.add_argument('--record-video', action=argparse.BooleanOptionalAction, default=True,
                        help='Record selected video-source with command arrows inside recording-root; red on line loss.')
    parser.add_argument('--video-source', choices=('bird_pair', 'binary', 'camera'), default='bird_pair',
                        help='Recorded image: paired BGR birdseye + final binary mask (default), binary only, or original camera')
    parser.add_argument('--video-fps', type=float, default=10.,
                        help='Recorded playback FPS, 1..30; repeats samples by timestamp, not camera processing rate.')
    parser.add_argument('--video-width', type=int, default=960,
                        help='Maximum recorded width, 64..1920; does not change detector resolution.')
    parser.add_argument('--camera-async', action=argparse.BooleanOptionalAction, default=True,
                        help='Capture continuously in a worker; process only the latest frame')
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAM_IDX", camera["index"])))
    parser.add_argument("--card-camera", type=int, default=-1,
                        help="Optional second /dev/videoN for stopped card classification only; disabled by default, set N to restore")
    parser.add_argument("--camera-device", default=None, help="Stable main capture path, e.g. /dev/v4l/by-path/...-video-index0")
    parser.add_argument("--card-camera-device", default=None, help="Stable stopped-card capture path; overrides card camera index")
    parser.add_argument("--card-camera-width", type=int, default=1280)
    parser.add_argument("--card-camera-height", type=int, default=720)
    parser.add_argument("--card-camera-height-cm", type=float, default=None,
                        help="Second camera mount height; defaults to --camera-height-cm")
    parser.add_argument("--card-camera-pitch-deg", type=float, default=None,
                        help="Second camera optical downward pitch; defaults to --camera-pitch-deg")
    parser.add_argument("--card-camera-vfov-deg", type=float, default=None,
                        help="Second camera vertical field of view; defaults to --camera-vfov-deg")
    parser.add_argument("--card-camera-photometric-mode",
                        choices=("raw", "threshold", "match", "normalize", "legacy"),
                        default="raw", help="Photometry for stopped-card camera only; primary-camera LUT is not shared")
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
    parser.add_argument("--vx", type=float, default=0.2,
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
    parser.add_argument("--start-gate", choices=("off", "qr", "shape", "both", "button"),
                        default="off",
                        help="Hold the robot at vx=wz=0 until the start valves pass. "
                             "'qr' = a QR code decoded to --start-gate-qr-payload; "
                             "'shape' = the first card classified on consecutive calls; "
                             "'both' = the competition setting, voting on recent card "
                             "readings and locking the winner only after QR; 'button' = STM32 PC2 "
                             "starts gait, with optional first-card PA3 indication. "
                             "In button mode gait is not launched while waiting. "
                             "In the other gate modes, while held the body is "
                             "also asked to stand upright: the policy's own stopped "
                             "pose leans back about 20 deg and the card geometry is "
                             "calibrated at the install angle, so without that the "
                             "classifier rejects every card with rej=ground and the "
                             "gate never opens. Off by default, and a run without it "
                             "behaves exactly as before")
    parser.add_argument("--start-gate-qr-payload", default="1",
                        help="The QR payload that opens the first valve")
    parser.add_argument("--start-gate-shape-frames", type=int, default=2,
                        help="Consecutive same-shape calls for --start-gate shape; "
                             "competition --start-gate both uses votes instead")
    parser.add_argument("--start-gate-shape-votes", type=int, default=8,
                        help="Minimum classified calls in the recent start-card vote "
                             "window; with --start-gate both, decide only after QR")
    parser.add_argument("--start-gate-vote-window", type=int, default=24,
                        help="Maximum recent classified calls kept for the first-card vote")
    parser.add_argument("--start-policy-on-gate", action="store_true",
                        help="After the both/button start gate passes, launch the enabled walking "
                             "policy and wait for its STM32-ready marker before moving")
    parser.add_argument("--start-policy-model",
                        default="humanoid_jetson_deploy/policy_49_max.onnx",
                        help="Walking ONNX model for the automatically started policy")
    parser.add_argument("--start-policy-one-foot-model",
                        default="humanoid_jetson_deploy/policy-one-foot-standing_old.onnx",
                        help="Legacy ignored option; vision leg cards run on STM32")
    parser.add_argument("--start-policy-python", default=None,
                        help="Python executable for the walking policy; defaults to "
                             "the repository .venv/bin/python when available")
    parser.add_argument("--start-policy-port", default="/dev/ttyACM0")
    parser.add_argument("--start-policy-max-seconds", type=float, default=1200.0)
    parser.add_argument('--startup-first-walk-s', type=float, default=.5,
                        help='Initial straight walk before executing the latched first card; finite >0 seconds.')
    parser.add_argument('--startup-sequence', default='',
                        help='Optional JSON array of duration_s/vx/wz steps before first card action; '
                             'overrides startup-first-walk-s; empty uses a single straight step.')
    parser.add_argument('--command-min-hold-s', type=float, default=0.0,
                        help='Minimum actual walking command duration in launched policy; '
                             'heading/segments use at least steering-command-window-s. Stops preempt.')
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
    parser.add_argument("--line-log-dir", default="",
                        help="Optional per-frame scalar measurement and command logs; "
                             "each run gets a new directory and source manifest")
    parser.add_argument("--dump-on-loss", default="",
                        help="Directory to write the frames around a bottom-lock "
                             "drop-out or a confidence collapse into. Empty is off. "
                             "Each run gets a new subdirectory and command manifest. "
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
                             "Each run gets a new subdirectory; old runs are kept. "
                             "Needs a card to be present, so a lap writes tens of pairs, "
                             "not thousands")
    parser.add_argument("--no-red-detect", action="store_true",
                        help="Ignore red completely: no red bar, no narrow gate, and "
                             "a red row no longer blocks the band scan or the bottom "
                             "lock. Temporary, for isolating red's effect on line "
                             "following")
    parser.add_argument("--card-hold-ms", type=float,
                        default=float(os.getenv("CARD_HOLD_MS", "3000")),
                        help="Minimum stop after an arm/head card event (default 3000 ms). "
                             "Leg cards run on STM32 and return to standing before "
                             "walking resumes")
    parser.add_argument("--card-stop-ms", type=float,
                        default=float(os.getenv("CARD_STOP_MS", "3000")),
                        help="Bounded wait for STM32 re-pose start from the proximity stop, "
                             "then a fresh timeout of the same duration from the first "
                             "observed active event. After DONE plus --card-settle-ms, "
                             "a separate fixed one-second vote window begins. "
                             "0 makes the feedback wait expire immediately")
    parser.add_argument("--card-trigger-frac", type=float, default=None,
                        help="Explicit override: confirmed cue centroid height in the frame "
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
                             "earlier does not change walking commands")
    parser.add_argument("--card-tilt-ms", type=float,
                        default=float(os.getenv("CARD_TILT_MS", "1800")),
                        help="Legacy wait after stop before reading a card, used only "
                             "when the attitude broadcaster does not report STM32 "
                             "re-pose status. Current deployments wait for DONE instead.")
    parser.add_argument("--card-settle-ms", type=float,
                        default=float(os.getenv("CARD_SETTLE_MS", "100")),
                        help="Wait this long after STM32 confirms card re-pose DONE "
                             "before taking the first shape vote. With an older "
                             "attitude broadcaster, add it to --card-tilt-ms instead.")
    parser.add_argument("--card-cold-start", action="store_true",
                        help="On the stopped-to-walking edge, wipe the controller's loop "
                             "state and the detector's memory instead of keeping them. "
                             "Off by default: the robot stops the moment it has "
                             "recognised a card, which is a point where the detector was "
                             "tracking the line well, and keeping its estimate of where "
                             "the lane is beats restarting from the image middle. Turn "
                             "it on to break a lock that had already gone wrong")
    parser.add_argument("--stop-cool-down", type=float,
                        default=float(os.getenv("STOP_COOL_DOWN", "5")),
                        help="Successful-card cooldown in seconds, starting when normal walking resumes")
    parser.add_argument("--card-failure-cooldown-s", type=float,
                        default=float(os.getenv("CARD_FAILURE_COOLDOWN_S", "0.5")),
                        help="Seconds after failed recognition returns to normal walking")
    parser.add_argument("--card-every-stopped", type=int,
                        default=max(1, int(os.getenv("CARD_EVERY_STOPPED", "2"))),
                        help="Sampling during the unchanged startup-card action only; "
                             "ordinary cards detect/classify every frame")
    parser.add_argument("--shape-proposal-mode", choices=("all", "staged"),
                        default="all",
                        help="Stopped-card box proposals. 'staged' tries component "
                             "and ring candidates first, then falls back to line/corner "
                             "proposals if no box verifies. Walking presence uses only cue detection.")
    parser.add_argument("--card-clear-calls", type=int,
                        default=max(1, int(os.getenv("CARD_CLEAR_CALLS", "4"))),
                        help="First startup card departure confirmation calls only")
    parser.add_argument("--shape-every", type=int,
                        default=max(1, int(os.getenv("SHAPE_EVERY", "2"))),
                        help="Startup card sampling period; ordinary walking cards run every frame")
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
    parser.add_argument("--wz-mode", choices=("heading", "segments", "continuous", "discrete"),
                        default="heading",
                        help="heading (default): ground heading + near offset, variable "
                             "wz follows accepted observations without a minimum hold "
                             "in this experiment. segments: measured lane target "
                             "with heading fallback and comparison telemetry; automatically "
                             "enables lane-fit-segments. discrete: legacy error-only "
                             "bursts. continuous: legacy PID. Use this branch's C, "
                             "which also removes the normal minimum hold")
    parser.add_argument("--heading-lookahead-cm", type=float, default=50.0,
                        help="heading mode: forward target plane, cm; beyond near band")
    parser.add_argument("--heading-far-cm", type=float,
                        default=os.getenv("HEADING_FAR_CM", "29"),
                        help="heading/segments: far observation centre in ground cm "
                             "(default 29; HEADING_FAR_CM); distinct from projected lookahead")
    parser.add_argument('--heading-near-cm', type=float,
                        default=os.getenv('HEADING_NEAR_CM') or None,
                        help='near observation centre in ground cm; unset keeps legacy ~25.07cm rows')
    parser.add_argument('--heading-regions-cm', default=os.getenv('HEADING_REGIONS_CM', ''),
                        help='JSON [[near_min,near_max],[far_min,far_max]] in ground cm; '
                             'eight rows inside each interval; overrides near/far centre settings')
    parser.add_argument('--steering-angle-wz-table', default=os.getenv('STEERING_ANGLE_WZ_TABLE', ''),
                        help='JSON [[lo_deg,hi_deg,signed_wz_rad_s],...]; signed final decision angle, '
                             '[lo,hi) bins covering -90 to +90; yaw-sign maps wire direction; empty uses legacy mapping')
    parser.add_argument('--segment-regions-cm', default=os.getenv('SEGMENT_REGIONS_CM') or '[[20,32],[32,44],[44,56]]',
                        help='JSON ground distance regions; 2–5 contiguous bins >=6cm within 20–70cm')
    parser.add_argument("--heading-right-tolerance-deg", type=float, default=12.0,
                        help="heading mode: positive target-bearing dead zone, degrees; "
                             "tolerates body pointing right before left correction")
    parser.add_argument("--heading-left-tolerance-deg", type=float, default=4.0,
                        help="heading mode: negative target-bearing dead zone, degrees")
    parser.add_argument("--heading-corridor-cm", type=float, default=8.0,
                        help="heading mode: near/forward lateral corridor; independent of angular tolerance")
    parser.add_argument("--heading-full-scale-deg", type=float, default=20.0,
                        help="heading mode: degrees beyond gate for full demand before level selection; "
                             "legacy PID/fire/stop/turn/gap settings do not apply")
    parser.add_argument("--heading-left-wz", type=float, nargs="+",
                        default=[0.37, 0.43, 0.5], metavar="WZ",
                        help="heading mode: left-turn levels, positive magnitudes in "
                             "increasing order; the smallest follows a bend, larger "
                             "ones escalate. Default 0.37 0.43 0.5")
    parser.add_argument("--heading-right-wz", type=float, nargs="+",
                        default=[0.3, 0.5], metavar="WZ",
                        help="heading mode: right-turn levels, positive magnitudes in "
                             "increasing order, published negated. Default 0.3 0.5")
    parser.add_argument("--heading-straight-wz", type=float, default=0.0,
                        help="heading mode: the wz published when no correction is "
                             "needed; must stay below the smallest turn level")
    parser.add_argument('--steering-position-blend-weight', type=float,
                        default=os.getenv('STEERING_POSITION_BLEND_WEIGHT', '0.3'),
                        help='Active heading angle table: C weight in (1-w)*F+w*C; finite [0,1]')
    for name, default in (('gain', 1.), ('dead-cm', 2.), ('lookahead-cm', 50.), ('max-deg', 12.),
                          ('recovery-cm', 8.), ('recovery-full-scale-cm', 12.)):
        parser.add_argument('--position-'+name, type=float, default=default,
                            help='heading/segments independent near-position '+name)
    parser.add_argument('--position-confirm-frames', type=int, default=2,
                        help='Fresh same-side frames before priority near-position recovery')
    parser.add_argument("--wz-fire-cm", type=float, default=5.0,
                        help="Dead band, cm: inside it the published wz is exactly "
                             "0 - no scaling, no half authority, straight. Below "
                             "this nothing happens at all. It is the one knob that "
                             "decides how straight a straight is, and the reason it "
                             "is 5 and not 3: the measured curve steady state is "
                             "+5~6 cm, so 3 had the robot pulsing almost "
                             "continuously even while it was basically on the line")
    parser.add_argument("--wz-stop-cm", type=float, default=None,
                        help="Where a running turn ends, cm. Signed, no range "
                             "limit: positive = on the OPPOSITE side (a left turn "
                             "runs until err reaches -stop-cm, mirrored for a right "
                             "turn); 0 = the moment err crosses the centre; "
                             "negative = stop early on the SAME side (err back "
                             "inside |stop-cm|). Unset = --wz-fire-cm, so the stop "
                             "line is the mirror of the fire line: +4.5 starts it, "
                             "-4.5 ends it. Added 2026-10-03 in two steps: the turn "
                             "used to be a fixed 2.5 s burst nothing could cut; then "
                             "a same-side stop cut it at 2 cm, the robot came out of "
                             "every turn still right of centre, gathered new right "
                             "error on the straight and left the track on the right")
    # 0.4 / 0.5 是**实车跑出来的好值**，而且挑的是"大且稳"那一端：关节在小角度
    # 上表现得比大角度还不稳，所以不能拿"刚好够用"的小量加精确时长去凑。
    # 不要拿去跟 vx/R 之类的算术比然后"修正"它 —— 2026-10-03 试过一版按
    # vx/0.776 推的（--vx 0.2 下推成 0.258），推出来的数在车上是错的。
    parser.add_argument("--wz-step", type=float, default=0.5,
                        help="How hard one turn is, rad/s, at most --max-wz. ONE "
                             "value: there is no second gear keyed off how big the "
                             "error is. Measured good on the robot; a fixed "
                             "constant, not derived from --vx")
    parser.add_argument("--wz-turn-s", type=float, default=1.0,
                        help="How long one turn lasts at most, seconds: a cap, not "
                             "a length (--wz-stop-cm can end it early). It has to be "
                             "long enough for the machine to actually carry it out: "
                             "at the 3:8 step rate a 0.15 s command is two or three "
                             "steps, over before the robot has acted on it, so the "
                             "same command turns a different amount every time")
    parser.add_argument("--wz-gap-s", type=float, default=2.5,
                        help="Coast forced after every turn, seconds. Without it a "
                             "turn that ends with |err| still over the threshold "
                             "re-fires on the very next frame, the turns run "
                             "together and the curve is a continuous turn instead "
                             "of the polygon")
    parser.add_argument("--wz-allow-right", action="store_true",
                        help="Enable right turns in legacy discrete mode. "
                             "Heading mode always enables asymmetric left/right levels")
    parser.add_argument("--anticipation-clip", type=float,
                        default=float(os.getenv("ANTICIPATION_CLIP", "0.5")),
                        help="Legacy option retained for command compatibility. "
                             "P1 uses a quality-gated geometric preview and no "
                             "longer clips preview by the near-error magnitude.")
    parser.add_argument("--line-preprocess", choices=("contrast", "legacy", "canny"),
                        default=os.getenv("LINE_PREPROCESS", "legacy"),
                        help="Lane candidate extraction: contrast uses limited local "
                             "equalization and preserves thin/oblique fragments; "
                             "legacy restores the previous binary preprocessing; "
                             "canny selects experimental filled dark-stroke evidence")
    parser.add_argument("--lane-observation-mode", choices=("continuous", "strict"),
                        default="continuous",
                        help="Boundary association and local support acceptance; strict restores the prior gates")
    parser.add_argument("--lane-strictness-pct", type=float, default=50.,
                        help="Continuous mode numeric gate interpolation: 0=current permissive, "
                             "100=old strict numeric values. All percentages retain "
                             "the same categorical fallbacks; use observation mode strict separately")
    parser.add_argument("--lane-gap-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="Join short tape gaps with compatible observed endpoint directions")
    parser.add_argument('--lane-visible-direction-fallback',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='In continuous mode, allow direction-only recovery after BOTH/single '
                             'paths fail. Independent of numeric lane-strictness-pct')
    parser.add_argument("--lane-gap-max-cm", type=float, default=12.,
                        help="Maximum endpoint gap for binary completion, in ground centimetres")
    parser.add_argument('--straight-road-guard', action=argparse.BooleanOptionalAction, default=True,
                        help='Strict near-image straight confirmation; restrict final walking yaw to straight and first gears')
    parser.add_argument('--straight-near-cm', type=float, default=10.)
    parser.add_argument('--straight-far-cm', type=float, default=40.)
    parser.add_argument('--straight-confirm-s', type=float, default=1.)
    parser.add_argument('--straight-exit-s', type=float, default=.6)
    parser.add_argument('--straight-hold-s', type=float, default=3.,
                        help='Maximum time to retain road state without supporting fresh observations')
    parser.add_argument("--initial-boundary-side", choices=("unknown", "left", "right"),
                        default="unknown",
                        help="Only for an explicitly known cold-start single boundary; "
                             "unknown refuses to guess its left/right identity")
    parser.add_argument("--single-boundary-tracking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable sustained real single-boundary path recovery; "
                             "--no-single-boundary-tracking restores paired-only control")
    parser.add_argument("--line-adaptive-c", type=float, default=-12.0,
                        help="Reference C for the legacy lane adaptive threshold; "
                             "threshold mode adapts it by ground std/reference std")
    parser.add_argument("--shape-preprocess", choices=("selective", "canny", "local_dark"),
                        default=os.getenv("SHAPE_PREPROCESS", "selective"),
                        help="Card ink candidates; canny is an explicit experimental alternative")
    parser.add_argument("--photometric-mode", choices=("raw", "threshold", "match", "normalize", "legacy", "lut"),
                        default=os.getenv("PHOTOMETRIC_MODE", "lut"),
                        help="Mutually exclusive image mapping: raw passthrough, threshold-only, "
                             "guarded per-frame match (legacy), or the calibrated fixed->auto LUT "
                             "(lut/normalize, the production default; never stacks with the match or with "
                             "statistics-based threshold compensation)")
    parser.add_argument("--photometric-lut-dir",
                        default=os.getenv("PHOTOMETRIC_LUT_DIR", _DEFAULT_LUT_DIR),
                        help="Fitted fixed->auto mapping directory (lut_bgr.npy/lut_gray.npy/"
                             "lut_max.npy/mapping.json "
                             "from fit_photometric_mapping.py); required by --photometric-mode lut")
    parser.add_argument("--lane-fit", action="store_true",
                        help="EXPERIMENTAL, and it changes nothing on its own: also "
                             "scan one tall band (20~70 cm instead of the two "
                             "50-row slices at 20~35) and fit a curve to the lane "
                             "centre, then read the centre at --lane-fit-near-cm and "
                             "--lane-fit-far-cm. Only adds fields to the log, so the "
                             "'far minus near' separation can be measured on dumps "
                             "before anything is built on it")
    parser.add_argument("--lane-fit-segments", action="store_true",
                        help="Fit separate 20-32, 32-44, 44-56 cm "
                             "ground-space segments from paired observations. Enables "
                             "--lane-fit and adds fit_seg_* telemetry; diagnostic only "
                             "unless --wz-mode segments is selected")
    parser.add_argument("--lane-fit-near-cm", type=float, default=25.0,
                        help="Where to read the near point off the fitted centre line")
    parser.add_argument("--lane-fit-far-cm", type=float, default=50.0,
                        help="Where to read the far point. On the R=0.776 m arc, "
                             "with the robot centred and tangent, the lane centre "
                             "sits R-sqrt(R^2-z^2) to the side: 11 cm (44 px) at "
                             "40 cm, 18 cm (68 px) at 50 cm, against 4 cm (18 px) at "
                             "25 cm, while a straight gives 0 at both. The ceiling is "
                             "~59 cm (past that the lane's outer line leaves the "
                             "birdseye and the row cannot be paired, so the fit stops "
                             "and reports nothing rather than extrapolating), but on "
                             "the real dumps even healthy frames stop at 54 cm - "
                             "50 leaves the margin")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="0 runs until Ctrl+C")
    from steering_filter import add_arguments, config_from_args, validate_ladder
    add_arguments(parser)
    from steering_recovery import add_arguments as add_recovery_arguments, config_from_args as recovery_from_args
    add_recovery_arguments(parser)
    from steering_command_window import add_arguments as add_window_arguments
    add_window_arguments(parser)
    args = parser.parse_args()
    if args.log_items is None:
        args.log_items = tuple(name for name in LOG_ITEMS if name != 'video' or args.record_video)
    else:
        args.record_video = 'video' in args.log_items
    if not math.isfinite(args.lane_strictness_pct) or not 0. <= args.lane_strictness_pct <= 100.:
        parser.error('lane-strictness-pct must be between 0 and 100')
    if not math.isfinite(args.line_adaptive_c):
        parser.error("--line-adaptive-c must be finite")
    try:
        SteeringCommandWindow(args.steering_command_window_s, args.steering_command_median)
    except ValueError as exc:
        parser.error(str(exc))
    from steering_config import validate_heading_regions
    try:
        args.heading_regions_cm = validate_heading_regions(args.heading_regions_cm)
        if args.heading_regions_cm is not None:
            if args.wz_mode in ('heading','segments') and args.heading_regions_cm[0][1] > args.heading_lookahead_cm:
                raise ValueError('heading near interval must be entirely below the lookahead target')
            args.heading_near_cm = sum(args.heading_regions_cm[0])/2
            args.heading_far_cm = sum(args.heading_regions_cm[1])/2
    except ValueError as exc:
        parser.error(str(exc))
    from steering_config import validate_position_blend
    try:
        validate_position_blend(args.steering_position_blend_weight, args.position_gain,
                                args.position_dead_cm, args.position_lookahead_cm, args.position_max_deg)
    except ValueError as exc:
        parser.error(str(exc))
    position_values = (args.position_recovery_cm, args.position_recovery_full_scale_cm)
    if (not all(math.isfinite(v) for v in position_values)
            or args.position_recovery_cm < 0 or args.position_recovery_full_scale_cm <= 0
            or args.position_confirm_frames < 1):
        parser.error('invalid near position settings')
    try:
        validate_camera_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.startup_first_walk_s) or args.startup_first_walk_s <= 0:
        parser.error('startup-first-walk-s must be finite and >0')
    from startup_sequence import parse_sequence
    try:
        parse_sequence(args.startup_sequence, args.startup_first_walk_s, args.vx, args.max_wz)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.video_fps) or not 1 <= args.video_fps <= 30:
        parser.error('video-fps must be finite and in [1, 30]')
    if not 64 <= args.video_width <= 1920:
        parser.error('video-width must be in [64, 1920]')
    try:
        filter_config = config_from_args(args)
        recovery_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.command_min_hold_s) or args.command_min_hold_s < 0:
        parser.error('command-min-hold-s must be finite and nonnegative')
    if args.steering_filter_mode != 'legacy' and args.wz_mode not in ('heading', 'segments'):
        parser.error('steering-filter-mode shadow/active requires heading or segments')
    if not 1 <= args.connector_port <= 65535:
        parser.error("connector-port must be between 1 and 65535")
    if not math.isfinite(args.lane_gap_max_cm) or not 0 < args.lane_gap_max_cm <= 15:
        parser.error("lane-gap-max-cm must be in (0, 15]")
    if (not all(math.isfinite(v) for v in (args.straight_near_cm,args.straight_far_cm,
             args.straight_confirm_s,args.straight_exit_s,args.straight_hold_s))
            or not 0 <= args.straight_near_cm < args.straight_far_cm <= 80
            or args.straight_far_cm-args.straight_near_cm < 15
            or min(args.straight_confirm_s,args.straight_exit_s) <= 0
            or args.straight_hold_s < max(args.straight_confirm_s,args.straight_exit_s)):
        parser.error('invalid straight road region or confirmation/hold durations')
    if args.card_camera < -1 or (not args.camera_device and not args.card_camera_device and args.card_camera == args.camera):
        parser.error("--card-camera must be -1 or a different camera index")
    if args.card_camera_device and args.no_shape_detect:
        parser.error("--card-camera requires shape detection")
    if args.card_camera_width <= 0 or args.card_camera_height <= 0:
        parser.error("second camera dimensions must be positive")
    for name, lo, hi in (("card_camera_height_cm", 0., float("inf")),
                         ("card_camera_pitch_deg", 0., 90.),
                         ("card_camera_vfov_deg", 0., 180.)):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and lo < value < hi):
            parser.error(f"--{name.replace('_', '-')} is outside its valid range")
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
    if not math.isfinite(args.card_failure_cooldown_s) or args.card_failure_cooldown_s < 0:
        parser.error("card-failure-cooldown-s must be finite and nonnegative")
    if not math.isfinite(args.stop_cool_down) or args.stop_cool_down < 0:
        parser.error("stop-cool-down must be finite and nonnegative")
    if not math.isfinite(args.card_tilt_ms) or args.card_tilt_ms < 0:
        parser.error("card-tilt-ms must be finite and nonnegative")
    if not math.isfinite(args.card_settle_ms) or args.card_settle_ms < 0:
        parser.error("card-settle-ms must be finite and nonnegative")
    if not math.isfinite(args.center_dead_cm) or args.center_dead_cm < 0:
        parser.error("center-dead-cm must be finite and nonnegative")
    if args.max_wz_right is not None and not (
            math.isfinite(args.max_wz_right)
            and 0 < args.max_wz_right <= args.max_wz):
        parser.error("max-wz-right must be in (0, max-wz] when set")
    if args.card_trigger_frac is not None and not (
            math.isfinite(args.card_trigger_frac) and 0 < args.card_trigger_frac <= 1):
        parser.error("card-trigger-frac must be in (0, 1]")
    if not math.isfinite(args.card_trigger_dist_cm) or args.card_trigger_dist_cm <= 0:
        parser.error("card-trigger-dist-cm must be positive")
    if args.card_clear_calls < 1:
        parser.error("card-clear-calls must be at least 1")
    if args.card_every_stopped < 1:
        parser.error("card-every-stopped must be at least 1")
    if args.shape_every < 1:
        parser.error("shape-every must be at least 1")
    if args.dump_on_loss_ring < 1:
        parser.error("dump-on-loss-ring must be at least 1")
    if not math.isfinite(args.dump_on_loss_cooldown) or args.dump_on_loss_cooldown < 0:
        parser.error("dump-on-loss-cooldown must be finite and nonnegative")
    if args.start_gate in ("shape", "both", "button") and args.no_shape_detect:
        parser.error("start-gate shape/both/button needs shape detection")
    if args.start_gate in ("qr", "both") and not args.start_gate_qr_payload.strip():
        parser.error("start-gate-qr-payload must not be empty")
    if args.start_gate_shape_frames < 1:
        parser.error("start-gate-shape-frames must be at least 1")
    if (args.start_gate_shape_votes < 1
            or args.start_gate_vote_window < args.start_gate_shape_votes):
        parser.error("start-gate-vote-window must be at least start-gate-shape-votes >= 1")
    if args.start_policy_on_gate and args.start_gate not in ("both", "button"):
        parser.error("start-policy-on-gate requires --start-gate both or button")
    if args.start_gate == "button" and not args.start_policy_on_gate:
        parser.error("start-gate button requires --start-policy-on-gate")
    if (not math.isfinite(args.start_policy_max_seconds)
            or args.start_policy_max_seconds <= 0):
        parser.error("start-policy-max-seconds must be positive")
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
    # 五个数都可以自由调：只查"是不是个能用的数"，不设形状边界。
    # 1.0 / 2.5 只是默认值，不是上限下限。
    if not (all(math.isfinite(v) for v in (args.wz_fire_cm, args.wz_step,
                                           args.wz_turn_s, args.wz_gap_s))
            and (args.wz_stop_cm is None or math.isfinite(args.wz_stop_cm))
            and 0 < args.wz_fire_cm
            and 0 < args.wz_step <= args.max_wz
            and args.wz_turn_s > 0.0
            and args.wz_gap_s >= 0.0):
        parser.error("need 0 < wz-fire-cm, a finite wz-stop-cm "
                     "(unset = wz-fire-cm, the mirror), 0 < wz-step <= max-wz, "
                     "wz-turn-s > 0, and wz-gap-s >= 0 (0 = no forced coast)")
    if args.wz_mode in ("heading", "segments") and args.wz_step > 0.5:
        parser.error("heading wz-step must be <=0.5: connector wire limit is +/-0.5")
    heading_values = (args.heading_far_cm, args.heading_lookahead_cm, args.heading_right_tolerance_deg,
                      args.heading_left_tolerance_deg, args.heading_full_scale_deg, args.heading_corridor_cm)
    if (not all(math.isfinite(v) for v in heading_values)
            or args.heading_far_cm <= 0 or args.heading_lookahead_cm <= 0 or args.heading_full_scale_deg <= 0
            or args.heading_corridor_cm <= 0
            or not 0 <= args.heading_right_tolerance_deg < 90
            or not 0 <= args.heading_left_tolerance_deg < 90):
        parser.error("heading far/lookahead/full-scale must be finite and positive; tolerances in [0,90)")
    if args.wz_mode in ('heading', 'segments') and args.heading_regions_cm is None:
        near = args.heading_near_cm if args.heading_near_cm is not None else 25.07
        if (not math.isfinite(near) or near < 20 or args.heading_far_cm-near < 3
                or args.heading_far_cm > 85 or near >= args.heading_lookahead_cm):
            parser.error('heading near must be >=20cm and below lookahead; far must be >=near+3cm and <=85cm; image fit is checked at setup')
    from steering_config import validate_angle_wz_table, validate_segment_regions
    try:
        cap = min(args.wz_step,args.max_wz)
        wire_right_cap = min(cap,args.max_wz_right if args.max_wz_right is not None else args.max_wz)
        left_cap,right_cap = (cap,wire_right_cap) if args.yaw_sign > 0 else (wire_right_cap,cap)
        args.steering_angle_wz_table = validate_angle_wz_table(args.steering_angle_wz_table, left_cap,
            right_cap=right_cap)
        args.segment_regions_cm = validate_segment_regions(args.segment_regions_cm)
        if args.wz_mode == 'segments':
            near = args.heading_near_cm if args.heading_near_cm is not None else 25.07
            if not args.segment_regions_cm[0][0] <= near < args.segment_regions_cm[0][1]:
                raise ValueError('first segment region must contain the heading near observation centre')
            if args.heading_regions_cm is not None and not (
                    args.segment_regions_cm[0][0] <= args.heading_regions_cm[0][0]
                    and args.heading_regions_cm[0][1] <= args.segment_regions_cm[0][1]):
                raise ValueError('first segment region must contain the entire configured heading near interval')
    except ValueError as exc:
        parser.error(str(exc))
    ladders = args.heading_left_wz + args.heading_right_wz
    if (not all(math.isfinite(v) and 0 < v <= 0.5 for v in ladders)
            or any(a >= b for a, b in zip(args.heading_left_wz, args.heading_left_wz[1:]))
            or any(a >= b for a, b in zip(args.heading_right_wz, args.heading_right_wz[1:]))):
        parser.error("heading left/right wz levels must be positive, increasing and "
                     "<=0.5 (connector wire limit)")
    if args.steering_angle_wz_table is None and (not math.isfinite(args.heading_straight_wz)
            or not 0 <= args.heading_straight_wz < min(args.heading_left_wz[0],
                                                       args.heading_right_wz[0])):
        parser.error("heading-straight-wz must be in [0, the smallest turn level)")
    if args.wz_mode in ("heading", "segments") and args.steering_angle_wz_table is None:
        max_wz_right = args.max_wz if args.max_wz_right is None else args.max_wz_right
        if args.yaw_sign > 0:
            left_cap, right_cap = args.wz_step, min(args.wz_step, max_wz_right)
        else:
            left_cap, right_cap = min(args.wz_step, max_wz_right), args.wz_step
        if min(args.heading_left_wz) > left_cap or min(args.heading_right_wz) > right_cap:
            parser.error("heading caps (wz-step / max-wz-right) must admit the "
                         "smallest left and right wz levels for corridor correction")
        if args.steering_filter_mode != 'legacy':
            try:
                for levels, cap in ((args.heading_left_wz, left_cap), (args.heading_right_wz, right_cap)):
                    validate_ladder(tuple(v for v in levels if v <= cap), cap,
                                    args.heading_full_scale_deg, filter_config.hysteresis_deg)
            except ValueError as exc:
                parser.error(str(exc))
    if args.wz_mode == "segments":
        args.lane_fit_segments = True
    if args.no_shape_detect:
        args.card_camera = -1
    return args


def fmt(value, spec):
    """Format one detector debug value; '-' when the detector did not report it."""
    return "-" if value is None else format(value, spec)


def main():
    args = parse_args()
    from startup_sequence import parse_sequence, StartupSequence
    startup_sequence = StartupSequence(parse_sequence(
        args.startup_sequence, args.startup_first_walk_s, args.vx, args.max_wz))
    recording_deferred = args.start_gate == 'button'
    recording_started = False
    recording_directory = None
    dump_options = (('vision', 'line_log_dir'), ('loss', 'dump_on_loss'), ('shape', 'shape_dump'))
    for name, option in dump_options:
        if name not in args.log_items:
            setattr(args, option, '')

    def prepare_recording_directory():
        nonlocal recording_directory
        if args.recording_root and args.log_items and recording_directory is None:
            from recording_session import create_session
            recording_directory = create_session(args.recording_root, log_items=args.log_items)
            for name, option in dump_options:
                setattr(args, option, str(recording_directory / name) if name in args.log_items else '')
            print(f'[recording] test directory: {recording_directory}', flush=True)

    if not recording_deferred:
        prepare_recording_directory()
    line_log = None
    command_video = None
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
            controller, fire_cm=args.wz_fire_cm, stop_cm=args.wz_stop_cm,
            turn_s=args.wz_turn_s, gap_s=args.wz_gap_s, step=args.wz_step,
            allow_right=args.wz_allow_right)
    elif args.wz_mode in ("heading", "segments"):
        from heading_steering import HeadingSteeringController
        controller_type = HeadingSteeringController
        if args.wz_mode == "segments":
            from segment_steering import SegmentSteeringController
            controller_type = SegmentSteeringController
        from steering_filter import configured_controller
        controller = configured_controller(controller_type, controller, dict(
            angle_wz_table=args.steering_angle_wz_table,
            **({'segment_regions_cm': args.segment_regions_cm} if args.wz_mode == 'segments' else {}),
            lookahead_cm=args.heading_lookahead_cm,
            right_tolerance_deg=args.heading_right_tolerance_deg,
            left_tolerance_deg=args.heading_left_tolerance_deg,
            full_scale_deg=args.heading_full_scale_deg, max_step=args.wz_step,
            allow_right=True, corridor_cm=args.heading_corridor_cm,
            left_levels=tuple(args.heading_left_wz),
            right_levels=tuple(args.heading_right_wz),
            straight_wz=args.heading_straight_wz,
            steering_position_blend_weight=args.steering_position_blend_weight,
            position_gain=args.position_gain, position_dead_cm=args.position_dead_cm,
            position_lookahead_cm=args.position_lookahead_cm, position_max_deg=args.position_max_deg,
            position_recovery_cm=args.position_recovery_cm,
            position_recovery_full_scale_cm=args.position_recovery_full_scale_cm,
            position_confirm_frames=args.position_confirm_frames), args)
    # Lazy imports keep --help and controller tests usable without a camera stack.
    import cv2
    from straight_road_guard import StraightRoadGuard
    straight_guard = StraightRoadGuard(
        enabled=args.straight_road_guard, confirm_s=args.straight_confirm_s,
        exit_s=args.straight_exit_s, hold_s=args.straight_hold_s,
        straight_wz=getattr(controller,'straight_wz',args.heading_straight_wz),
        left_levels=tuple(v for v in getattr(controller,'left_levels',args.heading_left_wz)
                          if v <= args.max_wz),
        right_levels=tuple(v for v in getattr(controller,'right_levels',args.heading_right_wz)
                          if v <= (args.max_wz if args.max_wz_right is None else args.max_wz_right)),
        yaw_sign=args.yaw_sign)
    from line_detector_v1_warp import LineDetector
    from utils import open_camera, show_debug_windows

    from shape_detector import trigger_frac_at_dist
    # Moving presence uses the cue height calibrated from stop distance.
    card_trigger_frac = (args.card_trigger_frac if args.card_trigger_frac is not None
                         else trigger_frac_at_dist(args.card_trigger_dist_cm))
    print(f"[card-stop] configured distance={args.card_trigger_dist_cm:g}cm; "
          f"cue-only cy>={card_trigger_frac:.3f}"
          f"{' (explicit cy override)' if args.card_trigger_frac is not None else ' (distance-derived)'}; "
          "cue score only confirms presence, it does not set stop distance", flush=True)

    from photometric_thresholds import load_lut_mapping, resolve_mode
    if resolve_mode(args.photometric_mode) == "lut":
        try:
            load_lut_mapping(args.photometric_lut_dir)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"photometric mode lut cannot load "
                             f"{args.photometric_lut_dir}: {exc}")
        print(f"[photometric] lut fixed->auto mapping: {args.photometric_lut_dir} "
              "(fixed camera settings must match the calibration)", flush=True)
    shape = precision_shape = shape_names = None
    card_camera_mount_pitch = (args.card_camera_pitch_deg
        if args.card_camera_pitch_deg is not None else args.camera_pitch_deg)
    if not args.no_shape_detect:
        from shape_detector import ShapeDetector
        shape = ShapeDetector()
        shape.preprocess_mode = args.shape_preprocess
        shape.photometric_mode = args.photometric_mode
        print(f"[shape-preprocess] {args.shape_preprocess}; photometric={args.photometric_mode} "
              f"effective={resolve_mode(args.photometric_mode)}", flush=True)
        shape_names = {number: name for name, number in shape.action_map.items()}
        shape_numbers = dict(shape.action_map)
        if args.card_camera_device or args.card_camera >= 0:
            precision_shape = ShapeDetector()
            precision_shape.preprocess_mode = args.shape_preprocess
            precision_shape.photometric_mode = args.card_camera_photometric_mode
            precision_shape.cfg["cam_height_cm"] = (args.card_camera_height_cm
                if args.card_camera_height_cm is not None else args.camera_height_cm)
            precision_shape.cfg["cam_pitch_deg"] = card_camera_mount_pitch
            precision_shape.cfg["cam_vfov_deg"] = (args.card_camera_vfov_deg
                if args.card_camera_vfov_deg is not None else args.camera_vfov_deg)
            precision_shape.cfg["area_min"] = int(precision_shape._card_area_at_dist(
                precision_shape.cfg["max_dist_cm"]))

    # 起跑门控。--help 依旧不碰 cv2：QrReader 只在真的要用二维码时才 import。
    start_gate = qr_reader = None
    if args.start_gate != "off":
        from start_gate import StartGate
        start_gate = StartGate(mode=args.start_gate,
                               expected_qr=args.start_gate_qr_payload,
                               shape_confirm=args.start_gate_shape_frames,
                               shape_vote_min=args.start_gate_shape_votes,
                               shape_vote_window=args.start_gate_vote_window,
                               shape_voting=args.start_gate == "both")
        if start_gate.require_qr:
            from qr_reader import QrReader
            qr_reader = QrReader(min_edge_px=args.qr_min_edge_px,
                                 max_edge_px=args.qr_max_edge_px,
                                 upscale=args.qr_upscale,
                                 max_side=args.qr_max_side)
    policy_launcher = startup_button_link = None
    if args.start_policy_on_gate:
        from policy_gate_launcher import PolicyGateLauncher
        policy_launcher = PolicyGateLauncher(args.start_policy_model,
                                             args.start_policy_port,
                                             args.start_policy_max_seconds,
                                             policy_python=args.start_policy_python,
                                             one_foot_model=args.start_policy_one_foot_model,
                                             command_min_hold_s=effective_policy_hold(args),
                                             log_items=args.log_items,
                                             **({'recording_directory': recording_directory}
                                                if recording_directory is not None else {}))
    if args.start_gate == "button":
        from startup_button_link import StartupButtonLink
        startup_button_link = StartupButtonLink(args.start_policy_port)

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
    camera_reader = None
    card_camera_reader = None
    try:
        from camera_devices import resolve_devices
        main_device, card_device = resolve_devices(args.camera, args.card_camera,
                                                   args.camera_device, args.card_camera_device)
        print(f"[cameras] main={main_device}; stopped-card={card_device}; secondary opens only during stopped voting", flush=True)
        cap = open_camera(main_device, args.width, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open main camera {main_device}; check capture node, permission and another process holding it")
        from camera_controls import apply_camera_controls
        camera_control_report = apply_camera_controls(main_device if isinstance(main_device,str) else f'/dev/video{main_device}',args)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        detector = LineDetector(width, height, cam_height_cm=args.camera_height_cm,
                                cam_pitch_deg=args.camera_pitch_deg,
                                cam_vfov_deg=args.camera_vfov_deg)
        if args.wz_mode in ("heading", "segments"):
            if args.heading_regions_cm is not None:
                detector.set_heading_regions_cm(args.heading_regions_cm)
            else:
                detector.set_heading_distances(args.heading_near_cm, args.heading_far_cm)
        detector.preprocess_mode = args.line_preprocess
        detector.set_lane_observation_mode(args.lane_observation_mode, args.lane_strictness_pct)
        detector.visible_direction_fallback = args.lane_visible_direction_fallback
        detector.lane_gap_fill = args.lane_gap_fill
        detector.lane_gap_max_cm = args.lane_gap_max_cm
        detector.straight_guard_enabled = args.straight_road_guard
        detector.straight_near_cm = args.straight_near_cm
        detector.straight_far_cm = args.straight_far_cm
        print(f'[straight-road] enabled={args.straight_road_guard}; observed near region '
              f'{args.straight_near_cm:g}..{args.straight_far_cm:g}cm; '
              f'confirm={args.straight_confirm_s:g}s exit={args.straight_exit_s:g}s '
              f'unknown-hold={args.straight_hold_s:g}s; '
              f'allowed geometric wz={straight_guard.allowed}', flush=True)
        print(f"[lane-observation] requested={args.lane_observation_mode}; "
              f"effective={detector.lane_observation_mode}; "
              f"strictness={args.lane_strictness_pct:g}%; "
              f"direction-fallback={detector.lane_observation_mode == 'continuous' and detector.visible_direction_fallback}; "
              f"gap-fill={args.lane_gap_fill} max={args.lane_gap_max_cm:g}cm", flush=True)
        detector.initial_boundary_side = (None if args.initial_boundary_side == "unknown"
                                          else args.initial_boundary_side)
        detector.single_boundary_tracking = args.single_boundary_tracking
        detector.adaptive_c = args.line_adaptive_c
        detector.photometric_mode = args.photometric_mode
        print(f"[line-preprocess] {detector.preprocess_mode}; "
              f"photometric-effective={resolve_mode(args.photometric_mode)}; "
              "candidate mask only; geometry quality checks retained", flush=True)
        print(f"[lane-boundary] enabled={args.single_boundary_tracking}; "
              f"cold-start-side={args.initial_boundary_side}; "
              f"equal-width-prior={getattr(detector, 'lane_width_true_cm', 'unknown')}cm; "
              "single edge requires real continuous observations", flush=True)
        if args.no_red_detect:
            detector.red_detect_enable = False
        detector.anticipation_clip = args.anticipation_clip
        detector.lane_fit_enable = args.lane_fit or args.lane_fit_segments
        detector.lane_segments_enable = args.lane_fit_segments
        detector.segment_regions_cm = args.segment_regions_cm
        detector.lane_fit_near_cm = args.lane_fit_near_cm
        detector.lane_fit_far_cm = args.lane_fit_far_cm
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
        if args.wz_mode in ("heading", "segments"):
            print(f'[steering-position] recovery={args.position_recovery_cm:g}cm; '
                  f'full_scale={args.position_recovery_full_scale_cm:g}cm; '
                  f'confirm={args.position_confirm_frames} fresh frames', flush=True)
            print(f'[steering-filter] mode={args.steering_filter_mode}; '
                  f'algorithm={args.steering_filter_algorithm}; '
                  f'angle_cutoff={args.steering_filter_min_hz:g}..{args.steering_filter_max_hz:g}Hz; '
                  f'beta={args.steering_filter_beta:g}; '
                  f'hysteresis={args.steering_hysteresis_deg:g}deg; '
                  f'entry/exit={args.steering_enter_deg:g}/{args.steering_exit_deg:g}deg; '
                  f'model_min_hold={effective_policy_hold(args):g}s (launched policy only)', flush=True)
            if args.steering_filter_algorithm == 'robust':
                print(f'[steering-filter] robust tau={args.steering_filter_robust_tau_s:g}s; '
                      f'window={args.steering_filter_robust_window_s:g}s; '
                      f'slew={args.steering_filter_robust_slew_deg_s:g}deg/s',flush=True)
            print(f'[steering-loss] mode={args.steering_loss_mode}; max={args.steering_loss_max_s:g}s; '
                  f'history={args.steering_loss_history_s:g}s; '
                  f'qualified_segment_fallback={args.steering_segment_fallback}',flush=True)
            if args.steering_loss_mode != 'legacy':
                print('[steering-loss] 丢线保持前进；无近期转向或沿用到期后，'
                      f'使用备用 WZ={args.steering_loss_fallback_wz:+g}（服从 yaw-sign 和方向幅值上限）。'
                      '起步闸、图卡停车和人工停止仍优先。', flush=True)
            left_text = "、".join(f"+{v:g}" for v in controller.left_levels)
            right_text = "、".join(f"-{v:g}" for v in controller.right_levels)
            if args.steering_angle_wz_table is not None:
                print(f'[steering-angle-table] {json.dumps(args.steering_angle_wz_table)}; '
                      f'input=steering_table_input_deg; active heading uses F+w*P, w={args.steering_position_blend_weight:g}; '
                      'F=filtered target demand, P=bounded position correction; output=signed rad/s; '
                      'stop/loss protection and connector windows retained', flush=True)
            if args.heading_regions_cm is not None:
                print(f'[heading-regions] {json.dumps(args.heading_regions_cm)} cm; '
                      'eight rows per region; centre settings overridden', flush=True)
            print(f"[wz] 视觉逐帧提交候选；connector 独立窗口 {args.steering_command_window_s:g}s，"
                  f"中位数 {args.steering_command_median}；自动启动模型保持 {effective_policy_hold(args):g}s；"
                  f"近端观测中心 {args.heading_near_cm if args.heading_near_cm is not None else 25.07:g}cm，"
                  f"远端观测中心 {args.heading_far_cm:g}cm，"
                  f"前视 {args.heading_lookahead_cm:g}cm，目标方位容忍区 "
                  f"[-{args.heading_left_tolerance_deg:g}, +{args.heading_right_tolerance_deg:g}]°，"
                  f"横向走廊 ±{args.heading_corridor_cm:g}cm；"
                  f"左档 {left_text}，右档 {right_text}，直行 {controller.straight_wz:+g}"
                  f"（实际共 {1 + len(controller.left_levels) + len(controller.right_levels)} 档；"
                  f"--heading-left-wz / --heading-right-wz / "
                  f"--heading-straight-wz 可调）。"
                  "观测趋势仅用于提前减小正在执行的转向。"
                  "旧 PID/bias/fire/stop/turn/gap 参数不参与本模式。"
                  "需配套新 connector 和 C；停车/失联可立即打断。", flush=True)
        if args.wz_mode == "segments":
            print(f'[segment-regions] {json.dumps(args.segment_regions_cm)} cm', flush=True)
            print("[segment-control] 已接入实际转向：连续 3 帧通过近端锚定、宽度、"
                  "残差和分段连续性检查后，使用观测范围内的前方目标；"
                  "不足时回退 heading。原 heading 的对照输出只写日志。", flush=True)
        if args.wz_mode == "discrete":
            levels = (f"{{0, ±{args.wz_step}}}" if args.wz_allow_right
                      else f"{{0, +{args.wz_step}}}")
            coast = (f"转完强制空 {args.wz_gap_s:.2f}s 才允许下一段"
                     if args.wz_gap_s > 0.0 else
                     "转完不强制滑行（--wz-gap-s 0），err 还在阈值上就接着开")
            stop_cm = (args.wz_fire_cm if args.wz_stop_cm is None
                       else args.wz_stop_cm)
            if args.wz_stop_cm is None:
                release = (f"err 翻到另一侧的 {stop_cm}cm 才收手"
                           f"（--wz-stop-cm 不写就是 fire 的镜像）")
            elif stop_cm > 0.0:
                release = f"err 翻到另一侧的 {stop_cm}cm 才收手"
            elif stop_cm < 0.0:
                release = f"err 回到同侧的 {-stop_cm}cm 以内就收手"
            else:
                release = "err 一翻过中心就收手"
            print(f"[wz] 离散模式：wz 只有 {levels} 两个状态。"
                  f"err ≥ {args.wz_fire_cm}cm 就开一段转向，{release}，最多 "
                  f"{args.wz_turn_s:.2f}s（--wz-turn-s 是上限，不是定长）；"
                  f"{coast}。五个数都自由可调。"
                  f"{'两边都能转' if args.wz_allow_right else '只在车身偏右（err>0）时才左转，偏左不转'}"
                  f"；--center-dead-cm / --steer-full-scale-cm / --bias-cm / "
                  f"PID 增益在这个模式下不影响输出。", flush=True)
            print(f"[wz] ⚠️ A 那边要用 --max-wz-accel 0，否则一段转向会被它的斜率"
                  f"限制削成三角形（默认 2.0 时 0→{args.wz_step} 要爬 "
                  f"{args.wz_step / 2.0:.2f}s）", flush=True)
            print(f"[wz] 幅度 {args.wz_step} 挑的是「大且稳」那一端：关节在小角度下"
                  f"比大角度还不稳，所以不拿小量加精确时长去凑。"
                  f"参考：--vx {args.vx} 下跟住一个弯要 {args.vx / LANE_RADIUS_M:.3f} "
                  f"rad/s（只是参考，幅度不跟着它走）。触发看原始 err=", flush=True)
        if start_gate is not None:
            if args.start_gate == "button":
                print("[button-start] 等待PC2按钮；首卡识别后PA3亮灯，未亮灯也可按按钮启动。"
                      "等待期间不启动步态，不发送电机COMMAND。", flush=True)
            else:
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
        line_pitch_until = 0.0
        last_cmd_vx = 0.0
        window_origin_s = None
        startup_next_stop_at = 0.0
        card_dbg = {}
        cards = CardFlow(args.stop_cool_down, args.card_failure_cooldown_s,
                         args.card_stop_ms / 1000., args.card_settle_ms / 1000.,
                         args.card_tilt_ms / 1000.)
        card_feedback = {}
        reported_card_decision = None
        steering_paired_count = 0
        steering_single_count = 0
        card_stop_cue_hit_count = 0
        card_stop_cue_trigger_count = 0
        last_counts_log = -math.inf
        lateral_warned = None    # None until the lateral bound first trips
        card_window_open = False
        # 门控的"上一帧"状态。闸门关着时 window_open 为真，于是窗口关闭那套交接
        # （drop_held_command / line_pitch_until）被原样复用，
        # 释放的那一帧走的就是停车窗口关闭走过的同一条路。
        gate_window_open = start_gate is not None
        gate_released = False
        start_released = False
        gate_last_log = -math.inf
        qr_odd_seen = set()
        # Competition start: classify the first card while stationary, walk briefly,
        # then act on that latched classification. Keep its ordinary stop gate closed
        # until it has been left behind and the next card approaches from afar.
        startup_first_card = -1
        startup_first_card_pending = False
        startup_first_card_lock = False
        startup_first_card_clear_calls = 0
        startup_card_present = False
        dumped = 0
        dump_metadata = {
            "argv": list(sys.argv),
            "arguments": vars(args).copy(),
            "camera_width": width, "camera_height": height,
            "camera_controls": camera_control_report,
        }
        dump_metadata['performance_environment'] = environment_metadata()
        dump_metadata["measurement_parameters"] = {
            name: getattr(detector, name) for name in (
                "lock_width_cv_max", "observation_rmse_max_px", "measurement_quality_min",
                "measurement_max_age_s", "single_width_max_age_s", "single_quality_max",
                "filter_tau_s", "shake_filter_tau_s", "preview_gain", "preview_max_cm",
                "heading_min_points", "heading_min_span_px",
            ) if isinstance(getattr(detector, name, None), (int, float, bool))
        }
        print("[vision] P1 geometry: quality-gated near/preview; elapsed-time EMA; "
              "legacy --anticipation-clip has no effect", flush=True)
        if args.log_items and (args.recording_root or args.shape_dump or args.dump_on_loss or args.line_log_dir):
            dump_metadata["source_sha256"] = {}
            for source_name in ("run_policy_vision.py", "line_detector_v1_warp.py",
                                "lane_boundary_path.py", "continuous_lane_heading.py",
                                "lane_gap_completion.py", "visible_lane_direction.py", "straight_road_guard.py",
                                "shape_detector.py", "card_flow.py", "policy_bridge.py", "card_vote_camera.py", "camera_devices.py",
                                "discrete_steering.py", "heading_steering.py", "camera_config.py",
                                "line_telemetry.py", "lane_segments.py", "segment_steering.py", "steering_config.py",
                                "steering_filter.py", "steering_recovery.py", "steering_command_window.py", "startup_sequence.py", "camera_controls.py",
                                "line_preprocess.py", "masked_ground.py", "latest_camera.py", "photometric_thresholds.py", "canny_candidates.py",
                                "recording_options.py", "recording_session.py", "performance_telemetry.py"):
                with open(os.path.join(os.path.dirname(__file__), source_name), "rb") as source:
                    dump_metadata["source_sha256"][source_name] = hashlib.sha256(source.read()).hexdigest()

        def begin_recording():
            nonlocal recording_directory, recording_started, line_log, command_video
            if recording_started:
                return
            prepare_recording_directory()
            if policy_launcher is not None and recording_directory is not None:
                policy_launcher.recording_directory = recording_directory
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}"
            dump_metadata.update(run_id=run_id,
                                 started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 arguments=vars(args).copy())
            if args.line_log_dir:
                line_log = LineTelemetry(_new_dump_run(args.line_log_dir, run_id, dump_metadata))
                print(f"[vision] per-frame log: {line_log.path}", flush=True)
                print(f"[vision] readable counts: {line_log.counts_path}", flush=True)
            if recording_directory is not None and args.record_video:
                try:
                    from command_video import CommandVideo
                    command_video = CommandVideo(recording_directory/'video', fps=args.video_fps,
                                                 width=args.video_width, max_wz=args.max_wz,
                                                 frame_source=args.video_source,
                                                 image_size=((detector.bird_w * (2 if args.video_source == 'bird_pair' else 1),
                                                              detector.bird_h)
                                                             if args.video_source != 'camera' else None))
                    print(f'[video] source={args.video_source} + vision sent commands: {recording_directory / "video"}; '
                          f'{args.video_fps:g} playback FPS; asynchronous encoder', flush=True)
                except Exception as exc:
                    print(f'[video] recording disabled: {exc}', flush=True)
            if args.shape_dump:
                args.shape_dump = _new_dump_run(args.shape_dump, run_id, dump_metadata)
                print(f"[shape] run dump: {args.shape_dump}", flush=True)
            if args.dump_on_loss:
                args.dump_on_loss = _new_dump_run(args.dump_on_loss, run_id, dump_metadata)
                print(f"[vision] run dump: {args.dump_on_loss}", flush=True)
            recording_started = True

        print(f"[recording] enabled logs: {' '.join(args.log_items) or 'none'}", flush=True)
        if not args.log_items:
            print('[recording] 测试文件记录已关闭；控制台和服务journal仍保留。', flush=True)
        elif recording_deferred:
            print('[recording] 等待PC2启动按钮；按下后才创建测试目录并开始日志和录像。', flush=True)
        else:
            begin_recording()
        # The frames around a lock drop-out, kept short so the last good frame before
        # the drop is still in the ring when it trips -- that is the one that shows
        # what the detector was looking at while it still agreed with itself.
        loss_ring = []
        loss_dumped = 0
        loss_next_ok = 0.0
        prev_pair = 0.0
        prev_conf = 0.0
        previous_loop_start_ns = None
        previous_loop_tail_ms = None
        runtime_counters = RuntimeCounters()
        if args.camera_async:
            from latest_camera import LatestCamera
            camera_reader = LatestCamera(cap).start()
            print('[camera] async capture; one latest-frame slot; stale frames dropped', flush=True)
        while not stopped:
            now = time.monotonic()
            if (policy_launcher is not None and start_released
                    and policy_launcher.process is not None
                    and policy_launcher.process.poll() is not None):
                status = policy_launcher.process.returncode
                if status != 0:
                    raise RuntimeError(policy_launcher.exit_report(status))
                print("[start-gate] policy process ended; vision is stopping", flush=True)
                break
            if args.max_seconds > 0 and now - start >= args.max_seconds:
                break
            # App read-attempt cadence, not the sensor's hardware frame timestamp.
            loop_start_ns = time.perf_counter_ns()
            loop_period_ms = (None if previous_loop_start_ns is None else
                              (loop_start_ns - previous_loop_start_ns) / 1_000_000.0)
            previous_loop_start_ns = loop_start_ns
            capture_metadata = {}
            if camera_reader is not None:
                ok, frame, capture_metadata = camera_reader.read()
            else:
                ok, frame = cap.read()
            read_return_ns = time.perf_counter_ns()
            camera_read_ms = (read_return_ns - loop_start_ns) / 1_000_000.0
            if not ok:
                controller.reset()
                window_origin_s = None
                event = ({"event_id": card_event_id, "event_action": card_action}
                         if card_event_id and now < card_until else {})
                if cards.event_id:
                    event = dict(event_id=cards.event_id, event_action=cards.action)
                client.publish(0.0, 0.0, -1, hold_upright=gate_window_open,
                               card_tilt=cards.tilt_requested, **event)
                if now - last_log >= 0.5:
                    print("[vision -> connector] camera read failed; vx=0 wz=0", flush=True)
                    last_log = now
                time.sleep(0.02)
                continue
            processed = time.monotonic()
            frame_observed_at = capture_metadata.get('camera_capture_started_s', processed)
            if attitude is not None:
                attitude.poll()
            card_feedback = ordinary_card_feedback(attitude, processed)
            cards.advance(processed, card_feedback)
            stop_window = cards.active or processed < card_until
            window_open = stop_window or gate_window_open
            if card_window_open and not window_open:
                controller.drop_held_command()
                if args.card_cold_start:
                    controller.reset(clear_hold=True)
                    detector.reset_state()
                line_pitch_until = processed + args.line_pitch_hold_s
                print(f"[vision] {'start gate released' if gate_released else 'card window closed'}; "
                      f"{'cold start' if args.card_cold_start else 'resuming frozen'}", flush=True)
                gate_released = False
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
                if not cards.tilt_requested:
                    effective_pitch = attitude.value
            if shape is not None:
                shape.set_camera_pitch_deg(effective_pitch)
            if precision_shape is not None:
                precision_shape.set_camera_pitch_deg(
                    card_camera_mount_pitch + effective_pitch - args.camera_pitch_deg)
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
            # Keep walking memory separate from observations made while stopped.
            # A re-pose changes the camera geometry and can put the card frame into
            # the lane search. Still process it for diagnostics and red-bar sensing,
            # but do not let it train the next walking frame's seed/width/EMA.
            tracking_before_frame = detector.snapshot_tracking_state()
            line_process_start_ns = time.perf_counter_ns()
            line_frame_token = zlib.crc32(frame)
            detector._pending_frame_token = line_frame_token
            startup_card_only = bool(gate_window_open and args.start_gate == 'button')
            if startup_card_only:
                # Lane commands cannot be used before button handoff. Prioritize
                # first-card classification and startup serial communication.
                confidence, visualization = 0.0, frame
                debug = dict(startup_card_only=True, measurement_valid=False,
                             lost_frames=0, base_err_cm=0.0,
                             tracking_mode='STARTUP_CARD_ONLY')
            else:
                _, _, confidence, visualization, debug = detector.process(
                    frame, dt=processed - previous)
            line_process_ms = (time.perf_counter_ns() - line_process_start_ns) / 1_000_000.0
            if window_open:
                detector.restore_tracking_state(tracking_before_frame)
            decision_start_ns = time.perf_counter_ns()
            qr_decode_ms = None
            shape_update_ms = None
            frames += 1
            log_frames += 1
            recognized_this_frame = False
            if recording_started and args.dump_on_loss and not startup_card_only:
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
            # Ordinary walking presence and stopped voting each run on every frame.
            # Startup retains its existing sampling and first-card lifecycle.
            shape_period = 1
            if processed < card_until:
                shape_period = args.card_every_stopped
            elif gate_window_open or startup_first_card_pending or startup_first_card_lock:
                shape_period = (min(args.shape_every, args.card_every_stopped)
                                if startup_card_present else args.shape_every)
            tilting = cards.active and not cards.can_vote(frame_observed_at)
            # Only a new classification may cast a vote; never reuse the previous frame.
            card_dbg = None
            card_detection_mode = None
            card_vote_decision = None
            card_frame_source = None
            card_camera_status = None
            card_camera_meta = {}
            card_primary_shape = None
            card_secondary_shape = None
            startup_shape_candidate = None
            startup_shape_source = None
            card_shape_source = None
            card_reach = card_reach_line = None
            # ── 阀1：二维码 ──
            # 只在还没扫到时跑，而且每 --qr-every 帧才解一次：detectAndDecode 是纯
            # CPU 的，1280x720 raw 十几毫秒、再放大一倍几十毫秒，逐帧跑会把主循环
            # 拖垮。扫到就锁存，之后一次都不再进来 —— 行进段一分钱都不付。
            if (qr_reader is not None and not start_gate.qr_passed
                    and frames % args.qr_every == 0):
                qr_decode_start_ns = time.perf_counter_ns()
                reading = qr_reader.decode(frame)
                qr_decode_ms = (time.perf_counter_ns() - qr_decode_start_ns) / 1_000_000.0
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
            startup_shape_needed = bool(
                gate_window_open and start_gate is not None
                and start_gate.require_shape and not start_gate.shape_passed
                and not (args.start_gate == 'button' and start_gate.button_passed))
            stopped_vote_needed = cards.can_vote(frame_observed_at)
            secondary_needed = card_device is not None and startup_shape_needed
            if secondary_needed and card_camera_reader is None:
                from latest_camera import LatestCamera
                from card_vote_camera import CardVoteCamera
                from camera_devices import verify_camera_roles
                verify_camera_roles(main_device, card_device)
                card_cap = open_camera(card_device, args.card_camera_width, args.card_camera_height)
                if not card_cap.isOpened():
                    card_cap.release()
                    raise RuntimeError(f"Cannot open stopped-card camera {card_device}")
                card_camera_reader = CardVoteCamera(LatestCamera(card_cap).start())
                print(f"[card-camera] opened {card_device} for "
                      f"{'startup shape gate' if startup_shape_needed else 'stopped recognition'}", flush=True)
            elif not secondary_needed and card_camera_reader is not None:
                if not card_camera_reader.close():
                    raise RuntimeError('Stopped-card capture did not terminate; refusing duplicate open')
                card_camera_reader = None
                print('[card-camera] released after stopped recognition', flush=True)
            if (shape is not None and not tilting and (frames % shape_period == 0)
                    and not (startup_card_only and start_gate.shape_passed)):
                classify_card = bool(
                    (gate_window_open and start_gate is not None
                     and start_gate.require_shape and not start_gate.shape_passed)
                    or stopped_vote_needed)
                card_detection_mode = "classify" if classify_card else "presence"
                shape_update_start_ns = time.perf_counter_ns()
                shape_frame = frame
                card_frame_source = "primary"
                dual_vote = startup_shape_needed
                primary_card_dbg = None
                if dual_vote:
                    # Each camera owns its classifier state. Fuse the two
                    # results into at most one vote for this capture cycle.
                    _shape_name, card_dbg = shape.update(
                        frame,
                        classify=True,
                        proposal_mode=args.shape_proposal_mode)
                    primary_card_dbg = card_dbg
                    card_primary_shape = card_dbg.get("shape")
                if dual_vote and card_camera_reader is not None:
                    # Startup retains its fresh secondary-camera sampling.
                    vote_start = 0.0
                    shape_frame, card_camera_meta, card_camera_status = (
                        card_camera_reader.sample(vote_start))
                    if shape_frame is not None:
                        card_frame_source = "secondary"
                    else:
                        shape_frame = frame
                if dual_vote and card_frame_source == "secondary":
                    _shape_name, card_dbg = precision_shape.update(
                        shape_frame,
                        classify=True,
                        proposal_mode=args.shape_proposal_mode)
                    card_secondary_shape = card_dbg.get("shape")
                    if primary_card_dbg.get("presence"):
                        # A card visible to either camera remains present while
                        # the stopped vote is active.
                        card_dbg = dict(card_dbg, presence=True)
                elif not dual_vote:
                    _shape_name, card_dbg = shape.update(
                        frame,
                        classify=classify_card,
                        proposal_mode=args.shape_proposal_mode)
                if dual_vote:
                    fused_shape, card_shape_source = resolve_card_shapes(
                        card_primary_shape, card_secondary_shape)
                    if startup_shape_needed:
                        startup_shape_candidate = fused_shape
                        startup_shape_source = card_shape_source
                    else:
                        card_dbg = dict(card_dbg, shape=fused_shape)
                shape_update_ms = (time.perf_counter_ns() - shape_update_start_ns) / 1_000_000.0
                # Count this call's two independent stop-gate evidence paths,
                # including early cue hits that have not filled its vote window.
                if (card_detection_mode == "presence" and not gate_window_open
                        and not startup_first_card_pending
                        and not cards.active and processed >= card_until
                        and last_cmd_vx > 0.0):
                    if card_dbg.get("cue_hit"):
                        card_stop_cue_hit_count += 1
                # Only current, history-confirmed cue geometry can stop walking.
                # Stopped/startup classifications never supply a width stop gate.
                card_reach = card_dbg.get("presence_cy_frac")
                card_reach_line = card_trigger_frac
                if recording_started and args.shape_dump and (card_dbg.get("card_found")
                                                               or card_dbg.get("presence")):
                    dumped += 1
                    stem = os.path.join(
                        args.shape_dump,
                        f"{dumped:04d}_{card_dbg.get('shape')}"
                        f"_cy{card_dbg.get('presence_cy_frac')}"
                        f"_g{card_dbg.get('card_found') and 1 or 0}")
                    cv2.imwrite(stem + ".jpg", shape_frame)
                    with open(stem + ".json", "w", encoding="utf-8") as handle:
                        json.dump({"t": processed, **card_dbg}, handle,
                                  indent=1, default=str)
                if gate_window_open or startup_first_card_lock:
                    startup_card_present = startup_card_present or bool(card_dbg.get('presence'))
                if startup_first_card_lock:
                    if (not startup_first_card_pending and processed >= card_until
                            and last_cmd_vx > 0.0
                            and not bool(card_dbg.get("presence"))):
                        startup_first_card_clear_calls += 1
                        if startup_first_card_clear_calls >= args.card_clear_calls:
                            startup_first_card_lock = False
                            startup_first_card_clear_calls = 0
                            startup_card_present = False
                            print("[start-gate] 首卡已离开；已允许下一张近距图卡停车（仍受冷却约束）", flush=True)
                    else:
                        startup_first_card_clear_calls = 0
                normal_walking = (not gate_window_open and not startup_first_card_pending
                    and not startup_first_card_lock and processed >= card_until
                    and processed >= startup_next_stop_at and not cards.active)
                if (normal_walking and cards.trigger(
                        processed, bool(card_dbg.get('presence')),
                        card_reach is not None and card_reach > card_reach_line,
                        card_feedback)):
                    detector.restore_tracking_state(tracking_before_frame)
                    card_stop_cue_trigger_count += 1
                    print(f"[card] 停车读卡：reach={card_reach:.3f}, "
                          f"threshold={card_reach_line:.3f}; 等待站定和重摆", flush=True)
                # Why a card that is plainly in view did not become an action: the
                # quad gates (found/closure), the classifier (shape/rules), or the
                # consecutive-frame latch. Only while a card is around, at 4 Hz.
                if ((card_dbg.get("card_found") or card_dbg.get("presence"))
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
                        f"raw={fmt(card_dbg.get('cue_score_raw'), '.2f')} "
                        f"photo={fmt(card_dbg.get('photometric_mean'), '.1f')}/"
                        f"{fmt(card_dbg.get('photometric_std'), '.1f')} "
                        f"scale={fmt(card_dbg.get('photometric_contrast_scale'), '.2f')} "
                        f"found={int(bool(card_dbg.get('card_found')))} "
                        f"shape={card_dbg.get('shape') or '-'} "
                        f"quad={card_dbg.get('quad_total', '?')}"
                        f"g{card_dbg.get('quad_geom', '?')}"
                        f"v{len(card_dbg.get('scores') or [])} "
                        f"rej={' '.join(f'{k}:{n}' for k, n in rej) or '-'} "
                        f"phase={cards.phase} "
                        f"mode={card_detection_mode} "
                        f"source={card_dbg.get('shape_proposal_source') or '-'} "
                        f"cameras={card_shape_source or '-'}",
                        flush=True)
            # Startup card confirmation and release retain the existing flow.
            if start_gate is not None:
                if (start_gate.require_shape and not start_gate.shape_passed
                        and not (args.start_gate == "button" and start_gate.button_passed)
                        and startup_shape_source == "conflict"):
                    start_gate.reject_shape_conflict()
                    print(f"[start-gate] 图卡双摄冲突：主={card_primary_shape} "
                          f"副={card_secondary_shape}；本帧弃票并清除旧票", flush=True)
                if (start_gate.require_shape and not start_gate.shape_passed
                        and not (args.start_gate == "button" and start_gate.button_passed)
                        and startup_shape_candidate):
                    if start_gate.observe_shape(startup_shape_candidate):
                        evidence = (f"最近 {len(start_gate.shape_votes)} 票 "
                                    f"{start_gate.vote_tally()}"
                                    if start_gate.shape_voting else
                                    f"连续 {start_gate.shape_streak} 帧")
                        print("\n" + "=" * 68, flush=True)
                        print(f"  ◆◆◆  阀2 通过：图卡 "
                              f"{CARD_NAMES_ZH.get(start_gate.last_shape, '?')}"
                              f"（{start_gate.last_shape}）"
                              f"    {evidence} 来源={startup_shape_source}", flush=True)
                        print("=" * 68 + "\n", flush=True)
                button_pressed = False
                if startup_button_link is not None:
                    startup_button_link.host_ready = client.poll_startup_ready()
                    button_pressed = startup_button_link.poll(start_gate.shape_passed)
                if (startup_button_link is not None
                        and start_gate.observe_button(button_pressed)):
                    # Initialize all outputs before C starts, so its console,
                    # control/IMU trace and motor positions share this session.
                    begin_recording()
                    print("[button-start] 收到PC2按钮；交接串口并启动步态。", flush=True)
                if startup_button_link is not None:
                    # Record transmission and STM32 state separately from the
                    # recognized class; the lamp itself has no readback bit.
                    startup_serial_debug = startup_button_link.diagnostics()
                    if isinstance(startup_serial_debug, dict):
                        debug.update(startup_serial_debug)
                if start_gate.passed and gate_window_open:
                    if policy_launcher is not None and not policy_launcher.started:
                        if startup_button_link is not None:
                            startup_button_link.close()
                            startup_button_link = None
                        policy_launcher.start()
                    if policy_launcher is None or policy_launcher.ready():
                        gate_released = True
                        start_released = True
                        if args.start_gate in ("both", "button") and start_gate.shape_passed:
                            startup_first_card = shape_numbers[start_gate.last_shape]
                            startup_first_card_pending = True
                            startup_first_card_lock = True
                            startup_first_card_clear_calls = 0
                            startup_sequence.reset()  # begins on first released camera frame
                            release_source = ("按钮已按下、首卡已锁存" if args.start_gate == "button"
                                              else "二维码和首卡均已锁存")
                            print(f"[start-gate] {release_source}："
                                  f"{start_gate.last_shape} -> {startup_first_card}; "
                                  f"启动序列 {startup_sequence.describe()}，再停车直接执行首卡（不等近距触发/二次投票）",
                                  flush=True)
                            if (len(startup_sequence.steps) > 1 and args.command_min_hold_s >
                                    min(s.duration_s for s in startup_sequence.steps)):
                                print('[start-gate] 注意：COMMAND_MIN_HOLD_S 大于序列最短段，'
                                      '模型可能延迟切换；请调小保持时间并检查 control CSV。', flush=True)
            startup_pair = None
            if startup_first_card_pending and not gate_window_open:
                startup_pair = startup_sequence.command(processed)
                debug.update(startup_sequence_index=startup_sequence.index,
                             startup_sequence_remaining_s=max(0., (startup_sequence.deadline or processed)-processed))
            if startup_first_card_pending and not gate_window_open and startup_pair is None:
                # The first card may already be inside the normal proximity line.
                # Its identity was confirmed before QR release, so there is no
                # benefit in walking farther just to trigger another vote.
                startup_first_card_pending = False
                startup_first_card_clear_calls = 0
                card_action = startup_first_card
                card_event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                card_until = processed + card_action_stop_s(card_action, args.card_hold_ms)
                startup_next_stop_at = card_until + STARTUP_CARD_RETRIGGER_GAP_S
                recognized_this_frame = True
                print(f"[start-gate] 首卡启动序列结束；停车并发送已锁存的 "
                      f"action={card_action} event={card_event_id}", flush=True)
            # Each newly classified frame casts at most one vote, never a cached result.
            if stopped_vote_needed and card_dbg is not None:
                completed_at = time.monotonic()
                cards.observe(completed_at, frame_observed_at, card_dbg.get('shape'), card_feedback)
                cards.advance(completed_at, card_feedback)
            if cards.phase == 'dispatch':
                event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                action_number = shape_numbers[cards.winner]
                cards.dispatch(time.monotonic(), event_id, action_number,
                               card_action_stop_s(action_number, args.card_hold_ms), card_feedback)
                recognized_this_frame = True
            card_vote_decision = cards.decision
            if card_vote_decision is not reported_card_decision:
                reported_card_decision = card_vote_decision
                print(f"[card] 投票结果 {card_vote_decision}; phase={cards.phase}", flush=True)

            if card_action != -1 and processed >= card_until:
                card_action = -1
                card_event_id = 0
            # Pause normal steering throughout reading, restoration and action.
            if (cards.active or processed < card_until or gate_window_open):
                # Not running the controller here is the whole point: it would be fed
                # the card-corrupted err for the length of the stop, and its integral
                # and filtered derivative would carry that corruption into the first
                # real frame. Idle it, and leave the state it brought in untouched - the
                # stop is a pause in the walking state, not the start of a new one.
                # Nothing else to do: the controller is not called here, so nothing it
                # holds can reach the wheels, and the handover below drops the stored
                # command before the first frame that does call it.
                vx, wz = 0.0, 0.0
            elif startup_first_card_pending:
                # Explicit first-card commands; normal line steering resumes after
                # the sequence/action. No model, protocol or command-hold changes.
                vx, wz = startup_pair
            else:
                vx, wz = controller.command(debug, confidence, processed - previous)
                debug.update(getattr(controller, "diagnostics", {}))
                # Printed on the transition, not every frame: one line per time the
                # near band hands over an offset the lane cannot produce. How often
                # this fires on a real lap is the measurement.
                if controller.rejected_lateral is not None:
                    if lateral_warned is None:
                        print(f"[vision] near band says "
                              f"{controller.rejected_lateral:+.1f}cm, past the "
                              f"{args.max_lateral_cm:.1f}cm lane half-width; discarding "
                              f"steering and retaining previous walking speed", flush=True)
                    lateral_warned = True
                else:
                    lateral_warned = False
            previous = processed
            visible_qr = (cards.action if cards.event_id else card_action) if recognized_this_frame else -1
            event = ({"event_id": cards.event_id, "event_action": cards.action}
                     if cards.event_id else
                     {"event_id": card_event_id, "event_action": card_action}
                     if card_event_id else {})
            # Ordinary cards own their pose request; startup never re-poses here.
            in_card_window = cards.active or processed < card_until
            # --hold-still：只观测不驱动。台架测"机身姿态对读数的影响"时要开
            # C（姿态是 C 从 STM32 读了广播的），但 C 一使能电机、B 这边一发
            # vx=0.2 车就走了。这里把命令压成 0 —— 检测、日志、图卡那套逻辑
            # 全都照跑，只有"发出去的 vx/wz"是零，于是策略原地站着（也就是
            # 那个后仰的站姿），读数还能正常观察。
            if args.hold_still:
                vx, wz = 0.0, 0.0
                controller.drop_held_command()
            if args.wz_mode in ("heading", "segments") and (in_card_window or gate_window_open or args.hold_still):
                debug.update(steering_reason="external_stop", steering_applied_wz=0.0,
                             command_hold_remaining_s=0.0)
                if args.wz_mode == "segments":
                    debug.update(segment_control_active=False, segment_applied_wz=0.0,
                                 segment_shadow_vx=0.0, segment_shadow_wz=0.0,
                                 segment_shadow_reason="external_stop",
                                 segment_gate_reason="external_stop", segment_confirm_frames=0)
            debug['steering_frame_wz'] = wz
            window_bypass = (in_card_window or gate_window_open or args.hold_still
                             or startup_first_card_pending)
            debug.update(steering_window_reason='external_bypass' if window_bypass else 'connector_collection',
                         steering_window_duration_s=args.steering_command_window_s,
                         steering_window_median_side=args.steering_command_median)
            road_before = straight_guard.state
            straight_guard.update(debug.get('straight_evidence','UNKNOWN'), processed,
                line_frame_token, paused=in_card_window or args.hold_still)
            wz_before_guard = wz
            straight_bypass = (in_card_window or gate_window_open
                or args.hold_still or startup_first_card_pending or vx <= 0.)
            wz = straight_guard.limit(wz, bypass=straight_bypass)
            debug.update(straight_guard.diagnostics(processed,wz_before_guard,wz))
            debug['straight_guard_bypass'] = straight_bypass
            if road_before != straight_guard.state:
                print(f'[straight-road] {road_before}->{straight_guard.state}; '
                      f"evidence={debug.get('straight_evidence','UNKNOWN')} "
                      f"reason={debug.get('straight_reject_reason','unavailable')}", flush=True)
            debug['steering_applied_wz'] = wz
            if args.wz_mode == 'segments':
                debug['segment_applied_wz'] = wz
            if (args.wz_mode in ('heading', 'segments') and vx > 0.0
                    and not in_card_window and not gate_window_open
                    and not args.hold_still and not startup_first_card_pending
                    and debug.get('measurement_valid')
                    and debug.get('steering_reason') in ('new_block', 'continue_block')):
                source = debug.get('steering_heading_source')
                if (debug.get('tracking_mode') == 'BOTH'
                        and source not in ('left_boundary', 'right_boundary')):
                    steering_paired_count += 1
                elif (debug.get('tracking_mode') in ('LEFT_ONLY', 'RIGHT_ONLY')
                      and source in ('left_boundary', 'right_boundary')):
                    steering_single_count += 1
            last_cmd_vx = vx
            # Aggregate policy/card logic; QR and shape fields below are sub-stages.
            decision_ms = (time.perf_counter_ns() - decision_start_ns) / 1_000_000.0
            # 门控期间按住直立：策略自己的站姿后仰约 20°，而相机 45° 是在直立时
            # 标定的，几何闸只认 38.6~59° —— 不扳直，阀2 会把每一张卡都拒掉，机器人
            # 永远不走。释放那帧仍然按着（gate_window_open 是上一帧的值），下一帧
            # 才落，card_tilt 那条线上不会和它撞在同一帧。
            udp_publish_start_ns = time.perf_counter_ns()
            # One completed visual frame = one candidate. Collection happens on
            # the connector clock, after all geometric limits, never backdated to
            # this frame's pre-detection timestamp. Card/startup stops bypass it.
            window_metadata = ({'steering_window': dict(seconds=args.steering_command_window_s,
                median=args.steering_command_median, valid=not line_lost(debug, confidence))}
                if args.wz_mode in ('heading', 'segments') and args.steering_command_window_s > 0
                and not window_bypass and vx > 0 else {})
            if window_metadata:
                if window_origin_s is None:
                    window_origin_s = processed
                window_metadata['steering_window']['origin_s'] = window_origin_s
            else:
                window_origin_s = None
            client.publish(vx, wz, visible_qr,
                           hold_upright=(gate_window_open and start_gate is not None
                                         and start_gate.require_shape),
                           card_tilt=cards.tilt_requested,
                           **({"startup_sequence_active": True}
                              if startup_first_card_pending and not gate_window_open
                              and not in_card_window and not args.hold_still else {}),
                           **({"command_mode": "held"} if args.wz_mode in ("heading", "segments") else {}),
                           **window_metadata,
                           **event)
            udp_publish_ms = (time.perf_counter_ns() - udp_publish_start_ns) / 1_000_000.0
            command_host_time_ns = time.time_ns()
            if capture_metadata:
                capture_metadata['camera_return_to_publish_ms'] = (
                    time.monotonic()-capture_metadata['camera_capture_returned_s'])*1000
            read_to_publish_ms = (time.perf_counter_ns() - read_return_ns) / 1_000_000.0
            recording_submit_ms = None
            if command_video is not None:
                recording_submit_start_ns = time.perf_counter_ns()
                try:
                    actual_command = None
                    if attitude is not None:
                        attitude.poll()
                        actual_command = attitude.executed_command()
                    # Startup has no lane geometry; keep recording its camera view.
                    video_frame, camera_fallback = recording_frame(frame, debug, args.video_source)
                    command_video.submit(video_frame, frame_id=frames, host_time_ns=command_host_time_ns,
                        monotonic_s=time.monotonic(), vx=vx, wz=wz, lost=line_lost(debug, confidence),
                        executed=actual_command, lane_debug=debug, camera_fallback=camera_fallback)
                except Exception as exc:
                    # Auxiliary diagnostics must not escape into the motor loop.
                    print(f'[video] frame submission disabled: {exc}', flush=True)
                    command_video.error = str(exc)
                recording_submit_ms = (
                    time.perf_counter_ns() - recording_submit_start_ns) / 1_000_000.0
            if line_log is not None:
                # Card detection is sampled; only attach this frame's actual
                # diagnostics, never relabel a previous detection as fresh.
                telemetry_debug = dict(debug)
                if card_dbg:
                    telemetry_debug.update({"card_" + key: value for key, value in card_dbg.items()
                        if key.startswith(("photometric_", "cue_", "shape_"))})
                repose_feedback_id = getattr(attitude, 'card_tilt_event_id', None)
                repose_feedback_done = getattr(attitude, 'card_tilt_done', None)
                # LOG=vision alone also records fresh adopted policy commands.
                # Drain feedback after perception; never substitute visual wz.
                if command_video is None and attitude is not None:
                    attitude.poll()
                execution = execution_log_fields(
                    attitude.executed_command() if attitude is not None else None)
                line_log.write(
                    telemetry_debug, frame=frames, host_time_ns=command_host_time_ns,
                    **runtime_counters.sample(),
                    previous_loop_tail_ms=previous_loop_tail_ms,
                    **execution,
                    process_monotonic_s=processed, confidence=confidence,
                    camera_async=args.camera_async, **capture_metadata,
                    loop_period_ms=loop_period_ms, camera_read_ms=camera_read_ms,
                    line_process_ms=line_process_ms, qr_decode_ms=qr_decode_ms,
                    shape_update_ms=shape_update_ms, decision_ms=decision_ms,
                    pre_line_ms=(line_process_start_ns-read_return_ns)/1e6,
                    decision_other_ms=max(0., decision_ms-(shape_update_ms or 0.)-(qr_decode_ms or 0.)),
                    card_detection_mode=card_detection_mode,
                    card_frame_source=card_frame_source,
                    card_primary_shape=card_primary_shape,
                    card_secondary_shape=card_secondary_shape,
                    startup_shape_source=startup_shape_source,
                    card_shape_source=card_shape_source,
                    card_camera_status=card_camera_status,
                    card_camera_sequence=card_camera_meta.get('camera_sequence'),
                    card_camera_frame_age_ms=card_camera_meta.get('camera_frame_age_ms'),
                    steering_paired_count=steering_paired_count,
                    steering_single_count=steering_single_count,
                    card_stop_box_hit_count=0,  # Legacy log field; width path removed.
                    card_stop_cue_hit_count=card_stop_cue_hit_count,
                    card_stop_box_trigger_count=0,
                    card_presence_mode="cue_only",
                    card_stop_cue_trigger_count=card_stop_cue_trigger_count,
                    card_stop_first_card_lock=bool(startup_first_card_lock),
                    card_stop_current_presence=(bool(card_dbg.get('presence')) if card_dbg else None),
                    card_stop_current_box=(bool(card_dbg.get('card_found')) if card_dbg else None),
                    card_stop_reach=card_reach, card_stop_reach_line=card_reach_line,
                    card_stop_reach_units='fraction',
                    **cards.diagnostics(processed),
                    card_vote_reason=(card_vote_decision or {}).get('reason'),
                    card_vote_winner=(card_vote_decision or {}).get('winner'),
                    udp_publish_ms=udp_publish_ms, read_to_publish_ms=read_to_publish_ms,
                    recording_submit_ms=recording_submit_ms,
                    vx=vx, wz=wz, mode=args.wz_mode,
                    body_track_deviation_deg=debug.get('heading_control_deg'),
                    body_track_deviation_valid=bool(debug.get('heading_control_valid', False)),
                    card_window=in_card_window, start_gate=gate_window_open,
                    start_gate_mode=args.start_gate, start_released=start_released,
                    qr_passed=bool(start_gate and start_gate.qr_passed),
                    shape_passed=bool(start_gate and start_gate.shape_passed),
                    button_passed=bool(start_gate and start_gate.button_passed),
                    hold_still=args.hold_still, event_id=event.get('event_id', 0),
                    camera_pitch_deg=effective_pitch,
                    turn_remaining_s=getattr(controller, "turn_left", None),
                    gap_remaining_s=getattr(controller, "gap_left", None),
                )
            gate_window_open = start_gate is not None and not start_released
            if start_gate is not None and processed - gate_last_log >= args.start_gate_log_s:
                gate_last_log = processed
                scans = qr_reader.scans if qr_reader is not None else 0
                rej = qr_reader.geom_rejects if qr_reader is not None else 0
                cost = qr_reader.last_cost_ms if qr_reader is not None else 0.0
                gate_state = "等待起步" if gate_window_open else "已放行"
                print(f"[start-gate] {start_gate.status()} | 扫 {scans} 次/"
                      f"{rej} 拒 单次 {cost:.0f}ms | 门控={gate_state} "
                      f"vx={vx:+.3f} wz={wz:+.3f}", flush=True)
            if processed - last_log >= 0.5:
                # Left of the bar is what the robot is doing; right of it is why.
                # Read only the left if it is behaving.
                print(
                    f"[vision] {log_frames / max(processed - last_log_at, 1e-6):4.0f}Hz "
                    f"vx={vx:+.3f} wz={wz:+.3f} "
                    f"err={debug.get('fused_err_cm', 0.0):+.1f}cm "
                    f"near={fmt(debug.get('near_error_cm'), '+.1f')}cm "
                    f"preview={fmt(debug.get('preview_error_cm'), '+.1f')}cm "
                    f"valid={int(bool(debug.get('measurement_valid', confidence > 0)))} "
                    f"reason={debug.get('steering_reason', 'legacy')} "
                    f"decision={debug.get('steering_decision', '-')} "
                    f"braked={int(bool(debug.get('steering_braked', False)))} "
                    f"age={fmt(debug.get('measurement_age_s'), '.3f')}s "
                    f"lock_w={fmt(debug.get('bottom_lock_weight'), '.2f')} "
                    f"conf={confidence:.2f} qr={visible_qr}"
                    # Same id the connector and policy log, so the three can be
                    # lined up by hand when an event goes missing in the middle.
                    + (f"/ev{event['event_action']}#{event['event_id']}" if event else "")
                    + " | "
                    f"steer={controller.last_steer:+.2f} "
                    f"eff={controller.last_err_eff:+.1f}{'deg' if args.wz_mode in ('heading', 'segments') else 'cm'} "
                    f"ground_ang={fmt(debug.get('heading_control_deg'), '+.1f')} "
                    f"geom={int(bool(debug.get('heading_control_valid', False)))} "
                    f"pts={debug.get('heading_control_points', 0)} "
                    f"rmse={fmt(debug.get('heading_control_pixel_rmse_px'), '.1f')}px "
                    f"near_z={fmt(debug.get('near_z_cm'), '.1f')}cm "
                    f"single={fmt(debug.get('single_edge_heading_deg') if debug.get('single_edge_valid', False) else None, '+.1f')} "
                    f"predict_ang={fmt(debug.get('steering_predicted_heading_deg'), '+.1f')} "
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
                if processed - last_counts_log >= 5.0:
                    print(f"[counts] {time.strftime('%H:%M:%S')} frame={frames} "
                          f"累计有效巡线帧: 双边={steering_paired_count} "
                          f"单边={steering_single_count} | "
                          f"累计行走中图卡候选命中（仅cue）: "
                          f"cue={card_stop_cue_hit_count} "
                          f"| 实际停车事件: "
                          f"cue高度={card_stop_cue_trigger_count} "
                          f"(候选命中不等于停车)", flush=True)
                    last_counts_log = processed
                last_log = processed
                last_log_at = processed
                log_frames = 0
                if args.lane_fit or args.lane_fit_segments:
                    # 只读诊断：整条车道拟合读出来的两点中心（px，相对画面中心）。
                    # "远 − 近" 只是整条曲线的一个粗略诊断；分段模式还分别记录
                    # 三个地面距离范围的局部方向。按检测器自己的 LUT 反算，
                    # 车在正中、对准切线时：直道 0；R=0.776m 的圆弧上前视
                    # 45cm 处车道中心偏 14cm=55px、65cm 处偏 35cm=120px，而拟合对
                    # 检测器自身读数的实测误差只有 5~11px。top 是链条真正爬到的地面
                    # 距离 —— 它小于 far-cm 时 far 那一点就是外推，代码会直接不给值
                    # （fit_ok 为假），所以 top 偏低只会丢帧，不会喂假数。
                    if debug.get("fit_ok"):
                        print(f"[lane-fit] pts={debug.get('fit_pts')} "
                              f"top={debug.get('fit_top_cm') or 0:.0f}cm "
                              f"near={debug.get('fit_near_px'):+.0f}px "
                              f"far={debug.get('fit_far_px'):+.0f}px "
                              f"远-近={debug.get('fit_curve_px'):+.0f}px", flush=True)
                    else:
                        print(f"[lane-fit] 拟合不出（pts={debug.get('fit_pts', 0)}, "
                              f"top={debug.get('fit_top_cm') or 0:.0f}cm）",
                              flush=True)
                if args.lane_fit_segments:
                    segment_angles = ' '.join(
                        f"seg{i}={fmt(debug.get(f'fit_seg{i}_heading_deg'), '+.1f')}°"
                        for i in range(len(args.segment_regions_cm)))
                    print(f"[lane-segments] anchored={int(bool(debug.get('fit_seg_anchored')))} "
                          f"segments={debug.get('fit_seg_count', 0)} "
                          f"{segment_angles} "
                          f"change={fmt(debug.get('fit_seg_heading_change_deg'), '+.1f')}° "
                          f"pattern={debug.get('fit_seg_pattern', 'insufficient_support')}",
                          flush=True)
                if args.wz_mode == "segments":
                    print(f"[segment-control] active={int(bool(debug.get('segment_control_active')))} "
                          f"confirm={debug.get('segment_confirm_frames', 0)} "
                          f"target={fmt(debug.get('segment_target_z_cm'), '.1f')}cm "
                          f"bearing={fmt(debug.get('segment_target_bearing_deg'), '+.1f')}° "
                          f"actual={wz:+.3f} heading_shadow={fmt(debug.get('segment_shadow_wz'), '+.3f')} "
                          f"gate={debug.get('segment_gate_reason')}", flush=True)
            if not args.headless:
                cv2.putText(frame, f"vx={vx:+.3f} wz={wz:+.3f} Q=quit", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Policy vision", frame)
                show_debug_windows(debug, visualization)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # Includes recording, telemetry, console and GUI after command send.
            previous_loop_tail_ms = (time.perf_counter_ns()-udp_publish_start_ns)/1e6 - udp_publish_ms
    finally:
        try:
            if line_log is not None:
                line_log.close()
        finally:
            if startup_button_link is not None:
                try:
                    startup_button_link.close()
                except (OSError, RuntimeError) as exc:
                    print(f"[button-start] closing startup serial: {exc}", flush=True)
            client.close()
            if policy_launcher is not None:
                policy_launcher.close()
            if attitude is not None:
                attitude.close()
            if card_camera_reader is not None:
                if not card_camera_reader.close():
                    print('[card-camera] capture read still blocked', flush=True)
            if camera_reader is not None:
                if not camera_reader.close():
                    print('[camera] capture read still blocked; worker will release on return', flush=True)
            elif cap is not None:
                cap.release()
            if not args.headless:
                cv2.destroyAllWindows()
            if command_video is not None:
                command_video.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
