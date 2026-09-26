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
import collections
import math
import os
import signal
import time

from camera_config import load as load_camera
from policy_bridge import ConnectorClient, SteeringController


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
    parser.add_argument("--vx", type=float, default=0.4,
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
                        default=float(os.getenv("STEER_BIAS_CM", "5")),
                        help="Standing trim added to fused_err_cm on curves, shifting where "
                             "the loop settles to cancel a one-sided lateral offset. Set it "
                             "to the err the log shows standing in the curve")
    parser.add_argument("--bias-gate-px", type=float,
                        default=float(os.getenv("STEER_BIAS_GATE_PX", "12")),
                        help="abs(curve_px) at which --bias-cm is fully applied; it fades "
                             "to zero by curve_px 0 so straights are untouched")
    parser.add_argument("--no-shape-detect", action="store_true",
                        help="Skip geometric card detection entirely; qr stays -1")
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
    parser.add_argument("--card-trigger-frac", type=float,
                        default=float(os.getenv("CARD_TRIGGER_FRAC", "0.5")),
                        help="Box centroid height in the frame (0=top, 1=bottom) at which "
                             "the robot stops and identifies the shape. 0.5 is the middle; "
                             "stopping later than that leaves the robot within 10 cm of the "
                             "card by the time the stop lands. Seeing a card earlier only "
                             "slows it down")
    parser.add_argument("--card-slow-vx", type=float,
                        default=float(os.getenv("CARD_SLOW_VX", "0.2")),
                        help="Forward speed while a card is in view but not yet close "
                             "enough to act on")
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
    parser.add_argument("--card-clear-s", type=float,
                        default=float(os.getenv("CARD_CLEAR_S", "2.0")),
                        help="After the action, drive past the card on the replayed "
                             "command for at most this long, until the line detector "
                             "recovers. The card is left 20-35 cm ahead, inside the near "
                             "band, so steering from those frames is garbage")
    parser.add_argument("--card-clear-conf", type=float,
                        default=float(os.getenv("CARD_CLEAR_CONF", "0.8")),
                        help="Confidence that counts as the detector having recovered")
    parser.add_argument("--card-replay-s", type=float,
                        default=float(os.getenv("CARD_REPLAY_S", "1.0")),
                        help="During the clearance, hold the average command from this "
                             "many seconds of normal driving before the stop instead of "
                             "going straight - on a curve that keeps the turn rate, and "
                             "the approach was already slowed by --card-slow-vx")
    parser.add_argument("--shape-every", type=int,
                        default=max(1, int(os.getenv("SHAPE_EVERY", "6"))),
                        help="Run card detection every N frames. Measured at 1280x720: "
                             "31 ms with no card in view, 95-122 ms with one, against "
                             "29 ms for the line detector alone. 6 keeps the loop near "
                             "29 Hz clear and 23 Hz while a card is visible")
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
    if not math.isfinite(args.card_trigger_frac) or not 0 < args.card_trigger_frac <= 1:
        parser.error("card-trigger-frac must be in (0, 1]")
    if args.card_clear_calls < 1:
        parser.error("card-clear-calls must be at least 1")
    if args.card_stable_frames < 1:
        parser.error("card-stable-frames must be at least 1")
    if not math.isfinite(args.card_clear_s) or args.card_clear_s < 0:
        parser.error("card-clear-s must be finite and nonnegative")
    if not math.isfinite(args.card_clear_conf) or not 0.0 <= args.card_clear_conf <= 1.0:
        parser.error("card-clear-conf must be in [0, 1]")
    if not math.isfinite(args.card_replay_s) or args.card_replay_s < 0:
        parser.error("card-replay-s must be finite and nonnegative")
    if args.shape_every < 1:
        parser.error("shape-every must be at least 1")
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
    )
    # Lazy imports keep --help and controller tests usable without a camera stack.
    import cv2
    from line_detector_v1_warp import LineDetector
    from utils import open_camera, show_debug_windows

    shape = shape_names = None
    if not args.no_shape_detect:
        from shape_detector import ShapeDetector
        # run_robot.py's cooldown, so a card cannot re-fire while it is still in view.
        shape = ShapeDetector(stable_frames=args.card_stable_frames,
                              cooldown_ms=3200, debug=False)
        shape_names = {number: name for name, number in shape.action_map.items()}

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
        print(f"Camera {args.camera}: {width}x{height}; UDP -> "
              f"{args.connector_host}:{args.connector_port}; vx={args.vx} m/s; "
              f"max_wz={args.max_wz} rad/s; yaw_sign={args.yaw_sign}", flush=True)
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
        card_flag = False        # a card is in view on this approach
        card_absent = 0          # consecutive detection calls without one
        card_triggered = False   # this card has already been acted on
        card_action_triggered = False
        clear_deadline = 0.0     # > 0 while driving past the card without steering
        clear_pending = False    # an action ran; clear the card once its window ends
        clear_good = 0           # consecutive healthy frames during the clearance
        history = collections.deque()   # (t, vx, wz) of published commands
        replay = (0.0, 0.0)      # average of the last --card-replay-s before the stop
        card_dbg = {}
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
            _, _, confidence, visualization, debug = detector.process(frame)
            processed = time.monotonic()
            frames += 1
            log_frames += 1
            recognized_this_frame = False
            # Stopped in front of a card, the camera is steady, so the classification
            # can have every frame. --shape-every only throttles the driving case.
            if shape is not None and (frames % args.shape_every == 0
                                      or processed < stop_until
                                      or processed < card_until):
                action, card_dbg = shape.update(
                    frame, lane_offset_cm=float(debug.get("base_err_cm", 0.0)))
                if action is not None and not card_action_triggered and card_event_id == 0:
                    card_action = action
                    card_event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                    card_until = processed + args.card_hold_ms / 1000.0
                    card_action_triggered = True
                    clear_pending = True
                    recognized_this_frame = True
                    print(f"[shape] qr={action} ({shape_names.get(action, '?')}) "
                          f"held {args.card_hold_ms:.0f} ms", flush=True)
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
                # Seeing a card only slows the robot down. Stopping waits until the box
                # centroid has come down to the trigger line, i.e. the card is close.
                cy = card_dbg.get("presence_cy_frac")
                if (card_flag and not card_triggered and cy is not None
                        and cy >= args.card_trigger_frac):
                    card_triggered = True
                    stop_until = processed + args.card_stop_ms / 1000.0
                    # Last look at what the robot was doing while it could still see
                    # the line. The approach was already at --card-slow-vx, so this
                    # carries the slowed speed and the curve's turn rate forward.
                    recent = [h for h in history
                              if h[0] >= processed - args.card_replay_s]
                    if recent:
                        replay = (sum(h[1] for h in recent) / len(recent),
                                  sum(h[2] for h in recent) / len(recent))
                    controller.reset()
                    print(f"[shape] box centroid at {cy:.2f} -> stand still "
                          f"{args.card_stop_ms:.0f} ms", flush=True)
                # Why a card that is plainly in view did not become an action: the
                # quad gates (found/closure), the classifier (shape/rules), or the
                # consecutive-frame latch. Only while a card is around, at 4 Hz.
                if ((card_dbg.get("card_found") or card_dbg.get("presence") or card_flag)
                        and processed - last_shape_log >= 0.25):
                    last_shape_log = processed
                    hu_name = card_dbg.get("hu_best")
                    print(
                        f"[shape] found={int(bool(card_dbg.get('card_found')))} "
                        f"presence={int(bool(card_dbg.get('presence')))} "
                        f"shape={card_dbg.get('shape')} "
                        f"rules={card_dbg.get('shape_rules')} "
                        f"hu={hu_name + ':' if hu_name else '-'}"
                        f"{fmt(card_dbg.get('hu_dist'), '.3f')} "
                        f"score={fmt(card_dbg.get('closure'), '.2f')} "
                        f"top={fmt(card_dbg.get('box_top_work'), '.0f')} "
                        f"cy={fmt(card_dbg.get('presence_cy_frac'), '.2f')} "
                        f"cue={fmt(card_dbg.get('presence_cue'), '.2f')} "
                        f"armed={int(bool(getattr(shape, 'armed', True)))} "
                        f"cand={getattr(shape, 'candidate', None)}"
                        f"x{getattr(shape, 'candidate_count', 0)}", flush=True)
            if card_action != -1 and processed >= card_until:
                card_action = -1
                card_event_id = 0
            vx, wz = controller.command(debug, confidence, processed - previous)
            previous = processed
            if card_flag and not card_triggered:
                vx = min(vx, args.card_slow_vx)
            # Two windows, whichever ends later: --card-stop-ms caps how long we wait
            # for a shape that may never settle, --card-hold-ms is the rules' action
            # window once we do know it.
            if processed < stop_until or processed < card_until:
                vx, wz = 0.0, 0.0
            # Once the action is over the card is still 20-35 cm ahead, which is the
            # line detector's near band (z 20-27 cm). The near band then tracks the
            # card's border instead of the lane and confidence collapses, so any
            # steering derived from those frames is garbage - on the robot it swung
            # between the yaw limits. Drive straight at the slow speed until the
            # detector is healthy again, rather than for a fixed distance.
            # clear_pending, not card_triggered: the armed latch drops out a few
            # detection calls after the action fires (presence is armed-gated), so
            # card_triggered is already false by the time the 5 s window closes.
            if clear_pending and clear_deadline == 0.0 and processed >= max(stop_until, card_until):
                clear_deadline = processed + args.card_clear_s
                clear_pending = False
                clear_good = 0
                controller.reset()
                print(f"[shape] past the card holding vx={replay[0]:+.3f} "
                      f"wz={replay[1]:+.3f}, up to {args.card_clear_s:.1f}s", flush=True)
            if clear_deadline > 0.0:
                clear_good = clear_good + 1 if confidence >= args.card_clear_conf else 0
                if clear_good >= 3 or processed >= clear_deadline:
                    print(f"[shape] line detector back at {confidence:.2f}", flush=True)
                    clear_deadline = 0.0
                else:
                    # The replayed average, not zero: zero straightens the robot out
                    # of a curve it has not left yet. Capped so a late trigger cannot
                    # replay full speed while the near band is still blind.
                    vx, wz = min(replay[0], args.card_slow_vx), replay[1]
            history.append((processed, vx, wz))
            while history and history[0][0] < processed - 2.0 * args.card_replay_s:
                history.popleft()
            visible_qr = card_action if recognized_this_frame else -1
            event = ({"event_id": card_event_id, "event_action": card_action}
                     if card_event_id else {})
            client.publish(vx, wz, visible_qr, **event)
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
                    f"lost={debug.get('lost_frames', '?')}", flush=True)
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
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
