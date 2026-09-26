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
    parser.add_argument("--max-wz-right", type=float,
                        default=float(os.getenv("MAX_WZ_RIGHT", "0.25")),
                        help="Yaw-rate limit for right turns, as opposed to --max-wz "
                             "for left. Negative wz is a right turn in the published "
                             "log. A right curve needing more than this runs wide")
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
    if args.card_every_stopped < 1:
        parser.error("card-every-stopped must be at least 1")
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
        max_lateral_cm=args.max_lateral_cm,
        max_wz_right=args.max_wz_right,
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
        if args.no_red_detect:
            detector.red_detect_enable = False
        print(f"Camera {args.camera}: {width}x{height}; UDP -> "
              f"{args.connector_host}:{args.connector_port}; vx={args.vx} m/s; "
              f"max_wz={args.max_wz} rad/s; yaw_sign={args.yaw_sign}; "
              f"red={'off' if args.no_red_detect else 'on'}", flush=True)
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
        card_armed = True        # and it has since been seen far enough to trigger
        card_action_triggered = False
        card_dbg = {}
        lateral_warned = None    # None until the lateral bound first trips
        card_window_open = False
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
                # Cold start on the stopped-to-walking edge, BEFORE this frame is
                # processed. Doing it after detector.process() left the first walking
                # frame computed from the polluted state, so only the second frame
                # was clean - and the first is the one that decides where it goes.
                # Both halves: the controller was already reset every stopped frame;
                # the detector was not, and its state is what keeps a bad lock alive
                # - last_lane_center_x is the next frame's scan hint and smoothed_err
                # is a long EMA. A fresh process tracks this same curve fine, so
                # start the frame the same way.
                controller.reset(clear_hold=True)
                detector.reset_state()
                print(f"[vision] card window closed; cold start "
                      f"(hold={controller.hold[0]:+.2f},{controller.hold[1]:+.2f} "
                      f"lost_s={controller.lost_s:.2f})", flush=True)
            card_window_open = window_open
            _, _, confidence, visualization, debug = detector.process(frame)
            frames += 1
            log_frames += 1
            recognized_this_frame = False
            # Stopped in front of a card, the camera is steady, so the classification
            # can have every frame. --shape-every only throttles the driving case.
            if shape is not None and (frames % (
                    args.card_every_stopped
                    if (processed < stop_until or processed < card_until)
                    else args.shape_every) == 0):
                action, card_dbg = shape.update(
                    frame, lane_offset_cm=float(debug.get("base_err_cm", 0.0)))
                # Phase two waits for phase one. The classifier is only reliable on a
                # card that is close, and the trigger line is what says it is. Acting
                # as soon as a shape appears classified a card 50 cm away - top=186,
                # cy=0.40 - on a small warp: rules called it diamond, then triangle,
                # while hu read circle at distance 0.01-0.04 throughout and was the
                # one that was right. The box only reaches that far down the frame
                # after --card-trigger-frac, so nothing may act before it.
                reach = card_dbg.get("presence_cy_frac")
                reached = reach is not None and reach >= args.card_trigger_frac
                if action is not None and reached and not card_action_triggered and card_event_id == 0:
                    card_action = action
                    card_event_id = max(1, (time.time_ns() // 1_000_000) & 0xFFFFFFFF)
                    card_until = processed + args.card_hold_ms / 1000.0
                    card_action_triggered = True
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
                # A trigger has to be earned again by seeing the card well above the
                # line. presence is armed-gated, so the action firing makes it read
                # false and card_absent clears card_triggered on its own; the card the
                # robot just drove past is still in frame at cy 0.84, well below the
                # line, and stopped the robot a second time mid-curve. A card that is
                # already low in the frame has not been approached, so it cannot fire.
                if cy is not None and cy < args.card_trigger_frac:
                    card_armed = True
                if (card_flag and not card_triggered and card_armed and cy is not None
                        and cy >= args.card_trigger_frac):
                    card_triggered = True
                    card_armed = False
                    stop_until = processed + args.card_stop_ms / 1000.0
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
                        f"quad={card_dbg.get('quad_total', '?')}"
                        f"g{card_dbg.get('quad_geom', '?')}"
                        f"v{len(card_dbg.get('scores') or [])} "
                        f"rej={card_dbg.get('geom_rejects') or {}} "
                        f"sc={[(s, round(c, 2)) for s, c in (card_dbg.get('scores') or [])[:3]]} "
                        f"top={fmt(card_dbg.get('box_top_work'), '.0f')} "
                        f"cy={fmt(card_dbg.get('presence_cy_frac'), '.2f')} "
                        f"cue={fmt(card_dbg.get('presence_cue'), '.2f')} "
                        f"cuecy={fmt(card_dbg.get('cue_cy_frac'), '.2f')} "
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
                # Do not run the controller here. It is fed the card-corrupted err for
                # the whole stop, and its integral and filtered derivative then carry
                # that corruption into the first real frame - the loop restarts off
                # the equilibrium its P/lookahead/bias terms had settled into on the
                # curve. Idle and reset instead: the first frame after the stop is a
                # clean start from the live frame.
                vx, wz = 0.0, 0.0
                # clear_hold: otherwise the first invalid frame after the stop
                # republishes the pre-stop (vx, wz) as the lost-line fallback, which
                # is the replayed command all over again.
                controller.reset(clear_hold=True)
            else:
                vx, wz = controller.command(debug, confidence, processed - previous)
                # Printed on the transition, not every frame: one line per time the
                # near band hands over an offset the lane cannot produce. How often
                # this fires on a real lap is the measurement.
                if controller.rejected_lateral is not None:
                    if lateral_warned is None:
                        print(f"[vision] near band says "
                              f"{controller.rejected_lateral:+.1f}cm, past the "
                              f"{args.max_lateral_cm:.1f}cm lane half-width; holding "
                              f"then stopping instead of steering on it", flush=True)
                    lateral_warned = True
                else:
                    lateral_warned = False
                # card_flag alone, not "and not card_triggered": a card that never got
                # classified lets --card-stop-ms expire, and the robot used to resume at
                # full speed straight past it - the one place the near band is fully
                # covered by the card. Creep instead, so the classifier still has frames
                # to work with before the card is behind the robot.
                if card_flag:
                    vx = min(vx, args.card_slow_vx)
            previous = processed
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
                    # Why a frame produced nothing: the bottom lock (the pairing
                    # check that keeps the near band on the right line) and how many
                    # rows it paired. conf=0 with pair=0 means no band at all; conf=0
                    # with lock=0 means the near band was rejected as asymmetric.
                    f"lock={int(bool(debug.get('bottom_lock_valid', False)))}"
                    f"pair={debug.get('bottom_pair_ratio', 0.0):.2f} "
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
