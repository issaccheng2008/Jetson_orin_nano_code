"""Decompose fused_err into its terms, with the robot parked where it is now."""
import sys

import cv2

sys.path.insert(0, "/home/isaac/jetson_orin_code/new_vision/jetson")
from line_detector_v1_warp import LineDetector

FRAMES = 90

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = LineDetector(1280, 720, cam_pitch_deg=45.0)
debug = None
for _ in range(FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    _, _, confidence, _, debug = detector.process(frame)
cap.release()

scale = detector.err_scale_cm
near_px = float(debug["near_err_px"])
far_px = float(debug["far_err_px"])
curve_px = float(debug.get("curve_px", 0.0) or 0.0)
angle = float(debug["angle_err_deg"])

near_norm = near_px / (0.5 * 1280)
far_norm = far_px / (0.5 * 1280)
curve_norm = curve_px / (0.5 * 1280)
turn_gate = min(abs(curve_px) / max(detector.curve_switch_px, 1.0), 1.0)
lookahead = detector.pix_lookahead_gain * (0.70 + 0.90 * turn_gate)

terms = [
    ("near   -near_norm", -near_norm),
    ("far    lookahead*(-far_norm)", lookahead * (-far_norm)),
    ("curve  pix_curve_gain*(-curve_norm)", detector.pix_curve_gain * (-curve_norm)),
    ("angle  pix_angle_gain*(angle/45)", detector.pix_angle_gain * (angle / 45.0)),
]
if curve_px < -detector.left_curve_outward_px:
    terms.append(("left-curve outward", detector.left_curve_outward_gain * curve_norm))

print("conf %.2f  pair %.2f  lock %s  curve_mode %s" % (
    confidence, debug["bottom_pair_ratio"], debug["bottom_lock_valid"],
    debug.get("curve_mode")))
print("near_err_px %+.1f   far_err_px %+.1f   curve_px %+.1f   angle_err %+.1f deg" % (
    near_px, far_px, curve_px, angle))
print("err_scale_cm %.3f   lookahead_dyn %.3f (gate %.2f)" % (
    scale, lookahead, turn_gate))
print()
print("  term                                    raw        in cm")
total = 0.0
for name, value in terms:
    total += value
    print("  %-38s %+9.5f  %+9.2f" % (name, value, value * scale))
print("  %-38s %+9.5f  %+9.2f" % ("SUM (pre-EMA)", total, total * scale))
print()
print("  fused_err_cm   (smoothed)              %+9.2f" % debug["fused_err_cm"])
print("  base_err_cm                             %+9.2f" % debug["base_err_cm"])
print("  -> steer = 0.83 * fused_err_cm = %+.2f  -> wz = %+.3f rad/s (%s)" % (
    0.83 * debug["fused_err_cm"], 0.83 * debug["fused_err_cm"] / 10.0 * 0.5,
    "LEFT" if debug["fused_err_cm"] > 0 else "RIGHT"))
