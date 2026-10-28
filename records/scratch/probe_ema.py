"""Watch smoothed_err converge with the robot parked and nothing touching it."""
import sys

import cv2

sys.path.insert(0, "/home/isaac/jetson_orin_code/new_vision/jetson")
from line_detector_v1_warp import LineDetector

FRAMES = 400

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = LineDetector(1280, 720, cam_pitch_deg=45.0)
debug = None
print("  frame   near_px  raw_sum_cm   smoothed(cM)   lateral_scale   conf")
for i in range(FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    _, _, confidence, _, debug = detector.process(frame)
    if i % 40 == 0 or i == FRAMES - 1:
        scale = detector.err_scale_cm
        near_px = float(debug["near_err_px"])
        far_px = float(debug["far_err_px"])
        curve_px = float(debug.get("curve_px", 0.0) or 0.0)
        angle = float(debug["angle_err_deg"])
        gate = min(abs(curve_px) / max(detector.curve_switch_px, 1.0), 1.0)
        look = detector.pix_lookahead_gain * (0.70 + 0.90 * gate)
        raw = (-near_px / 640.0
               + look * (-far_px / 640.0)
               + detector.pix_curve_gain * (-curve_px / 640.0)
               + detector.pix_angle_gain * (angle / 45.0))
        print("  %5d  %+8.1f  %+9.2f  %+11.2f  %13.3f  %5.2f" % (
            i, near_px, raw * scale, debug["fused_err_cm"],
            detector.lateral_scale, confidence))
cap.release()
