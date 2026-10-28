"""One-shot: what does the detector read with the robot parked where it is?

Runs both pitch candidates through the same frames so lateral_scale converges in
each (a fresh LineDetector starts at 1.0 and the running system sits near the clamp).
"""
import sys

import cv2

sys.path.insert(0, "/home/isaac/jetson_orin_code/new_vision/jetson")
from line_detector_v1_warp import LineDetector

PITCHES = (45.0, 50.0)
FRAMES = 90

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detectors = [LineDetector(1280, 720, cam_pitch_deg=p) for p in PITCHES]
last = [None] * len(PITCHES)
count = 0
frame = None
for _ in range(FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
    for i, detector in enumerate(detectors):
        _, _, confidence, _, debug = detector.process(frame)
        last[i] = (confidence, debug)
cap.release()

if frame is not None:
    cv2.imwrite("/home/isaac/_probe_frame.jpg", frame)

print("frames read: %d" % count)
for pitch, detector, (confidence, debug) in zip(PITCHES, detectors, last):
    print("--- camera pitch %.1f ---" % pitch)
    print("  base_err_cm    %+8.2f   (positive = robot right of the lane centre)"
          % debug["base_err_cm"])
    print("  near_err_px    %+8.1f" % debug["near_err_px"])
    print("  fused_err_cm   %+8.2f" % debug["fused_err_cm"])
    print("  angle_err_deg  %+8.1f" % debug["angle_err_deg"])
    print("  conf %.2f  pair %.2f  lock %s  left %s right %s"
          % (confidence, debug["bottom_pair_ratio"], debug["bottom_lock_valid"],
             debug.get("left_seen"), debug.get("right_seen")))
    print("  lateral_scale  %6.3f   (clamp %.2f..%.2f)"
          % (detector.lateral_scale, detector.lateral_scale_min,
             detector.lateral_scale_max))
    print("  lane_width_px  %6.1f   true %.1f cm"
          % (debug.get("lane_width_px", float("nan")), detector.lane_width_true_cm))
    print("  cm_per_px@%d  %6.4f" % (detector.NEAR_BAND_ROW,
                                     detector.cm_per_px_at(detector.NEAR_BAND_ROW)))
    print("  z@near         %6.1f cm" % detector.z_cm_at(detector.NEAR_BAND_ROW))
    print("  err_scale_cm   %6.3f" % detector.err_scale_cm)
