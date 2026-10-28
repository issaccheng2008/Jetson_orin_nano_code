#!/usr/bin/env python3
"""Draw calibrated ground-distance ticks on an offline bird view.

This reads one image and runs only the lane detector. It does not open a camera,
connector, serial port, or policy process.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision" / "jetson"))
from camera_config import load as load_camera  # noqa: E402
from line_detector_v1_warp import LineDetector  # noqa: E402


def annotated(image, detector, title):
    height, width = image.shape[:2]
    left, right = 74, 130
    canvas = np.full((height, left + width + right, 3), (28, 28, 28), np.uint8)
    canvas[:, left:left + width] = image
    mask = detector.ground_valid_mask
    ticks = {}
    for distance in range(10, 81, 10):
        row = int(np.argmin(np.abs(detector._lut_z_cm - distance)))
        ticks[str(distance)] = {"row": row, "actual_cm": detector.z_cm_at(row)}
        xs = np.flatnonzero(mask[row])
        if len(xs):
            cv2.line(canvas, (left + int(xs[0]), row),
                     (left + int(xs[-1]), row), (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (left - 6, row), (left, row), (0, 230, 255), 1)
        cv2.putText(canvas, f"{distance} cm", (3, max(13, min(height - 3, row + 4))),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 230, 255), 1, cv2.LINE_AA)
    for index, (lo, hi) in enumerate(detector.segment_regions_cm):
        top = int(np.argmin(np.abs(detector._lut_z_cm - hi)))
        bottom = int(np.argmin(np.abs(detector._lut_z_cm - lo)))
        x = left + width + 10
        cv2.line(canvas, (x, top), (x, bottom), (120, 220, 120), 2)
        cv2.line(canvas, (x, top), (x + 6, top), (120, 220, 120), 2)
        cv2.line(canvas, (x, bottom), (x + 6, bottom), (120, 220, 120), 2)
        cv2.putText(canvas, f"S{index} {lo:g}-{hi:g}",
                    (x + 9, (top + bottom) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (120, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left + 8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas, ticks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--photometric-mode", default="legacy",
                        choices=("legacy", "raw", "threshold", "match"))
    args = parser.parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error(f"cannot read {args.image}")
    camera = load_camera()
    width, height = int(camera["width"]), int(camera["height"])
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    calib = camera["distance_calib"]
    detector = LineDetector(
        cam_w=width, cam_h=height,
        cam_height_cm=float(camera["mount_height_cm"]),
        cam_pitch_deg=float(camera["pitch_deg"]),
        cam_vfov_deg=float(camera["vfov_deg"]),
        z_calib=(float(calib["a"]), float(calib["b"])),
        lane_width_cm=float(camera["lane_width_cm"]),
    )
    detector.photometric_mode = args.photometric_mode
    debug = detector.process(frame, dt=.1)[-1]
    color, ticks = annotated(debug["bird_color"], detector, "bird BGR")
    binary = cv2.cvtColor(debug["binary"], cv2.COLOR_GRAY2BGR)
    binary, _ = annotated(binary, detector, "lane mask: white=line")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = (("bird_color_10cm.png", color), ("bird_binary_10cm.png", binary),
             ("bird_pair_10cm.png", np.hstack((color, binary))))
    for name, result in files:
        if not cv2.imwrite(str(args.output_dir / name), result):
            raise RuntimeError(f"cannot write {args.output_dir / name}")
    meta = {
        "source": str(args.image.resolve()),
        "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "camera_profile": camera["profile"],
        "source_resize": [width, height],
        "bird_size": [detector.bird_w, detector.bird_h],
        "photometric_mode": args.photometric_mode,
        "effective_photometric_mode": debug["photometric_effective_mode"],
        "line_preprocess_mode": detector.preprocess_mode,
        "distance_reference": "camera optical center projected vertically onto ground",
        "top_cm": detector.z_cm_at(0),
        "bottom_cm": detector.z_cm_at(detector.bird_h - 1),
        "cm_per_row": detector.cm_per_px,
        "ticks": ticks,
        "segment_regions_cm": detector.segment_regions_cm,
        "valid_fraction": float(detector.ground_valid_mask.mean()),
    }
    (args.output_dir / "distance_grid.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
