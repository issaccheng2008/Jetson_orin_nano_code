"""Offline frame replay for the 2026-10-08 lane segment audit.

Run on one image collection at a time. Frames within a clip/event retain
detector history; each new clip/event starts with a fresh detector.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson"))
from line_detector_v1_warp import LineDetector


def detector(full_bands=False, segments=False):
    value = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
    value.startup_force_simple_bottom = False
    value.lane_fit_enable = segments
    value.lane_segments_enable = segments
    if full_bands:
        value.band_rows_low = value.band_rows_mid = 25
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--pattern", default="**/*_frame.jpg")
    parser.add_argument("--group", choices=("parent", "video_prefix"), default="parent",
                        help="Use video_prefix for *_flip_XXXX_raw.jpg samples")
    args = parser.parse_args()
    paths = sorted(args.image_dir.glob(args.pattern))
    if not paths:
        parser.error("no matching images")
    counts = Counter()
    current_group = None
    for path in paths:
        group = (str(path.parent) if args.group == "parent"
                 else path.name.split("_flip_")[0])
        if group != current_group:
            base = detector(segments=True)
            full = detector(full_bands=True)
            current_group = group
        image = cv2.imread(str(path))
        if image is None:
            counts["unreadable"] += 1
            continue
        if image.shape[:2] != (720, 1280):
            image = cv2.resize(image, (1280, 720))
        a = base.process(image, dt=.1)[-1]
        b = full.process(image, dt=.1)[-1]
        counts["frames"] += 1
        for name in ("measurement_valid", "heading_control_valid", "fit_seg_anchored"):
            counts["base_" + name] += bool(a.get(name))
        counts["three_segments"] += a.get("fit_seg_count") == 3
        counts["full_measurement_valid"] += bool(b.get("measurement_valid"))
        if a.get("fit_seg_anchored"):
            counts["anchored_" + a["fit_seg_pattern"]] += 1
        if a.get("heading_control_valid") and b.get("heading_control_valid"):
            difference = abs(a["heading_control_deg"] - b["heading_control_deg"])
            counts["heading_delta_gt5"] += difference > 5
            counts["heading_delta_gt15"] += difference > 15
        if a.get("measurement_valid") and b.get("measurement_valid"):
            counts["near_delta_gt5"] += abs(a["near_error_cm"] - b["near_error_cm"]) > 5
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
