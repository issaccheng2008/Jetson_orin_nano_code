"""Replay real camera loss dumps through legacy and optional track observers.

No motor/UDP code is loaded. One directory is one temporal group; the detector
starts fresh between groups. The CSV is for manual image review, not an
automatic claim that one estimate is ground truth.

Example:
  python new_vision/scripts/replay_track_geometry.py records/loss_dump \
    --output /tmp/track_replay.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson"))
from line_detector_v1_warp import LineDetector


FIELDS = ("path", "legacy_valid", "legacy_near_cm", "legacy_heading_deg",
          "track_valid", "track_reason", "track_mode", "track_phase",
          "track_near_cm", "track_heading_deg", "track_target_z_cm",
          "track_target_bearing_deg", "track_width_cm", "track_confidence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--pattern", default="**/*_frame.jpg")
    parser.add_argument("--output", type=Path, help="Optional per-frame CSV")
    args = parser.parse_args(argv)
    paths = sorted(args.image_dir.glob(args.pattern))
    if not paths:
        parser.error("no matching camera frames")
    counts = Counter()
    current_group = None
    output = args.output.open("w", newline="", encoding="utf-8") if args.output else None
    try:
        writer = csv.DictWriter(output, FIELDS) if output else None
        if writer:
            writer.writeheader()
        for path in paths:
            if path.parent != current_group:
                detector = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35.)
                detector.startup_force_simple_bottom = False
                detector.track_geometry_enable = True
                current_group = path.parent
            frame = cv2.imread(str(path))
            if frame is None:
                counts["unreadable"] += 1
                continue
            if frame.shape[:2] != (720, 1280):
                frame = cv2.resize(frame, (1280, 720))
            debug = detector.process(frame, dt=.1)[-1]
            row = {"path": str(path),
                   "legacy_valid": bool(debug.get("heading_control_valid")),
                   "legacy_near_cm": debug.get("near_error_cm"),
                   "legacy_heading_deg": debug.get("heading_control_deg"),
                   **{field: debug.get(field) for field in FIELDS if field.startswith("track_")}}
            if writer:
                writer.writerow(row)
            counts["frames"] += 1
            counts["legacy_valid"] += row["legacy_valid"]
            counts["track_valid"] += bool(row["track_valid"])
            if row["track_valid"]:
                counts["track_" + str(row["track_mode"])] += 1
            else:
                counts["reject_" + str(row["track_reason"])] += 1
    finally:
        if output:
            output.close()
    print(dict(sorted(counts.items())))
    if args.output:
        print(args.output)


if __name__ == "__main__":
    main()
