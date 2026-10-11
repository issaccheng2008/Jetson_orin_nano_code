#!/usr/bin/env python3
"""Replay the birdseye half of an existing paired diagnostic AVI offline.

This is a diagnostic replay, not a camera-level performance benchmark: the AVI
already contains an older birdseye projection, lossy encoding, and a HUD. The
script removes the right-hand binary view and bottom HUD, then compares the
current detector with its single-boundary rollback on identical frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
from line_detector_v1_warp import LineDetector


def _detector(source_width: int, source_height: int, *, single: bool) -> LineDetector:
    detector = LineDetector(z_calib=(1., 0.))
    # Source is already an old birdseye. Only resize it to the current grid;
    # running the current perspective transform a second time would be invalid.
    detector.M = np.array([[detector.bird_w / source_width, 0., 0.],
                           [0., detector.bird_h / source_height, 0.],
                           [0., 0., 1.]], dtype=float)
    detector.M_inv = np.linalg.inv(detector.M)
    detector.ground_valid_mask[:] = True
    detector.startup_force_simple_bottom = False
    detector.bottom_lock_enable = False
    detector.red_detect_enable = False
    detector.preprocess_mode = "contrast"
    detector.photometric_mode = "raw"
    detector.single_boundary_tracking = single
    return detector


def replay(video: Path, output_dir: Path, *, bird_height: int,
           max_frames: int | None = None) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open recording: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width < 2 or not 0 < bird_height <= height or fps <= 0:
        raise ValueError("invalid paired-video dimensions, crop height, or FPS")
    source_width = width // 2
    enabled = _detector(source_width, bird_height, single=True)
    rollback = _detector(source_width, bird_height, single=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "time_s", "mode", "valid", "reject_reason",
              "left_real_points", "right_real_points", "near_cm", "heading_deg",
              "path_quality", "paired_rows", "rollback_valid", "rollback_mode",
              "rollback_reject_reason"]
    counts: Counter[str] = Counter()
    rollback_counts: Counter[str] = Counter()
    rescued_single = 0
    rejected_when_rollback_valid = 0
    accepted_singles: list[int] = []
    rows = 0
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        while max_frames is None or rows < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            bird = frame[:bird_height, :source_width]
            if bird.shape[:2] != (bird_height, source_width):
                raise ValueError("recorded frame size changed")
            result = enabled.process(bird, dt=1. / fps, frame_token=rows)[-1]
            old = rollback.process(bird, dt=1. / fps, frame_token=rows)[-1]
            mode = str(result.get("tracking_mode", "UNKNOWN"))
            counts[mode] += 1
            rollback_counts[str(old.get("tracking_mode", "UNKNOWN"))] += 1
            if mode in ("LEFT_ONLY", "RIGHT_ONLY") and result.get("measurement_valid"):
                accepted_singles.append(rows)
                if not old.get("measurement_valid"):
                    rescued_single += 1
            if not result.get("measurement_valid") and old.get("measurement_valid"):
                rejected_when_rollback_valid += 1
            writer.writerow({
                "frame": rows, "time_s": round(rows / fps, 3),
                "mode": mode, "valid": int(bool(result.get("measurement_valid"))),
                "reject_reason": result.get("reject_reason"),
                "left_real_points": result.get("observed_left_points"),
                "right_real_points": result.get("observed_right_points"),
                "near_cm": result.get("near_error_cm"),
                "heading_deg": result.get("heading_control_deg"),
                "path_quality": result.get("path_quality"),
                "paired_rows": sum(row.get("source") == "paired" for row in
                                   result.get("heading_rows", ())),
                "rollback_valid": int(bool(old.get("measurement_valid"))),
                "rollback_mode": old.get("tracking_mode"),
                "rollback_reject_reason": old.get("reject_reason"),
            })
            rows += 1
    capture.release()
    summary = {
        "source": str(video.resolve()), "source_size": [width, height],
        "source_fps": fps, "crop": [0, 0, source_width, bird_height],
        "current_bird_grid": [enabled.bird_w, enabled.bird_h],
        "frames": rows, "modes": dict(counts),
        "rollback_modes": dict(rollback_counts),
        "single_valid_while_rollback_invalid": rescued_single,
        "new_invalid_while_rollback_valid": rejected_when_rollback_valid,
        "accepted_single_frame_indices": accepted_singles,
        "limits": "Older compressed and annotated birdseye; resized, never reprojected. "
                  "Current metric positions are illustrative, not ground truth.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bird-height", type=int, required=True,
                        help="Rows above the video HUD; for camera_commands.avi use 358")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    print(json.dumps(replay(args.video, args.output_dir,
                            bird_height=args.bird_height,
                            max_frames=args.max_frames), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
