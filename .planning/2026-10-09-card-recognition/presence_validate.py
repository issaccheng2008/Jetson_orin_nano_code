"""Reproducible helper-only probes; presence candidates are not actions/labels.

Run with the OpenCV environment from the repository root. This script uses the
persisted manually reviewed intervals and optionally full ROI annotations.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_vision/jetson"))
from card_presence import find_broken_border_presence
from photometric_thresholds import measure

CFG = dict(cue_side_min=30, cue_side_max=340, cue_aspect_max=2.6,
           cue_edge_margin=4, presence_top_frac=1 / 3,
           cue_core_gray_min=100., cue_contrast_min=4., cue_score_min=1.2)


def variants(gray):
    yield "original", gray
    yield "gain0.4", (gray.astype(np.float32) * .4).astype(np.uint8)
    for axis in ("horizontal", "vertical"):
        kernel = np.zeros((11, 11), np.float32)
        if axis == "horizontal":
            kernel[5] = 1 / 11
        else:
            kernel[:, 5] = 1 / 11
        yield "motion11_" + axis, cv2.filter2D(gray, -1, kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("presence_validation.json"))
    args = parser.parse_args()
    intervals = json.loads(Path(__file__).with_name("presence_validation_intervals.json").read_text())
    data_root = Path(intervals["source_root"])
    if args.annotations:
        annotated = json.loads(args.annotations.read_text())
        sequences = annotated["sequences"]
    else:
        sequences = [dict(s, truth="none") for s in intervals["negative_intervals"]]
        sequences += intervals["positive_intervals"]
    rows = []

    def evaluate(source, frame, image, truth, keys=None):
        gray = cv2.cvtColor(cv2.resize(image, (960, 540)), cv2.COLOR_BGR2GRAY)
        for transform, changed in variants(gray):
            photo = measure(changed)
            started = time.perf_counter()
            box, score, diagnostics = find_broken_border_presence(changed, photo, CFG)
            elapsed = (time.perf_counter() - started) * 1000
            row = dict(source=str(source), frame=frame, truth=truth, variant=transform,
                       box=box, score=score, ms=elapsed, diagnostics=diagnostics)
            if keys:
                keyframes = [k["frame"] for k in keys]
                gt = np.array([np.interp(frame, keyframes, [k["xyxy"][i] for k in keys])
                               for i in range(4)]) * [960, 540, 960, 540]
                iou = 0.
                if box:
                    x, y, w, h = box
                    corners = np.array([x, y, x + w, y + h])
                    area = np.prod(np.maximum(0, np.minimum(gt[2:], corners[2:])
                                              - np.maximum(gt[:2], corners[:2])))
                    iou = float(area / (np.prod(gt[2:] - gt[:2]) + w * h - area))
                row["rough_roi_iou"] = iou
            rows.append(row)

    for sequence in sequences:
        source = Path(sequence["source"])
        if not source.is_absolute():
            source = data_root / source
        capture = cv2.VideoCapture(str(source))
        for frame in range(sequence["start"], sequence["end"] + 1, 3):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("cannot read %s frame %s" % (source, frame))
            truth = "none" if sequence["truth"] in ("negative", "none") else sequence["truth"]
            evaluate(source, frame, image, truth, sequence.get("box_keyframes"))
        capture.release()
    negatives = [data_root / p for p in intervals["independent_negative_stills"]]
    negatives += [ROOT / p for p in intervals["repository_negative_stills"]]
    for path in negatives:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError("cannot read %s" % path)
        evaluate(path, None, image, "none")
    summary = {}
    for transform in ("original", "gain0.4", "motion11_horizontal", "motion11_vertical"):
        selected = [r for r in rows if r["variant"] == transform]
        negative = [r for r in selected if r["truth"] == "none"]
        positive = [r for r in selected if r["truth"] != "none"]
        summary[transform] = dict(negative_count=len(negative),
                                  negative_candidates=sum(r["box"] is not None for r in negative),
                                  positive_count=len(positive),
                                  positive_candidates=sum(r["box"] is not None for r in positive),
                                  positive_rough_roi_matches=sum(r.get("rough_roi_iou", 0.) >= .25 for r in positive),
                                  milliseconds_percentiles=np.percentile([r["ms"] for r in selected], [50, 95]).tolist())
    report = dict(opencv=cv2.__version__, summary=summary, rows=rows,
                  limitations=["Shared development data, no unseen holdout.",
                               "Positive candidate count is not recall without ROI matching.",
                               "Linear interpolation of rough card boxes can miss rapid image motion.",
                               "Gain and blur are counterfactuals, not new camera recordings.",
                               "Desktop timings do not establish Nano timings or physical stopping."])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
