#!/usr/bin/env python3
"""Score the six-shape card classifier against labelled real photographs.

`6card/` holds 31 real camera frames, one card each, labelled by the first
character of the filename. Nothing in either repo had ever scored the
classifier on real data - the only offline checks were six synthetic cards and
a "did it return a shape at all" count - so every judgement about it so far has
been an inference from tallies in a field log.

Three numbers are reported separately on purpose, because they fail for
different reasons:

    box found    the finder located a card at all. Nothing downstream can work
                 if this is low, and the field reports of a card that never
                 stopped the robot are this stage.
    classified   a shape came back rather than None.
    correct      the shape matches the label.

    python new_vision/scripts/evaluate_shape_cards.py --dump /tmp/card_after

`--dump` writes each card's rectified 200x200 warp and the binary it came from,
named by label and prediction, which is also how the filename-to-label mapping
gets confirmed by eye rather than assumed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path):
    """cv2.imread goes through the ANSI code page on Windows and fails on the
    Chinese filenames this set is labelled with."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if ok:
        encoded.tofile(str(path))
    return ok

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "jetson"))

from shape_detector import ShapeDetector  # noqa: E402

# The first character of each filename. 正 is 正方形, not 正三角形 - the print
# sheet these photographs come from lays out 圆形/五角星/正方形/菱形/十字/正三角形
# in a 2x3 grid, and 五 draws a five-pointed star rather than a pentagon.
FILENAME_LABEL = {"圆": "circle", "五": "pentagon", "正": "square",
                  "菱": "diamond", "十": "cross", "三": "triangle"}

DEFAULT_DIR = r"D:\用户\Lenovo\桌面\Robocup\6card"


def label_of(path: Path):
    for name in path.stem:
        if name in FILENAME_LABEL:
            return FILENAME_LABEL[name]
    return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--dump", default="", help="write warps here")
    parser.add_argument("--limit", type=int, default=0, help="0 is all")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.dir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"no images under {args.dir}")
        return 1

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    detector = ShapeDetector()
    classes = sorted(detector.action_map)
    known = set(classes)

    # truth -> predicted (or "-" for no shape back)
    confusion = {t: Counter() for t in classes}
    per_truth = {t: Counter() for t in classes}
    unknown_label = []

    for index, path in enumerate(images, start=1):
        truth = label_of(path)
        frame = imread_unicode(path)
        if frame is None:
            print(f"  unreadable: {path.name}")
            continue
        if truth is None or truth not in known:
            unknown_label.append(path.name)
            continue

        _action, dbg = detector.update(frame)
        found = dbg.get("quad") is not None or bool(dbg.get("presence"))
        shape = dbg.get("shape")
        predicted = shape if shape in known else "-"

        per_truth[truth]["n"] += 1
        per_truth[truth]["box"] += 1 if found else 0
        per_truth[truth]["classified"] += 1 if predicted != "-" else 0
        per_truth[truth]["correct"] += 1 if predicted == truth else 0
        confusion[truth][predicted] += 1

        if args.dump:
            stem = f"{index:03d}_{truth}_pred-{predicted}"
            for key in ("warp", "binary"):
                image = dbg.get(key)
                if image is not None:
                    imwrite_unicode(os.path.join(args.dump, f"{stem}_{key}.png"), image)

    total = sum(per_truth[t]["n"] for t in classes)
    print(f"{len(images)} images under {args.dir}"
          + (f"  ({len(unknown_label)} with no readable label: "
             f"{', '.join(unknown_label[:4])})" if unknown_label else ""))
    print()

    print(f"{'label':10s} {'n':>3} {'box':>4} {'cls':>4} {'correct':>8} {'recall':>7}")
    for truth in classes:
        row = per_truth[truth]
        if not row["n"]:
            continue
        print(f"{truth:10s} {row['n']:3d} {row['box']:4d} {row['classified']:4d} "
              f"{row['correct']:8d} {row['correct'] / row['n']:6.0%}")
    boxes = sum(per_truth[t]["box"] for t in classes)
    classified = sum(per_truth[t]["classified"] for t in classes)
    correct = sum(per_truth[t]["correct"] for t in classes)
    print(f"{'TOTAL':10s} {total:3d} {boxes:4d} {classified:4d} {correct:8d} "
          f"{correct / max(total, 1):6.0%}")
    print()

    header = [c[:8] for c in classes] + ["-"]
    print("confusion (rows truth, cols predicted, - = no shape back)")
    print(f"{'':10s}" + "".join(f"{h:>9s}" for h in header))
    for truth in classes:
        if not per_truth[truth]["n"]:
            continue
        cells = [confusion[truth][c] for c in classes] + [confusion[truth]["-"]]
        print(f"{truth:10s}" + "".join(f"{v:9d}" for v in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
