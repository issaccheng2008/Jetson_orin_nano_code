#!/usr/bin/env python3
"""Read-only Canny photo/frame comparison; artifacts only under --output.

The card detector baseline is loaded from --baseline-ref, insulated from working
tree runtime edits. Two experiments replace selective binary only, or binary
and ring ink together. Presence fallback stays as in that baseline in both.
Lane labels are sparse visually estimated centers, not expert pixel masks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "new_vision/jetson"))
sys.path.insert(0, str(REPO / "new_vision/scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from canny_candidates import card_canny, lane_canny
from evaluate_lane_extraction import annotated_regions, aggregate, detector, score
from evaluate_shape_cards import label_of


def snapshot_module(ref, filename, name):
    source = subprocess.check_output(
        ["git", "show", ref + ":new_vision/jetson/" + filename], cwd=REPO)
    module = types.ModuleType(name)
    module.__file__ = str(REPO / "new_vision/jetson" / filename)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labeled(image, caption, width=320, height=180):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(output, (0, 0), (width, 23), (0, 0, 0), -1)
    cv2.putText(output, caption, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (0, 255, 255), 1, cv2.LINE_AA)
    return output


def cards(args):
    module, baseline_sha256 = snapshot_module(
        args.baseline_ref, "shape_detector.py", "shape_baseline_canny_evaluation")
    original_ink = module._card_ink
    paths = [path for path in sorted((args.data_dir / "6card").iterdir())
             if path.suffix.lower() in (".jpg", ".jpeg", ".png")
             and label_of(path) is not None]
    rows, tiles, identities = [], [], []
    for index, path in enumerate(paths, 1):
        source_bytes = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Cannot read " + str(path))
        truth = label_of(path)
        identities.append(dict(name=path.name, sha256=hashlib.sha256(source_bytes).hexdigest()))
        outputs = {}
        for mode in ("baseline", "binary_canny", "both_canny"):
            current = module.ShapeDetector()
            if mode != "baseline":
                current._binary_selective = lambda gray: card_canny(gray)[0]
            module._card_ink = (lambda gray: card_canny(gray)[0]) if mode == "both_canny" else original_ink
            try:
                _, debug = current.update(frame.copy())
            finally:
                module._card_ink = original_ink
            shape = debug.get("shape")
            classified = shape in current.action_map
            quad_found = debug.get("quad") is not None
            rows.append(dict(index=index, filename=path.name, truth=truth, mode=mode,
                             found=quad_found or bool(debug.get("presence")),
                             quad_found=quad_found, presence=bool(debug.get("presence")),
                             classified=classified, prediction=shape if classified else "-",
                             correct=classified and shape == truth,
                             wrong=classified and shape != truth,
                             quad_candidates=int(debug.get("quad_total", 0))))
            outputs[mode] = debug
        gray = outputs["baseline"]["gray"]
        mask, diagnostics = card_canny(gray)
        original = labeled(frame, "%02d %s" % (index, truth))
        cells = [original, labeled(diagnostics["raw_edges"], "raw Canny borders"),
                 labeled(mask, "Canny supported ink")]
        for mode, debug in outputs.items():
            shown = debug["binary"].copy()
            if debug.get("quad_work") is not None:
                shown = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
                cv2.polylines(shown, [debug["quad_work"].astype(np.int32)], True, (0, 255, 0), 2)
            cells.append(labeled(shown, mode + " => " + str(debug.get("shape", "-"))))
        comparison = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:])))
        cv2.imwrite(str(args.output / ("card_%02d_%s.jpg" % (index, truth))), comparison)
        tiles.append(cv2.resize(comparison, (600, 225), interpolation=cv2.INTER_AREA))
        print("card", index, path.name,
              [(r["mode"], r["prediction"]) for r in rows[-3:]], flush=True)
    columns = 3
    tiles.extend(np.zeros_like(tiles[0]) for _ in range((-len(tiles)) % columns))
    sheet = np.vstack([np.hstack(tiles[start:start + columns])
                       for start in range(0, len(tiles), columns)])
    cv2.imwrite(str(args.output / "allphotos_contact_sheet.jpg"), sheet)
    save_csv(args.output / "card_scores.csv", rows)
    summaries = []
    for mode in ("baseline", "binary_canny", "both_canny"):
        group = [row for row in rows if row["mode"] == mode]
        summaries.append(dict(mode=mode, images=len(group), **{
            key: sum(bool(row[key]) for row in group)
            for key in ("found", "quad_found", "presence", "classified", "correct", "wrong")}))
    return dict(summary=summaries, images=identities,
                baseline_shape_sha256=baseline_sha256,
                experiments={"binary_canny": "replace selective binary only; original Otsu ring ink and presence cue retained",
                             "both_canny": "replace selective binary and ring ink; original presence cue retained"},
                changes=[row for row in rows if row["mode"] != "baseline" and (
                    row["prediction"] != next(r["prediction"] for r in rows
                                              if r["index"] == row["index"] and r["mode"] == "baseline"))])


def lanes(args):
    baseline, baseline_sha256 = snapshot_module(
        args.baseline_ref, "line_preprocess.py", "lane_baseline_canny_evaluation")
    annotations_bytes = args.annotations.read_bytes()
    annotations = json.loads(annotations_bytes)
    rows, identities = [], []
    for annotation in annotations["frames"]:
        path = args.data_dir / annotation["video"]
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotation["frame"])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Cannot decode " + str(path))
        current = detector(frame.shape[1], frame.shape[0], annotations["meta"]["calibration"])
        bird = cv2.warpPerspective(frame, current.M, (320, 400))
        gray = np.max(bird, axis=2)
        mask, diagnostics = lane_canny(gray)
        masks = {"legacy": baseline.extract_lane_candidates(gray, "legacy")[1],
                 "contrast": baseline.extract_lane_candidates(gray, "contrast")[1], "canny": mask}
        identities.append(dict(key=annotation["key"],
                               decoded_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                               bird_gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
        cells = [labeled(bird, annotation["key"], 320, 400)]
        centers, valid = annotated_regions(annotation)
        for mode, image in masks.items():
            shown = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            shown[centers] = (0, 255, 0)
            shown[~valid] = (shown[~valid] // 2 + np.array([50, 0, 0])).astype(np.uint8)
            cells.append(labeled(shown, mode + " + sparse centers", 320, 400))
            if annotation["usable_for_precision"]:
                rows.append(dict(key=annotation["key"], video=annotation["video"],
                                 frame=annotation["frame"], method=mode,
                                 **score(image, centers, valid)))
        cells.append(labeled(diagnostics["raw_edges"], "raw Canny borders", 320, 400))
        cv2.imwrite(str(args.output / (annotation["key"] + "_comparison.jpg")), np.hstack(cells))
    save_csv(args.output / "lane_scores.csv", rows)
    summaries = []
    for subset in ("all", "latest", "older", "activity_selected"):
        for mode in ("legacy", "contrast", "canny"):
            group = [row for row in rows if row["method"] == mode and (
                subset == "all" or
                (subset == "latest" and row["video"] == "camera_commands3.avi") or
                (subset == "older" and row["video"] != "camera_commands3.avi") or
                (subset == "activity_selected" and row["video"] == "camera_commands3.avi"
                 and row["frame"] in (536, 670, 805, 1073)))]
            summaries.append(dict(subset=subset, method=mode, **aggregate(group)))
    per_activity = [dict(key=row["key"], method=row["method"], **aggregate([row]))
                    for row in rows if row["video"] == "camera_commands3.avi"
                    and row["frame"] in (536, 670, 805, 1073)]
    return dict(summary=summaries, activity_frames=per_activity, frames=identities,
                baseline_preprocess_sha256=baseline_sha256,
                annotation_sha256=hashlib.sha256(annotations_bytes).hexdigest(),
                metrics="pixel-weighted sparse visual center coverage within 4px; candidate precision within 10px; ignore regions excluded",
                limitations=annotations["meta"]["limitations"],
                excluded=[a["key"] for a in annotations["frames"] if not a["usable_for_precision"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/canny-evaluation"))
    parser.add_argument("--baseline-ref", default="3260323934ae3df830a2ff8c146aa606ec32eca9")
    parser.add_argument("--annotations", type=Path,
                        default=REPO / "docs/audit_2026-10-09/line_extraction/annotations.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(baseline_ref=args.baseline_ref, opencv=cv2.__version__,
                  canny_source_sha256=hashlib.sha256((REPO / "new_vision/jetson/canny_candidates.py").read_bytes()).hexdigest(),
                  cards=cards(args), lanes=lanes(args),
                  limitations=["Existing photos and selected frames, not independent test scenes or vehicle validation.",
                               "Canny is an optional comparison; these probes do not select or modify runtime defaults."])
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(cards=report["cards"]["summary"], lanes=report["lanes"]["summary"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
