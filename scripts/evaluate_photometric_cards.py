#!/usr/bin/env python3
"""Read-only production normalization comparison with real cards and frames.

Gain 0.6/0.4 multiplies decoded uint8 BGR, then truncates: a same-scene
counterfactual, not a photograph captured under new lighting/exposure.
Each evaluation reconstructs its detector; no temporal confirmation is scored.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "new_vision/jetson"))
from evaluate_canny_candidates import labeled, snapshot_module
from shape_detector import ShapeDetector
from photometric_thresholds import measure

LABELS = {"圆": "circle", "五": "pentagon", "正": "square",
          "菱": "diamond", "十": "cross", "三": "triangle"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result(factory, frame, mode):
    measured = measure(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    instance = factory()
    if mode != "baseline":
        instance.photometric_mode = "normalize"
        instance.preprocess_mode = "canny" if mode == "canny" else "selective"
    _, debug = instance.update(frame.copy())
    classified = debug.get("shape") in instance.action_map
    quad = debug.get("quad") is not None
    data = dict(found=quad or bool(debug.get("presence")), quad_found=quad,
                presence=bool(debug.get("presence")), classified=classified,
                prediction=debug.get("shape") if classified else "-",
                cue_scored=not quad,
                cue_gate_hit=not quad and debug.get("cue_candidate_cy_frac") is not None,
                score_raw=float(debug.get("cue_score_raw", debug.get("presence_cue", 0))),
                score_ref=float(debug.get("presence_cue", 0)),
                score_gate_ref=float(instance.cfg["cue_score_min"]),
                score_gate_raw=float(debug.get("cue_score_threshold_raw", instance.cfg["cue_score_min"])),
                adaptive_c_raw=float(debug.get("shape_adaptive_c", instance.cfg["adaptive_c"])),
                adaptive_c_applied=mode != "canny",
                ground_mean_raw=debug.get("photometric_mean", measured.mean),
                ground_std_raw=debug.get("photometric_std", measured.std),
                ground_stats_source="offline diagnostic only" if mode == "baseline" else "detector source ROI",
                ground_std_used=debug.get("photometric_std_used"),
                contrast_scale=float(debug.get("photometric_contrast_scale", 1)),
                cue_ink_gate_raw=float(debug.get("cue_ink_threshold", instance.cfg["cue_ink_thresh"])),
                cue_core_gray_gate_raw=float(debug.get("cue_core_gray_threshold", instance.cfg["cue_core_gray_min"])),
                cue_contrast_gate_raw=float(debug.get("cue_contrast_threshold", instance.cfg["cue_contrast_min"])))
    return data, debug


def panel(frame, identity, truth, outputs):
    caption = identity + " truth=" + truth
    if truth == "no six-class GT":
        caption = "f" + identity.rsplit("_", 1)[-1] + ": no six-class GT"
    elif truth == "negative/no card":
        caption = "background: negative/no card"
    cells = [labeled(frame, caption)]
    for mode, (data, debug) in outputs.items():
        shown = cv2.cvtColor(debug["binary"], cv2.COLOR_GRAY2BGR)
        if debug.get("quad_work") is not None:
            cv2.polylines(shown, [debug["quad_work"].astype(np.int32)],
                          True, (0, 255, 0), 2)
        cells.append(labeled(shown, mode + " pred=" + data["prediction"]))
    return np.hstack(cells)


def sheet(path, tiles):
    cells = [cv2.resize(tile, (640, 135), interpolation=cv2.INTER_AREA)
             for tile in tiles]
    cells.extend(np.zeros_like(cells[0]) for _ in range((-len(cells)) % 2))
    cv2.imwrite(str(path), np.vstack([np.hstack(cells[index:index + 2])
                                     for index in range(0, len(cells), 2)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/photometric-card-evaluation"))
    parser.add_argument("--baseline-ref", default="3260323934ae3df830a2ff8c146aa606ec32eca9")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, baseline_hash = snapshot_module(
        args.baseline_ref, "shape_detector.py", "photometric_card_baseline")
    hashes = {name: hashlib.sha256((REPO / "new_vision/jetson" / name).read_bytes()).hexdigest()
              for name in ("shape_detector.py", "photometric_thresholds.py", "canny_candidates.py")}
    paths = [path for path in sorted((args.data_dir / "6card").glob("*"))
             if path.suffix.lower() in (".jpg", ".jpeg", ".png")
             and path.stem[0] in LABELS]
    rows, canny_rows, frame_rows, identities = [], [], [], []
    sheets = {gain: [] for gain in (1.0, .6, .4)}
    for index, path in enumerate(paths, 1):
        image_bytes = path.read_bytes()
        source = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError("Cannot read " + str(path))
        truth = LABELS[path.stem[0]]
        identities.append(dict(name=path.name, sha256=hashlib.sha256(image_bytes).hexdigest()))
        for gain in (1.0, .6, .4):
            frame = np.clip(source.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            outputs = {}
            for mode, factory in (("baseline", baseline.ShapeDetector), ("normalize", ShapeDetector)):
                data, debug = result(factory, frame, mode)
                outputs[mode] = (data, debug)
                rows.append(dict(index=index, filename=path.name, truth=truth, gain=gain,
                                 mode=mode, **data, correct=data["prediction"] == truth,
                                 wrong=data["classified"] and data["prediction"] != truth))
            if gain == 1:
                data, debug = result(ShapeDetector, frame, "canny")
                outputs["canny"] = (data, debug)
                canny_rows.append(dict(index=index, filename=path.name, truth=truth,
                                       **data, correct=data["prediction"] == truth,
                                       wrong=data["classified"] and data["prediction"] != truth))
            comparison = panel(frame, "%02d gain%.1f" % (index, gain), truth, outputs)
            cv2.imwrite(str(args.output / ("card_%02d_%s_gain%.1f.jpg" % (index, truth, gain))), comparison)
            sheets[gain].append(comparison)
        print("processed", index, path.name, flush=True)

    extras = [("background_negative", args.data_dir / "6card/背景.png", "negative/no card")]
    for name, path, truth in extras:
        image_bytes = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        outputs = {}
        for mode, factory in (("baseline", baseline.ShapeDetector), ("normalize", ShapeDetector), ("canny", ShapeDetector)):
            data, debug = result(factory, frame, mode)
            outputs[mode] = (data, debug)
            frame_rows.append(dict(key=name, truth=truth, mode=mode, **data))
        cv2.imwrite(str(args.output / (name + ".jpg")), panel(frame, name, truth, outputs))
    cap = cv2.VideoCapture(str(args.data_dir / "camera_commands3.avi"))
    for index in (1130, 1150, 1220, 1230):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode actual frame %d" % index)
        name = "camera_commands3_%d" % index
        identities.append(dict(name=name, decoded_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
        outputs = {}
        for mode, factory in (("baseline", baseline.ShapeDetector), ("normalize", ShapeDetector), ("canny", ShapeDetector)):
            data, debug = result(factory, frame, mode)
            outputs[mode] = (data, debug)
            frame_rows.append(dict(key=name, truth="no six-class GT", mode=mode, **data))
        cv2.imwrite(str(args.output / (name + ".jpg")), panel(frame, name, "no six-class GT", outputs))
    cap.release()
    for gain, tiles in sheets.items():
        sheet(args.output / ("allphotos_gain%.1f.jpg" % gain), tiles)
    write_csv(args.output / "photo_scores.csv", rows)
    write_csv(args.output / "canny_optional_original.csv", canny_rows)
    write_csv(args.output / "actual_frames_and_negative.csv", frame_rows)
    summaries = []
    for gain in (1.0, .6, .4):
        for mode in ("baseline", "normalize"):
            group = [row for row in rows if row["gain"] == gain and row["mode"] == mode]
            summaries.append(dict(gain=gain, mode=mode, images=len(group), **{
                key: sum(bool(row[key]) for row in group)
                for key in ("found", "quad_found", "classified", "correct", "wrong")}))
    report = dict(baseline_ref=args.baseline_ref, baseline_shape_sha256=baseline_hash,
                  current_source_sha256=hashes, opencv=cv2.__version__,
                  summary=summaries, actual_frames_and_negative=frame_rows, inputs=identities,
                  optional_canny_scope="gain1 only; actual current preprocess_mode=canny replaces both binary and ring ink; production normalize remains selective",
                  units={"score_raw": "inner bright-hole fraction times raw uint8 grayscale contrast; zero if quad exists and fallback was not scored",
                         "score_ref": "score_raw / contrast_scale; reference grayscale contrast units; exported presence_cue",
                         "baseline_scores": "legacy contrast_scale=1; score_raw and score_ref coincide without reference normalization",
                         "adaptive_c_raw": "signed raw blackhat gray levels in OpenCV threshold=Gaussian local mean minus C; negative C requires response above local mean by abs(C); applied only when adaptive_c_applied=true, Canny skips adaptive C",
                         "normalize_c_formula": "-16 * min(1, max(ground_std,4) / 20.665240198035168)",
                         "ground_mean_raw/ground_std_raw": "raw BGR2GRAY gray levels in central ROI x25%-75%, y35%-75%, measured before work resize/padding; baseline statistics are offline diagnostic only, baseline does not adapt its thresholds",
                         "cue_ink_gate_raw/cue_contrast_gate_raw": "raw grayscale differences; reference gates times contrast_scale",
                         "cue_core_gray_gate_raw": "absolute raw grayscale level; ground_mean + contrast_scale*(100-reference_mean)"},
                  limitations=["Gain0.6/0.4 are same-scene decoded-pixel counterfactuals, not new real lighting/exposure captures; sensor noise/blur/exposure response is not simulated.",
                               "Per-image fresh detectors measure frame-level finding/classification, not temporal stopping/action behavior. cue_gate_hit can be true while presence/found stays false because cue history confirmation starts empty.",
                               "31 cards are existing regression photos; the complete set, including misses/errors, is exported.",
                               "Actual late video frames have no six-class truth; predictions and photometry are diagnostics, not accuracy.",
                               "One background negative does not establish false-positive rate; no hardware validation."])
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
