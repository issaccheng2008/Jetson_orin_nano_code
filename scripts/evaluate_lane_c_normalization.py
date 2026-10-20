#!/usr/bin/env python3
"""Offline comparison of original legacy masks and legacy with normalized C.

Uses the existing manually annotated real frames, with unchanged calibration.
No camera, sockets, serial ports or control programs are opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from line_preprocess import extract_lane_candidates
from photometric_thresholds import measure, MAX_CHANNEL_REFERENCE
from evaluate_lane_extraction import detector, annotated_regions, score, aggregate, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'docs/audit_2026-10-09/line_extraction/annotations.json'
    annotations = json.loads(source.read_text())
    rows, samples = [], []
    cv2.setNumThreads(1)
    for annotation in annotations['frames']:
        cap = cv2.VideoCapture(str(args.data_dir / annotation['video']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotation['frame'])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot decode {annotation['key']}")
        d = detector(frame.shape[1], frame.shape[0], annotations['meta']['calibration'], 'legacy')
        bird = cv2.warpPerspective(frame, d.M, (320, 400))
        gray = np.max(bird, axis=2)
        photo = measure(np.max(frame, axis=2), reference=MAX_CHANNEL_REFERENCE)
        old = extract_lane_candidates(gray, 'legacy')
        normalized = extract_lane_candidates(gray, 'legacy', photometry=photo)
        samples.append(dict(key=annotation['key'], **photo.diagnostics(),
                            **normalized[3], old_threshold=old[2], normalized_threshold=normalized[2]))
        panels = [bird]
        for label, result in [('fixed C=-12', old), ('normalized C', normalized)]:
            shown = cv2.cvtColor(result[1], cv2.COLOR_GRAY2BGR)
            cv2.putText(shown, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cv2.putText(shown, f"C={result[3]['preprocess_adaptive_c_effective']:.2f}",
                        (5, 42), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            panels.append(shown)
        cv2.imwrite(str(args.output / (annotation['key'] + '.png')), np.hstack(panels))
        if annotation['usable_for_precision']:
            centers, valid = annotated_regions(annotation)
            for name, result in [('fixed', old), ('normalized', normalized)]:
                rows.append(dict(key=annotation['key'], video=annotation['video'], method=name,
                                 **score(result[1], centers, valid)))
    write_csv(args.output / 'scores.csv', rows)
    summary = []
    for subset in ('all', 'latest', 'older'):
        for method in ('fixed', 'normalized'):
            selected = [r for r in rows if r['method'] == method and
                        (subset == 'all' or (r['video'] == 'camera_commands3.avi') == (subset == 'latest'))]
            summary.append(dict(subset=subset, method=method, **aggregate(selected)))
    report = dict(opencv=cv2.__version__, numpy=np.__version__,
                  annotation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reference=MAX_CHANNEL_REFERENCE, calibration=annotations['meta']['calibration'],
                  changed='legacy adaptive C only; final threshold bounds, morphology and component gates preserved',
                  samples=samples, summary=summary,
                  limitations=['Shared development recordings; not independent field trials.',
                               'Tolerance metrics use existing manual centerlines and ignore regions.',
                               'Global ROI standard deviation is affected by scene content, not just exposure.',
                               'Fixed final threshold and minimum component height can still remove weak/horizontal lines.'])
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for item in summary:
        print(item)


if __name__ == '__main__':
    main()
