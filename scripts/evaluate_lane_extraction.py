#!/usr/bin/env python3
"""Read-only real-video extraction evaluation. No camera, serial, UDP or motors.

Example: python scripts/evaluate_lane_extraction.py --data-dir /path/to/Robocup
         --output /tmp/lane-evaluation --replay
Uses fixed annotation calibration, native video dimensions and original frames;
never crops/rescales the command HUD into a different camera geometry.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from line_preprocess import extract_lane_candidates


def detector(width, height, calibration, mode='contrast'):
    d = LineDetector(width, height, cam_height_cm=calibration['height_cm'],
                     cam_pitch_deg=calibration['pitch_deg'],
                     cam_vfov_deg=calibration['vfov_deg'],
                     z_calib=calibration['z_calib'], lane_width_cm=calibration['lane_width_cm'])
    d.preprocess_mode = mode
    d.startup_force_simple_bottom = False
    d.lane_fit_enable = d.lane_segments_enable = True
    return d


def annotated_regions(annotation):
    centers = np.zeros((400, 320), np.uint8)
    valid = np.ones_like(centers)
    for line in annotation['lane_center_polylines']:
        cv2.polylines(centers, [np.array(line, np.int32)], False, 1, 1)
    for polygon in annotation.get('ignore_polygons', []):
        cv2.fillPoly(valid, [np.array(polygon, np.int32)], 0)
    for x0, y0, x1, y1 in annotation.get('ignore_rects', []):
        valid[y0:y1, x0:x1] = 0
    return centers.astype(bool), valid.astype(bool)


def score(mask, centers, valid, recall_tolerance=4, precision_tolerance=10):
    candidates, truth = (mask > 0) & valid, centers & valid
    distance_to_candidates = cv2.distanceTransform((~candidates).astype(np.uint8),
                                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_to_truth = cv2.distanceTransform((~truth).astype(np.uint8),
                                             cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dict(center_hits=int(np.count_nonzero(truth & (distance_to_candidates <= recall_tolerance))),
                center_pixels=int(truth.sum()),
                near_lane_pixels=int(np.count_nonzero(candidates & (distance_to_truth <= precision_tolerance))),
                candidate_pixels=int(candidates.sum()))


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in
              ('center_hits', 'center_pixels', 'near_lane_pixels', 'candidate_pixels')}
    return dict(frames=len(rows), **counts,
                center_recall=counts['center_hits'] / max(1, counts['center_pixels']),
                tolerance_precision=counts['near_lane_pixels'] / max(1, counts['candidate_pixels']))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(video=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def replay(path, calibration, output):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detectors = {m: detector(width, height, calibration, m) for m in ('legacy', 'contrast')}
    hashes, rows, index = set(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hashes.add(hashlib.sha256(frame.tobytes()).digest())
        for mode, d in detectors.items():
            start = time.perf_counter()
            debug = d.process(frame, dt=1 / fps)[-1]
            rows.append(dict(video=path.name, frame=index, time_s=index/fps, mode=mode,
                             process_ms=(time.perf_counter()-start)*1000,
                             foreground=debug['preprocess_foreground_pixels'],
                             black_th=debug['black_th'], valid=debug.get('measurement_valid', False),
                             heading_valid=debug.get('heading_control_valid', False),
                             fit_segments=debug.get('fit_seg_count', 0)))
        index += 1
        if index % 200 == 0:
            print(f'{path.name}: processed {index} frames', flush=True)
    cap.release()
    write_csv(output / (path.stem + '_replay.csv'), rows)
    summary = []
    for mode in detectors:
        group = [r for r in rows if r['mode'] == mode]
        elapsed = [r['process_ms'] for r in group]
        summary.append(dict(video=path.name, mode=mode, decoded_frames=index,
                            unique_decoded_frames=len(hashes), fps=fps,
                            empty_mask_frames=sum(r['foreground'] == 0 for r in group),
                            measurement_valid_frames=sum(bool(r['valid']) for r in group),
                            heading_valid_frames=sum(bool(r['heading_valid']) for r in group),
                            process_ms_median=float(np.median(elapsed)),
                            process_ms_p95=float(np.percentile(elapsed, 95))))
    print(f'{path.name}: {index} decoded, {len(hashes)} unique', flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--annotations', type=Path,
                   default=REPO/'docs/audit_2026-10-09/line_extraction/annotations.json')
    p.add_argument('--replay', action='store_true', help='Also process all five videos sequentially')
    args = p.parse_args()
    annotation_bytes = args.annotations.read_bytes()
    annotations = json.loads(annotation_bytes)
    calibration = annotations['meta']['calibration']
    videos = list(dict.fromkeys(a['video'] for a in annotations['frames']))
    input_videos = [input_identity(args.data_dir / video) for video in videos]
    args.output.mkdir(parents=True, exist_ok=True)
    rows, stress = [], []
    for a in annotations['frames']:
        cap = cv2.VideoCapture(str(args.data_dir / a['video']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, a['frame'])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read {a['video']} frame {a['frame']}")
        d = detector(frame.shape[1], frame.shape[0], calibration)
        bird = cv2.warpPerspective(frame, d.M, (d.bird_w, d.bird_h))
        gray = np.max(bird, axis=2)
        methods = {'legacy': extract_lane_candidates(gray, 'legacy')[1],
                   'contrast': extract_lane_candidates(gray, 'contrast')[1],
                   'gain_2.4': extract_lane_candidates(
                       np.clip(gray.astype(np.float32)*2.4, 0, 255).astype(np.uint8), 'legacy')[1]}
        cells = [bird]
        for name, mask in methods.items():
            shown = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(shown, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            cells.append(shown)
        cv2.imwrite(str(args.output/(a['key']+'_comparison.jpg')), np.hstack(cells))
        if not a['usable_for_precision']:
            stress.append(dict(key=a['key'], note=a['notes']))
            continue
        centers, valid = annotated_regions(a)
        for name, mask in methods.items():
            rows.append(dict(key=a['key'], video=a['video'], split=a['split'], method=name,
                             **score(mask, centers, valid)))
    write_csv(args.output/'annotated_scores.csv', rows)
    summary = []
    for subset in ('all', 'early', 'later', 'latest', 'older'):
        for name in ('legacy', 'contrast', 'gain_2.4'):
            group = [r for r in rows if r['method'] == name and (
                subset == 'all' or r['split'] == subset or
                (subset == 'latest' and r['video'] == 'camera_commands3.avi') or
                (subset == 'older' and r['video'] != 'camera_commands3.avi'))]
            summary.append(dict(subset=subset, method=name, **aggregate(group)))
    replay_summary = []
    if args.replay:
        for video in videos:
            replay_summary.extend(replay(args.data_dir/video, calibration, args.output))
    report = dict(input_videos=input_videos,
                  annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                  source_sha256={f: hashlib.sha256((REPO/'new_vision/jetson'/f).read_bytes()).hexdigest()
                                 for f in ('line_preprocess.py', 'line_detector_v1_warp.py')},
                  opencv=cv2.__version__, numpy=np.__version__, calibration=calibration,
                  metrics=dict(center_recall_tolerance_px=4, candidate_precision_tolerance_px=10,
                               aggregation='pixel-weighted; ignore regions excluded',
                               caveat='centerline tolerance metrics, not pixel-accurate masks or independent trials'),
                  annotated_summary=summary, stress_frames=stress, replay_summary=replay_summary)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for r in summary:
        print(r['subset'], r['method'], f"recall={r['center_recall']:.3f}",
              f"precision={r['tolerance_precision']:.3f}")


if __name__ == '__main__':
    main()
