#!/usr/bin/env python3
"""Replay manually labeled continuous card/negative intervals, without hardware.

Example: python scripts/evaluate_card_sequences.py --annotations annotations.json
  --ref c7ad975 --output /tmp/cards-sequence-validation/baseline

Frames within one encounter are correlated. Rough interpolated ROIs support
localization checks, not pixel precision claims. Reset state between intervals.
"""
import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


class ReplayClock:
    """Virtual wall time for confirmation; timing diagnostics keep real clocks."""
    def __init__(self, original, replay_time):
        self.original = original
        self.replay_time = replay_time

    def time(self):
        return self.replay_time

    def __getattr__(self, name):
        return getattr(self.original, name)


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_path(ref, output):
    if ref == 'current':
        destination = output / 'source_snapshot' / 'new_vision'
        for subtree in ('jetson', 'config'):
            shutil.copytree(REPO / 'new_vision' / subtree, destination / subtree,
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        return destination / 'jetson'
    destination = output / 'source_snapshot'
    destination.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(
        ['git', 'archive', ref, 'new_vision/jetson', 'new_vision/config'], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        # Git paths are restricted to this repository subtree.
        archive.extractall(destination, filter='data')
    return destination / 'new_vision/jetson'


def gt_box(sequence, frame):
    keys = sequence.get('box_keyframes', [])
    if not keys:
        return None
    frames = [key['frame'] for key in keys]
    if sequence.get('box_annotation_method') == 'manual_every_detection_call' and frame not in frames:
        raise RuntimeError('Frame %d of %s has no manually reviewed ROI; use annotated sampling' %
                           (frame, sequence['id']))
    return np.array([np.interp(frame, frames, [key['xyxy'][i] for key in keys])
                     for i in range(4)])


def detection_box(quad, width, height):
    if quad is None:
        return None
    quad = np.asarray(quad)
    return np.array([quad[:, 0].min() / width, quad[:, 1].min() / height,
                     quad[:, 0].max() / width, quad[:, 1].max() / height])


def match_box(predicted, truth):
    if predicted is None or truth is None:
        return False, 0.0
    intersection = np.maximum(0, np.minimum(predicted[2:], truth[2:])
                              - np.maximum(predicted[:2], truth[:2])).prod()
    area = np.maximum(0, predicted[2:] - predicted[:2]).prod()
    truth_area = np.maximum(0, truth[2:] - truth[:2]).prod()
    iou = float(intersection / max(area + truth_area - intersection, 1e-12))
    center = (predicted[:2] + predicted[2:]) / 2
    # A loose ROI accommodates hand annotation and shake between keyframes.
    inside = bool(np.all(center >= truth[:2]) and np.all(center <= truth[2:]))
    return inside and iou >= .15, iou


def gt_tile(frame, truth, caption):
    """Ground truth contact sheet cell; detector outputs do not select images."""
    shown = frame.copy()
    height, width = shown.shape[:2]
    if truth is not None:
        cv2.rectangle(shown, (int(truth[0] * width), int(truth[1] * height)),
                      (int(truth[2] * width), int(truth[3] * height)), (0, 255, 0), 3)
    tile = np.full((200, 320, 3), 255, np.uint8)
    tile[20:] = cv2.resize(shown, (320, 180), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def save_gt_sheets(output, name, tiles):
    if not tiles:
        return
    directory = output / 'gt_sheets'
    directory.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(tiles), 20)):
        cells = list(tiles[start:start + 20])
        cells.extend(np.full_like(cells[0], 255) for _ in range((-len(cells)) % 4))
        sheet = np.vstack([np.hstack(cells[index:index + 4])
                           for index in range(0, len(cells), 4)])
        if not cv2.imwrite(str(directory / (name + '_%02d.jpg' % page)), sheet):
            raise RuntimeError('Cannot write GT contact sheet ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ref', default='current')
    parser.add_argument('--every', type=int, default=3)
    parser.add_argument('--sequence', action='append', help='Optional exact sequence ID')
    parser.add_argument('--preprocess', choices=['selective', 'canny'])
    parser.add_argument('--save-gt-sheets', action='store_true', help='Save raw sampled frames with manual GT ROI only')
    args = parser.parse_args()
    if args.every < 1:
        parser.error('--every must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    code = source_path(args.ref, args.output)
    sys.path.insert(0, str(code))
    module = importlib.import_module('shape_detector')
    camera_config = importlib.import_module('camera_config')
    config = code.parent / 'config'
    camera_path = config / 'cameras.json'
    if not camera_path.is_file():
        raise RuntimeError('Missing camera calibration: ' + str(camera_path))
    camera_bytes = camera_path.read_bytes()
    expected_bytes = ((REPO / 'new_vision/config/cameras.json').read_bytes()
                      if args.ref == 'current' else subprocess.check_output(
                          ['git', 'show', args.ref + ':new_vision/config/cameras.json'], cwd=REPO))
    if camera_bytes != expected_bytes:
        raise RuntimeError('Snapshot calibration differs from requested source ref')
    camera_json = json.loads(camera_bytes)
    profile = os.environ.get('CAMERA_PROFILE') or camera_json['active']
    expected_cam = dict(camera_config._DEFAULTS, profile=profile)
    expected_cam.update({key: value for key, value in camera_json['cameras'][profile].items()
                         if value is not None})
    if module._CAM != expected_cam or camera_config.load() != expected_cam:
        raise RuntimeError('Detector camera parameters differ from source calibration')
    print('Verified camera profile', profile, 'distance_calib', module._CAM['distance_calib'], flush=True)
    annotations = json.loads(args.annotations.read_text())
    rows, summaries, still_rows = [], [], []
    original_clock = module.time
    try:
        for sequence in annotations['sequences']:
            if args.sequence and sequence['id'] not in args.sequence:
                continue
            detector = module.ShapeDetector(debug=False)
            if args.preprocess:
                detector.preprocess_mode = args.preprocess
            cap = cv2.VideoCapture(sequence['source'])
            if not cap.isOpened():
                raise RuntimeError('Cannot open ' + sequence['source'])
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, sequence['start'])
            first = len(rows)
            previous_scene = None
            scene_hashes = set()
            gt_tiles = []
            for frame_index in range(sequence['start'], sequence['end'] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Cannot decode frame %d of %s' %
                                       (frame_index, sequence['source']))
                if (frame_index - sequence['start']) % args.every:
                    continue
                module.time = ReplayClock(original_clock, 1000 + frame_index / fps)
                started = time.perf_counter()
                action, debug = detector.update(frame)
                elapsed = (time.perf_counter() - started) * 1000
                debug = debug or {}
                truth = gt_box(sequence, frame_index)
                height, width = frame.shape[:2]
                if args.save_gt_sheets:
                    gt_tiles.append(gt_tile(frame, truth, '%s f%d GT=%s' %
                                             (sequence['id'], frame_index, sequence['truth'])))
                # The new AVI has a historical HUD in its bottom quarter.
                # Image-scene hashes and motion diagnostics omit that overlay.
                scene = cv2.resize(cv2.cvtColor(frame[:int(height * .75)], cv2.COLOR_BGR2GRAY), (240, 135))
                scene_hash = hashlib.sha256(scene.tobytes()).hexdigest()
                duplicate_scene = scene_hash in scene_hashes
                scene_hashes.add(scene_hash)
                scene_change = float(np.mean(np.abs(scene.astype(float) - previous_scene))) if previous_scene is not None else None
                previous_scene = scene.astype(float)
                precise = detection_box(debug.get('quad'), width, height)
                presence = bool(debug.get('presence'))
                present_box = detection_box(debug.get('quad') if debug.get('card_found')
                                           else debug.get('presence_box'), width, height)
                matched, iou = match_box(present_box, truth)
                precise_match, precise_iou = match_box(precise, truth)
                gate = debug.get('cue_candidate_cy_frac') is not None
                cue_quad = getattr(detector, '_cue_box', None)
                cue_box = detection_box(cue_quad[1], width, height) if gate and cue_quad else None
                cue_match, cue_iou = match_box(cue_box, truth)
                negative = sequence['truth'] == 'negative'
                eligible = not negative and truth is not None and truth[1] >= detector.cfg['presence_top_frac']
                prediction = debug.get('shape') or ''
                rows.append(dict(sequence=sequence['id'], source=sequence['source'],
                                 frame=frame_index, saved_video_seconds=frame_index / fps,
                                 scene_sha256=scene_hash, exact_duplicate_scene=duplicate_scene,
                                 scene_change_mean_gray=scene_change,
                                 truth=sequence['truth'], eligible_by_rough_top=bool(eligible),
                                 gt_box=json.dumps(truth.tolist()) if truth is not None else '',
                                 presence=presence, presence_box_match=presence and matched,
                                 presence_box_iou=iou, card_found=bool(debug.get('card_found')),
                                 precise_box_match=precise_match, precise_box_iou=precise_iou,
                                 prediction=prediction,
                                 class_correct=precise_match and prediction == sequence['truth'],
                                 class_wrong=precise_match and bool(prediction) and prediction != sequence['truth'],
                                 negative_presence=negative and presence,
                                 negative_cue_gate=negative and gate, cue_gate_hit=gate,
                                 cue_presence_path=debug.get('cue_presence_path', ''),
                                 cue_repair_diagnostics=json.dumps({key: value for key, value in debug.items()
                                                                   if key.startswith('cue_broken_')}, sort_keys=True),
                                 cue_box_match=cue_match, cue_box_iou=cue_iou,
                                 cue_score=debug.get('cue_score_reference', debug.get('presence_cue', 0)),
                                 cue_score_threshold=detector.cfg['cue_score_min'],
                                 cue_hist_hits=sum(getattr(detector, '_cue_hist', [])),
                                 cue_hist_calls=len(getattr(detector, '_cue_hist', [])),
                                 cue_window_confirmed=bool(detector._cue_confirmed()),
                                 presence_cy=debug.get('presence_cy_frac'), action=action,
                                 elapsed_ms=elapsed))
            cap.release()
            if args.save_gt_sheets:
                save_gt_sheets(args.output, sequence['id'], gt_tiles)
            group = rows[first:]
            cue_hit = next((row for row in group if row['cue_gate_hit'] and row['cue_box_match']), None)
            confirmed = next((row for row in group if row['presence_box_match']), None)
            summaries.append(dict(sequence=sequence['id'], truth=sequence['truth'], calls=len(group),
                                  unique_scene_calls=len(scene_hashes),
                                  exact_duplicate_scene_calls=sum(row['exact_duplicate_scene'] for row in group),
                                  moving_scene_calls=sum(row['scene_change_mean_gray'] is not None and
                                                         row['scene_change_mean_gray'] >= 1 for row in group),
                                  eligible_calls=sum(row['eligible_by_rough_top'] for row in group),
                                  presence_calls=sum(row['presence'] for row in group),
                                  matched_presence_calls=sum(row['presence_box_match'] for row in group),
                                  matched_precise_calls=sum(row['precise_box_match'] for row in group),
                                  correct_calls=sum(row['class_correct'] for row in group),
                                  wrong_calls=sum(row['class_wrong'] for row in group),
                                  negative_presence_calls=sum(row['negative_presence'] for row in group),
                                  negative_cue_gate_calls=sum(row['negative_cue_gate'] for row in group),
                                  first_match_frame=confirmed['frame'] if confirmed else None,
                                  first_cue_match_frame=cue_hit['frame'] if cue_hit else None,
                                  cue_to_presence_saved_seconds=(confirmed['saved_video_seconds'] - cue_hit['saved_video_seconds'])
                                  if cue_hit and confirmed and confirmed['frame'] >= cue_hit['frame'] else None,
                                  onset_to_presence_saved_seconds=(confirmed['frame'] - sequence['start']) / fps
                                  if confirmed else None))
            print(sequence['id'], summaries[-1], flush=True)
        module.time = original_clock
        still_tiles = []
        for still in annotations.get('stills', []):
            if args.sequence:
                break
            frame = cv2.imread(still['source'])
            if frame is None:
                raise RuntimeError('Cannot open ' + still['source'])
            detector = module.ShapeDetector(debug=False)
            if args.preprocess:
                detector.preprocess_mode = args.preprocess
            _, debug = detector.update(frame)
            if args.save_gt_sheets:
                still_tiles.append(gt_tile(frame, np.array(still['box']) if still.get('box') else None,
                                            '%s GT=%s' % (Path(still['source']).stem, still['truth'])))
            quad = detection_box(debug.get('quad'), frame.shape[1], frame.shape[0])
            matched, iou = match_box(quad, np.array(still['box']) if still.get('box') else None)
            still_rows.append(dict(source=still['source'], truth=still['truth'],
                                   presence=bool(debug.get('presence')), card_found=bool(debug.get('card_found')),
                                   precise_box_match=matched, precise_box_iou=iou,
                                   class_correct=matched and debug.get('shape') == still['truth'],
                                   class_wrong=matched and bool(debug.get('shape')) and debug.get('shape') != still['truth'],
                                   prediction=debug.get('shape'),
                                   cue_gate_hit=debug.get('cue_candidate_cy_frac') is not None,
                                   cue_score=debug.get('cue_score_reference', debug.get('presence_cue', 0))))
        if args.save_gt_sheets:
            save_gt_sheets(args.output, 'stills', still_tiles)
    finally:
        module.time = original_clock
    csv_write(args.output / 'frames.csv', rows)
    csv_write(args.output / 'sequences.csv', summaries)
    csv_write(args.output / 'stills.csv', still_rows)
    report = dict(ref=args.ref, preprocess_override=args.preprocess, every=args.every,
                  opencv=cv2.__version__, annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},
                  config_sha256={str(p.relative_to(config)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in config.rglob('*') if p.is_file()},
                  camera_profile_environment=os.environ.get('CAMERA_PROFILE'),
                  camera_config_path=str(camera_path), camera_loaded=module._CAM,
                  camera_verified_against_requested_ref=True,
                  sequences=summaries, stills=still_rows,
                  negative_calls=sum(row['truth'] == 'negative' for row in rows),
                  negative_presence_calls=sum(row['negative_presence'] for row in rows),
                  negative_cue_gate_calls=sum(row['negative_cue_gate'] for row in rows),
                  elapsed_ms_percentiles=dict(zip(['p50', 'p95', 'max'], np.percentile(
                      [row['elapsed_ms'] for row in rows], [50, 95, 100]).tolist())) if rows else {},
                  limitations=annotations.get('limitations', []))
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
