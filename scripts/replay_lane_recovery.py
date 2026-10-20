#!/usr/bin/env python3
"""Same-frame replay of recorded bird BGR; never re-project an existing bird view."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--mode', choices=('baseline', 'continuous'), default='baseline')
    p.add_argument('--gap-fill', action='store_true')
    p.add_argument('--strictness-pct', type=float, default=0.)
    p.add_argument('--visible-direction-fallback', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--photometric-mode', default='lut')
    p.add_argument('--max-frames', type=int)
    p.add_argument('--no-save-masks', action='store_true',
                   help='Counts-only replay; skip per-frame PNG output')
    args = p.parse_args()
    sys.path.insert(0, str(args.code_root/'new_vision/jetson'))
    from line_detector_v1_warp import LineDetector
    from photometric_thresholds import load_lut_mapping
    # Record the exact code/data shared by a same-frame sweep. Revision alone
    # does not identify an uncommitted detector used during development.
    code_hashes = {name: hashlib.sha256((args.code_root/'new_vision/jetson'/name).read_bytes()).hexdigest()
        for name in ('line_detector_v1_warp.py', 'continuous_lane_heading.py',
                     'lane_boundary_path.py', 'visible_lane_direction.py',
                     'lane_gap_completion.py', 'line_preprocess.py',
                     'masked_ground.py', 'camera_config.py', 'photometric_thresholds.py')}
    if args.photometric_mode == 'lut':
        load_lut_mapping(args.code_root/'new_vision/config/photometric_mapping')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    width, height = int(cap.get(3)), int(cap.get(4))
    index_path = args.video.with_name('frames.jsonl')
    index = [json.loads(line) for line in index_path.read_text().splitlines()] if index_path.exists() else None
    detector = LineDetector()
    detector.set_heading_regions_cm(((20.,22.),(24.,26.)))
    detector.preprocess_mode = 'contrast'
    detector.photometric_mode = args.photometric_mode
    detector.lane_gap_fill = args.gap_fill
    detector.visible_direction_fallback = args.visible_direction_fallback
    if args.mode != 'baseline':
        detector.set_lane_observation_mode(args.mode, args.strictness_pct)
    crop_w, crop_h = width//2, height-110
    detector.M = np.diag([detector.bird_w/crop_w,detector.bird_h/crop_h,1.])
    detector.M_inv = np.linalg.inv(detector.M)
    args.output.mkdir(parents=True, exist_ok=True)
    masks = args.output/'masks'
    if not args.no_save_masks:
        masks.mkdir(exist_ok=True)
    counts, moving_counts, rejects = Counter(), Counter(), Counter()
    selected={56,62,70,86,110,124,295,347,432,454,466}
    seen = set()
    rows=[]
    previous=None
    video_i=0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meta = index[video_i] if index is not None else {'source_frame':video_i+1,'command_monotonic_s':video_i/10.,'vx':.2}
        video_i += 1
        token=meta['source_frame']
        if token in seen:
            continue
        seen.add(token)
        stamp=meta['command_monotonic_s']
        dt=.1 if previous is None else max(.001, stamp-previous)
        previous=stamp
        bird=frame[:crop_h,:crop_w]
        saved=detector.snapshot_tracking_state()
        started=time.perf_counter()
        debug=detector.process(bird,dt=dt,frame_token=token)[-1]
        cost=(time.perf_counter()-started)*1000
        if meta['vx']<=0:
            detector.restore_tracking_state(saved)
        mode=debug['tracking_mode']
        counts[mode]+=1
        if meta['vx']>0:
            moving_counts[mode]+=1
        if not debug['measurement_valid']:
            rejects[debug['reject_reason']]+=1
        rows.append(dict(source_frame=token,video_frame=video_i-1,
            time_s=stamp-(index[0]['command_monotonic_s'] if index else 0),
            moving=meta['vx']>0,valid=bool(debug['measurement_valid']),
            mode=mode,reason=debug['reject_reason'],
            tracker_reason=debug.get('boundary_tracker_reject_reason'),
            scan_valid=debug.get('lane_scan_valid'),
            scan_reason=debug.get('lane_scan_reject_reason'),
            position_valid=debug.get('position_valid',debug['measurement_valid']),
            near_cm=debug['near_error_cm'],heading_deg=debug['heading_control_deg'],
            heading_span_cm=debug.get('heading_control_z_span_cm'),
            heading_points=debug.get('heading_control_points'),
            bridges=debug.get('lane_gap_bridges',0),
            added_pixels=debug.get('lane_gap_added_pixels',0),process_ms=cost))
        # Preserve masks for all frames; makes any reported improvement reviewable.
        if not args.no_save_masks:
            cv2.imwrite(str(masks/f'{token:04d}.png'),debug['binary'])
        if token in selected and not args.no_save_masks:
            cv2.imwrite(str(args.output/f'{token:04d}_bird.jpg'),bird)
            cv2.imwrite(str(args.output/f'{token:04d}_view.jpg'),debug['bird'])
            if debug.get('binary_gap_added') is not None:
                cv2.imwrite(str(args.output/f'{token:04d}_added.png'),debug['binary_gap_added'])
        if len(rows)%100==0:
            print(f'{args.mode} fill={args.gap_fill}: {len(rows)} unique frames',flush=True)
        if args.max_frames is not None and len(rows)>=args.max_frames:
            break
    cap.release()
    (args.output/'frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary=dict(video=str(args.video),sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        code_root=str(args.code_root),mode=args.mode,
        effective_mode=detector.lane_observation_mode,gap_fill=args.gap_fill,
        code_sha256=code_hashes,
        index_sha256=(hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None),
        fixed_settings=dict(line_preprocess=detector.preprocess_mode,
            photometric_mode=detector.photometric_mode,
            lut_sha256=({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted((args.code_root/'new_vision/config/photometric_mapping').glob('*'))
                if f.is_file()} if args.photometric_mode == 'lut' else {}),
            heading_regions_cm=[[20.,22.],[24.,26.]],
            bird_dimensions=[detector.bird_w,detector.bird_h],
            gap_max_cm=detector.lane_gap_max_cm,
            visible_direction_fallback=(detector.lane_observation_mode=='continuous'
                                        and detector.visible_direction_fallback),
            tracker_continuous=detector._boundary_path_tracker.continuous),
        strictness_pct=args.strictness_pct,
        unique_frames=len(rows),video_frames_read=video_i,crop=[0,0,crop_w,crop_h],
        modes=dict(counts),moving_modes=dict(moving_counts),invalid_reasons=dict(rejects),
        bridges=sum(r['bridges'] for r in rows),added_pixels=sum(r['added_pixels'] for r in rows),
        mean_process_ms=float(np.mean([r['process_ms'] for r in rows])),
        limitations='Compressed recorded bird view, resized to native grid; no second IPM. '
        'Repeated source frames skipped. Centimetres are replay estimates. '
        'Validity/visibility continuity does not measure driving success or geometric accuracy.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
