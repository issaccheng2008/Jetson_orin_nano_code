#!/usr/bin/env python3
"""Offline strict near-road classification on the original BGR half of a bird-pair AVI.

No second perspective transform. The output compares recorded vision commands
with the final guard only; it does not simulate body motion or rerun a policy.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from line_preprocess import extract_lane_candidates
from photometric_thresholds import measure, max_channel, MAX_CHANNEL_REFERENCE, load_lut_mapping
from straight_road_guard import measure_near_road, StraightRoadGuard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--line-log',type=Path)
    p.add_argument('--photometric-mode',default='lut')
    p.add_argument('--straight-wz',type=float,default=.1)
    p.add_argument('--left-wz',type=float,nargs='+',default=[.25,.5])
    p.add_argument('--right-wz',type=float,nargs='+',default=[.1,.2])
    args=p.parse_args()
    if args.photometric_mode=='lut':load_lut_mapping(ROOT/'new_vision/config/photometric_mapping')
    video_manifest=json.loads(args.video.with_name('manifest.json').read_text())
    if video_manifest['frame_source']!='bird_pair':raise ValueError('Requires bird_pair recording')
    index=[json.loads(s) for s in args.video.with_name('frames.jsonl').read_text().splitlines()]
    log={}
    if args.line_log:
        log={r['frame']:r for r in (json.loads(s) for s in args.line_log.read_text().splitlines())}
    capture=cv2.VideoCapture(str(args.video))
    if not capture.isOpened():raise ValueError('Cannot open video')
    width,height=int(capture.get(3)),int(capture.get(4))
    source_w,source_h=width//2,height-110
    detector=LineDetector()
    guard=StraightRoadGuard(straight_wz=args.straight_wz,left_levels=args.left_wz,right_levels=args.right_wz)
    args.output.mkdir(parents=True,exist_ok=True)
    seen=set(); rows=[]; samples={}; evidence=Counter(); states=Counter(); rejects=Counter()
    i=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        meta=index[i];i+=1
        token=meta['source_frame']
        if token in seen:continue
        seen.add(token)
        row=log.get(token,{})
        bird=cv2.resize(frame[:source_h,:source_w],(detector.bird_w,detector.bird_h))
        photo=measure(max_channel(bird),args.photometric_mode,MAX_CHANNEL_REFERENCE,lut_input='max_channel')
        gray=photo.apply(max_channel(bird))
        mask=extract_lane_candidates(gray,'contrast',photometry=photo,
                                    valid_mask=detector.ground_valid_mask)[1]
        mask[~detector.ground_valid_mask]=0
        scale=float(row.get('measurement',{}).get('lateral_scale',1.))
        obs=measure_near_road(mask,detector.ground_valid_mask,detector._px_to_ground_cm,
            row_z_cm=detector._lut_z_cm,row_x_scale_cm=detector._lut_cm_per_px*scale,
            center_x=detector.center_x)
        stamp=meta['command_monotonic_s']
        paused=bool(row.get('card_window',meta['vx']<=0))
        state=guard.update(obs['straight_evidence'],stamp,token,paused=paused)
        bypass=paused or bool(row.get('start_gate',False)) or meta['vx']<=0
        final=guard.limit(meta['wz'],bypass=bypass)
        evidence[obs['straight_evidence']]+=1;states[state]+=1;rejects[obs['straight_reject_reason']]+=1
        rows.append(dict(source_frame=token,time_s=stamp-index[0]['command_monotonic_s'],
            evidence=obs['straight_evidence'],state=state,reason=obs['straight_reject_reason'],
            edges=obs['straight_edges'],turn_deg=obs['straight_max_turn_deg'],
            rmse_cm=obs['straight_max_rmse_cm'],width_cm=obs['straight_width_cm'],
            paused=paused,vx=meta['vx'],recorded_wz=meta['wz'],guarded_wz=final))
        if obs['straight_evidence'] not in samples:
            display=bird.copy()
            for z in (10,40):
                y=int(np.argmin(np.abs(detector._lut_z_cm-z)))
                cv2.line(display,(0,y),(display.shape[1]-1,y),(255,0,255),1)
            view=np.hstack((display,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)))
            cv2.putText(view,f'{token} {obs["straight_evidence"]} {obs["straight_reject_reason"]}',
                        (5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            samples[obs['straight_evidence']]=view
    capture.release()
    if i!=len(index):raise ValueError('Video and index length mismatch')
    with (args.output/'frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,view in samples.items():cv2.imwrite(str(args.output/(name.lower()+'.jpg')),view)
    summary=dict(source=str(args.video),source_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        video_frames=i,unique_source_frames=len(rows),evidence=dict(evidence),states=dict(states),
        rejects=dict(rejects),changed_recorded_commands=sum(r['guarded_wz']!=r['recorded_wz'] for r in rows),
        limitation='Compressed original BGR birdseye crop, resized; one LUT and contrast extraction, '
        'no second IPM. Counterfactual restriction of recorded WZ only, not driving validation. '
        'Single-camera pose and geometry remain replay estimates; no road-shape ground-truth labels.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
