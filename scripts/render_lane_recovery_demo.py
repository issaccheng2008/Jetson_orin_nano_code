#!/usr/bin/env python3
"""Render saved same-frame masks beside the recorded bird image."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('results',type=Path);p.add_argument('--after',default='completed');p.add_argument('--name',default='comparison');a=p.parse_args()
rows={r['video_frame']:r for r in map(json.loads,(a.results/a.after/'frames.jsonl').read_text().splitlines())}
old={r['source_frame']:r for r in map(json.loads,(a.results/'baseline/frames.jsonl').read_text().splitlines())}
cap=cv2.VideoCapture(a.video);out=cv2.VideoWriter(str(a.results/(a.name+'.avi')),cv2.VideoWriter_fourcc(*'MJPG'),5,(960,276))
i=0;saved=0
while True:
 ok,f=cap.read()
 if not ok:break
 r=rows.get(i);i+=1
 if r is None:continue
 token=r['source_frame'];panels=[]
 bird=f[:358,:480]; before=cv2.imread(str(a.results/'baseline/masks'/f'{token:04d}.png'));after=cv2.imread(str(a.results/a.after/'masks'/f'{token:04d}.png'))
 for im,label in ((bird,'Recorded bird'),(before,'Before: '+old[token]['mode']),(after,'After: '+r['mode'])):
  panel=np.zeros((276,320,3),np.uint8);panel[32:]=cv2.resize(im,(320,244));cv2.putText(panel,label,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);panels.append(panel)
 canvas=np.hstack(panels);out.write(canvas)
 if token in (56,62,86,124,295,347):cv2.imwrite(str(a.results/f'{a.name}_{token:04d}.jpg'),canvas)
cap.release();out.release()
