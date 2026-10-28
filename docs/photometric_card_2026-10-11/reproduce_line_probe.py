import csv,sys
from pathlib import Path
import cv2,numpy as np
sys.path[:0]=['/tmp/robocup_photo_deps',str(Path(sys.argv[1])/'new_vision'/'jetson')]
from line_detector_v1_warp import LineDetector
root=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[]
images=sorted(Path('/mnt/d/用户/lenovo/桌面/Robocup/6card').glob('*.jpg'))+[root/'latest_square_card_capture_20261010.jpg']
for p in images:
 im=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
 for mode in ('raw','match'):
  d=LineDetector();d.photometric_mode=mode;dev,heading,conf,_,b=d.process(im,dt=.1)
  rows.append(dict(file=p.name,group='dark_unverified' if p.name.startswith('latest') else 'old_bright',mode=mode,confidence=conf,paired_ratio=b.get('bottom_pair_ratio'),deviation_px=dev,heading_deg=heading,preprocess_foreground_pixels=b.get('preprocess_foreground_pixels'),preprocess_components=b.get('preprocess_components'),visual_GT_line_pair='not_annotated'))
with (out/'line_diagnostic.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for group in ('old_bright','dark_unverified'):
 for mode in ('raw','match'):
  rs=[r for r in rows if r['group']==group and r['mode']==mode]
  print(group,mode,len(rs),'conf>0.5',sum(float(r['confidence'])>.5 for r in rs),'pair>=0.5',sum(float(r['paired_ratio'] or 0)>=.5 for r in rs))
