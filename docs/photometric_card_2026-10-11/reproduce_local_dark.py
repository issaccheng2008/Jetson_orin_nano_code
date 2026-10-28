import csv,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path[:0]=['/tmp/robocup_photo_deps',str(Path(sys.argv[1])/'new_vision'/'jetson')]
from shape_detector import ShapeDetector
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(exist_ok=True)
images=sorted(Path('/mnt/d/用户/lenovo/桌面/Robocup/6card').glob('*.jpg'))+[root/'latest_square_card_capture_20261010.jpg']
truths={'圆':'circle','五':'pentagon','正':'square','菱':'diamond','十':'cross','三':'triangle'}
rows=[]
for p in images:
 frame=cv2.imdecode(np.fromfile(str(p),np.uint8),1); truth=truths.get(p.name[0],'square')
 for mode in ('raw','match'):
  d=ShapeDetector();d.photometric_mode=mode;d.preprocess_mode='local_dark'
  _,b=d.update(frame)
  stage='no_candidate' if not b['quad_total'] else 'geometry' if not b['quad_geom'] else 'verify_or_rank' if not b['card_found'] else 'classify' if not b['shape'] else 'classified'
  rows.append(dict(file=p.name,group='dark_unverified' if p.name.startswith('latest') else 'old_bright',truth=truth,mode=mode,shape=b['shape'],found=b['card_found'],stage=stage,reason=b.get('shape_reason'),threshold=b.get('shape_local_dark_threshold'),noise=b.get('shape_local_dark_noise'),quad_total=b['quad_total'],quad_geom=b['quad_geom']))
  print(p.name,mode,stage,b['shape'],flush=True)
with (dest/'card_local_dark.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for group in ('old_bright','dark_unverified'):
 for mode in ('raw','match'):
  rs=[r for r in rows if r['group']==group and r['mode']==mode]
  summary[f'{group}/{mode}']=dict(n=len(rs),box=sum(r['found'] for r in rs),correct=sum(r['shape']==r['truth'] for r in rs),wrong=sum(bool(r['shape']) and r['shape']!=r['truth'] for r in rs),no_class=sum(not r['shape'] for r in rs))
(dest/'local_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
