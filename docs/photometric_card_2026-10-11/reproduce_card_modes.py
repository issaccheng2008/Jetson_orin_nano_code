import csv,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path[:0]=['/tmp/robocup_photo_deps',str(Path(sys.argv[1])/'new_vision'/'jetson')]
from shape_detector import ShapeDetector
from photometric_thresholds import measure
root=Path(sys.argv[1]); dest=Path(sys.argv[2]); dest.mkdir(parents=True,exist_ok=True)
sources=sorted(Path('/mnt/d/用户/lenovo/桌面/Robocup/6card').glob('*.jpg'))
sources+=[root/'latest_square_card_capture_20261010.jpg']
labels={'圆':'circle','五':'pentagon','正':'square','菱':'diamond','十':'cross','三':'triangle'}
rows=[]
for path in sources:
 frame=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
 truth=labels.get(path.name[0], 'square' if path.name.startswith('latest_square') else None)
 for mode in ('raw','match','threshold'):
  d=ShapeDetector();d.photometric_mode=mode
  _,dbg=d.update(frame)
  p=measure(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),mode)
  gray=p.apply(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
  roi=lambda a:a[int(.35*a.shape[0]):int(.75*a.shape[0]),int(.25*a.shape[1]):int(.75*a.shape[1])]
  original=roi(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)); after=roi(gray)
  plane=original.astype(np.float32)*p.match_scale+p.match_offset if p.match_allowed else original.astype(np.float32)
  row=dict(file=path.name,origin=('latest_unverified_dark_capture' if path.name.startswith('latest_') else 'old_real_bright'),truth=truth,mode=mode,
   shape=dbg.get('shape'),card_found=bool(dbg.get('card_found')),presence=bool(dbg.get('presence')),
   quad_total=dbg.get('quad_total'),quad_geom=dbg.get('quad_geom'),geom_rejects=json.dumps(dbg.get('geom_rejects'),ensure_ascii=False),
   appearance_reason=dbg.get('shape_appearance_reason'),class_path=dbg.get('shape_classification_path'),
   pre_mean=float(original.mean()),pre_std=float(original.std()),post_mean=float(after.mean()),post_std=float(after.std()),
   gain=p.match_scale if p.match_allowed else 1.,bias=p.match_offset if p.match_allowed else 0.,
   black_clip=float((plane<0).mean()),white_clip=float((plane>255).mean()),match_allowed=p.match_allowed)
  row['stage']=('no_candidate' if not row['quad_total'] else 'geometry' if not row['quad_geom'] else 'verify_or_rank' if not row['card_found'] else 'classify' if not row['shape'] else 'classified')
  rows.append(row)
  print(path.name,mode,row['stage'],row['shape'],flush=True)
with (dest/'card_modes.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for origin in sorted(set(r['origin'] for r in rows)):
 for mode in ('raw','match','threshold'):
  subset=[r for r in rows if r['origin']==origin and r['mode']==mode]
  summary[f'{origin}/{mode}']=dict(n=len(subset),box=sum(r['card_found'] for r in subset),correct=sum(r['shape']==r['truth'] for r in subset),wrong_class=sum(r['shape'] is not None and r['shape']!=r['truth'] for r in subset),no_class=sum(r['shape'] is None for r in subset),stages={s:sum(r['stage']==s for r in subset) for s in sorted(set(r['stage'] for r in subset))})
(dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
