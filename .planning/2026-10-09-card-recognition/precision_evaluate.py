"""Development comparison: current box finder, old/new symbol extraction.

Labels are used only for scoring. Extra WIN images were visually labelled
triangle. This is shared development validation, not an unseen holdout.
"""
import contextlib,csv,hashlib,io,json,platform,subprocess,sys,time
from datetime import datetime, timezone
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'new_vision/jetson'))
from shape_detector import ShapeDetector, _CAM
from card_appearance import card_appearance
cv2.setNumThreads(1)
BASE=Path('/mnt/d/用户/Lenovo/桌面/Robocup')
LABELS={'三':'triangle','五':'pentagon','圆':'circle','正':'square','菱':'diamond','十':'cross'}
rows=[]
for dataset,paths,gains in [('photos31',sorted((BASE/'6card').glob('*')),(1.,.6,.4)),('WIN11',sorted(BASE.glob('WIN_20260922*.jpg')),(1.,))]:
 for gain in gains:
  for path in paths:
   truth=LABELS.get(path.stem[0]) if dataset=='photos31' else 'triangle'
   if not truth:continue
   frame=cv2.imdecode(np.fromfile(path,np.uint8),1)
   frame=np.clip(frame.astype(float)*gain,0,255).astype('uint8')
   detector=ShapeDetector()
   with contextlib.redirect_stdout(io.StringIO()):_,dbg=detector.update(frame)
   # Isolate symbol extraction on the SAME current box/preprocessed masks.
   # update() already uses the new module after integration, so its final
   # shape is never the old extractor baseline.
   old=None;new=None;reason='no_quad';ms=0;contrast=None
   if dbg.get('quad_work') is not None:
    old=detector._classify(dbg['warp'], {})
    start=time.perf_counter();appearance=card_appearance(dbg['gray'],dbg['quad_work'])
    ms=(time.perf_counter()-start)*1000
    reason=appearance.reason;contrast=appearance.contrast
    if appearance.symbol is not None:new=detector._classify_contour(appearance.symbol)
   rows.append(dict(dataset=dataset,file=path.name,gain=gain,truth=truth,quad=dbg.get('quad_work') is not None,old=old,new=new,combined=new if new is not None else old,reason=reason,contrast=contrast,appearance_ms=ms))
  print(dataset,gain,flush=True)
DEST=Path('/mnt/d/用户/lenovo/桌面/github repository/jetson_orin_nano_code/.planning/2026-10-09-card-recognition')
config=ROOT/'new_vision/config/cameras.json'
if not config.is_file():
 raise RuntimeError('Required camera config missing; refusing silent default calibration')
source_paths=['new_vision/jetson/shape_detector.py','new_vision/jetson/card_appearance.py',
              'new_vision/jetson/card_presence.py','new_vision/jetson/photometric_thresholds.py',
              'new_vision/jetson/camera_config.py','new_vision/config/cameras.json',
              '.planning/2026-10-09-card-recognition/precision_evaluate.py']
summary=[]
for dataset,gain in sorted(set((r['dataset'],r['gain']) for r in rows)):
 subset=[r for r in rows if (r['dataset'],r['gain'])==(dataset,gain)]
 summary.append(dict(dataset=dataset,gain=gain,n=len(subset),**{
  name:dict(correct=sum(r[name]==r['truth'] for r in subset),
            wrong=sum(r[name] is not None and r[name]!=r['truth'] for r in subset),
            no_shape=sum(r[name] is None for r in subset))
  for name in ['old','new','combined']}))
metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
 comparison='old/new symbol extraction on same current box finder and preprocessing; not a full historical detector comparison',
 git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                for path in source_paths},
 camera=_CAM,detector_cfg=ShapeDetector().cfg,
 software=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__,
               platform=platform.platform(),opencv_threads=cv2.getNumThreads(),
               opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()),
 summary=summary)
(DEST/'precision_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
with (DEST/'precision_scores.csv').open('w') as file:
 writer=csv.DictWriter(file,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
for dataset,gain in sorted(set((r['dataset'],r['gain']) for r in rows)):
 subset=[r for r in rows if (r['dataset'],r['gain'])==(dataset,gain)]
 print(dataset,gain,len(subset),{name:dict(correct=sum(r[name]==r['truth'] for r in subset),wrong=sum(r[name] is not None and r[name]!=r['truth'] for r in subset),no_shape=sum(r[name] is None for r in subset)) for name in ['old','new','combined']})
 print('disagree',[(r['file'],r['old'],r['new'],r['reason']) for r in subset if r['old']!=r['new']])
