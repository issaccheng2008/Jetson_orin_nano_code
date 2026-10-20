import sys,glob,time,json,hashlib,statistics
import cv2,numpy as np
sys.path.insert(0,sys.argv[1])
from line_detector_v1_warp import LineDetector
paths=sorted(glob.glob('/home/isaac/jetson_orin_code/records/tests/2026-10-10/test_202752*/loss/*/*_frame.jpg'))[::4]
images=[cv2.imread(p) for p in paths];results={}
for mode in ['contrast','legacy']:
 ts=[];checks=[]
 for repeat in range(4):
  d=LineDetector();d.preprocess_mode=mode;d.photometric_mode='legacy';d.set_heading_regions_cm([[20,22],[24,26]])
  for im in images:
   t=time.perf_counter();v=d.process(im,dt=.2);elapsed=(time.perf_counter()-t)*1000
   if repeat:
    ts.append(elapsed);dbg=v[-1];checks.append({'binary':hashlib.sha256(dbg['binary'].tobytes()).hexdigest(),'geometry':{k:dbg.get(k) for k in ['measurement_valid','near_error_cm','heading_control_deg','heading_control_valid','base_err_cm']}})
 results[mode]={'mean_ms':statistics.mean(ts),'median_ms':statistics.median(ts),'frames':len(ts),'checks':checks}
print(json.dumps({'opencv':cv2.__version__,'source':sys.argv[1],'results':results}))
