import sys,time,json,glob,statistics,cProfile,pstats,io
from collections import defaultdict
import cv2,numpy as np
sys.path.insert(0,'/home/isaac/jetson_orin_code/new_vision/jetson')
import line_detector_v1_warp as lm
import line_preprocess as lp
from photometric_thresholds import Photometry

paths=sorted(glob.glob('/home/isaac/jetson_orin_code/records/tests/2026-10-10/test_202752*/loss/*/*_frame.jpg'))[::4]
images=[cv2.imread(p) for p in paths]
timings=defaultdict(list)
def wrap(obj,name,label):
 old=getattr(obj,name)
 def fn(*a,**kw):
  st=time.perf_counter();v=old(*a,**kw);timings[label].append((time.perf_counter()-st)*1000);return v
 setattr(obj,name,fn)
for obj,name,label in [(lm,'measure','photometry_measure'),(Photometry,'match_image','photometry_match'),(lm,'extract_lane_candidates','preprocess_total'),(lp,'masked_clahe','masked_clahe'),(lp,'masked_morphology','masked_morphology'),(lp,'masked_gaussian','masked_gaussian'),(cv2,'warpPerspective','warp'),(cv2,'connectedComponentsWithStats','components'),(lm,'trace_heading','trace_heading')]:
 if hasattr(obj,name):wrap(obj,name,label)
for name in ['_detect_two_band_lanes','_detect_bottom_lock','_build_visualization','_centroid_pair_center','_scan_band_midline']:
 if hasattr(lm.LineDetector,name):wrap(lm.LineDetector,name,name)
def create():
 d=lm.LineDetector();d.preprocess_mode='contrast';d.photometric_mode='legacy';d.set_heading_regions_cm([[20,22],[24,26]]);return d
d=create()
for im in images[:3]:d.process(im,dt=.2)
timings.clear();total=[]
for _ in range(3):
 d=create()
 for im in images:
  t=time.perf_counter();d.process(im,dt=.2);total.append((time.perf_counter()-t)*1000)
report={'samples':paths,'frames':len(total),'opencv':cv2.__version__,'python':sys.executable,'threads':cv2.getNumThreads(),'total_mean_ms':statistics.mean(total),'stages':{k:{'calls':len(v),'mean_per_call_ms':statistics.mean(v),'mean_per_frame_ms':sum(v)/len(total)} for k,v in timings.items()}}
print(json.dumps(report))
pr=cProfile.Profile();d=create();pr.enable()
for im in images:d.process(im,dt=.2)
pr.disable();o=io.StringIO();pstats.Stats(pr,stream=o).sort_stats('tottime').print_stats(25);print(o.getvalue())
