import cv2,numpy as np,sys,time,glob,json,statistics
sys.path.insert(0,'/home/isaac/jetson_orin_code/new_vision/jetson')
from line_detector_v1_warp import LineDetector
from photometric_thresholds import Photometry
paths=sorted(glob.glob('/home/isaac/jetson_orin_code/records/tests/2026-10-10/test_202752*/loss/*/*_frame.jpg'))[::4];images=[cv2.imread(p) for p in paths]
def run():
 ts=[];outputs=[]
 for round in range(3):
  d=LineDetector();d.preprocess_mode='contrast';d.photometric_mode='legacy';d.set_heading_regions_cm([[20,22],[24,26]])
  for im in images:
   t=time.perf_counter();v=d.process(im,dt=.2);ts.append((time.perf_counter()-t)*1000)
   outputs.append((v[-1]['binary'].copy(),{k:v[-1].get(k) for k in ['measurement_valid','near_error_cm','heading_control_deg','heading_control_valid','base_err_cm']}))
 return ts,outputs
base,ob=run();oldmax=np.max;oldmatch=Photometry.match_image
def fastmax(a,*args,**kw):
 if isinstance(a,np.ndarray) and a.ndim==3 and a.shape[2]==3 and a.dtype==np.uint8 and args==() and kw=={'axis':2}:
  b,g,r=cv2.split(a);return cv2.max(cv2.max(b,g),r)
 return oldmax(a,*args,**kw)
def fastmatch(self,image):
 if isinstance(image,np.ndarray) and image.dtype==np.uint8:
  levels=np.arange(256,dtype=np.float32);lut=np.clip(np.rint((levels-self.mean)*self.match_scale+self.reference_mean),0,255).astype(np.uint8);return cv2.LUT(image,lut)
 return oldmatch(self,image)
np.max=fastmax;Photometry.match_image=fastmatch
fast,of=run()
print(json.dumps({'frames':len(base),'baseline_mean_ms':statistics.mean(base),'optimized_mean_ms':statistics.mean(fast),'baseline_median_ms':statistics.median(base),'optimized_median_ms':statistics.median(fast),'binary_changed_frames':sum(not np.array_equal(a[0],b[0]) for a,b in zip(ob,of)),'geometry_changed_frames':sum(a[1]!=b[1] for a,b in zip(ob,of)),'scope':'in-memory experiment only; current production files untouched'}))
