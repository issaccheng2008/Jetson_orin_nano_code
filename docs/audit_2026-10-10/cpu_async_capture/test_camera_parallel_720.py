import sys,time,json,statistics
import cv2
sys.path.insert(0,'/tmp/vision_cpu_async_review/new_vision/jetson')
from latest_camera import LatestCamera
from line_detector_v1_warp import LineDetector
# Read existing camera format/settings; do not set exposure, gain, white balance.
from utils import open_camera
cap=open_camera(0,1280,720)
if not cap.isOpened():raise RuntimeError('Camera unavailable')
report={'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),'driver_fps':cap.get(cv2.CAP_PROP_FPS),'fourcc':int(cap.get(cv2.CAP_PROP_FOURCC))}
r=LatestCamera(cap).start()
try:
 for name in ['simulated_150ms_processing','actual_optimized_line']:
  d=LineDetector(int(report['width']),int(report['height']));d.preprocess_mode='contrast';d.photometric_mode='legacy';d.set_heading_regions_cm([[20,22],[24,26]])
  start=time.monotonic();first=None;last=None;ages=[];times=[];count=0
  while time.monotonic()-start<5:
   ok,im,meta=r.read()
   if not ok:continue
   if first is None:first=meta
   last=meta;ages.append(meta['camera_frame_age_ms']);count+=1
   t=time.monotonic()
   if name.startswith('simulated'):time.sleep(.15)
   else:d.process(im,dt=.15)
   times.append((time.monotonic()-t)*1000)
  duration=time.monotonic()-start
  report[name]={'processed_frames':count,'duration_s':duration,'process_throughput_hz':count/duration,'capture_throughput_hz':(last['camera_sequence']-first['camera_sequence'])/(last['camera_capture_returned_s']-first['camera_capture_returned_s']),'frame_wait_age_mean_ms':statistics.mean(ages),'process_mean_ms':statistics.mean(times),'last_meta':last}
finally:report['camera_closed']=r.close()
print(json.dumps(report))
