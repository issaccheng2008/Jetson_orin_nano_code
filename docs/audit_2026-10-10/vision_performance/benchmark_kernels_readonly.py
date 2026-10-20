import cv2,numpy as np,sys,glob,time,json,statistics
im=cv2.imread(sorted(glob.glob('/home/isaac/jetson_orin_code/records/tests/2026-10-10/test_202752*/loss/*/*_frame.jpg'))[4])
def stat(v):return {'median_ms':float(np.median(v)),'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95))}
def host(fn,n=30):
 for _ in range(5):fn()
 v=[]
 for _ in range(n):
  t=time.perf_counter();fn();v.append((time.perf_counter()-t)*1000)
 return stat(v)
r={'opencv':cv2.__version__,'python':sys.executable,'threads':cv2.getNumThreads(),'results':{}}
lut=np.clip(np.rint((np.arange(256,dtype=np.float32)-50)*2.2+154),0,255).astype(np.uint8)
def maxcv():
 b,g,rr=cv2.split(im);return cv2.max(cv2.max(b,g),rr)
for name,a,b in [('max_channel',lambda:np.max(im,axis=2),maxcv),('lut',lambda:lut[im],lambda:cv2.LUT(im,lut))]:
 r['results'][name]={'numpy':host(a),'opencv':host(b),'exact':bool(np.array_equal(a(),b()))}
if '--gpu' in sys.argv:
 cv2.cuda.setDevice(0)
 gray=cv2.resize(maxcv(),(623,466));gpu=cv2.cuda_GpuMat();gpu.upload(gray)
 cases=[('gray',im,lambda a:cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),lambda a:cv2.cuda.cvtColor(a,cv2.COLOR_BGR2GRAY))]
 for name,size,shape in [('ellipse31',31,cv2.MORPH_ELLIPSE),('rect51',51,cv2.MORPH_RECT)]:
  k=cv2.getStructuringElement(shape,(size,size));f=cv2.cuda.createMorphologyFilter(cv2.MORPH_BLACKHAT,cv2.CV_8UC1,k)
  cases.append((name,gray,lambda a,k=k:cv2.morphologyEx(a,cv2.MORPH_BLACKHAT,k),lambda a,f=f:f.apply(a)))
 for name,a,cpu,op in cases:
  g=cv2.cuda_GpuMat();g.upload(a)
  def rt():
   g.upload(a);out=op(g);return out.download()
  for _ in range(5):rt()
  event_times=[]
  start=cv2.cuda_Event();end=cv2.cuda_Event()
  for _ in range(30):
   start.record();out=op(g);end.record();end.waitForCompletion();event_times.append(cv2.cuda_Event.elapsedTime(start,end))
  actual=rt();expected=cpu(a)
  r['results'][name]={'cpu_host':host(lambda:cpu(a)),'gpu_roundtrip':host(rt),'gpu_resident_events':stat(event_times),'mismatch_pixels':int(np.count_nonzero(actual!=expected))}
print(json.dumps(r))
