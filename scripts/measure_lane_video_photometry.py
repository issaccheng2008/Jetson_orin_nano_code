"""Read-only decoded-video photometry; output artifacts only under --output. Fixed sample validity applies to the five reviewed recordings only."""
from pathlib import Path
import argparse,csv,json,hashlib,platform
import cv2,numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
ROOT=args.data_dir
FILES=[ROOT/'camera_commands3.avi']+[ROOT/name for name in (
 'WIN_20260919_18_10_34_Pro_flip.mp4','WIN_20260919_18_11_50_Pro_flip.mp4',
 'WIN_20260919_18_13_14_Pro_flip.mp4','WIN_20260919_18_13_53_Pro_flip.mp4')]
INVALID={
 'camera_commands3.avi':{**{f:'start camera/scene obscured' for f in [0,20]},**{f:'foot/person overlaps measurement ROI' for f in [80,120,140,160,180]},**{f:'non-track floor overlaps ROI' for f in [740,760,780]},**{f:'power strip/cable object overlaps ROI' for f in [1040,1060,1080,1100]},**{f:'foot/person/non-track floor overlaps ROI' for f in range(1180,1341,20)}},
 'WIN_20260919_18_10_34_Pro_flip.mp4':{f:'non-track floor overlaps ROI' for f in [110,120,130]},
 'WIN_20260919_18_11_50_Pro_flip.mp4':{130:'non-track floor overlaps ROI'},
 'WIN_20260919_18_13_14_Pro_flip.mp4':{**{f:'power strip overlaps ROI' for f in [0,10,20,30,40]},**{f:'non-track floor overlaps ROI' for f in [160,170]}},
 'WIN_20260919_18_13_53_Pro_flip.mp4':{**{f:'power strip overlaps ROI' for f in [0,10]},**{f:'foreground cable overlaps ROI' for f in [80,110,120]}}
}
def metrics(im):
 y=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);flat=y.ravel();b,g,r=im.mean(axis=(0,1))
 p=np.percentile(flat,[1,5,10,25,50,75,90,95,99])
 return {'gray_mean':float(flat.mean()),'gray_std_spatial':float(flat.std()),**{'gray_p'+str(k):float(v) for k,v in zip([1,5,10,25,50,75,90,95,99],p)},'gray_ge250_fraction':float(np.mean(y>=250)),'gray_le5_fraction':float(np.mean(y<=5)),'any_channel_ge250_fraction':float(np.mean(np.max(im,axis=2)>=250)),'all_channel_le5_fraction':float(np.mean(np.max(im,axis=2)<=5)),'hsv_s_mean':float(hsv[:,:,1].mean()),'hsv_s_ge250_fraction':float(np.mean(hsv[:,:,1]>=250)),'b_mean':float(b),'g_mean':float(g),'r_mean':float(r),'r_over_g':float(r/g) if g else None,'b_over_g':float(b/g) if g else None,'laplacian_variance':float(cv2.Laplacian(y,cv2.CV_64F,ksize=1).var()),'pixel_count':int(flat.size)}
rows=[];metadata=[]
for path in FILES:
 cap=cv2.VideoCapture(str(path))
 if not cap.isOpened():raise RuntimeError(f'Cannot decode {path}')
 n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));interval=2 if path.suffix=='.avi' else .5
 ids=set(round(t*fps) for t in np.arange(0,n/fps,interval));decoded=0;used=[]
 for idx in range(n):
  ok,im=cap.read()
  if not ok:break
  decoded+=1
  if idx not in ids:continue
  used.append(idx); im=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
  scopes={'whole_raw':im,'whole_common_without_hud':im[:430,:],'ground_roi':im[189:405,240:720]}
  for scope,region in scopes.items():
   invalid=INVALID[path.name].get(idx,'') if scope=='ground_roi' else ''
   rows.append({'video':path.name,'frame_index':idx,'time_s':idx/fps,'scope':scope,'ground_valid':not bool(invalid),'invalid_reason':invalid,**metrics(region)})
 cap.release()
 st=path.stat();metadata.append({'name':path.name,'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'width':w,'height':h,'fps':fps,'reported_frames':n,'decoded_frames':decoded,'duration_s':n/fps,'sample_interval_s':interval,'sample_indices':used,'sample_count':len(used),'valid_ground_samples':sum(idx not in INVALID[path.name] for idx in used)})
 print(path.name,len(used),'ground valid',metadata[-1]['valid_ground_samples'],flush=True)
with (OUT/'frame_metrics.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
metric_keys=[k for k in rows[0] if k not in ['video','frame_index','time_s','scope','ground_valid','invalid_reason','pixel_count']]
summary=[]
for meta in metadata:
 for scope in ['whole_raw','whole_common_without_hud','ground_roi']:
  group=[r for r in rows if r['video']==meta['name'] and r['scope']==scope and r['ground_valid']]
  entry={'video':meta['name'],'scope':scope,'samples':len(group),'aggregate':{}}
  for key in metric_keys:
   a=np.array([r[key] for r in group],dtype=float);entry['aggregate'][key]={'mean_over_frames':float(a.mean()),'median_over_frames':float(np.median(a)),'std_over_frames':float(a.std()),'p05_over_frames':float(np.percentile(a,5)),'p95_over_frames':float(np.percentile(a,95)),'min':float(a.min()),'max':float(a.max())}
  summary.append(entry)
report={'software':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__},'sampling':{'mode':'deterministic uniformly spaced timestamps starting at zero, converted to nearest frame index; sequential video decoding','latest_interval_s':2,'older_interval_s':.5,'maximum_time_is_last_frame_before_declared_duration':True,'aggregation':'equal weight to each sampled valid frame; percentiles are per-frame spatial percentiles then averaged, not pooled pixel percentiles'},'measurement':{'common_resolution':[960,540],'resize':'INTER_AREA for old 1920x1080 files; latest already 960x540','gray':'OpenCV BGR2GRAY uint8, approximate gamma-encoded grayscale intensity, range 0..255, not calibrated luminance','whole_raw':[0,0,960,540],'whole_common_without_hud':[0,0,960,430],'ground_roi':[240,189,720,405],'ground_roi_normalized':[.25,.35,.75,.75],'latest_hud':'Visible dark status overlay starts near y=430 and extends to y=540, ~20.4% of frame; excluded by whole_common_without_hud and ground_roi','manual_validity':'Reviewed each sampled frame in saved inspection contact sheets. Exclude entire ground-ROI sample when obvious foreground objects, people, obscuration or non-track floor intersect. Ground ROI still contains black tape, markings, cards, shadows and specular reflections. No threshold-based deletion of dark pixels.'},'metadata':metadata,'manual_ground_invalid':INVALID,'summary':summary,'limitations':['Videos have unequal lengths, viewpoint, poses, motion, compression, and scene occupancy. Sampling is descriptive, not matched scene design or independent repeats.','Spatial grayscale standard deviation measures all content variability and is not black tape versus adjacent floor contrast.','Laplacian variance depends on scene texture, tape occupancy, pixel intensity, blur, compression and resizing; it does not directly identify focus or motion blur.','Cross-video differences in decoded brightness cannot establish changed exposure settings, illumination, gain, white balance, gamma, codec pipeline, or physical line visibility. No camera metadata was supplied.','Ground validity annotation is conservative visual screening at sampled instants, not semantic pixel segmentation and not a guarantee about unsampled frames.']}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (OUT/'summary.csv').open('w',newline='') as f:
 keys=['video','scope','samples','gray_mean','gray_std_spatial','gray_p5','gray_p50','gray_p95','gray_ge250_fraction','gray_le5_fraction','laplacian_variance','r_over_g','b_over_g'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
 for r in summary:writer.writerow({'video':r['video'],'scope':r['scope'],'samples':r['samples'],**{k:r['aggregate'][k]['mean_over_frames'] for k in keys[3:]}})
print('output',OUT/'report.json')
# Descriptive blocks make scene/time variability explicit; these are not matched-scene exposure tests.
temporal=[]
for lo,hi in [(0,20),(20,40),(40,60),(60,80),(80,100),(100,120),(120,135)]:
 group=[r for r in rows if r['video']=='camera_commands3.avi' and r['scope']=='ground_roi' and r['ground_valid'] and lo<=r['time_s']<hi]
 temporal.append({'video':'camera_commands3.avi','time_start_s':lo,'time_end_s':hi,'valid_samples':len(group),**{k:float(np.mean([r[k] for r in group])) if group else None for k in ['gray_mean','gray_std_spatial','laplacian_variance','r_over_g','b_over_g']}})
report['latest_temporal_ground_summary']=temporal
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (OUT/'latest_temporal.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(temporal[0]));writer.writeheader();writer.writerows(temporal)
