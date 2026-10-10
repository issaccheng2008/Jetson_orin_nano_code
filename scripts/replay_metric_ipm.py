"""Replay real ground images through the production metric lane detector."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from camera_config import load
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
cfg=load();records=[];thumbs=[]
def tile(im,w=640,h=440):
    scale=min(w/im.shape[1],(h-25)/im.shape[0]);small=cv2.resize(im,None,fx=scale,fy=scale)
    canvas=np.full((h,w,3),35,np.uint8);canvas[25:25+small.shape[0],:small.shape[1]]=small
    return canvas
for path in sorted(args.images.iterdir()):
    if path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
    im=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if im is None:continue
    h,w=im.shape[:2];d=LineDetector(cam_w=w,cam_h=h,cam_height_cm=cfg['mount_height_cm'],cam_pitch_deg=cfg['pitch_deg'],cam_vfov_deg=cfg['vfov_deg'])
    old=SimpleNamespace(cam_w=w,cam_h=h,cam_height=d.cam_height,cam_pitch=d.cam_pitch,cam_vfov_deg=d.cam_vfov_deg,bird_w=320,bird_h=400)
    oldM=LineDetector._build_legacy_birdseye_matrix(old,(20,80))
    # First frame, then two more to check update-state integration; no robot I/O.
    for _ in range(3):result=d.process(im,dt=.1)
    dbg=result[-1]
    actual=cv2.warpPerspective(im,d.M,(d.bird_w,d.bird_h));actual[~d.ground_valid_mask]=0
    panels=[tile(im),tile(cv2.warpPerspective(im,oldM,(320,400))),tile(actual)]
    for panel,title in zip(panels,['Raw','Old warp','Production metric warp (equal x/z scale)']):cv2.putText(panel,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    combined=np.hstack(panels);index=len(records);cv2.imwrite(str(args.output/f'{index:02d}.jpg'),combined)
    thumbs.append(cv2.resize(combined,(960,220)))
    records.append(dict(file=str(path),bird_size=[d.bird_w,d.bird_h],cm_per_px=d.cm_per_px,valid=bool(dbg['measurement_valid']),heading_valid=bool(dbg['heading_control_valid']),heading=float(dbg['heading_control_deg']),near_cm=float(dbg['near_error_cm']),valid_fraction=float(d.ground_valid_mask.mean())))
for start in range(0,len(thumbs),8):cv2.imwrite(str(args.output/f'contact_{start//8}.jpg'),np.vstack(thumbs[start:start+8]))
(args.output/'summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(dict(images=len(records),measurement_valid=sum(x['valid'] for x in records),heading_valid=sum(x['heading_valid'] for x in records))))
