from pathlib import Path
import argparse
import json,numpy as np
from PIL import Image,ImageDraw
parser=argparse.ArgumentParser(description='Offline legacy versus metric IPM audit; no control changes')
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((Path(__file__).resolve().parents[1]/'new_vision/config/cameras.json').read_text())['cameras']['usb_main']
def xy(H,q):
 p=np.column_stack((q,np.ones(len(q))))@H.T
 return p[:,:2]/p[:,2:]
def hom(s,d):
 A=[];b=[]
 for (x,y),(u,v) in zip(s,d):
  A.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]]);b.extend([u,v])
 return np.append(np.linalg.solve(A,b),1).reshape(3,3)
def warp(im,H,size):
 H=np.linalg.inv(H);H/=H[2,2]
 return im.transform(size,Image.Transform.PERSPECTIVE,tuple(H.flat)[:8],Image.Resampling.BILINEAR)
report=[];thumbs=[]
for path in sorted(args.images.iterdir()):
 if path.suffix.lower() not in ('.jpg','.png','.jpeg'):continue
 im=Image.open(path).convert('RGB');w,h=im.size
 f=h/(2*np.tan(np.deg2rad(cfg['vfov_deg']/2)));c,s=np.cos(np.deg2rad(cfg['pitch_deg'])),np.sin(np.deg2rad(cfg['pitch_deg']));height=cfg['mount_height_cm']
 P=np.array([[f,w/2*c,w/2*height*s],[0,h/2*c-f*s,height*(f*c+h/2*s)],[0,c,height*s]])
 W=2*80*np.tan(np.deg2rad(cfg['vfov_deg']/2))*w/h*.7
 src=xy(P,[[W/2,20],[-W/2,20],[-W/2,80],[W/2,80]])
 clipped=np.clip(src,[0,0],[w-1,h-1]);old=hom(clipped,[[319,399],[0,399],[0,0],[319,0]])
 a,b=cfg['distance_calib']['a'],cfg['distance_calib']['b']
 metric=np.array([[4,0,220],[0,-4*a,360-4*b],[0,0,1]])@np.linalg.inv(P)
 canvas=Image.new('RGB',(1160,435),(35,35,35));d=ImageDraw.Draw(canvas)
 raw=im.copy();raw.thumbnail((400,400));canvas.paste(raw,(0,30));canvas.paste(warp(im,old,(320,400)),(400,30));canvas.paste(warp(im,metric,(440,360)),(720,30))
 for x,txt in [(0,f'{len(report):02d} Raw'),(400,'Existing 320x400'),(720,'Metric candidate 4 px/cm; assumed pose')]:d.text((x+5,5),txt,fill='white')
 for z in (20,40,60,80):
  y=390-4*z;d.line((720,y,1159,y),fill=(40,130,40));d.text((725,y),f'{z} cm',fill='yellow')
 name=f'{len(report):02d}.jpg';canvas.save(out/name,quality=90);canvas.thumbnail((580,218));thumbs.append(canvas)
 G=np.linalg.inv(P)@np.linalg.inv(old);scales=[]
 for y in (0,100,200,300,398):
  q=xy(G,[[160,y],[161,y],[160,y+1]]);scales.append([y,float(q[1,0]-q[0,0]),float(a*(q[0,1]-q[2,1]))])
 t=np.linspace(0,2*np.pi,180,endpoint=False);circle=xy(metric@P,np.column_stack((10*np.cos(t),(50+10*np.sin(t)-b)/a)))
 error=float(np.max(np.abs(np.linalg.norm(circle-circle.mean(axis=0),axis=1)-40)));assert error<1e-7
 report.append(dict(file=str(path),size=[w,h],source_corners=src.tolist(),clipped_corners=clipped.tolist(),row_x_z_scale=scales,circle_test_error_px=error))
for start in range(0,len(thumbs),8):
 sheet=Image.new('RGB',(1160,872),(35,35,35))
 for i,thumb in enumerate(thumbs[start:start+8]):sheet.paste(thumb,((i%2)*580,(i//2)*218))
 sheet.save(out/f'contact_{start//8}.jpg')
(out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(len(report));print(json.dumps(report[0],ensure_ascii=False))
