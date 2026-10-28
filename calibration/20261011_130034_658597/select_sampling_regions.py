"""Session-specific sampling envelopes; excludes paper and background.
The top of the board is cropped, so these are safe envelopes whose central
cell boxes lie inside visible physical patches, not reconstructed paper corners.
"""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'new_vision/jetson'))
from photometric_lut import read_image,write_png,gray_from_bgr,extract_board_cells
from fit_photometric_mapping import _sampling_overlay
session=Path(__file__).resolve().parent
spec={'by_file':{},'note':'Visible interior sampling envelopes, not paper corners; top of board cropped. Separate per-frame geometry. Only central boxes are sampled.'}
thumbs=[]
for group in ('auto','fixed'):
 for path in sorted((session/group).glob('*.png')):
  image=read_image(path); gray=gray_from_bgr(image).astype(float)
  # Horizontal black separator: dark trough relative to both adjacent rows.
  score=np.mean((gray[:-12,300:900]+gray[12:,300:900])/2-gray[6:-6,300:900],axis=1)
  sep=int(np.argmax(score[70:260])+76)
  # Left edge above the separator, right edge below it (both dark patches).
  ytop=15; ylower=sep+100
  lefts=[];rights=[]
  for y in (ytop, max(20,sep-30)):
   profile=gray[max(0,y-4):y+5].mean(axis=0)
   lefts.append(int(np.argmin(np.diff(profile)[80:230])+80))
  for y in (sep+40,ylower):
   profile=gray[y-4:y+5].mean(axis=0)
   rights.append(int(np.argmax(np.diff(profile)[960:1060])+960))
  leftslope=(lefts[1]-lefts[0])/(max(20,sep-30)-ytop)
  rightslope=(rights[1]-rights[0])/(ylower-(sep+40))
  bottom=min(sep+220, int(ytop+2.1*(sep-ytop)))
  corners=[[lefts[0]+2,ytop],[rights[0]+rightslope*(ytop-sep-40)-2,ytop],
           [rights[0]+rightslope*(bottom-sep-40)-2,bottom],
           [lefts[0]+leftslope*(bottom-ytop)+2,bottom]]
  spec['by_file'][f'{group}/{path.name}']=corners
  cells,_,H=extract_board_cells(image,corners)
  overlay=_sampling_overlay(image,cells,H,48)
  write_png(session/f'sampling_{group}_{path.name}',overlay)
  thumbs.append(overlay[::4,::4])
  print(group,path.name,'separator',sep,'corners',np.round(corners,1).tolist())
spec['auto_corners']=spec['by_file']['auto/000.png'];spec['fixed_corners']=spec['by_file']['fixed/000.png']
(session/'sampling_corners.json').write_text(json.dumps(spec,indent=2)+'\n')
write_png(session/'sampling_contact_sheet.png',np.concatenate([np.concatenate(thumbs[i:i+5],axis=1) for i in range(0,20,5)],axis=0))
