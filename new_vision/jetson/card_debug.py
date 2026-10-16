"""Camera-only card recognition dashboard; no policy, serial or network output."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from camera_config import load as load_camera
from camera_controls import add_arguments, apply_camera_controls, validate_args
from latest_camera import LatestCamera
from photometric_thresholds import load_lut_mapping, resolve_mode
from shape_detector import ShapeDetector
from shape_demo import put_text, _load_font
from utils import open_camera

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('none', 'circle', 'pentagon', 'square', 'diamond', 'cross', 'triangle')
ZH = ('无卡', '圆形', '五角星', '正方形', '菱形', '十字', '三角形')
WINDOW = 'Card recognition debug - camera only'


def label(name):
    if name in NAMES and _load_font(24) is not None:
        return f'{ZH[NAMES.index(name)]} / {name}'
    return name


def scalar_debug(dbg):
    """Keep all scalar diagnostics and small geometry, omit image buffers."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size <= 64 else None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return {k: convert(v) for k, v in dbg.items()
            if not isinstance(v, np.ndarray) or v.size <= 64}


def state(dbg):
    # A weak presence cue never counts as a verified frame or a class label.
    if dbg.get('card_found'):
        return ('RECOGNIZED' if dbg.get('shape') else 'BOX FOUND / UNKNOWN',
                (60, 220, 70) if dbg.get('shape') else (0, 200, 255))
    return ('PRESENCE CUE ONLY' if dbg.get('presence') else 'NO CARD', (0, 180, 255))


def check_display(parser):
    if not sys.platform.startswith('linux') or os.getenv('QT_QPA_PLATFORM')=='offscreen':
        return
    if not (os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')):
        parser.error('No desktop display. Run in a Jetson desktop terminal, or use --headless --max-frames 10.')
    if os.getenv('DISPLAY'):
        # Qt aborts the process on an unauthorised X connection. Detect it first
        # so an SSH shell gets an actionable error instead of a core dump.
        import ctypes
        import ctypes.util
        library=ctypes.util.find_library('X11')
        if library:
            x11=ctypes.CDLL(library)
            x11.XOpenDisplay.argtypes=[ctypes.c_char_p]
            x11.XOpenDisplay.restype=ctypes.c_void_p
            x11.XCloseDisplay.argtypes=[ctypes.c_void_p]
            connection=x11.XOpenDisplay(None)
            if not connection:
                parser.error('Cannot access desktop display. Log in as isaac on the Jetson screen and launch from its terminal, or use --headless.')
            x11.XCloseDisplay(connection)


class Statistics:
    def __init__(self):
        self.frames = self.boxes = self.classified = self.labelled = self.correct = 0
        self.confusion = Counter()

    def observe(self, dbg, expected):
        self.frames += 1
        self.boxes += bool(dbg.get('card_found'))
        shape = dbg.get('shape') if dbg.get('card_found') else None
        self.classified += shape is not None
        prediction = shape or ('unknown' if dbg.get('card_found') or dbg.get('presence') else 'none')
        if expected != 'unlabelled':
            self.labelled += 1
            self.correct += prediction == expected
            self.confusion[(expected, prediction)] += 1

    def report(self):
        return dict(frames=self.frames, verified_boxes=self.boxes, classified=self.classified,
                    labelled_frames=self.labelled, correct_frames=self.correct,
                    frame_accuracy=self.correct / self.labelled if self.labelled else None,
                    confusion=[dict(expected=a, predicted=b, frames=n)
                               for (a, b), n in sorted(self.confusion.items())])


def tile(image, title, width, height, quad=None):
    canvas = np.full((height, width, 3), 24, np.uint8)
    if image is not None:
        view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        if quad is not None:
            cv2.polylines(view, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        scale = min(width / view.shape[1], (height - 32) / view.shape[0])
        view = cv2.resize(view, (max(1, round(view.shape[1]*scale)), max(1, round(view.shape[0]*scale))))
        y, x = 32 + (height-32-view.shape[0])//2, (width-view.shape[1])//2
        canvas[y:y+view.shape[0], x:x+view.shape[1]] = view
    else:
        cv2.putText(canvas, 'No current card data', (12, height//2), cv2.FONT_HERSHEY_SIMPLEX, .5, (140,140,140), 1)
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230), 1)
    return canvas


def dashboard(frame, dbg, stats, expected, fps, streak, paused, modes):
    canvas = np.full((1000, 1440, 3), 18, np.uint8)
    status, color = state(dbg)
    titles={'RECOGNIZED':'已识别图卡','BOX FOUND / UNKNOWN':'找到黑框，形状未知',
            'PRESENCE CUE ONLY':'仅有图卡存在信号','NO CARD':'未检测到图卡'}
    title=f'{titles[status]} / {status}' if _load_font(32) is not None else status
    put_text(canvas, f'{title}    {label(dbg.get("shape") or "-")}', (15, 8), color, 32)
    accuracy = f'{100*stats.correct/stats.labelled:.1f}%' if stats.labelled else 'N/A (set expected label)'
    lines = [f'Expected: {label(expected)} | frame accuracy: {accuracy} ({stats.correct}/{stats.labelled})',
             f'Frames {stats.frames} | boxes {stats.boxes} | classified {stats.classified} | same-class streak {streak} | {fps:.1f} FPS | detector {dbg.get("shape_total_ms",0):.1f} ms',
             f'{"PAUSED | " if paused else ""}{modes} | presence={bool(dbg.get("presence"))} | source={dbg.get("shape_proposal_source")} | closure={dbg.get("closure",0):.3f}',
             '0=no card  1=circle  2=star  3=square  4=diamond  5=cross  6=triangle  U=unlabelled  P=pause  S=save  R=reset  Q=quit']
    for i, text in enumerate(lines):
        put_text(canvas, text, (15, 52+i*27), (220,220,220), 20)
    raw = frame.copy()
    if dbg.get('quad') is not None:
        cv2.polylines(raw, [np.asarray(dbg['quad'],np.int32).reshape(-1,1,2)], True, (0,255,0), 3)
    if dbg.get('presence_box') is not None:
        # ShapeDetector returns four original-frame corner points, just like
        # quad. A cue box is not an (x, y, width, height) tuple.
        cue_quad = np.asarray(dbg['presence_box'],np.int32).reshape(4,1,2)
        cv2.polylines(raw,[cue_quad],True,(0,180,255),2)
    canvas[170:656,:864] = tile(raw,'Camera: green=verified black frame, amber=presence cue',864,486)
    for i,(key,title) in enumerate((('gray','Detector grayscale'),('appearance_gray','Rectified grayscale'),
                                  ('appearance_binary','Local symbol binary'),('warp','Legacy classifier warp'))):
        y,x = 170+(i//2)*243,864+(i%2)*288
        canvas[y:y+243,x:x+288] = tile(dbg.get(key),title,288,243)
    for i,(key,title) in enumerate((('binary','Selective binary + verified frame'),('ink','Ring / card ink + verified frame'))):
        canvas[656:926,i*720:(i+1)*720] = tile(dbg.get(key),title,720,270,dbg.get('quad_work'))
    rejects = sorted((dbg.get('geom_rejects') or {}).items(),key=lambda p:-p[1])[:5]
    details = [f'Candidates {dbg.get("quad_total",0)} -> geometry {dbg.get("quad_geom",0)} -> scores {len(dbg.get("scores") or [])} | rejects: {rejects}',
               f'Local={dbg.get("shape_local_candidate")} | legacy={dbg.get("shape_legacy_candidate")} | path={dbg.get("shape_classification_path")} | appearance={dbg.get("shape_appearance_reason")}']
    for i,text in enumerate(details):
        put_text(canvas,text,(12,932+28*i),(200,200,200),19)
    return canvas


def save_snapshot(directory, index, frame, view, dbg, row):
    destination = directory / f'snapshot_{index:06d}_{time.time_ns()}'
    destination.mkdir()
    images = dict(camera=frame, dashboard=view)
    images.update({k:v for k,v in dbg.items() if isinstance(v,np.ndarray) and v.ndim in (2,3) and v.size>64})
    for name,image in images.items():
        if not cv2.imwrite(str(destination / f'{name}.png'),image):
            raise OSError(f'Could not save {name}')
    (destination / 'diagnostics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[snapshot] {destination}',flush=True)


def main(argv=None):
    cam = load_camera()
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--camera', default='/dev/robot_main_camera' if Path('/dev/robot_main_camera').exists() else str(cam['index']))
    sources.add_argument('--video',type=Path)
    sources.add_argument('--image',type=Path)
    parser.add_argument('--width',type=int,default=cam['width'])
    parser.add_argument('--height',type=int,default=cam['height'])
    parser.add_argument('--shape-preprocess',choices=('selective','canny','local_dark'),default='selective')
    parser.add_argument('--photometric-mode',choices=('legacy','raw','threshold','match','lut','normalize'),default='legacy')
    parser.add_argument('--photometric-lut-dir',type=Path,default=ROOT/'new_vision/config/photometric_mapping')
    parser.add_argument('--expected',choices=(*NAMES,'unlabelled'),default='unlabelled')
    parser.add_argument('--output',type=Path,default=ROOT/'records/card_debug')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-frames',type=int,default=0,help='0 means until quit/end of input')
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.width<=0 or args.height<=0 or args.max_frames<0:
        parser.error('width/height must be positive; max-frames must be nonnegative')
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.headless:
        check_display(parser)
    if resolve_mode(args.photometric_mode)=='lut':
        load_lut_mapping(args.photometric_lut_dir)
    detector = ShapeDetector()
    detector.preprocess_mode, detector.photometric_mode = args.shape_preprocess,args.photometric_mode
    session = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    stats, expected = Statistics(),args.expected
    meta = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                camera_profile=cam,detector_config=detector.cfg,
                note='Per processed-frame accuracy; correlated frames are not independent card trials. No action is issued.')
    (session/'session.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    capture = latest = None
    paused, previous, streak, total = False,None,0,0
    row = {}
    print(f'[card-debug] camera only; no robot commands. Output: {session}',flush=True)
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise RuntimeError(f'Cannot read image: {args.image}')
        else:
            device = int(args.camera) if args.camera.isdigit() else args.camera
            if args.video:
                capture = cv2.VideoCapture(str(args.video))
            else:
                report = apply_camera_controls(f'/dev/video{device}' if isinstance(device,int) else device,args)
                (session/'camera_controls.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                capture = open_camera(device,args.width,args.height)
            if not capture.isOpened():
                raise RuntimeError('Cannot open camera/video; check USB/device or another program using the camera')
            if not args.video:
                latest = LatestCamera(capture).start()
        if not args.headless:
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW,1280,889)
        failures,last_print = 0,0.
        with (session/'frames.jsonl').open('w',encoding='utf-8',buffering=1) as log:
            while True:
                if not paused:
                    started = time.monotonic()
                    camera_meta = {}
                    if not args.image:
                        if latest:
                            ok,frame,camera_meta = latest.read(timeout=1.)
                        else:
                            ok,frame = capture.read()
                        if not ok:
                            if args.video:
                                break
                            failures += 1
                            if not args.headless:
                                # Hide the previous recognition immediately on camera failure.
                                blank = np.zeros((540,960,3),np.uint8)
                                cv2.putText(blank,'CAMERA UNAVAILABLE - NO CURRENT RESULT',(20,260),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
                                cv2.imshow(WINDOW,blank)
                                if cv2.waitKey(1)&255 in (27,ord('q')):
                                    break
                            if failures>=5:
                                raise RuntimeError('Camera stopped delivering frames')
                            continue
                    failures = 0
                    _,dbg = detector.update(frame,classify=True)
                    total += 1
                    current = dbg.get('shape')
                    streak = streak+1 if current is not None and current==previous else int(current is not None)
                    previous = current
                    stats.observe(dbg,expected)
                    fps = 1/max(1e-6,time.monotonic()-started)
                    row = dict(frame=total,timestamp=datetime.now().isoformat(),expected=expected,
                               status=state(dbg)[0],camera=camera_meta,debug=scalar_debug(dbg))
                    log.write(json.dumps(row,ensure_ascii=False)+'\n')
                    if time.monotonic()-last_print>1:
                        print(f'[{total}] {state(dbg)[0]} shape={current} expected={expected} {fps:.1f} FPS',flush=True)
                        last_print=time.monotonic()
                view = dashboard(frame,dbg,stats,expected,fps,streak,paused,
                                 f'{args.shape_preprocess}/{args.photometric_mode}')
                if args.headless:
                    key=-1
                else:
                    cv2.imshow(WINDOW,view)
                    key=cv2.waitKey(30 if paused else 1)&255
                    if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                        break
                if key in (27,ord('q')):
                    break
                if key==ord('s'):
                    save_snapshot(session,total,frame,view,dbg,row)
                elif key==ord('p'):
                    paused=not paused
                elif key==ord('r'):
                    stats=Statistics()
                    log.write(json.dumps(dict(event='reset_statistics',after_frame=total))+'\n')
                elif ord('0')<=key<=ord('6') or key==ord('u'):
                    expected = 'unlabelled' if key==ord('u') else NAMES[key-ord('0')]
                    print(f'[label] Next processed frames: {expected}',flush=True)
                if args.image or (args.max_frames and total>=args.max_frames):
                    save_snapshot(session,total,frame,view,dbg,row)
                    if args.image and not args.headless:
                        cv2.waitKey(0)
                    break
    finally:
        if latest:
            latest.close()
        elif capture is not None:
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
        (session/'summary.json').write_text(json.dumps(stats.report(),ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[summary] {session}/summary.json',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError,ValueError,RuntimeError,cv2.error) as exc:
        print(f'[card-debug] {exc}',file=sys.stderr)
        raise SystemExit(1)
