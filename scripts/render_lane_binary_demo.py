#!/usr/bin/env python3
"""Render recorded-camera footage and the exact final legacy/contrast binary masks.

Offline dependencies: numpy, opencv-python, pillow, imageio-ffmpeg.
No camera/serial/motor access; original video is read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from line_preprocess import extract_lane_candidates


def writer(path, size, fps, crf=18):
    stream = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec='libx264',
                                         pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                                         macro_block_size=1, ffmpeg_log_level='error',
                                         output_params=['-crf', str(crf), '-movflags', '+faststart'])
    stream.send(None)
    return stream


def panel(frame, bird, old, new, font, timestamp, duration, cut):
    image = Image.new('RGB', (1600, 600), (20, 24, 31))
    draw = ImageDraw.Draw(image)
    draw.text((20, 8), '实车录像：最终二值化对照（离线回放）', font=font, fill='white')
    labels = [(0, '原始画面'), (640, '原始鸟瞰'), (960, '旧版 legacy'), (1280, '新版 contrast')]
    for x, label in labels:
        draw.text((x+10, 55), label, font=font, fill=(120, 230, 190) if x==1280 else 'white')
    image.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)), (0, 105))
    image.paste(Image.fromarray(cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)), (640, 95))
    image.paste(Image.fromarray(old[1]).convert('RGB'), (960, 95))
    image.paste(Image.fromarray(new[1]).convert('RGB'), (1280, 95))
    draw.text((20, 480), f'录像时间 {timestamp:05.1f}s / {duration:.1f}s'+ ('  |  片段剪辑' if cut else ''), font=font, fill='white')
    draw.text((970, 505), f'阈值 {old[2]:.0f}  像素 {np.count_nonzero(old[1])}', font=font, fill='white')
    draw.text((1290, 505), f'阈值 {new[2]:.0f}  像素 {np.count_nonzero(new[1])}', font=font, fill=(120, 230, 190))
    draw.text((20, 550), '黑底白线：白色为最终候选；固定俯角45°，仍需几何配对判断赛道身份。', font=font, fill=(200, 205, 210))
    return np.asarray(image)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True, help='Chinese TTF/TTC font')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 24)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {args.video}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    calibration = json.loads((ROOT/'docs/audit_2026-10-09/line_extraction/annotations.json').read_text())['meta']['calibration']
    d = LineDetector(width, height, cam_height_cm=calibration['height_cm'],
                     cam_pitch_deg=calibration['pitch_deg'], cam_vfov_deg=calibration['vfov_deg'],
                     z_calib=calibration['z_calib'], lane_width_cm=calibration['lane_width_cm'])
    full = writer(args.output/'comparison_full.mp4', (1600, 600), fps)
    binary = writer(args.output/'binary_contrast_full.mp4', (320, 400), fps, crf=0)
    highlights = writer(args.output/'comparison_24s.mp4', (1600, 600), fps)
    cuts, gif, selected = [(10, 18), (38, 46), (77, 85)], [], {134, 402, 805}
    index, highlight_count = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bird = cv2.warpPerspective(frame, d.M, (320, 400))
            gray = np.max(bird, axis=2)
            old = extract_lane_candidates(gray, 'legacy')
            new = extract_lane_candidates(gray, 'contrast')
            timestamp = index / fps
            full.send(panel(frame, bird, old, new, font, timestamp, count/fps, False))
            binary.send(cv2.cvtColor(new[1], cv2.COLOR_GRAY2RGB))
            if any(lo <= timestamp < hi for lo, hi in cuts):
                highlights.send(panel(frame, bird, old, new, font, timestamp, count/fps, True))
                highlight_count += 1
            if 78.5 <= timestamp < 82.5 and index % 2 == 0:
                shown = Image.fromarray(panel(frame, bird, old, new, font, timestamp, count/fps, True))
                gif.append(shown.resize((1000, 375), Image.Resampling.LANCZOS))
            if index in selected:
                cv2.imwrite(str(args.output/f'binary_frame_{index}.png'), new[1])
                Image.fromarray(panel(frame, bird, old, new, font, timestamp, count/fps, False)).save(args.output/f'comparison_frame_{index}.png')
            index += 1
            if index % 200 == 0:
                print(f'Rendered {index}/{count} frames', flush=True)
    finally:
        cap.release()
        full.close()
        binary.close()
        highlights.close()
    if index != count:
        raise RuntimeError(f'Incomplete video: decoded {index}, expected {count}')
    if gif:
        gif[0].save(args.output/'preview.gif', save_all=True, append_images=gif[1:],
                    duration=round(2000/fps), loop=0)
    digest = hashlib.sha256()
    with args.video.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    metadata = dict(video=str(args.video), video_sha256=digest.hexdigest(), frames=index,
                    source_size=[width, height], fps=fps, calibration=calibration,
                    comparison_size=[1600, 600], binary_size=[320, 400],
                    highlights_intervals_s=cuts, highlight_frames=highlight_count,
                    source_sha256={name:hashlib.sha256((ROOT/'new_vision/jetson'/name).read_bytes()).hexdigest()
                                   for name in ['line_preprocess.py', 'line_detector_v1_warp.py']},
                    note='Final masks directly from production extraction; MP4/GIF are display formats, PNGs preserve exact binary pixels. No IMU pose reconstruction.')
    (args.output/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed {index} frames; output {args.output}', flush=True)


if __name__ == '__main__':
    main()
