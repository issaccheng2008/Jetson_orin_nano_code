#!/usr/bin/env python3
"""Render original/old inset/local-gray card extraction from recorded photos."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from shape_detector import ShapeDetector
from card_appearance import card_appearance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 19)
    # Explicit representative evidence; filenames never reach the detector.
    names = [args.data_dir / '6card/正2.png',
             args.data_dir / 'WIN_20260922_20_32_48_Pro.jpg',
             args.data_dir / 'WIN_20260922_20_33_43_Pro.jpg']
    for path in names:
        if not path.exists():
            matches = list(path.parent.glob(path.stem + '.*'))
            if not matches:
                raise FileNotFoundError(path)
            path = matches[0]
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        d = ShapeDetector()
        _, debug = d.update(frame)
        if debug.get('quad_work') is None:
            raise RuntimeError(f'No verified quad in representative frame {path.name}')
        appearance = card_appearance(debug['gray'], debug['quad_work'])
        old = d._classify(debug['warp'], {})
        local = d._classify_contour(appearance.symbol) if appearance.symbol is not None else None
        shown = cv2.cvtColor(appearance.binary, cv2.COLOR_GRAY2BGR)
        if appearance.symbol is not None:
            cv2.drawContours(shown, [appearance.symbol], -1, (0, 255, 0), 2)
        canvas = Image.new('RGB', (1200, 335), '#14181f')
        draw = ImageDraw.Draw(canvas)
        columns = [(0, 480, '原始实拍', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                   (480, 240, f'旧裁边二值：{old}', debug['warp']),
                   (720, 240, '整卡原灰度矫正', appearance.gray),
                   (960, 240, f'局部分割：{local}', cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))]
        for x, width, label, pixels in columns:
            draw.text((x+6, 8), label, font=font, fill='white')
            canvas.paste(Image.fromarray(pixels).convert('RGB').resize((width, 250)), (x, 38))
        draw.text((8, 303), f'{path.name} | {appearance.reason} | 绿色为实际送入分类器的轮廓',
                  font=font, fill='#9be7c5')
        canvas.save(args.output / (path.stem + '.png'))


if __name__ == '__main__':
    main()
