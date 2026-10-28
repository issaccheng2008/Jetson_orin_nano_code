"""生成 Hu 矩模板签名 — 6 类标准卡的图形轮廓。

模板与运行时同分布：合成场景 → 找框 → 矫正 warp200 → 同款轮廓提取
（close5+dilate 桥接 → 最大轮廓）。Hu 矩本身对平移/旋转/缩放不变，
模板只需一张标准正面图。

输出: jetson/shape_hu_templates.npz（每类一条轮廓点集）

用法: python jetson/make_hu_templates.py
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "6_pictures"))

import cv2
import numpy as np

from generate_synthetic_cards import SHAPES, render_card, compose_scene
from shape_detector import ShapeDetector


def extract_main_contour(warp):
    """与 _classify_shape 同款：close5+dilate 桥接 → 最大轮廓。"""
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    w = cv2.morphologyEx(warp, cv2.MORPH_CLOSE, k5)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    w = cv2.dilate(w, k3, iterations=1)
    contours, _ = cv2.findContours(w, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def main():
    sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False)
    out = {}
    for idx, (name, _cls, _act) in enumerate(SHAPES):
        # 保守参数场景（贴近比赛摆放），取多张里面积最完整的一条
        best = None
        for trial in range(3):
            card, quad = render_card(idx, size=256,
                                     offset_frac=0.04 if trial else None)
            scene, _, _ = compose_scene(card, quad, 960, 540, "homography",
                                        conservative=True)
            _act, dbg = sd.update(scene)
            warp = dbg.get("warp")
            if warp is None:
                continue
            c = extract_main_contour(warp)
            if c is not None and (best is None
                                  or cv2.contourArea(c) > cv2.contourArea(best)):
                best = c
        if best is None:
            print(f"[FAIL] {name}: 未取到轮廓")
            return
        out[name] = best.astype(np.int32)
        print(f"[ok] {name}: {len(best)} 点, 面积 {cv2.contourArea(best):.0f}")

    path = os.path.join(_SCRIPT_DIR, "shape_hu_templates.npz")
    np.savez(path, **out)
    print(f"已保存: {path}")


if __name__ == "__main__":
    main()
