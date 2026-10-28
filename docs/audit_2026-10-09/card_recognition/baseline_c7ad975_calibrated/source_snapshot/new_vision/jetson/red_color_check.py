"""红条颜色诊断 — 实时显示画面最红像素与当前判红阈值的对比。

用法:
    python jetson/red_color_check.py --cam 1

对着红条看输出：
  - "最红 BGR" = 画面里最红的像素值
  - "红度" = R − max(G,B)
  - "过阈值像素" = 满足判红条件的像素数（需 ≥ red_min_pixels 才检出）

若过阈值像素远小于下限 → 红条颜色达不到阈值（需降阈值）；
若过阈值像素大但画面里没有红条 → 误检（需升阈值）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from line_detector_v1_warp import LineDetector

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1, help="摄像头索引")
args = ap.parse_args()

ld = LineDetector()
print(f"判红阈值(HSV): H≤{ld.red_h_max} 或 ≥{ld.red_h_min}, "
      f"S≥{ld.red_s_min}, V≥{ld.red_v_min}，红像素需 ≥{ld.red_min_pixels}")
print("Q/ESC 退出\n")

cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    print(f"打不开摄像头 {args.cam}")
    sys.exit(1)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    b = frame[:, :, 0].astype(np.int32)
    g = frame[:, :, 1].astype(np.int32)
    r = frame[:, :, 2].astype(np.int32)
    score = r - np.maximum(g, b)
    i = np.unravel_index(int(np.argmax(score)), score.shape)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0].astype(np.int32)
    ss = hsv[:, :, 1].astype(np.int32)
    vv = hsv[:, :, 2].astype(np.int32)
    mask = (((hh <= ld.red_h_max) | (hh >= ld.red_h_min))
            & (ss >= ld.red_s_min) & (vv >= ld.red_v_min))
    n = int(mask.sum())
    res = ld._detect_red_bar(frame)
    z = f"{res[2]:.0f}cm" if res else "未检出"
    print(f"\r最红 BGR=({b[i]},{g[i]},{r[i]}) 红度={score[i]:>3} | "
          f"过阈值像素={n:>6} (需≥{ld.red_min_pixels}) | 检测={z}      ",
          end="")
    vis = frame.copy()
    if res:
        cv2.line(vis, (0, int(res[1])), (1279, int(res[1])), (0, 0, 255), 2)
    cv2.imshow("red check", cv2.resize(vis, (640, 360)))
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
print()
