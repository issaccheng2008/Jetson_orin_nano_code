"""距离标定拟合 — 用地面尺子实测把相机模型的距离读数校准回来。

地面沿前方摆一把尺，0cm 对准相机正下方；红棍下沿依次对准 --start 起、每 --step 一档。
本脚本检红棍下沿 → 用相机模型反算距离 → 最小二乘出线性映射
    Z_true = a · Z_est + b
a/b 写进 config/cameras.json 的 distance_calib 后，生产代码按此校正距离读数。

用法:
    python scripts/fit_camera_distance.py
    python scripts/fit_camera_distance.py --dir 线性拟合 --start 20 --step 5
    python scripts/fit_camera_distance.py --write      # 把结果写回 cameras.json
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "jetson"))
from camera_config import load as load_camera  # noqa: E402

_CAMERAS_JSON = os.path.join(_ROOT, "config", "cameras.json")


def imread_u(path):
    """cv2.imread 在 Windows 上读不了中文路径。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_red_bottom(img):
    """红棍下沿行号（画面中列附近取中位，抗棍倾斜）。返回 None 表示没检出。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (((h < 12) | (h > 168)) & (s > 90) & (v > 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    if stats[i, 4] < 2000:
        return None
    comp = (labels == i)
    xs = np.nonzero(comp.any(axis=0))[0]
    x0, x1 = int(xs.min()), int(xs.max())
    bottoms = []
    for frac in (0.15, 0.5, 0.85):
        col = int(x0 + frac * (x1 - x0))
        rows = np.nonzero(comp[:, col])[0]
        if rows.size:
            bottoms.append(int(rows.max()))
    if not bottoms:
        return None
    return float(np.median(bottoms))


def z_from_row(v, img_h, vfov_deg, h_cm, pitch_deg):
    """相机模型：行号 → 地面距离 cm。

    与 line_detector_v1_warp.py / shape_detector.py 同一套合成针孔模型
    （cy = 图高/2，fy 由 vfov 反推）。
    """
    fy = img_h / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    cy = img_h / 2.0
    r = (v - cy) / fy
    th = math.radians(pitch_deg)
    return h_cm * (math.cos(th) - r * math.sin(th)) / (r * math.cos(th) + math.sin(th))


def fit_linear(z_est, z_true):
    a, b = np.linalg.lstsq(
        np.vstack([z_est, np.ones_like(z_est)]).T, z_true, rcond=None)[0]
    resid = a * z_est + b - z_true
    return float(a), float(b), float(np.sqrt(np.mean(resid ** 2))), float(np.max(np.abs(resid)))


def fit_physical(v, z_true, img_h, vfov_deg):
    """重拟 (h, θ)，仅供查根因参考 —— 生产不用（会动 IPM 鸟瞰几何）。"""
    best = None
    for h in np.arange(20.0, 45.0, 0.05):
        for t in np.arange(30.0, 60.0, 0.05):
            e = z_from_row(v, img_h, vfov_deg, h, t) - z_true
            e = float(np.sqrt(np.mean(e ** 2)))
            if best is None or e < best[0]:
                best = (e, h, t)
    return best[1], best[2], best[0]


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="线性拟合", help="尺子照片目录")
    ap.add_argument("--start", type=float, default=20.0, help="最近一档的真值 cm")
    ap.add_argument("--step", type=float, default=5.0, help="每档间隔 cm（照片按文件名时间序）")
    ap.add_argument("--write", action="store_true", help="把拟合结果写回 cameras.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(_ROOT, args.dir, "*.jpg")))
    files += sorted(glob.glob(os.path.join(_ROOT, args.dir, "*.png")))
    if not files:
        print(f"[ERROR] {args.dir} 里没有图片")
        sys.exit(1)

    cam = load_camera()
    h_cm, pitch, vfov = cam["mount_height_cm"], cam["pitch_deg"], cam["vfov_deg"]
    print(f"相机 {cam['profile']}: h={h_cm}cm 俯角={pitch}° vfov={vfov}°")
    print(f"照片 {len(files)} 张，真值 {args.start:.0f} ~ "
          f"{args.start + args.step * (len(files) - 1):.0f}cm 每 {args.step:.0f}cm")
    print()

    rows, z_est, z_true = [], [], []
    for k, path in enumerate(files):
        img = imread_u(path)
        if img is None:
            print(f"  [跳过] 读不了 {os.path.basename(path)}")
            continue
        v = find_red_bottom(img)
        if v is None:
            print(f"  [跳过] 没检出红棍 {os.path.basename(path)}")
            continue
        rows.append(v)
        z_est.append(z_from_row(v, img.shape[0], vfov, h_cm, pitch))
        z_true.append(args.start + args.step * k)

    if len(rows) < 3:
        print("[ERROR] 有效样本不足，无法拟合")
        sys.exit(1)

    rows = np.array(rows)
    z_est = np.array(z_est)
    z_true = np.array(z_true)

    print("   真值cm   下沿行    模型cm    误差cm   误差%")
    for zt, v, ze in zip(z_true, rows, z_est):
        print("   %6.1f   %6.0f   %7.2f   %+6.2f   %+5.1f%%"
              % (zt, v, ze, ze - zt, 100.0 * (ze - zt) / zt))
    raw_rms = float(np.sqrt(np.mean((z_est - z_true) ** 2)))
    print()
    print(f"   校正前 RMS = {raw_rms:.2f} cm  最大 {np.max(np.abs(z_est - z_true)):.2f} cm")

    a, b, rms, mx = fit_linear(z_est, z_true)
    print(f"   线性映射 Z_true = {a:.4f}·Z_est {b:+.3f}   "
          f"RMS = {rms:.3f} cm  最大残差 {mx:.2f} cm")

    if len(rows) >= 6:
        fh, ft, fe = fit_physical(rows, z_true, imread_u(files[0]).shape[0], vfov)
        print(f"   [参考] 物理重拟 h={fh:.2f}cm θ={ft:.2f}°  RMS = {fe:.3f} cm"
              f"  —— 会动 IPM 几何，不用于生产")

    if args.write:
        with open(_CAMERAS_JSON, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["cameras"][cam["profile"]]["distance_calib"] = {
            "_note": f"{args.dir}/ 共 {len(rows)} 张，真值 {z_true[0]:.1f}~{z_true[-1]:.1f}cm",
            "a": round(a, 5),
            "b": round(b, 4),
            "rms_cm": round(rms, 3),
            "raw_rms_cm": round(raw_rms, 2),
        }
        with open(_CAMERAS_JSON, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print()
        print(f"   已写入 {os.path.relpath(_CAMERAS_JSON, _ROOT)}")


if __name__ == "__main__":
    main()
