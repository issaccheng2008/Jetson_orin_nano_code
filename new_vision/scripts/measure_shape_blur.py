"""图卡识别的模糊耐受度 — 在真实图卡照片上施加渐变模糊，量检出率怎么掉。

基准数据是实物拍摄的图卡照片（同一张卡、30~80cm、清晰静止），只做模糊，
不做几何变换/缩放 —— 保持原始像素，避免引入合成渲染的锯齿伪影。

用法:
    python scripts/measure_shape_blur.py
    python scripts/measure_shape_blur.py --kind motion --sigmas 0,3,5,7,9,13 --angle 90
    python scripts/measure_shape_blur.py --ref "WIN_20260919_*_flip.mp4"
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "jetson"))
from shape_detector import ShapeDetector  # noqa: E402
from camera_config import to_true_z  # noqa: E402

LV_W = 960          # 清晰度指标统一在 960 宽上算（与曝光档位分析同一口径）


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def sharpness(bgr):
    g = cv2.cvtColor(cv2.resize(bgr, (LV_W, int(bgr.shape[0] * LV_W / bgr.shape[1]))),
                     cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def blur(bgr, amount, kind="gauss", angle_deg=90.0):
    """amount：gauss 是 σ，motion 是运动核长（px）。angle_deg：核方向，90=竖直。"""
    if amount <= 0:
        return bgr
    if kind == "gauss":
        k = int(amount * 6) | 1
        return cv2.GaussianBlur(bgr, (k, k), amount)
    n = max(3, int(amount) | 1)
    kern = np.zeros((n, n), np.float32)
    kern[n // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((n / 2.0 - 0.5, n / 2.0 - 0.5), angle_deg, 1.0)
    kern = cv2.warpAffine(kern, m, (n, n))
    total = float(kern.sum())
    if total < 1e-6:
        return bgr
    return cv2.filter2D(bgr, -1, kern / total)


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="WIN_20260922_*.jpg", help="图卡照片（glob）")
    ap.add_argument("--width", type=int, default=1280, help="处理分辨率（生产档位）")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--sigmas", default="0,0.5,1,1.5,2,3,4,6")
    ap.add_argument("--kind", default="gauss", choices=["gauss", "motion"],
                    help="gauss=各向同性，motion=方向性运动模糊（更接近真实抖动）")
    ap.add_argument("--angle", type=float, default=90.0,
                    help="motion 的核方向，90=竖直（走路俯仰摆动在画面里是竖直的）")
    ap.add_argument("--ref", default=None, help="参考视频 glob，打印它的清晰度分布")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(_ROOT, args.dir)))
    if not files:
        print(f"[ERROR] {args.dir} 没有图片")
        sys.exit(1)
    imgs = [cv2.resize(imread_u(f), (args.width, args.height), interpolation=cv2.INTER_AREA)
            for f in files]
    kind_cn = "各向同性高斯" if args.kind == "gauss" else f"方向性运动（{args.angle:.0f}°）"
    print(f"真实图卡照片 {len(imgs)} 张（{args.width}×{args.height}），"
          f"只加{kind_cn}模糊，不做几何变换")
    print()

    sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False)
    sigmas = [float(x) for x in args.sigmas.split(",")]

    # 每张照片对应的真实距离：用 σ=0 帧检出的框反投影到地面（模型值过距离标定）
    dists = []
    for im in imgs:
        a, dbg = sd.update(im)
        z = None
        qw = dbg.get("quad_work")
        if qw is not None:
            g = sd._quad_to_ground(np.asarray(qw, np.float32))
            if g:
                z = to_true_z(float(np.mean([p[1] for p in g])))
        dists.append(z)
    order = sorted(range(len(imgs)), key=lambda i: (dists[i] is None, dists[i] or 0))

    print("  模糊%s   " % ("σ" if args.kind == "gauss" else "核长px")
          + "".join("%6.1f" % s for s in sigmas) + "     清晰度lv")
    for i in order:
        cells = []
        for s in sigmas:
            b = blur(imgs[i], s, args.kind, args.angle)
            a, dbg = sd.update(b)
            cells.append("    ✓" if dbg.get("card_found") else "    ×")
        lv = sharpness(blur(imgs[i], sigmas[0], args.kind, args.angle))
        print("  %5s cm " % ("%.0f" % dists[i] if dists[i] else "?")
              + "".join(cells) + "     %6.1f" % lv)

    print()
    unit = "σ" if args.kind == "gauss" else "核长px"
    print("  模糊%s   清晰度lv   找框成功   判对形状   说明" % unit)
    for s in sigmas:
        lvs, found, correct = [], 0, 0
        for im in imgs:
            b = blur(im, s, args.kind, args.angle)
            lvs.append(sharpness(b))
            a, dbg = sd.update(b)
            if dbg.get("card_found"):
                found += 1
                # 基准照片是同一张卡，照片间相对关系不作为判据；只看"有没有形状"
                if dbg.get("shape"):
                    correct += 1
        n = len(imgs)
        print("   %4.1f    %8.1f     %2d/%-2d      %2d/%-2d     %s" % (
            s, float(np.median(lvs)), found, n, correct, n,
            "清晰" if s == 0 else ("还能用" if found >= n * 0.7 else
                                   "明显退化" if found > 0 else "全瞎")))

    if args.ref:
        vids = sorted(glob.glob(os.path.join(_ROOT, args.ref)))
        if vids:
            print()
            ref = []
            for v in vids:
                cap = cv2.VideoCapture(v)
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    ref.append(sharpness(f))
                cap.release()
            if ref:
                a = np.array(ref)
                print("  参考：机器人机载视频 %d 帧的清晰度分位" % len(a))
                print("    p10=%.0f  p25=%.0f  p50=%.0f  p75=%.0f  p90=%.0f   ≥150 占 %.0f%%" % (
                    np.percentile(a, 10), np.percentile(a, 25), np.median(a),
                    np.percentile(a, 75), np.percentile(a, 90),
                    100.0 * np.sum(a >= 150) / len(a)))


if __name__ == "__main__":
    main()
