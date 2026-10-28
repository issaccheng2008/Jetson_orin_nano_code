"""向下摄像头 — 虚线计数（视觉里程）。

摄像头垂直向下看赛道内侧的刻度虚线，相邻虚线间距 period_cm（默认 10cm）。
每条虚线跨过画面中线一次 → 计数 +1 → 里程 = 计数 × 间距。

每条虚线必须是一个**细长连通域**：方向接近竖直（转弯时允许偏 max_angle_deg），
长度 ≥ min_len_px，长宽比 ≥ aspect_min。这样一条歪掉的线仍是一个整体，
不会裂成两条；横向边缘线、数字等块状物被角度和长宽比挡掉。

用法:
    python jetson/dash_counter.py --cam 0              # 实时预览
    python jetson/dash_counter.py --image frame.png    # 单图调试
    python jetson/dash_counter.py --cam 0 --min-len 25 --max-angle 25

按键: S 保存当前帧 + 检测图, Q/ESC 退出
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

from utils import open_camera, imread_unicode


class DashCounter:
    """二值化 → 连通域 → 角度/长度/长宽比筛选 → 跨中线计数。"""

    def __init__(self, period_cm=10.0, orient="v"):
        self.period_cm = float(period_cm)
        self.orient = orient

        # 二值化（白底黑线 → 线=255）
        # T = 邻域均值 − C；C 为正 → 只有明显比背景暗的像素算线
        self.block = 31
        self.C = 12

        # 虚线筛选
        self.min_len_px = 15       # 长度下限（同一条虚线碎片合并后的总长，px）
        self.max_angle_deg = 30.0  # 与竖直方向的夹角上限（转弯时线会歪）
        self.aspect_min = 2.5      # 长宽比下限（线段 vs 数字/噪点块）

        # 同一条虚线的碎片聚类容差（px）：相邻碎片中心距在此以内算一条线
        self.cluster_tol = 25.0

        # 帧间匹配最大位移（单条虚线时占画面尺寸比例）
        self.match_max_frac = 0.5

        self.count = 0
        self._prev_pos = []
        self._orient_locked = None

    def _binarize(self, gray):
        # 3x3 模糊即可：虚线只有几像素宽，大核会把对比度抹平
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # 不做开运算：3x3 腐蚀会把细虚线整条吃掉
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, self.block, self.C)

    def _guess_orient(self, binary):
        """连通域长宽比投票；贯穿画面的大块（场地边缘线）不参与。"""
        h, w = binary.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        n_h = n_v = 0
        for i in range(1, n):
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw > 0.9 * w or bh > 0.9 * h:
                continue
            if bw >= 1.5 * bh:
                n_h += 1
            elif bh >= 1.5 * bw:
                n_v += 1
        if n_h == 0 and n_v == 0:
            return self._orient_locked or "v"
        return "h" if n_h >= n_v else "v"

    def _find_dash_fragments(self, binary, orient):
        """返回所有形状/方向合格的碎片（长度不在这里判，留给聚类后判）。"""
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        frags = []
        for c in cnts:
            if len(c) < 3:
                continue
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
            length = max(rw, rh)
            width = min(rw, rh)
            if length / max(width, 1e-6) < self.aspect_min:
                continue
            fl = np.ravel(cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01))
            vx, vy = float(fl[0]), float(fl[1])
            if orient == "v":
                dev = abs(math.degrees(math.atan2(vx, vy)))
            else:
                dev = abs(math.degrees(math.atan2(vy, vx)))
            if dev > self.max_angle_deg:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            frags.append({"cx": float(cx), "cy": float(cy),
                          "length": float(length),
                          "x0": x, "y0": y, "x1": x + bw, "y1": y + bh})
        return frags

    def _cluster_fragments(self, frags, orient):
        """同一条虚线的碎片聚成一条线（按垂直于虚线的方向排序后链式聚类）。"""
        if not frags:
            return []
        key = (lambda L: L["cx"]) if orient == "v" else (lambda L: L["cy"])
        frags = sorted(frags, key=key)

        groups, cur = [], [frags[0]]
        for L in frags[1:]:
            if key(L) - key(cur[-1]) <= self.cluster_tol:
                cur.append(L)
            else:
                groups.append(cur)
                cur = [L]
        groups.append(cur)

        out = []
        for g in groups:
            wsum = sum(x["length"] for x in g)
            cx = sum(x["cx"] * x["length"] for x in g) / wsum
            cy = sum(x["cy"] * x["length"] for x in g) / wsum
            total_len = sum(x["length"] for x in g)
            if total_len < self.min_len_px:
                continue
            out.append({"center": float(cx if orient == "v" else cy),
                        "cx": float(cx), "cy": float(cy),
                        "length": float(total_len),
                        "n_frag": len(g),
                        "x0": min(x["x0"] for x in g),
                        "y0": min(x["y0"] for x in g),
                        "x1": max(x["x1"] for x in g),
                        "y1": max(x["y1"] for x in g)})
        return out

    def _update_count(self, cur, span, center):
        """跟踪每条虚线的位置；跨过中线（任一方向）计一次。"""
        if len(cur) >= 2:
            gaps = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
            match_max = max(2.0, 0.5 * float(np.median(gaps)))
        else:
            match_max = self.match_max_frac * span

        prev = self._prev_pos
        if prev:
            used = set()
            for p in cur:
                best_j, best_d = -1, None
                for j, q in enumerate(prev):
                    if j in used:
                        continue
                    dd = abs(q - p)
                    if best_d is None or dd < best_d:
                        best_d, best_j = dd, j
                if best_j < 0 or best_d is None or best_d > match_max:
                    continue
                used.add(best_j)
                q = prev[best_j]
                if (q < center <= p) or (q >= center > p):
                    self.count += 1
        self._prev_pos = cur

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = self._binarize(gray)
        h, w = binary.shape

        orient = self.orient
        if orient == "auto":
            orient = self._guess_orient(binary)
        if orient != self._orient_locked:
            self._prev_pos = []
            self._orient_locked = orient

        frags = self._find_dash_fragments(binary, orient)
        lines = self._cluster_fragments(frags, orient)
        cur = sorted(L["center"] for L in lines)
        span = h if orient == "h" else w
        self._update_count(cur, span, span / 2.0)

        return {"count": self.count,
                "dist_cm": self.count * self.period_cm,
                "n_lines": len(lines),
                "orient": orient,
                "lines": lines,
                "binary": binary}


def draw(binary, res):
    """在送进检测的二值化图上叠加检测结果（不是原图）。"""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    if res["orient"] == "h":
        cv2.line(vis, (0, h // 2), (w - 1, h // 2), (0, 255, 255), 1)
    else:
        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 255, 255), 1)
    for L in res["lines"]:
        cv2.rectangle(vis, (L["x0"], L["y0"]), (L["x1"], L["y1"]),
                      (0, 255, 0), 1)
        cv2.putText(vis, f"{L['n_frag']}", (L["x1"] + 2, L["y0"] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
    txt = (f"count={res['count']}  dist={res['dist_cm']:.0f}cm  "
           f"n={res['n_lines']}  orient={res['orient']}")
    cv2.rectangle(vis, (0, 0), (w, 24), (30, 30, 30), -1)
    cv2.putText(vis, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return vis


def main():
    ap = argparse.ArgumentParser(description="向下摄像头虚线计数")
    ap.add_argument("--cam", type=int, default=0, help="摄像头索引")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--image", type=str, default=None, help="单图调试")
    ap.add_argument("--period", type=float, default=10.0, help="虚线间距 cm")
    ap.add_argument("--orient", choices=["auto", "h", "v"], default="v",
                    help="虚线方向 h=横 v=纵（默认 v）")
    ap.add_argument("--min-len", type=int, default=15,
                    help="虚线最短长度 px")
    ap.add_argument("--max-angle", type=float, default=30.0,
                    help="与竖直方向的夹角上限（度）")
    ap.add_argument("--cluster-tol", type=float, default=25.0,
                    help="同一条虚线的碎片聚类容差 px")
    ap.add_argument("--outdir", type=str, default=None,
                    help="保存帧目录（默认脚本同目录）")
    args = ap.parse_args()

    dc = DashCounter(period_cm=args.period, orient=args.orient)
    dc.min_len_px = args.min_len
    dc.max_angle_deg = args.max_angle
    dc.cluster_tol = args.cluster_tol

    if args.image:
        frame = imread_unicode(args.image)
        if frame is None:
            print(f"读不到图片: {args.image}")
            return
        res = dc.update(frame)
        print(f"count={res['count']} dist={res['dist_cm']:.0f}cm "
              f"n_lines={res['n_lines']} orient={res['orient']}")
        for L in res["lines"]:
            print(f"  cx={L['cx']:.0f} cy={L['cy']:.0f} "
                  f"len={L['length']:.0f} frags={L['n_frag']}")
        cv2.imshow("dash detect (binary input)", draw(res["binary"], res))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = open_camera(args.cam, args.width, args.height)
    if not cap.isOpened():
        print(f"打不开摄像头 {args.cam}")
        return
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头 {args.cam}: {aw}x{ah}   间距 {args.period}cm   "
          f"min_len={args.min_len}px  max_angle={args.max_angle}deg")
    print("S 保存帧, Q/ESC 退出\n")

    outdir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    fps_t0, fps_n, fps_val = time.time(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        res = dc.update(frame)
        fps_n += 1
        if fps_n % 30 == 0:
            fps_val = 30.0 / max(time.time() - fps_t0, 1e-3)
            fps_t0 = time.time()
        vis = draw(res["binary"], res)
        if fps_val:
            cv2.putText(vis, f"FPS={fps_val:.0f}", (aw - 90, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("dash detect (binary input)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            stamp = time.strftime("%H%M%S")
            p1 = os.path.join(outdir, f"dash_frame_{stamp}.png")
            p2 = os.path.join(outdir, f"dash_detect_{stamp}.png")
            cv2.imwrite(p1, frame)
            cv2.imwrite(p2, vis)
            print(f"saved {p1} / {p2}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
