"""在 loss_dump 上离线评巡线：复现率 + 航向质量。

每个 dump 是一帧原图 + 真跑时落盘的标量。逐帧跑检测器，先看哪些量能复现
（复现不了的说明它依赖跨帧状态），再看 ang 的分布 —— 改融合前先量这个。

用法:
    python new_vision/scripts/evaluate_line_dumps.py --dir ~/loss_dump_2
    python new_vision/scripts/evaluate_line_dumps.py --dir DIR --baseline   # 只看复现率
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jetson"))

import cv2
import numpy as np

from line_detector_v1_warp import LineDetector

# 落盘字段 -> debug 键
EXACT = [("bottom_pair_ratio", "bottom_pair_ratio"),
         ("n_roi_results", "n_roi_results"),
         ("band_mask", "band_mask"),
         ("lost_frames", "lost_frames")]
NEAR = [("angle_err_deg", "angle_err_deg"),
        ("avg_conf", "avg_conf"),
        ("near_err_px", "near_err_px")]


def load(dirpath):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        fp = jp[:-5] + "_frame.jpg"
        if not os.path.exists(fp):
            continue
        out.append((os.path.basename(jp)[:-5], fp, json.load(open(jp, encoding="utf-8"))))
    return out


def run_one(fp, disagree_max=None):
    # startup 分支走的是另一条检测路径，离线复现要关掉
    det = LineDetector()
    det.startup_settle_frames = 0
    if disagree_max is not None:
        det.band_disagree_max_px = disagree_max
    img = cv2.imread(fp)
    if img is None:
        return None
    _, _, _, _, dbg = det.process(img)
    return dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("TEMP", "/tmp"),
                                                  "rb", "loss"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bad-deg", type=float, default=20.0,
                    help="|ang| 超过这个度数算坏帧")
    ap.add_argument("--disagree-max", type=float, default=None,
                    help="覆盖 band_disagree_max_px；给 1e9 等于关掉守卫")
    args = ap.parse_args()

    items = load(args.dir)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"没找到 dump: {args.dir}")
        return

    rec, got = [], []
    for name, fp, saved in items:
        dbg = run_one(fp, args.disagree_max)
        if dbg is None:
            continue
        rec.append(saved)
        got.append((name, dbg))

    print(f"帧数 {len(rec)}   dir={args.dir}")
    print()
    print("复现率（落盘值 vs 离线重跑）:")
    for key, dkey in EXACT:
        hit = sum(1 for s, (_, d) in zip(rec, got)
                  if abs(float(s.get(key, -99)) - float(d.get(dkey, -99))) < 1e-6)
        print(f"  {key:<20} {hit:>4}/{len(rec)}")
    for key, dkey in NEAR:
        diffs = [abs(float(s.get(key, 0.0)) - float(d.get(dkey, 0.0)))
                 for s, (_, d) in zip(rec, got)]
        print(f"  {key:<20} |Δ| 中位 {st.median(diffs):7.2f}   90分位 "
              f"{sorted(diffs)[int(0.9 * (len(diffs) - 1))]:7.2f}")

    print()
    for lbl, vals in (("落盘", [s["angle_err_deg"] for s in rec]),
                      ("重跑", [d["angle_err_deg"] for _, d in got])):
        bad = sum(1 for v in vals if abs(v) > args.bad_deg)
        print(f"  {lbl} |ang|: 中位 {st.median(abs(v) for v in vals):6.1f}°  "
              f"峰 {max(abs(v) for v in vals):6.1f}°  "
              f"坏帧 {bad}/{len(vals)}  ({100.0 * bad / len(vals):.0f}%)")

    print()
    conf_o = [s["avg_conf"] for s in rec]
    conf_n = [d["avg_conf"] for _, d in got]
    print(f"  conf: 落盘中位 {st.median(conf_o):.2f} -> 重跑 {st.median(conf_n):.2f}")
    lock_o = sum(1 for s in rec if s["bottom_lock_valid"])
    lock_n = sum(1 for _, d in got if d["bottom_lock_valid"])
    print(f"  lock 有效: 落盘 {lock_o}/{len(rec)} -> 重跑 {lock_n}/{len(rec)}")

    # ── 虚假指令：实测基本居中，融合却下达大修正 ──
    W = 160.0
    calm, loud = 0.08, 0.25   # ≈4.2cm 算居中；≈13.2cm 算大修正
    spur, amp = 0, []
    for _, d in got:
        n = abs(d.get("near_err_px", 0.0)) / W
        f = abs(d.get("fused_err_raw", 0.0))
        amp.append(f / max(n, 0.02))
        if n <= calm and f >= loud:
            spur += 1
    amp.sort()
    print()
    print(f"  虚假指令（实测<{calm * W * 0.332:.1f}cm 却下达>{loud * W * 0.332:.1f}cm）: "
          f"{spur}/{len(got)}  ({100.0 * spur / len(got):.0f}%)")
    print(f"  融合放大倍数 |fused|/|near|: 中位 {st.median(amp):.2f}  "
          f"90分位 {amp[int(0.9 * (len(amp) - 1))]:.2f}  峰 {amp[-1]:.1f}")


if __name__ == "__main__":
    main()
