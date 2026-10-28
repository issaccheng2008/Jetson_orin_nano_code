"""几何图卡识别 Demo — 支持照片和视频。

用法:
    python jetson/shape_demo.py --image path/to/photo.jpg
    python jetson/shape_demo.py --video path/to/video.mp4
    python jetson/shape_demo.py --image 6_pictures/圆形.png

分类走纯 CV 规则法（多边形拟合+几何特征判定）。YOLO 方案用 --yolo 单独启用。
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shape_detector import ShapeDetector
from shape_preprocess import preprocess_for_yolo
from utils import open_camera, imread_unicode

# 动作映射（与规则一致）
ACTION_NAMES = {
    1: "举左手", 2: "举右手", 3: "抬左腿",
    4: "抬右腿", 5: "举双手", 6: "摇头",
}
SHAPE_NAMES = {
    "circle": "圆形", "pentagon": "五角星", "square": "正方形",
    "diamond": "菱形", "cross": "十字", "triangle": "三角形",
}
# 分类过了但没触发时，_confirm 的闸门名 → 人话
REJECT_ZH = {
    "trusted": "连续帧数不够",
    "disarmed": "已触发过·等卡离开",
    "no_box": "没有框",
    "too_far": "框还没压到下方 2/3",
    "cooldown": "冷却中",
    "fallback": "走的是存在信号兜底",
}
# YOLO类别 → 动作号
YOLO_CLS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
YOLO_NAMES = {0: "circle", 1: "pentagon", 2: "square",
              3: "diamond", 4: "cross", 5: "triangle"}

# ── 中文绘制（cv2.putText 不支持中文，用 PIL）──
_FONT_CACHE = {}


def _load_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        from PIL import ImageFont
    except Exception:
        _FONT_CACHE[size] = None
        return None
    for p in ("C:/Windows/Fonts/msyh.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                pass
    _FONT_CACHE[size] = None
    return None


def put_text(img, text, org, color_bgr=(0, 0, 255), size=26):
    """绘制文本（支持中文；无字体时回退英文近似）。"""
    font = _load_font(size)
    if font is None:
        cv2.putText(img, text.encode("ascii", "replace").decode(), org,
                    cv2.FONT_HERSHEY_SIMPLEX, size / 36.0, color_bgr, 2)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(org, text, font=font,
                             fill=tuple(reversed(color_bgr)))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def run_cv(img, detector):
    """找框+分类（纯 CV 规则法）。返回 (shape, action, conf, quad, dbg)。"""
    action, dbg = detector.update(img)
    shape = dbg.get("shape")
    conf = 1.0 if shape is not None else 0.0
    return shape, action, conf, dbg.get("quad"), dbg


def run_yolo(img, model):
    """YOLO方案。返回 (shape_name, action, conf, box)"""
    results = model.predict(img, conf=0.3, verbose=False)
    best = None
    for r in results:
        for box in r.boxes:
            if box.cls is None:
                continue
            conf = float(box.conf[0])
            if best is None or conf > best[1]:
                best = (int(box.cls[0]), conf, box.xyxy[0].cpu().numpy())
    if best is None:
        return None, None, 0.0, None
    cls_id, conf, xyxy = best
    name = YOLO_NAMES[cls_id]
    return name, YOLO_CLS[cls_id], conf, xyxy


def main():
    parser = argparse.ArgumentParser(description="几何图卡识别 Demo")
    parser.add_argument("--image", type=str, default=None, help="照片路径")
    parser.add_argument("--video", type=str, default=None, help="视频路径")
    parser.add_argument("--yolo", action="store_true",
                        help="启用YOLO方案（需ultralytics+权重）")
    parser.add_argument("--camera", action="store_true",
                        help="摄像头实时识别（默认纯CV）")
    parser.add_argument("--cam", type=int, default=0, help="摄像头索引")
    parser.add_argument("--save-mid", type=str, default=None,
                        help="保存找框输入图/warp 的目录"
                             "（摄像头模式按 S 键保存当前帧）")
    parser.add_argument("--preprocess", action="store_true",
                        help="YOLO前先做传统CV预处理（灰度/CLAHE/Otsu/形态学）")
    parser.add_argument("--save-pre", type=str, default=None,
                        help="保存预处理中间图到指定路径（调试用）")
    args = parser.parse_args()

    detector = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False,
                             roi_ratio=1.0)

    # YOLO模型（可选）
    model = None
    if args.yolo:
        try:
            from ultralytics import YOLO
            weights = os.path.join(_SCRIPT_DIR, "..", "6_pictures",
                                   "shape_yolo_best.pt")
            model = YOLO(os.path.abspath(weights))
            print(f"[yolo] 模型加载: {os.path.abspath(weights)}")
        except Exception as e:
            print(f"[yolo] 加载失败（跳过YOLO）: {e}")

    _mid_dir = args.save_mid or os.path.join(
        _SCRIPT_DIR, "..", "generated", "shape_debug")

    def save_debug(dbg, idx):
        """保存找框输入图（二值化）+ 矫正后的 warp。"""
        os.makedirs(_mid_dir, exist_ok=True)
        ts = time.strftime("%H%M%S")
        n = 0
        if dbg.get("binary") is not None:
            p = os.path.join(_mid_dir, f"findbox_input_{ts}_{idx}.png")
            cv2.imencode(".png", dbg["binary"])[1].tofile(p)
            print(f"  找框输入图 → {p}")
            n += 1
        if dbg.get("warp") is not None:
            p = os.path.join(_mid_dir, f"warp_{ts}_{idx}.png")
            cv2.imencode(".png", dbg["warp"])[1].tofile(p)
            print(f"  矫正warp    → {p}")
            n += 1
        if n == 0:
            print("  未检测到图卡，无中间图可存")

    def process_frame(frame, frame_idx=0):
        # 找框 + 分类
        cv_shape, cv_action, cv_conf, cv_quad, cv_dbg = run_cv(frame, detector)
        rules_s = cv_dbg.get("shape_rules")
        # 方案B: YOLO（可选预处理：真实帧→白底黑线→喂模型）
        yo_shape, yo_action, yo_conf, yo_box = (None, None, 0.0, None)
        if model is not None:
            yolo_input = frame
            if args.preprocess:
                save_pre = None
                if args.save_pre and frame_idx == 0:
                    save_pre = args.save_pre
                yolo_input = preprocess_for_yolo(frame, save_debug=save_pre)
            yo_shape, yo_action, yo_conf, yo_box = run_yolo(yolo_input, model)

        # 混合决策：取置信度高者
        if cv_action is not None and yo_action is not None:
            if yo_conf > 0.6:
                final = ("YOLO", yo_shape, yo_action, yo_conf)
            else:
                final = ("CV", cv_shape, cv_action, cv_conf)
        elif cv_action is not None:
            final = ("CV", cv_shape, cv_action, cv_conf)
        elif yo_action is not None:
            final = ("YOLO", yo_shape, yo_action, yo_conf)
        else:
            final = None

        # 可视化底图：两张二值图并排，框画在两张上。
        # 左边 _binary_selective（压巡线那条，Hough/LSD/corner/CC 用），
        # 右边 _card_ink（原始灰度 Otsu，ring 通道用）。卡框在右边那张是完整的，
        # 在左边那张是碎的 —— 并排看才明白为什么要有第二条。
        b = cv_dbg.get("binary")
        ink = cv_dbg.get("ink")
        qw = cv_dbg.get("quad_work")

        def panel(img, title):
            v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if qw is not None:
                cv2.polylines(v, [qw.astype(np.int32).reshape(-1, 1, 2)],
                              True, (0, 255, 0), 2)
            cv2.putText(v, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            cv2.putText(v, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            return v

        if b is not None or ink is not None:
            left = panel(b, "_binary_selective (只喂 Hough/LSD/corner/CC)")
            if ink is None:
                disp = left
            else:
                disp = np.hstack([left, panel(ink, "_card_ink (只喂 ring)")])
        else:
            disp = frame.copy()
        if yo_box is not None:
            x1, y1, x2, y2 = [int(v) for v in yo_box]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 3)

        hu_d = cv_dbg.get("hu_dist")
        if final is not None:
            src, shape, action, conf = final
            if hu_d is not None:
                conf_s = f"Hu距离 {hu_d:.3f}"   # 纯CV无概率，用模板距离
            else:
                conf_s = f"置信度 {conf:.2f}"
            label = (f"{SHAPE_NAMES.get(shape, shape)} 动作{action} "
                     f"{ACTION_NAMES[action]} | {conf_s}")
        else:
            # 分出来了却没触发 = _confirm 的闸门没过，和分类器判不出是两回事。
            # 混成一句话看的人会去查分类器，而问题在触发闸门。
            if cv_shape is not None:
                why = REJECT_ZH.get(cv_dbg.get("confirm_reject"),
                                    str(cv_dbg.get("confirm_reject")))
                label = f"已分类 {SHAPE_NAMES.get(cv_shape, cv_shape)}，未触发（{why}）"
            elif cv_dbg.get("card_found"):
                label = "找到框但分类不出"
            else:
                label = "未找到框"
        # 一行流水：候选 -> 过几何闸 -> 进打分表，加上前两位拒绝理由。
        # g0 就是"一个候选都没过" —— 实车 2026-09-30 每帧都是这个。
        rej = sorted((cv_dbg.get("geom_rejects") or {}).items(),
                     key=lambda kv: -kv[1])[:2]
        label += (f"   [{cv_dbg.get('quad_total', '?')}"
                  f"g{cv_dbg.get('quad_geom', '?')}"
                  f"v{len(cv_dbg.get('scores') or [])}"
                  f" {' '.join(f'{k}:{n}' for k, n in rej)}]")
        put_text(disp, label, (10, 8), (0, 0, 255), 26)

        def _n(s):
            return SHAPE_NAMES.get(s, s) if s else "-"
        hu_b, hu_d = cv_dbg.get("hu_best"), cv_dbg.get("hu_dist")
        info = f"规则: {_n(rules_s)}"
        if hu_b:
            info += f" | Hu: {_n(hu_b)}({hu_d:.3f})"
        if model is not None:
            info += f" | YOLO: {_n(yo_shape)}"
        put_text(disp, info, (10, 46), (200, 200, 0), 22)
        return disp, final, (rules_s, hu_b, hu_d), cv_dbg

    # ── 摄像头实时模式 ──
    if args.camera:
        cap = open_camera(args.cam, 1280, 720)
        if not cap.isOpened():
            print(f"无法打开摄像头 {args.cam}")
            avail = []
            for i in range(4):
                c = cv2.VideoCapture(i)
                if c.isOpened():
                    ok0, _ = c.read()
                    if ok0:
                        avail.append(i)
                c.release()
            if avail:
                print(f"  当前可用索引: {avail}  → 用 --cam {avail[0]}")
            else:
                print("  未检测到可用摄像头（检查 USB 连接，"
                      "或是否有其他程序占用）")
            return
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"=== 摄像头 {args.cam} 实时识别（纯CV） {aw}x{ah} ===")
        if aw < 1000:
            print("  提示：分辨率偏低，图卡拿近些更容易找到框")
        print("Q/ESC 退出，S 保存当前找框输入图+warp")
        frame_idx = 0
        fails = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails == 5:
                    print("[WARN] 读帧失败，重试中…")
                if fails > 30:
                    print("[WARN] 摄像头持续无响应，退出"
                          "（可重插 USB 或换 --cam 索引）")
                    break
                time.sleep(0.02)
                continue
            fails = 0
            disp, final, paths, dbg = process_frame(frame, frame_idx)
            cv2.imshow("Find-box input (binary)", disp)
            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                save_debug(dbg, frame_idx)
        cap.release()
        cv2.destroyAllWindows()
        print(f"  共处理 {frame_idx} 帧")

    # ── 照片模式 ──
    elif args.image:
        img = imread_unicode(args.image)
        if img is None:
            print(f"无法读取 {args.image}")
            return
        disp, final, paths, dbg = process_frame(img)
        rules_s, hu_b, hu_d = paths
        print(f"\n=== {os.path.basename(args.image)} ===")
        print(f"  规则(纯CV): {SHAPE_NAMES.get(rules_s, '-') if rules_s else '-'}")
        if hu_b:
            print(f"  Hu矩: {SHAPE_NAMES.get(hu_b, hu_b)} (距离 {hu_d:.4f}，辅助参考)")
        if final:
            src, shape, action, conf = final
            print(f"  识别: {SHAPE_NAMES.get(shape, shape)} (动作{action} {ACTION_NAMES[action]})")
            print(f"  方案: {src}  置信度: {conf:.2f}")
        else:
            print("  未识别")
        save_debug(dbg, 0)
        out = os.path.join(_SCRIPT_DIR, "..", "6_pictures", "demo_result.jpg")
        cv2.imencode(".jpg", disp)[1].tofile(os.path.abspath(out))
        print(f"  结果图: {os.path.abspath(out)}")
        cv2.imshow("Result", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── 视频模式 ──
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"无法打开视频 {args.video}")
            return
        print(f"=== {os.path.basename(args.video)} ===")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            disp, final, paths, dbg = process_frame(frame, frame_idx)
            if final:
                src, shape, action, conf = final
                print(f"  帧{frame_idx:>4}: {SHAPE_NAMES.get(shape, shape)} 动作{action} "
                      f"({src} conf={conf:.2f})  [规则:{paths[0] or '-'} "
                      f"Hu:{paths[1] or '-'}]")
            else:
                print(f"  帧{frame_idx:>4}: 未识别  [规则:{paths[0] or '-'} "
                      f"Hu:{paths[1] or '-'}]")
            cv2.imshow("Video", disp)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print(f"  共处理 {frame_idx} 帧")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
