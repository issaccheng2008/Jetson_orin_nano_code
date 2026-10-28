"""图卡检测预处理 — YOLO输入前（照搬巡线完整管线版）。

本分支 = 完整照搬 line_detector_v1_warp.py 的预处理（含black_th二次阈值、
形态学4步、连通域过滤），仅连通域阈值按分辨率比例调整。

实测检出率21%（对比main分支简化版88%）——保留本分支供继续调参。
"""
import cv2
import numpy as np

# ── 固定工作分辨率（方案A）──
WORK_W = 960
WORK_H = 540

# ── 预处理参数：照搬巡线（line_detector_v1_warp.py 914-957行）──
# 巡线工作分辨率 320×400，本模块 960×540 → 面积比≈4倍
# 连通域过滤阈值按面积比例调整（300→1200, 80→110）
BH_KERNEL = 31          # blackhat核（椭圆，巡线同款）
ADAPTIVE_BLOCK = 31     # 自适应阈值窗口（巡线同款）
ADAPTIVE_C = -12        # 阈值偏移（原-8，调参中）
TH_OFFSET = -2          # black_th = median(mask) + offset（巡线同款）
TH_MIN = 25             # black_th 限幅（巡线同款）
TH_MAX = 80
CC_AREA_MIN = 1200      # 连通域过滤（巡线300 × 面积比4）
CC_HEIGHT_MIN = 110     # 连通域过滤（巡线80 × 高比1.35）


def preprocess_for_yolo(bgr, invert=False, save_debug=None,
                        method="adaptive"):
    """真实帧 → YOLO友好的白底黑线图（3通道BGR，960×540）。

    管线（照搬巡线 line_detector_v1_warp.py 预处理，含全部4步形态学）:
        resize到960×540 → 灰度
        → blackhat(31×31椭圆)
        → 自适应阈值(GAUSSIAN, 31, C=-8) 取线mask
        → black_th = median(mask)+(-2), clamp(25,80)
        → 二值化 threshold(gray_detect, black_th)
        → CLOSE(5×5椭圆) → OPEN(5×5) → CLOSE(5×5) → OPEN(3×3)
        → 连通域过滤（area/heigh按分辨率比例调整）
        → 反转成白底黑线（训练分布）→ BGR3通道 → YOLO

    Args:
        bgr: 彩色帧 (任意分辨率, H, W, 3)
        invert: 额外反转（黑底白线），默认False
        save_debug: 若给路径，保存中间结果图（调试用）
        method: "adaptive"=巡线同款管线（默认）
                "otsu"=CLAHE+Otsu全图阈值（备用）

    Returns:
        (540, 960, 3) BGR图，喂给YOLO
    """
    # 0. 统一到固定工作分辨率（方案A：参数与分辨率解耦）
    if bgr.shape[1] != WORK_W or bgr.shape[0] != WORK_H:
        bgr = cv2.resize(bgr, (WORK_W, WORK_H))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        # 备用：CLAHE + Otsu（光照均匀场景）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, bw = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = 255 - bw  # 白底黑线
        bw = _morphology_and_cc(bw)
    else:
        # 照搬巡线预处理（line_detector_v1_warp.py 914-957行）
        # 1. blackhat：抑制宽阴影、增强细黑线→变亮
        k31 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (BH_KERNEL, BH_KERNEL))
        gray_detect = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k31)

        # 2. 自适应阈值（Gaussian，巡线同款 C=-8）
        adaptive_binary = cv2.adaptiveThreshold(
            gray_detect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
        # blackhat后线变亮 → THRESH_BINARY 把线判为255
        adaptive_mask = (adaptive_binary == 255)
        if np.count_nonzero(adaptive_mask) > 100:
            black_th = np.median(gray_detect[adaptive_mask]) + TH_OFFSET
        else:
            black_th = np.median(gray_detect) + TH_OFFSET
        black_th = float(np.clip(black_th, TH_MIN, TH_MAX))

        # 3. 二值化（线区域 → 255）
        bw = cv2.threshold(gray_detect, black_th, 255,
                           cv2.THRESH_BINARY)[1]

        # 4. 形态学4步（巡线同款：close5→open5→close5→open3）
        bw = _morphology_and_cc(bw)

        # 5. 反转成白底黑线（训练分布；blackhat输出是黑底白线）
        bw = 255 - bw

    if invert:
        bw = 255 - bw

    if save_debug is not None:
        cv2.imencode('.jpg', bw)[1].tofile(save_debug)

    # 转回3通道（YOLO输入要求）
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _morphology_and_cc(bw):
    """形态学：close(3×3) 一次，无开运算（几何计算推荐）。"""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=1)
    return bw


if __name__ == "__main__":
    # 自测：跑视频一帧看预处理效果
    import os
    video = os.path.join(os.path.dirname(__file__), "..", "tests",
                         "6-shape-test", "video-1.mp4")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(7) * 0.5))
    ok, frame = cap.read()
    cap.release()
    out = preprocess_for_yolo(frame,
                              save_debug=os.path.join(
                                  os.path.dirname(__file__), "..",
                                  "tests", "6-shape-test",
                                  "preprocess_adaptive.jpg"))
    print(f"预处理完成: 输入{frame.shape} → 输出{out.shape}")
    print(f"调试图: tests/6-shape-test/preprocess_adaptive.jpg")
