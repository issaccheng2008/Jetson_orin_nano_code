"""跨模块公共工具：数值裁剪、中文路径图像读取、相机打开。"""
import sys
import cv2
import numpy as np


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def open_camera(idx, width, height, warmup=3):
    """打开摄像头：Windows 默认后端失败回退 DSHOW，Linux 用 V4L2。"""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    # 顺序不能反：DSHOW 下先设 FOURCC 再设分辨率会被驱动重新协商踢回 YUY2。
    # USB2.0 带宽下 YUY2 720p 只有 10fps，MJPG 有 22fps+。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(warmup):
        ok, _ = cap.read()
        if ok:
            break
    return cap


def show_debug_windows(dbg, vis_bird):
    """巡线中间结果三窗口（鸟瞰/二值/拟合）。"""
    if dbg.get("bird") is not None:
        bird_bgr = cv2.cvtColor(dbg["bird"], cv2.COLOR_GRAY2BGR)
        cv2.imshow("2.Warp (birdseye)",
                   cv2.resize(bird_bgr, (320, 400),
                              interpolation=cv2.INTER_NEAREST))
    if dbg.get("binary_raw") is not None:
        b_raw = cv2.cvtColor(dbg["binary_raw"], cv2.COLOR_GRAY2BGR)
        cv2.imshow("3.Adaptive (binary)",
                   cv2.resize(b_raw, (320, 400),
                              interpolation=cv2.INTER_NEAREST))
    if vis_bird is not None:
        cv2.imshow("4.Close+Fit",
                   cv2.resize(vis_bird, (320, 400),
                              interpolation=cv2.INTER_NEAREST))
