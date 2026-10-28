"""二维码读取（起跑门控的阀1）。

只用 ``cv2.QRCodeDetector``（opencv-python 自带，不需要 contrib）。双策略照搬
`archive/legacy_2026/jetson/qr_detector.py` 实测出来的那条：raw 先试，失败再
LANCZOS4 放大 —— 小码/远码靠后者，命中率从 ~30% 提到 ~65%。

它只负责"解出来是什么"，不判该不该认。白名单在 `StartGate`：扫到别的 payload
要在日志里看得见，而不是在这里被静默吃掉。

放大之前先把长边压到 ``max_side``：Jetson 上 2560×1440 直接放大到 5120×2880
是几百毫秒一次，压回 1280 之后 5cm 的码在 35cm 处还有 ~97px，照样能解。
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class QrReading:
    payload: str
    strategy: str          # "raw" | "upscale"
    edge_px: float         # 原图尺度上的平均边长
    cost_ms: float
    corners: np.ndarray    # (4, 2) 原图坐标


class QrReader:
    def __init__(self, min_edge_px=15.0, max_edge_px=450.0,
                 upscale=2.0, max_side=1280):
        if min_edge_px <= 0 or max_edge_px <= min_edge_px:
            raise ValueError("need 0 < min_edge_px < max_edge_px")
        if upscale < 1.0:
            raise ValueError("upscale must be at least 1")
        self.min_edge = float(min_edge_px)
        self.max_edge = float(max_edge_px)
        self.upscale = float(upscale)
        self.max_side = int(max_side)
        self.detector = cv2.QRCodeDetector()
        self.scans = 0
        self.hits = 0
        self.geom_rejects = 0
        self.last_cost_ms = 0.0

    def decode(self, frame):
        """返回 QrReading 或 None。None 也计入 scans（现场要看扫了多少次）。"""
        started = time.monotonic()
        self.scans += 1
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame)
        scale = 1.0
        if self.max_side > 0 and max(gray.shape[:2]) > self.max_side:
            # INTER_AREA 缩小时等价于按像素平均，不会像 INTER_LINEAR 那样在
            # 二维码的黑白格上产生中间灰度。
            fit = self.max_side / max(gray.shape[:2])
            gray = cv2.resize(gray, None, fx=fit, fy=fit,
                              interpolation=cv2.INTER_AREA)
            scale = fit
        reading = self._try(gray, scale, "raw", started)
        if reading is None and self.upscale > 1.0:
            big = cv2.resize(gray, None, fx=self.upscale, fy=self.upscale,
                             interpolation=cv2.INTER_LANCZOS4)
            reading = self._try(big, scale * self.upscale, "upscale", started)
        self.last_cost_ms = (time.monotonic() - started) * 1000.0
        if reading is not None:
            self.hits += 1
        return reading

    def _try(self, gray, scale, strategy, started):
        try:
            payload, points, _ = self.detector.detectAndDecode(gray)
        except cv2.error:
            return None
        if not payload or not payload.strip() or points is None:
            return None
        corners = points.reshape(-1, 2) / scale
        if corners.shape != (4, 2):
            return None
        edge = float(np.mean(np.linalg.norm(
            np.roll(corners, -1, axis=0) - corners, axis=1)))
        if not self.min_edge <= edge <= self.max_edge:
            self.geom_rejects += 1
            return None
        return QrReading(payload.strip(), strategy, edge,
                         (time.monotonic() - started) * 1000.0, corners)
