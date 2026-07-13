"""QR 检测模块 — Jetson Nano / Windows 通用。
使用 OpenCV QRCodeDetector，双策略：raw → 2x upscale。
实测数据：upscale 命中率 ~65%，raw ~30%，binary/clahe 无效。
比 OpenMV find_qrcodes 快 8-20 倍，1280x720 下轻松检测 5cm 码。
"""
import cv2
import time
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class QRDetector:
    def __init__(
        self,
        stable_frames=1,       # 连续确认帧数（1=首帧即发）
        cooldown_ms=3200,       # 发送冷却
        min_edge_px=15,         # QR 最小边长（像素）
        max_edge_px=450,        # QR 最大边长
        lens_k=None,            # 畸变系数 [k1,k2,p1,p2,k3] 或 None
        camera_matrix=None,     # 相机内参 (3x3)
        cam_w=640,
        cam_h=480,
        debug=True,
    ):
        self.stable_frames = stable_frames
        self.cooldown_ms = cooldown_ms
        self.min_edge = min_edge_px
        self.max_edge = max_edge_px
        self.lens_k = lens_k
        self.camera_matrix = camera_matrix
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.debug = debug

        self.detector = cv2.QRCodeDetector()
        self.candidate = None
        self.candidate_count = 0
        self.last_send_ms = None
        self.first_candidate_ms = None

        self._map_x = None
        self._map_y = None
        if lens_k is not None and camera_matrix is not None:
            self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                camera_matrix, lens_k, None,
                camera_matrix, (cam_w, cam_h), cv2.CV_16SC2)

    def preprocess(self, bgr):
        """可选预处理：去畸变 -> 灰度 -> CLAHE 增强对比度。"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._map_x is not None:
            gray = cv2.remap(gray, self._map_x, self._map_y, cv2.INTER_LINEAR)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        return gray

    def decode_one(self, gray):
        """单帧解码 (payload, [4 corners] or None)"""
        try:
            data, pts, _ = self.detector.detectAndDecode(gray)
        except cv2.error:
            return None, None
        if data is None or not data.strip() or pts is None:
            return None, None
        payload = data.strip()
        if payload not in ("1","2","3","4","5","6"):
            return None, None
        corners = pts.reshape(-1, 2)
        w = float(np.linalg.norm(corners[1] - corners[0]))
        h = float(np.linalg.norm(corners[2] - corners[1]))
        edge = max(w, h)
        if edge < self.min_edge or edge > self.max_edge:
            return None, None
        return payload, corners

    def update(self, bgr_or_gray):
        """返回 (action_number, debug_dict) 或 (None, None)。"""
        if len(bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_or_gray

        # S1: raw — QR 大/近时最快
        payload, corners = self.decode_one(gray)
        if payload is not None:
            return self._confirm(payload, corners, "raw")

        # S2: 2x upscale — QR 小/远时主导（实测命中率最高）
        h, w = gray.shape[:2]
        up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        payload, corners = self.decode_one(up)
        if payload is not None and corners is not None:
            corners = corners * 0.5
            return self._confirm(payload, corners, "upscale")

        self.candidate = None
        self.candidate_count = 0
        return None, None

    def _confirm(self, payload, corners, strategy):
        """确认逻辑 + 冷却，返回 (action, {debug})。"""
        now = int(time.time() * 1000)
        if payload == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = payload
            self.candidate_count = 1
            self.first_candidate_ms = now
            if self.debug:
                w = int(np.linalg.norm(corners[1] - corners[0]))
                h = int(np.linalg.norm(corners[2] - corners[1]))
                print(f"  [qr] NEW pl={payload} {w}x{h}px  strat={strategy}")

        if self.candidate_count >= self.stable_frames:
            ready = (self.last_send_ms is None
                     or (now - self.last_send_ms) >= self.cooldown_ms)
            if ready:
                u = int(payload)
                self.last_send_ms = now
                self.candidate_count = 0
                latency = now - (self.first_candidate_ms or now)
                dbg = {"action": u, "strategy": strategy, "latency_ms": latency}
                w = int(np.linalg.norm(corners[1] - corners[0]))
                h = int(np.linalg.norm(corners[2] - corners[1]))
                dbg["w"] = w
                dbg["h"] = h
                if self.debug:
                    print(f"  [qr] >>> SEND action={u}  latency={latency}ms  {w}x{h}px")
                return u, dbg
        return None, None

    def draw(self, frame, dbg):
        """在 frame 上画 QR 框。"""
        if dbg is None:
            return
