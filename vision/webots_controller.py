"""Webots 控制器 — 使用 jetson_vision 模块（与 Jetson Nano 同一套代码）

巡线 + QR检测 + 红条检测 -> UART协议V2输出 -> 电机控制
Webots 里 e-puck 的 camera 输出 BGRA bytes，转成 OpenCV BGR 后
直接走和真机一样的 LineDetector / QRDetector。

用法：Webots -> e-puck controller 设为此文件。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import atexit
import numpy as np

# Webots 特有的 import（保护真机运行不会炸）
try:
    from controller import Robot
    _IS_WEBOTS = True
except ImportError:
    _IS_WEBOTS = False
    Robot = object

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# jetson_vision 模块路径
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qr_detector import QRDetector
from line_detector import LineDetector


# ── 配置加载 ──
def _cfg_get(cfg, path, default):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_cfg():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "line_follow_params.json"),
        os.path.abspath("line_follow_params.json"),
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


CFG = _load_cfg()

# ── 参数 ──
CAM_PITCH = float(_cfg_get(CFG, "camera.pitch_deg", 45.0))
CAM_HEIGHT = float(_cfg_get(CFG, "camera.height_cm", 40.0))
TRACK_WIDTH = float(_cfg_get(CFG, "roi.min_track_width", 35.5))  # actually track width in cm
BASE_SPEED = float(_cfg_get(CFG, "webots.base_speed", 3.2))
STEER_TO_WHEEL = float(_cfg_get(CFG, "webots.steer_to_wheel", 0.04))
MAX_SPEED = float(_cfg_get(CFG, "webots.max_speed", 6.28))

KP = float(_cfg_get(CFG, "pid.straight.kp", 0.7))
KI = float(_cfg_get(CFG, "pid.straight.ki", 0.015))
KD = float(_cfg_get(CFG, "pid.straight.kd", 0.12))
I_CLAMP = float(_cfg_get(CFG, "pid.i_clamp", 60.0))

# ── 初始化 Webots ──
robot = Robot()
TIMESTEP = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera_ext")
camera.enable(TIMESTEP)
img_w, img_h = camera.getWidth(), camera.getHeight()

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ── jetson_vision 检测器（与真机同一行代码） ──
qr = QRDetector(stable_frames=1, cooldown_ms=2000, min_edge_px=20, max_edge_px=300, debug=True)

line = LineDetector(
    cam_height_cm=CAM_HEIGHT, cam_pitch_deg=CAM_PITCH,
    cam_w=img_w, cam_h=img_h, track_width_cm=TRACK_WIDTH,
    inner_radius_cm=59.75, outer_radius_cm=95.25,
)

# ── PID 状态 ──
pid_state = {"integral": 0.0, "last_err": 0.0, "last_steer": 0.0}


def webots_bgra_to_bgr(raw_bytes, w, h):
    """Webots Camera BGRA -> OpenCV BGR numpy array."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(buf, cv2.COLOR_BGRA2BGR)


# ── 主循环 ──
print("[webots_controller] jetson_vision bridge active")
print(f"  camera: {img_w}x{img_h}  pitch={CAM_PITCH}deg  height={CAM_HEIGHT}cm")
print(f"  QR detector: stable={qr.stable_frames}  cooldown={qr.cooldown_ms}ms")
print(f"  Line detector: warp+w=160 h=200  PID kp={KP} ki={KI} kd={KD}")

while robot.step(TIMESTEP) != -1:
    raw = camera.getImage()
    if raw is None:
        continue

    # Webots BGRA -> OpenCV BGR（和真机 USB 摄像头一样的格式）
    bgr = webots_bgra_to_bgr(raw, img_w, img_h)

    # ── 巡线 ──
    deviation_px, heading_deg, conf, vis, dbg = line.process(bgr)

    steer = 0.0
    if deviation_px is not None and conf > 0.15:
        # 简单 PID 控制
        err = -deviation_px / 80.0  # normalize
        dt = TIMESTEP / 1000.0
        pid_state["integral"] += err * dt
        pid_state["integral"] = max(-I_CLAMP, min(I_CLAMP, pid_state["integral"]))
        derr = (err - pid_state["last_err"]) / max(dt, 1e-3)
        pid_state["last_err"] = err
        steer = 70.0 * math.tanh((KP * err + KI * pid_state["integral"] + KD * derr) / 22.0)
    else:
        # 丢线：继续上次方向
        steer = pid_state["last_steer"] * 0.85
    pid_state["last_steer"] = steer

    # 速度 + 差速
    target_speed = BASE_SPEED
    delta = max(-2.8, min(2.8, steer * STEER_TO_WHEEL))
    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, target_speed - delta)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, target_speed + delta)))

    # ── QR 检测 ──
    action, qr_dbg = qr.update(bgr)

    # ── 红条检测 ──
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_ratio = cv2.countNonZero(mask) / (img_w * img_h)

    # ── 协议输出（与 OpenMV 相同格式） ──
    # TODO: UART 写入 / 文件 dump

    # ── 日志 ──
    t = robot.getTime()
    if int(t * 4) != int((t - TIMESTEP / 1000.0) * 4):
        print(f"t={t:.1f}s steer={steer:.1f} ex={deviation_px:.1f}px conf={conf:.2f} red={red_ratio:.3f}")
