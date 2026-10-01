"""接收策略进程广播的机身姿态，算出相机光轴的真实俯角。

视觉这边只知道自己配置里的安装角，而机身走路时俯仰会摆 30~40° —— 光轴相对
水平面到底低多少，靠静态配置是猜的。策略进程有 IMU（串口独占），它把 policy
帧下的 world-down 发过来，这里和自己配置的安装角组合。

**低通是必须的，不是保险**：步态 1.7Hz、摆幅 30~40°，实时值滞后几十毫秒就完全
跟不上，直接喂进几何比用静态值还糟。但这个摆动是周期信号、均值为零，而机身真正
的前倾（站姿、负重）是慢变量 —— 长时间常数低通留下的正好是有用的那部分。停下
做图卡动作时姿态本来就稳，低通几个时间常数就跟上了。

时间常数是按摆动幅度定的，不是拍的：τ=0.4s 时 1.7Hz 只衰减到 23%，残留还有 ±8°；
τ=1.2s 衰减到 7.8%，残留 ±2.7°，而 3 秒的停车够走 2.5 个时间常数（收敛 92%）。

帧约定和 config.IMU_TO_POLICY 一致：x 前、y 左、z 上，直立时 g=(0,0,-1)。
"""

from __future__ import annotations

import json
import math
import socket
import time

from utils import clamp


def camera_pitch_deg(gravity, mount_pitch_deg):
    """机身姿态 + 安装角 → 光轴低于水平面的角度（度）。

    光轴在 policy 帧里是固定的 (cos m, 0, −sin m)，它和世界"上"的夹角余弦
    就是把它投到 world-down 上：sin(θ) = g_x·cos m − g_z·sin m。
    直立时 g=(0,0,−1) 退化成 sin(θ)=sin m，也就是安装角本身。

    对任意姿态都精确，包括 roll —— 而且 roll 不能省：光轴在 policy 帧里本来就有
    z 分量，机身一侧倾，光轴跟着偏出铅垂面，俯角就变了（实测 pitch 10° + roll 30°
    时是 46.6°，只有 pitch 的话是 55°）。拿 Euler pitch 加安装角是错的。
    """
    gx, _gy, gz = (float(v) for v in gravity)
    mount = math.radians(float(mount_pitch_deg))
    return math.degrees(math.asin(clamp(
        gx * math.cos(mount) - gz * math.sin(mount), -1.0, 1.0)))


class AttitudeInput:
    """收姿态包，维护一个低通过的"当前光轴俯角"。"""

    def __init__(self, port, mount_pitch_deg, bind="127.0.0.1", tau_s=1.2):
        if tau_s <= 0.0:
            raise ValueError("low-pass time constant must be positive")
        self.mount_pitch_deg = float(mount_pitch_deg)
        self.tau_s = float(tau_s)
        self.value = float(mount_pitch_deg)   # 没有姿态时的退路：就是静态安装角
        self.received = 0
        self.last_packet_s = 0.0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        try:
            self.socket.bind((bind, int(port)))
        except OSError:
            self.socket.close()
            raise

    def poll(self, now=None):
        """把积压的包一次读完，只留最新的那个点。返回是否收到过包。"""
        now = time.monotonic() if now is None else float(now)
        seen = False
        while True:
            try:
                payload, _address = self.socket.recvfrom(512)
            except (BlockingIOError, OSError):
                break
            try:
                message = json.loads(payload.decode("utf-8"))
                gravity = message["g"]
                if len(gravity) != 3:
                    raise ValueError("gravity must have three components")
                target = camera_pitch_deg(gravity, self.mount_pitch_deg)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
                    ValueError, OverflowError):
                continue
            dt = (now - self.last_packet_s) if self.received else 0.0
            alpha = 1.0 if dt <= 0.0 else 1.0 - math.exp(-dt / self.tau_s)
            if not self.received:
                self.value = target
            else:
                self.value += alpha * (target - self.value)
            self.received += 1
            self.last_packet_s = now
            seen = True
        return seen

    def close(self):
        self.socket.close()
