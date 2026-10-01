"""把机身姿态发给视觉进程。

视觉那边只能从配置文件里读一个静态的安装俯角，但走路时机身俯仰会摆 30~40°，
停下来做图卡动作时又是另外一个姿态 —— 用静态值算出来的几何，在这两个状态下
至少有一个是错的。IMU 在策略进程手里（串口是独占的），所以这里把它广播出去。

发的是 policy 帧下的 world-down（单位向量），不是角度：安装角是视觉那边的事，
光轴到底比水平低多少由它拿自己的配置去组合，这边不需要知道。

帧约定和 config.IMU_TO_POLICY 一致 —— x 前、y 左、z 上。直立时 g=(0,0,-1)，
机身前倾 φ 时 g_x=sin(φ)。
"""

from __future__ import annotations

import json
import socket

import numpy as np


class AttitudeBroadcaster:
    """UDP 广播，丢了不补 —— 姿态是连续量，视觉那边低通着用。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5007) -> None:
        self.address = (str(host), int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, projected_gravity, elapsed_s: float) -> None:
        gravity = np.asarray(projected_gravity, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(gravity)):
            return
        message = {
            "g": [round(float(value), 5) for value in gravity],
            "t": round(float(elapsed_s), 3),
        }
        self.socket.sendto(
            json.dumps(message, separators=(",", ":")).encode("utf-8"), self.address
        )

    def close(self) -> None:
        self.socket.close()
