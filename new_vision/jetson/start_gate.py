"""起跑门控：两个阀都过了才允许发速度。

赛场上三个进程提前跑着，机器人必须站着不动，直到：
  阀1 —— 扫到二维码（2025 规则那套，payload 是字符串 "1"）；
  阀2 —— 认出第一张图卡是什么形状（分类成功，不是"看到有卡"）。

阀2 用连续同形状帧数而不是单帧：分类器在单帧上是会跳的，多站半秒的代价换一个
不会自己把自己卡死的门。`None` 是 no-op —— 没跑检测的那些帧（`card_dbg` 为空、
重摆期间）不该把连续计数打断。

这里不 import cv2、不读时间、不碰 IO，纯状态机，可以脱离相机栈单测。
"""

from __future__ import annotations

MODES = ("off", "qr", "shape", "both", "button")


class StartGate:
    def __init__(self, mode="both", expected_qr="1", shape_confirm=2):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if shape_confirm < 1:
            raise ValueError("shape_confirm must be at least 1")
        self.mode = mode
        self.expected_qr = str(expected_qr).strip()
        self.shape_confirm = int(shape_confirm)
        self.require_qr = mode in ("qr", "both")
        # In button mode shape recognition drives the lamp, never permission to walk.
        self.require_shape = mode in ("shape", "both", "button")
        self.button_passed = False
        self.qr_passed = False
        self.shape_passed = False
        self.last_qr = None
        self.last_shape = None
        self.shape_streak = 0

    @property
    def passed(self):
        if self.mode == "button":
            return self.button_passed
        return ((not self.require_qr or self.qr_passed)
                and (not self.require_shape or self.shape_passed))

    def observe_button(self, pressed):
        if self.mode != "button" or self.button_passed or not pressed:
            return False
        self.button_passed = True
        return True

    def observe_qr(self, payload):
        """返回本帧是否锁存了阀1。None = 这一帧没解码，什么都不算。"""
        if payload is None:
            return False
        text = str(payload).strip()
        self.last_qr = text
        if self.qr_passed or text != self.expected_qr:
            return False
        self.qr_passed = True
        return True

    def observe_shape(self, name):
        """返回本帧是否锁存了阀2。None 不打断连续计数。"""
        if name is None:
            return False
        if name == self.last_shape:
            self.shape_streak += 1
        else:
            self.last_shape = name
            self.shape_streak = 1
        if self.shape_passed or self.shape_streak < self.shape_confirm:
            return False
        self.shape_passed = True
        return True

    def status(self):
        if self.mode == "button":
            card = self.last_shape if self.shape_passed else "未识别（仍可按按钮）"
            return f"按钮={'已按下' if self.button_passed else '等待PC2'} | 首卡={card}"
        qr = "不需要" if not self.require_qr else (
            "已过" if self.qr_passed else f"等二维码 最近={self.last_qr}")
        shape = "不需要" if not self.require_shape else (
            "已过" if self.shape_passed else
            f"等图卡 最近={self.last_shape} 连 {self.shape_streak}/{self.shape_confirm}")
        return f"阀1={qr} | 阀2={shape}"
