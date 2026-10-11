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

from collections import Counter, deque

MODES = ("off", "qr", "shape", "both", "button")


class StartGate:
    def __init__(self, mode="both", expected_qr="1", shape_confirm=2,
                 *, shape_vote_min=8, shape_vote_window=24, shape_voting=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if shape_confirm < 1:
            raise ValueError("shape_confirm must be at least 1")
        if shape_vote_min < 1 or shape_vote_window < shape_vote_min:
            raise ValueError("shape vote window must hold at least the minimum votes")
        self.mode = mode
        self.expected_qr = str(expected_qr).strip()
        self.shape_confirm = int(shape_confirm)
        self.shape_voting = bool(shape_voting)
        self.shape_vote_min = int(shape_vote_min)
        self.shape_votes = deque(maxlen=int(shape_vote_window))
        self.require_qr = mode in ("qr", "both")
        # In button mode shape recognition drives the lamp, never permission to walk.
        self.require_shape = mode in ("shape", "both", "button")
        self.button_passed = False
        self.qr_passed = False
        self.shape_passed = False
        self.last_qr = None
        self.last_shape = None
        self.shape_streak = 0

    def vote_tally(self):
        return dict(Counter(self.shape_votes))

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
        """返回本帧是否锁存了阀2；比赛模式在扫码后才结算近期票。"""
        if name is None:
            return False
        if self.shape_voting:
            if self.shape_passed:
                return False
            self.shape_votes.append(name)
            # Before QR arrives, keep collecting recent votes rather than
            # freezing the first two matching readings of a stationary card.
            if self.require_qr and not self.qr_passed:
                return False
            counts = Counter(self.shape_votes)
            total = len(self.shape_votes)
            winner, votes = counts.most_common(1)[0]
            runner_up = max((count for label, count in counts.items()
                             if label != winner), default=0)
            if (total < self.shape_vote_min or votes * 3 < total * 2
                    or votes - runner_up < 2):
                return False
            self.last_shape = winner
            self.shape_passed = True
            return True
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
            (f"等图卡 投票={self.vote_tally()} / 至少{self.shape_vote_min}票"
             if self.shape_voting else
             f"等图卡 最近={self.last_shape} 连 {self.shape_streak}/{self.shape_confirm}"))
        return f"阀1={qr} | 阀2={shape}"
