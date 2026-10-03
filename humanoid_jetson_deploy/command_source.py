"""Fixed or UDP velocity commands for the walking policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import threading
import time

import numpy as np


@dataclass(frozen=True)
class CommandSnapshot:
    velocity: np.ndarray
    qr: int = -1
    event_id: int = 0
    event_action: int = -1
    hold_upright: bool = False
    card_tilt: bool = False


# 偏航指令的硬上限。原来写死 0.5：视觉那条路自己会夹（--max-wz 默认 0.5），
# 但固定指令模式（--command-source fixed --wz X）也走这个函数，想试更大的
# 偏航就被这里挡掉了。放宽到 1.0 —— 视觉那条路的行为完全不变（它自己先夹）。
# ⚠️ 策略的训练指令范围是未知的，超出它可能做不出对应动作甚至不稳，
# 往大调的时候一次加一点。
MAX_WZ = 1.0


def clamp_command(command) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(command)):
        raise ValueError("velocity command must contain only finite values")
    return np.array(
        [
            np.clip(command[0], 0.0, 1.0),
            0.0,
            np.clip(command[2], -MAX_WZ, MAX_WZ),
        ],
        dtype=np.float32,
    )


class FixedCommandSource:
    def __init__(self, vx: float, wz: float) -> None:
        self.command = clamp_command([vx, 0.0, wz])

    def get(self) -> np.ndarray:
        return self.command.copy()

    def close(self) -> None:
        pass


class ScriptedCommandSource:
    """A fixed open-loop timeline: (duration_s, vx, wz) legs, zero after the last.

    FixedCommandSource holds one command forever; this one is for "straight, then
    turn a whole curve, then straight again". The clock starts on the **first
    get()** rather than on construction or process start, because main.py builds
    the source and then has to open the serial link and wait for state packets --
    counting from construction would eat the first leg while the robot is still
    standing. main.py ticks get() at 50 Hz from the moment it is ready to walk,
    so the first call is the first walking tick.
    """

    def __init__(self, legs) -> None:
        if not legs:
            raise ValueError("a scripted timeline needs at least one leg")
        self._legs = []
        edge = 0.0
        for duration_s, vx, wz in legs:
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                raise ValueError("leg durations must be finite and positive")
            self._legs.append((edge, edge + duration_s, clamp_command([vx, 0.0, wz])))
            edge += duration_s
        self.total_s = edge
        self._t0 = None

    def get(self) -> np.ndarray:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        for start, end, command in self._legs:
            if elapsed < end:
                return command.copy()
        return clamp_command([0.0, 0.0, 0.0])

    def close(self) -> None:
        pass


def curve_legs(straight_s: float, turn_s: float, vx: float, turn_wz: float):
    """A stadium bulge as an open-loop timeline: straight, one 180 turn, straight.

    Defaults come straight off the track: centreline R = 0.776 m, so a semicircle
    is pi*R = 2.438 m and a straight is (6.140 - 2*2.438)/2 = 0.632 m. At
    vx = 0.2 that is 3.16 s of straight, and holding the arc needs
    omega = v/R = 0.258 rad/s for pi/0.258 = 12.19 s of turn.
    """
    return [(straight_s, vx, 0.0),
            (turn_s, vx, turn_wz),
            (straight_s, vx, 0.0)]


def parse_legs(spec: str):
    """`3.2:0.2:0; 12.2:0.2:0.258; 3.2:0.2:0` -> [(3.2, 0.2, 0.0), ...].

    Seconds, m/s, rad/s. Semicolons or commas between legs so both a shell-quoted
    string and a bare one work.
    """
    legs = []
    for chunk in spec.replace(",", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"leg {chunk!r} is not seconds:vx:wz")
        legs.append(tuple(float(p) for p in parts))
    if not legs:
        raise ValueError("no legs in the timeline")
    return legs


class UdpCommandSource:
    """Receive connector JSON and expose velocity plus shape metadata."""

    def __init__(self, port: int, timeout_s: float = 0.25, bind: str = "127.0.0.1") -> None:
        self.timeout_s = timeout_s
        self.command = np.zeros(3, dtype=np.float32)
        self.qr = -1
        self.event_id = 0
        self.event_action = -1
        self.hold_upright = False
        self.card_tilt = False
        self.last_update = 0.0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.1)
        self.sock.bind((bind, port))
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while not self.stop.is_set():
            try:
                data, _address = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            try:
                message = json.loads(data.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("command must be a JSON object")
                command = clamp_command([message["vx"], message.get("vy", 0.0), message["wz"]])
                qr = int(message.get("qr", -1))
                if qr not in (-1, 1, 2, 3, 4, 5, 6):
                    raise ValueError("invalid qr")
                event_id = int(message.get("event_id", 0))
                event_action = int(message.get("event_action", -1))
                if event_id < 0 or event_id > 0xFFFFFFFF or (event_id > 0 and event_action not in (1, 2, 3, 4, 5, 6)):
                    raise ValueError("invalid shape event")
                hold_upright = bool(message.get("hold_upright", False))
                card_tilt = bool(message.get("card_tilt", False))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
                continue
            with self.lock:
                self.command = command
                self.qr = qr
                self.event_id = event_id
                self.event_action = event_action if event_id else -1
                self.hold_upright = hold_upright
                self.card_tilt = card_tilt
                self.last_update = time.monotonic()

    def get(self) -> np.ndarray:
        return self.get_snapshot().velocity

    def get_snapshot(self) -> CommandSnapshot:
        with self.lock:
            command = self.command.copy()
            qr, event_id, event_action = self.qr, self.event_id, self.event_action
            hold_upright = self.hold_upright
            card_tilt = self.card_tilt
            age = time.monotonic() - self.last_update
        if age > self.timeout_s:
            return CommandSnapshot(np.zeros(3, dtype=np.float32))
        return CommandSnapshot(command, qr, event_id, event_action, hold_upright,
                               card_tilt)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.sock.close()
