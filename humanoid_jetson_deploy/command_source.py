"""Fixed or UDP velocity commands for the walking policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def clamp_command(command) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(command)):
        raise ValueError("velocity command must contain only finite values")
    return np.array(
        [
            np.clip(command[0], 0.0, 1.0),
            0.0,
            np.clip(command[2], -0.5, 0.5),
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


class UdpCommandSource:
    """Receive connector JSON and expose velocity plus shape metadata."""

    def __init__(self, port: int, timeout_s: float = 0.25, bind: str = "127.0.0.1") -> None:
        self.timeout_s = timeout_s
        self.command = np.zeros(3, dtype=np.float32)
        self.qr = -1
        self.event_id = 0
        self.event_action = -1
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
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
                continue
            with self.lock:
                self.command = command
                self.qr = qr
                self.event_id = event_id
                self.event_action = event_action if event_id else -1
                self.last_update = time.monotonic()

    def get(self) -> np.ndarray:
        return self.get_snapshot().velocity

    def get_snapshot(self) -> CommandSnapshot:
        with self.lock:
            command = self.command.copy()
            qr, event_id, event_action = self.qr, self.event_id, self.event_action
            age = time.monotonic() - self.last_update
        if age > self.timeout_s:
            return CommandSnapshot(np.zeros(3, dtype=np.float32))
        return CommandSnapshot(command, qr, event_id, event_action)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.sock.close()
