"""Fixed or UDP velocity commands for the walking policy."""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np


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
    """Receive connector JSON; extra fields such as ``qr`` are ignored."""

    def __init__(self, port: int, timeout_s: float = 0.25, bind: str = "127.0.0.1") -> None:
        self.timeout_s = timeout_s
        self.command = np.zeros(3, dtype=np.float32)
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
                command = clamp_command([message["vx"], message.get("vy", 0.0), message["wz"]])
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            with self.lock:
                self.command = command
                self.last_update = time.monotonic()

    def get(self) -> np.ndarray:
        with self.lock:
            command = self.command.copy()
            age = time.monotonic() - self.last_update
        return command if age <= self.timeout_s else np.zeros(3, dtype=np.float32)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.sock.close()
