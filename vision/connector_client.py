"""Small UDP publisher used by the vision process to reach connector.py."""

from __future__ import annotations

import json
import socket


class ConnectorClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5006) -> None:
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, vx: float, wz: float, qr: int = -1) -> None:
        message = {
            "vx": float(vx),
            "vy": 0.0,
            "wz": float(wz),
            "qr": int(qr),
        }
        self.socket.sendto(
            json.dumps(message, separators=(",", ":")).encode("utf-8"),
            self.address,
        )

    def close(self) -> None:
        self.socket.close()
