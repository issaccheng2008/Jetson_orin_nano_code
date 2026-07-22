from __future__ import annotations

import json
import socket
import time
import unittest

import numpy as np

from command_source import UdpCommandSource, clamp_command


class CommandSourceTests(unittest.TestCase):
    def test_clamp_rejects_non_finite_command(self) -> None:
        with self.assertRaises(ValueError):
            clamp_command([float("nan"), 0.0, 0.0])

    def test_udp_feedback_reaches_policy_and_stale_feedback_stops_it(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        source = UdpCommandSource(port, timeout_s=0.2)
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            payload = json.dumps({"vx": 0.3, "vy": 0.7, "wz": -0.2, "qr": 3}).encode()
            publisher.sendto(payload, ("127.0.0.1", port))

            deadline = time.monotonic() + 0.5
            command = source.get()
            while time.monotonic() < deadline and not np.allclose(command, [0.3, 0.0, -0.2]):
                time.sleep(0.005)
                command = source.get()
            np.testing.assert_allclose(command, [0.3, 0.0, -0.2])

            time.sleep(0.21)
            np.testing.assert_array_equal(source.get(), np.zeros(3, dtype=np.float32))
        finally:
            publisher.close()
            source.close()

    def test_udp_ignores_invalid_feedback(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        source = UdpCommandSource(port, timeout_s=0.1)
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            publisher.sendto(b'{"vx":NaN,"wz":0.2}', ("127.0.0.1", port))
            time.sleep(0.02)
            np.testing.assert_array_equal(source.get(), np.zeros(3, dtype=np.float32))
        finally:
            publisher.close()
            source.close()


if __name__ == "__main__":
    unittest.main()
