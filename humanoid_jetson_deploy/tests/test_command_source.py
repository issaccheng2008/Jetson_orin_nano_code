from __future__ import annotations

import json
import socket
import time
import unittest

import numpy as np

from command_source import (ScriptedCommandSource, UdpCommandSource, clamp_command,
                            curve_legs, parse_legs)


class ScriptedCommandSourceTests(unittest.TestCase):
    """开环时序：直行 -> 转一个整弯 -> 直行。没有反馈，所以时序本身要能钉住。"""

    def test_legs_parse_with_either_separator(self) -> None:
        for spec in ("3.16:0.2:0; 12.19:0.2:0.258; 3.16:0.2:0",
                     "3.16:0.2:0, 12.19:0.2:0.258, 3.16:0.2:0"):
            self.assertEqual(parse_legs(spec),
                             [(3.16, 0.2, 0.0), (12.19, 0.2, 0.258), (3.16, 0.2, 0.0)])
        for bad in ("", "  ", "3.16:0.2", "a:b:c", "1:2:3:4"):
            with self.subTest(spec=bad), self.assertRaises(ValueError):
                parse_legs(bad)

    def test_each_leg_holds_and_zero_follows_the_last(self) -> None:
        source = ScriptedCommandSource([(0.05, 0.2, 0.0), (0.05, 0.2, 0.4)])
        self.assertAlmostEqual(source.total_s, 0.1, places=6)
        np.testing.assert_allclose(source.get(), [0.2, 0.0, 0.0])   # 第一腿
        time.sleep(0.07)
        np.testing.assert_allclose(source.get(), [0.2, 0.0, 0.4])   # 第二腿
        time.sleep(0.06)
        np.testing.assert_allclose(source.get(), [0.0, 0.0, 0.0])   # 走完发零

    def test_the_clock_starts_on_the_first_tick_not_on_construction(self) -> None:
        """main.py 构造完指令源之后还要开串口、等 STM32 的状态包，那段可能要好几秒。
        从构造起算的话第一腿会在这段时间里被吃掉 —— 实测见过 7 秒的启动间隔。"""
        source = ScriptedCommandSource([(0.4, 0.2, 0.0), (0.4, 0.2, 0.4)])
        time.sleep(0.5)                     # 模拟"建好源之后还没开始走"
        np.testing.assert_allclose(source.get(), [0.2, 0.0, 0.0])

    def test_curve_legs_are_the_straight_turn_straight_default(self) -> None:
        """三个参数就是"直线多少秒、转多少秒、转多快"，默认值直接是场地几何：
        0.632m 直道 / 0.2 = 3.16s，180° / (0.2/0.776) = 12.19s。"""
        legs = curve_legs(3.16, 12.19, 0.2, 0.258)
        self.assertEqual([(d, round(v, 3), round(w, 3)) for d, v, w in legs],
                         [(3.16, 0.2, 0.0), (12.19, 0.2, 0.258), (3.16, 0.2, 0.0)])
        source = ScriptedCommandSource(legs)
        self.assertAlmostEqual(source.total_s, 18.51, places=6)

    def test_a_leg_needs_a_positive_length(self) -> None:
        for legs in ([], [(0.0, 0.2, 0.0)], [(-1.0, 0.2, 0.0)],
                     [(float("nan"), 0.2, 0.0)]):
            with self.subTest(legs=legs), self.assertRaises(ValueError):
                ScriptedCommandSource(legs)

    def test_commands_are_clamped_like_every_other_source(self) -> None:
        source = ScriptedCommandSource([(0.1, 9.9, 9.9)])
        np.testing.assert_allclose(source.get(), [1.0, 0.0, 1.0])   # MAX_WZ = 1.0


class CommandSourceTests(unittest.TestCase):
    def test_shape_event_snapshot_survives_repeated_udp_packets_and_expires(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        source = UdpCommandSource(port, timeout_s=0.1)
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            data = json.dumps({"vx": 0, "wz": 0, "qr": -1,
                               "event_id": 123, "event_action": 4}).encode()
            publisher.sendto(data, ("127.0.0.1", port))
            deadline = time.monotonic() + 0.5
            while source.get_snapshot().event_id != 123 and time.monotonic() < deadline:
                time.sleep(0.005)
            snapshot = source.get_snapshot()
            self.assertEqual((snapshot.qr, snapshot.event_id, snapshot.event_action), (-1, 123, 4))
            publisher.sendto(data, ("127.0.0.1", port))
            time.sleep(0.12)
            self.assertEqual(source.get_snapshot().event_id, 0)
        finally:
            publisher.close()
            source.close()

    def test_hold_upright_arrives_and_defaults_off(self) -> None:
        """A packet without the field must read as "don't", so a vision older than the
        field keeps behaving exactly as it did."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        source = UdpCommandSource(port, timeout_s=0.5)
        publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def wait_for(expected):
            deadline = time.monotonic() + 0.5
            while (source.get_snapshot().hold_upright is not expected
                   and time.monotonic() < deadline):
                time.sleep(0.005)
            return source.get_snapshot().hold_upright

        try:
            publisher.sendto(json.dumps({"vx": 0, "wz": 0}).encode(), ("127.0.0.1", port))
            self.assertFalse(wait_for(False))
            publisher.sendto(json.dumps({"vx": 0, "wz": 0, "hold_upright": True}).encode(),
                             ("127.0.0.1", port))
            self.assertTrue(wait_for(True))
        finally:
            publisher.close()
            source.close()

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
