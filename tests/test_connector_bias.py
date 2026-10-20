"""Yaw compensation through command processing and the real connector UDP loop."""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import connector


class ConnectorBiasTests(unittest.TestCase):
    def test_signed_bias_changes_only_yaw_and_preserves_metadata(self):
        message = {"vx": 0.2, "wz": -0.2, "qr": 3, "command_mode": "held",
                   "event_id": 77, "event_action": 3}
        original = message.copy()
        for bias, expected in ((0.1, -0.1), (-0.1, -0.3)):
            with self.subTest(bias=bias):
                result = connector.process_vision_output(message, wz_bias=bias)
                self.assertAlmostEqual(result["wz"], expected)
                self.assertEqual(result["vx"], 0.2)
                self.assertEqual((result["qr"], result["event_id"], result["event_action"]),
                                 (3, 77, 3))
                self.assertEqual(result["command_mode"], "held")
                self.assertEqual(message, original)

    def test_straight_walking_and_in_place_turn_can_be_compensated(self):
        for vx, wz, expected in ((0.2, 0.0, 0.1), (0.0, -0.3, -0.2)):
            with self.subTest(vx=vx, wz=wz):
                result = connector.process_vision_output({"vx": vx, "wz": wz}, wz_bias=0.1)
                self.assertEqual(result["vx"], vx)
                self.assertAlmostEqual(result["wz"], expected)

    def test_bias_is_applied_before_yaw_clamp(self):
        for wz, bias, expected in ((0.45, 0.1, 0.5), (-0.45, -0.1, -0.5),
                                   (0.6, -0.1, 0.5), (-0.6, 0.1, -0.5)):
            with self.subTest(wz=wz, bias=bias):
                result = connector.process_vision_output({"vx": 0.2, "wz": wz}, wz_bias=bias)
                self.assertAlmostEqual(result["wz"], expected)

    def test_stop_and_posture_requests_stop_immediately_with_bias(self):
        for mode in (None, "held", "continuous"):
            for flag in (None, "hold_upright", "card_tilt"):
                for bias in (0.1, -0.1):
                    with self.subTest(mode=mode, flag=flag, bias=bias):
                        smoother = connector.CommandSmoother(1.0, 2.0)
                        smoother.update({"vx": 0.2, "wz": 0.5, "qr": -1,
                                         "command_mode": "held"}, 0.02)
                        message = {"vx": 0.2 if flag else 0.0,
                                   "wz": 0.3 if flag else 0.0, "qr": 3,
                                   "event_id": 77, "event_action": 3}
                        if mode:
                            message["command_mode"] = mode
                        if flag:
                            message[flag] = True
                        target = connector.process_vision_output(message, wz_bias=bias)
                        output = smoother.update(target, 0.0)
                        self.assertEqual((output["vx"], output["wz"]), (0.0, 0.0))
                        self.assertEqual(output["event_id"], 77)
                        self.assertEqual(output.get("command_mode"), mode)

    def test_watchdog_zero_and_held_ticks_do_not_accumulate_bias(self):
        latest = connector.process_vision_output(
            {"vx": 0.2, "wz": 0.2, "command_mode": "held"}, wz_bias=0.1)
        smoother = connector.CommandSmoother(1.0, 2.0)
        for tick in range(10):
            target, fresh, _ = connector.select_output(latest, 1.0, 1.0 + tick * 0.02, 0.25)
            self.assertTrue(fresh)
            self.assertAlmostEqual(smoother.update(target, 0.02)["wz"], 0.3)
        target, fresh, _ = connector.select_output(latest, 1.0, 1.251, 0.25)
        self.assertFalse(fresh)
        self.assertEqual(smoother.update(target, 0.0)["wz"], 0.0)

    def test_default_zero_matches_existing_command_processing(self):
        for message in ({"vx": 0.2, "wz": 0.0}, {"vx": 0.0, "wz": 0.0},
                        {"vx": 2.0, "wz": -2.0}, {"vx": 0.2, "wz": 0.3, "card_tilt": True}):
            with self.subTest(message=message):
                self.assertEqual(connector.process_vision_output(message),
                                 connector.process_vision_output(message, wz_bias=0.0))

    def test_rejects_non_finite_bias(self):
        for bias in (math.nan, math.inf, -math.inf):
            with self.subTest(bias=bias), self.assertRaisesRegex(ValueError, "bias.*finite"):
                connector.process_vision_output({"vx": 0.2, "wz": 0.0}, wz_bias=bias)

    def test_cli_default_signed_bias_and_non_finite_rejection(self):
        for extra, expected in (([], 0.0), (["--wz-bias", "0.1"], 0.1),
                                (["--wz-bias", "-0.1"], -0.1)):
            with self.subTest(extra=extra), patch.object(sys, "argv", ["connector.py", *extra]):
                self.assertEqual(connector.parse_args().wz_bias, expected)
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), patch.object(sys, "argv", ["connector.py", f"--wz-bias={value}"]):
                with self.assertRaisesRegex(SystemExit, "wz-bias must be finite"):
                    connector.main()

    def test_real_udp_loop_offsets_turns_and_keeps_stop_and_stale_zero(self):
        for bias in (0.1, -0.1):
            with self.subTest(bias=bias), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", 0))
                receiver.settimeout(0.1)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    vision_port = reservation.getsockname()[1]
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(connector.__file__)),
                     "--vision-port", str(vision_port), "--policy-port", str(receiver.getsockname()[1]),
                     "--max-wz-accel", "0", "--vision-timeout", "0.15", "--wz-bias", str(bias)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                        def wait_for(expected_wz, expected_vx, message=None):
                            deadline = time.monotonic() + 3.0
                            while time.monotonic() < deadline:
                                if message is not None:
                                    sender.sendto(json.dumps(message).encode(), ("127.0.0.1", vision_port))
                                try:
                                    packet = json.loads(receiver.recv(4096))
                                except socket.timeout:
                                    if process.poll() is not None:
                                        stdout, stderr = process.communicate()
                                        self.fail(f"Connector exited: {stdout}\n{stderr}")
                                    continue
                                if (abs(packet["wz"] - expected_wz) < 1e-9
                                        and abs(packet["vx"] - expected_vx) < 1e-9):
                                    return packet
                            self.fail(f"No UDP command vx={expected_vx}, wz={expected_wz}")

                        wait_for(0.0, 0.0)  # Startup without vision stays stopped.
                        for wz in (0.0, 0.2, -0.2, 0.48, -0.48):
                            expected = max(-0.5, min(0.5, wz + bias))
                            packet = wait_for(expected, 0.2, {"vx": 0.2, "wz": wz,
                                                             "qr": 3, "command_mode": "held"})
                            self.assertEqual(packet["qr"], 3)
                            self.assertEqual(packet["command_mode"], "held")
                        wait_for(0.0, 0.0)  # Stop sending vision: watchdog cancels bias.
                        wait_for(bias, 0.2, {"vx": 0.2, "wz": 0.0, "command_mode": "held"})
                        wait_for(0.0, 0.0, {"vx": 0.0, "wz": 0.0})
                        wait_for(0.0, 0.0, {"vx": 0.2, "wz": 0.3, "card_tilt": True})
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
