"""Safety behavior and a real HTTP-to-UDP integration without robot hardware."""
import http.client
import json
import math
import socket
import time
import unittest
from unittest.mock import Mock, patch

from remote_control.control import ControlError, ControlState
from remote_control.server import RemoteService


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.state = ControlState(clock=lambda: self.now)

    def test_forward_only_limits_and_slew(self):
        session = self.state.arm()["session"]
        self.state.command(session, 1, 5, -8)
        status = self.state.tick(0.02)
        self.assertEqual((status["target_vx"], status["target_wz"]), (0.4, -0.5))
        self.assertAlmostEqual(status["vx"], 0.02)
        self.assertAlmostEqual(status["wz"], -0.04)
        self.state.command(session, 2, -1, 0)
        self.assertEqual(self.state.snapshot()["target_vx"], 0)

    def test_exclusive_control_and_stale_sequence(self):
        session = self.state.arm()["session"]
        with self.assertRaises(ControlError):
            self.state.arm()
        self.state.command(session, 2, 0.3, 0.2)
        with self.assertRaises(ControlError):
            self.state.command(session, 1, 0.4, -0.2)
        self.assertEqual(self.state.snapshot()["target_wz"], 0.2)

    def test_timeout_invalidates_late_commands_and_requires_new_arm(self):
        session = self.state.arm()["session"]
        self.state.command(session, 1, 0.4, 0.5)
        self.state.tick(0.02)
        self.now += 0.301
        status = self.state.tick(0.02)
        self.assertFalse(status["active"])
        self.assertEqual((status["vx"], status["wz"]), (0, 0))
        with self.assertRaises(ControlError):
            self.state.command(session, 2, 0.4, 0.5)
        self.assertNotEqual(self.state.arm()["session"], session)

    def test_stop_is_immediate_and_old_requests_cannot_affect_new_owner(self):
        old = self.state.arm()["session"]
        self.state.command(old, 1, 0.4, 0.5)
        self.state.tick(0.02)
        self.state.stop(old)
        self.assertEqual(self.state.snapshot()["vx"], 0)
        new = self.state.arm()["session"]
        for operation in (lambda: self.state.command(old, 2, .4, .5),
                          lambda: self.state.stop(old)):
            with self.assertRaises(ControlError):
                operation()
        self.state.command(new, 1, 0.2, 0)
        self.assertTrue(self.state.snapshot()["active"])

    def test_invalid_values_do_not_refresh_heartbeat(self):
        session = self.state.arm()["session"]
        for value in (math.nan, math.inf, True, "0.2", None):
            self.now = 10.2
            with self.subTest(value=value), self.assertRaises(ControlError):
                self.state.command(session, 1, value, 0)
        self.now = 10.301
        self.assertFalse(self.state.snapshot()["active"])

    def test_sequence_must_be_positive_integer(self):
        session = self.state.arm()["session"]
        for seq in (True, 0, -1, 1.5, "1", 2**53):
            with self.subTest(seq=seq), self.assertRaises(ControlError):
                self.state.command(session, seq, 0.2, 0)

    def test_sign_flip_and_stalled_tick_stay_bounded(self):
        state = ControlState(yaw_sign=-1)
        session = state.arm()["session"]
        state.command(session, 1, .4, .5)
        status = state.tick(10)
        self.assertEqual(status["target_wz"], -.5)
        self.assertLessEqual(status["vx"], .04)
        self.assertGreaterEqual(status["wz"], -.08)

    def test_bad_configuration_rejected(self):
        for options in ({"max_vx": 1.1}, {"max_wz": .6}, {"timeout": math.nan},
                        {"vx_accel": 0}, {"yaw_sign": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ControlState(**options)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(1)
        self.service = RemoteService("127.0.0.1", 0, "test-token",
                                     self.receiver.getsockname()[1])
        self.service.start()
        self.port = self.service.address[1]

    def tearDown(self):
        self.service.close()
        self.receiver.close()

    def request(self, path, body=None, token="test-token", **extra):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        headers = {"Authorization": f"Bearer {token}", **extra}
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn.request("GET" if body is None else "POST", path,
                     None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    def udp_until(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            message = json.loads(self.receiver.recv(4096))
            if predicate(message):
                return message
        self.fail("expected UDP output was not received")

    def test_http_commands_reach_udp_and_expire_to_zero(self):
        code, body = self.request("/api/arm", {})
        self.assertEqual(code, 200)
        session = json.loads(body)["session"]
        code, _ = self.request("/api/command", {"session": session, "seq": 1,
                                               "vx": .3, "wz": -.2})
        self.assertEqual(code, 200)
        moving = self.udp_until(lambda m: m["vx"] > 0 and m["wz"] < 0)
        self.assertEqual(moving["vy"], 0)
        self.assertEqual(moving["qr"], -1)
        self.udp_until(lambda m: m["vx"] == m["wz"] == 0)
        code, _ = self.request("/api/command", {"session": session, "seq": 2,
                                               "vx": .3, "wz": -.2})
        self.assertEqual(code, 409)

    def test_auth_origin_and_malformed_payload(self):
        self.assertEqual(self.request("/api/arm", {}, token="wrong")[0], 401)
        self.assertEqual(self.request("/api/arm", {}, Origin="http://evil.example")[0], 403)
        self.assertEqual(self.request("/api/arm", [1])[0], 400)
        self.assertEqual(self.request("/api/command", {})[0], 400)
        self.assertEqual(self.request("/api/arm", {}, **{"Content-Length": "999999"})[0], 413)
        self.assertFalse(self.service.state.snapshot()["active"])

    def test_public_status_does_not_leak_control_session(self):
        self.request("/api/arm", {})
        code, body = self.request("/api/status")
        self.assertEqual(code, 200)
        self.assertNotIn("session", json.loads(body))

    def test_publisher_shutdown_sends_zero_and_revokes_lease(self):
        session = json.loads(self.request("/api/arm", {})[1])["session"]
        self.request("/api/command", {"session": session, "seq": 1, "vx": .3, "wz": .2})
        self.udp_until(lambda message: message["vx"] > 0)
        self.service.stopped.set()
        self.service.publisher_thread.join(timeout=1)
        self.assertFalse(self.service.publisher_thread.is_alive())
        self.udp_until(lambda message: message["vx"] == message["wz"] == 0)
        self.assertFalse(self.service.state.snapshot()["active"])
        self.assertEqual(self.request("/api/arm", {})[0], 503)

    def test_publisher_failure_revokes_control_and_rejects_new_commands(self):
        session = json.loads(self.request("/api/arm", {})[1])["session"]
        self.request("/api/command", {"session": session, "seq": 1, "vx": .3, "wz": .2})
        self.udp_until(lambda message: message["vx"] > 0)
        failed_socket = Mock()
        failed_socket.sendto.side_effect = OSError("simulated UDP failure")
        with patch.object(self.service, "sock", failed_socket):
            self.service.publisher_thread.join(timeout=1)
            self.assertFalse(self.service.publisher_thread.is_alive())
        self.assertIn("simulated UDP failure", self.service.failure)
        self.assertFalse(self.service.state.snapshot()["active"])
        self.assertEqual(self.service.state.snapshot()["vx"], 0)
        self.assertEqual(self.request("/api/arm", {})[0], 503)


if __name__ == "__main__":
    unittest.main()
