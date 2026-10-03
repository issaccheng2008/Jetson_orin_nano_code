from __future__ import annotations

import json
import math
import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision" / "jetson"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "humanoid_jetson_deploy"))

import numpy as np

from attitude_input import AttitudeInput, camera_pitch_deg
from attitude_broadcast import AttitudeBroadcaster


def policy_axes(body_pitch_deg, roll_deg=0.0):
    """policy 三轴在世界坐标里（世界 = 前/左/上）。"""
    p = math.radians(body_pitch_deg)
    r = math.radians(roll_deg)
    forward = np.array([math.cos(p), 0.0, -math.sin(p)])
    left = np.array([math.sin(p) * math.sin(r), math.cos(r), math.cos(p) * math.sin(r)])
    return forward, left, np.cross(forward, left)


def gravity_for(body_pitch_deg, roll_deg=0.0):
    """policy 帧下的 world-down：x 前、y 左、z 上。直立时 (0,0,-1)。"""
    return [-float(np.dot([0.0, 0.0, 1.0], axis))
            for axis in policy_axes(body_pitch_deg, roll_deg)]


def optical_depression(body_pitch_deg, roll_deg, mount_deg):
    """独立算法：把光轴转进世界坐标，直接取它离水平面多低。"""
    forward, left, up = policy_axes(body_pitch_deg, roll_deg)
    m = math.radians(mount_deg)
    axis = math.cos(m) * forward - math.sin(m) * up
    return math.degrees(math.asin(-axis[2]))


class CameraPitchTests(unittest.TestCase):
    def test_upright_body_leaves_the_mount_angle_alone(self):
        self.assertAlmostEqual(camera_pitch_deg([0.0, 0.0, -1.0], 45.0), 45.0, places=6)

    def test_a_forward_lean_adds_to_the_mount_angle(self):
        # 光轴相对水平面低了 45+15=60 度 —— 低头看，卡在画面里更靠下。
        self.assertAlmostEqual(camera_pitch_deg(gravity_for(15.0), 45.0), 60.0, places=4)
        self.assertAlmostEqual(camera_pitch_deg(gravity_for(-20.0), 45.0), 25.0, places=4)

    def test_the_formula_matches_the_optical_axis_for_any_attitude(self):
        for pitch in (-25.0, 0.0, 15.0):
            for roll in (-30.0, 0.0, 30.0):
                self.assertAlmostEqual(
                    camera_pitch_deg(gravity_for(pitch, roll), 45.0),
                    optical_depression(pitch, roll, 45.0), places=4)

    def test_roll_is_not_optional(self):
        # 光轴在 policy 帧里本来就带 z 分量，机身一侧倾它就偏出铅垂面。
        # 拿 Euler pitch 加安装角（这里会算成 55°）是错的，真值 46.6°。
        leaned = camera_pitch_deg(gravity_for(10.0, 30.0), 45.0)
        self.assertAlmostEqual(leaned, 46.6, delta=0.2)
        self.assertGreater(abs(leaned - 55.0), 8.0, "比 pitch+安装角 差出 8° 以上")


class AttitudeBroadcastTests(unittest.TestCase):
    def test_a_packet_lands_on_the_listening_side(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        try:
            broadcaster = AttitudeBroadcaster("127.0.0.1", port)
            broadcaster.publish(gravity_for(12.0), 3.5)
            payload, _ = listener.recvfrom(512)
            message = json.loads(payload.decode("utf-8"))
        finally:
            broadcaster.close()
            listener.close()
        self.assertAlmostEqual(message["t"], 3.5, places=3)
        self.assertAlmostEqual(camera_pitch_deg(message["g"], 45.0), 57.0, places=3)
        self.assertEqual(message["card_tilt_event_id"], 0)
        self.assertFalse(message["card_tilt_done"])

    def test_re_pose_completion_reaches_the_vision_listener(self):
        receiver = AttitudeInput(0, 45.0)
        sender = AttitudeBroadcaster("127.0.0.1", receiver.socket.getsockname()[1])
        try:
            sender.publish(gravity_for(0.0), 1.0, card_tilt_event_id=7,
                           card_tilt_done=False)
            time.sleep(0.02)
            self.assertTrue(receiver.poll())
            self.assertTrue(receiver.card_tilt_status_seen)
            self.assertEqual(receiver.card_tilt_event_id, 7)
            self.assertFalse(receiver.card_tilt_done)

            sender.publish(gravity_for(0.0), 1.1, card_tilt_event_id=7,
                           card_tilt_done=True)
            time.sleep(0.02)
            self.assertTrue(receiver.poll())
            self.assertEqual(receiver.card_tilt_event_id, 7)
            self.assertTrue(receiver.card_tilt_done)
        finally:
            sender.close()
            receiver.close()

    def test_non_finite_gravity_is_not_sent(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(0.2)
        try:
            broadcaster = AttitudeBroadcaster("127.0.0.1", listener.getsockname()[1])
            broadcaster.publish([float("nan"), 0.0, -1.0], 1.0)
            with self.assertRaises(socket.timeout):
                listener.recvfrom(512)
        finally:
            broadcaster.close()
            listener.close()


class AttitudeInputTests(unittest.TestCase):
    def _input(self, tau_s=0.4):
        return AttitudeInput(0, 45.0, tau_s=tau_s)

    def test_without_any_packet_the_pitch_stays_at_the_mount_angle(self):
        receiver = self._input()
        try:
            self.assertFalse(receiver.poll())
            self.assertEqual(receiver.value, 45.0)
        finally:
            receiver.close()

    def test_the_first_packet_is_taken_as_is(self):
        receiver = self._input()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(json.dumps({"g": gravity_for(20.0), "t": 0.0}).encode(),
                          ("127.0.0.1", receiver.socket.getsockname()[1]))
            time.sleep(0.02)
            self.assertTrue(receiver.poll())
            self.assertAlmostEqual(receiver.value, 65.0, places=3)
        finally:
            receiver.close()
            sender.close()

    def _follow_gait(self, tau_s, cycles=3.0):
        """按 50Hz 喂 1.7Hz、均值 10°、摆幅 ±35° 的机身俯仰，返回读到的极值。"""
        receiver = self._input(tau_s=tau_s)
        port = receiver.socket.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        samples = int(50.0 * cycles / 1.7)
        try:
            now = 1000.0
            for k in range(samples):
                pitch = 10.0 + 35.0 * math.sin(2.0 * math.pi * 1.7 * k / 50.0)
                sender.sendto(json.dumps({"g": gravity_for(pitch)}).encode(),
                              ("127.0.0.1", port))
                receiver.poll(now)
                now += 0.02
                time.sleep(0.0005)
            tail = []
            for k in range(samples, samples + 40):
                pitch = 10.0 + 35.0 * math.sin(2.0 * math.pi * 1.7 * k / 50.0)
                sender.sendto(json.dumps({"g": gravity_for(pitch)}).encode(),
                              ("127.0.0.1", port))
                receiver.poll(now)
                tail.append(receiver.value)
                now += 0.02
                time.sleep(0.0005)
            return receiver, tail
        except Exception:
            receiver.close()
            sender.close()
            raise

    def test_the_gait_swing_is_averaged_out_not_followed(self):
        # 步态 1.7Hz、摆 ±35°：实时跟会烂掉，低通要的正是它的均值 10°。
        receiver, tail = self._follow_gait(tau_s=1.2)
        try:
            self.assertAlmostEqual(sum(tail) / len(tail), 55.0, delta=1.0)
            # τ=0.4s 只剩 23% 衰减，残留 ±8°，比判据的容差还大 —— 这就是
            # 默认值必须长于 0.4s 的理由。
            self.assertLess(max(tail) - min(tail), 6.0)
        finally:
            receiver.close()

    def test_a_short_time_constant_would_let_the_swing_through(self):
        receiver, tail = self._follow_gait(tau_s=0.2)
        try:
            self.assertGreater(max(tail) - min(tail), 10.0)
        finally:
            receiver.close()

    def test_a_broken_packet_is_ignored(self):
        receiver = self._input()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = receiver.socket.getsockname()[1]
        try:
            sender.sendto(b"not json", ("127.0.0.1", port))
            sender.sendto(json.dumps({"g": [1.0, 2.0]}).encode(), ("127.0.0.1", port))
            sender.sendto(json.dumps({"nope": 1}).encode(), ("127.0.0.1", port))
            time.sleep(0.02)
            receiver.poll()
            self.assertEqual(receiver.value, 45.0)
            self.assertEqual(receiver.received, 0)
        finally:
            receiver.close()
            sender.close()


def parse_with(argv, parser):
    original = sys.argv
    sys.argv = list(argv)
    try:
        return parser()
    finally:
        sys.argv = original


class AttitudePortWiringTests(unittest.TestCase):
    """两端默认端口/地址对不上就静默失效 —— 这种坑不该靠上机才发现。"""

    def test_the_broadcaster_and_the_listener_agree_on_port_and_host(self):
        import main as deploy_main
        import run_policy_vision
        listener = parse_with(["run_policy_vision.py"], run_policy_vision.parse_args)
        broadcaster = parse_with(
            ["main.py", "--model", "x.onnx", "--no-plot"], deploy_main.parse_args)
        self.assertGreater(listener.attitude_port, 0)
        self.assertEqual(listener.attitude_port, broadcaster.attitude_port)
        self.assertEqual(listener.attitude_bind, broadcaster.attitude_bind)

    def test_zero_port_turns_the_channel_off_on_both_ends(self):
        import run_policy_vision
        listener = parse_with(["run_policy_vision.py", "--attitude-port", "0"],
                              run_policy_vision.parse_args)
        self.assertEqual(listener.attitude_port, 0)


class ShapeDetectorPitchTests(unittest.TestCase):
    def test_the_live_pitch_replaces_the_static_mount_angle(self):
        from shape_detector import ShapeDetector
        detector = ShapeDetector()
        try:
            before = detector.cfg["cam_pitch_deg"]
            detector.set_camera_pitch_deg(61.5)
            self.assertEqual(detector.cfg["cam_pitch_deg"], 61.5)
            self.assertNotEqual(before, 61.5)
        finally:
            pass

    def test_the_ground_square_judgement_moves_with_the_pitch(self):
        # 同一块四边形，按对的俯角反投影是正方形，按错的俯角就不是 ——
        # 这正是"机身一倾、图卡就认不出来"的来源。
        from shape_detector import ShapeDetector

        def quad_seen_at(z_cm, pitch_deg, half_cm=5.0):
            h, vfov = 32.5, math.radians(55.876)
            th = math.radians(pitch_deg)
            fy = 540.0 / (2.0 * math.tan(vfov / 2.0))
            hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * 960.0 / 540.0)
            fx = 960.0 / (2.0 * math.tan(hfov / 2.0))
            pts = []
            for wx, wz in ((-half_cm, z_cm - 5.0), (half_cm, z_cm - 5.0),
                           (half_cm, z_cm + 5.0), (-half_cm, z_cm + 5.0)):
                zc = h * math.sin(th) + wz * math.cos(th)
                pts.append([fx * wx / zc + 480.0,
                            fy * (h * math.cos(th) - wz * math.sin(th)) / zc + 270.0])
            return np.float32(pts)

        detector = ShapeDetector()
        seen_at_45 = quad_seen_at(20.0, 45.0)
        detector.set_camera_pitch_deg(45.0)
        self.assertTrue(detector._square_on_ground(seen_at_45))
        # 画面没变、相机模型变了：反投影出来的不再是正方形，边比涨上去就拒了。
        detector.set_camera_pitch_deg(25.0)
        self.assertFalse(detector._square_on_ground(seen_at_45))

    def test_the_pitch_window_the_gate_tolerates_is_pinned(self):
        # 实测：20cm 处判据只在假设俯角 33~68° 之间认得出。机身一趟摆 ±15~20°，
        # 所以是"贴边、偶尔认不出"，不是"完全不认"—— 别把它当成图卡的主因。
        from shape_detector import ShapeDetector

        def quad_seen_at(z_cm, pitch_deg, half_cm=5.0):
            h, vfov = 32.5, math.radians(55.876)
            th = math.radians(pitch_deg)
            fy = 540.0 / (2.0 * math.tan(vfov / 2.0))
            hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * 960.0 / 540.0)
            fx = 960.0 / (2.0 * math.tan(hfov / 2.0))
            pts = []
            for wx, wz in ((-half_cm, z_cm - 5.0), (half_cm, z_cm - 5.0),
                           (half_cm, z_cm + 5.0), (-half_cm, z_cm + 5.0)):
                zc = h * math.sin(th) + wz * math.cos(th)
                pts.append([fx * wx / zc + 480.0,
                            fy * (h * math.cos(th) - wz * math.sin(th)) / zc + 270.0])
            return np.float32(pts)

        detector = ShapeDetector()
        quad = quad_seen_at(20.0, 45.0)
        for assumed in (34.0, 45.0, 67.0):
            detector.set_camera_pitch_deg(assumed)
            self.assertTrue(detector._square_on_ground(quad), f"{assumed}° 该过")
        for assumed in (25.0, 75.0):
            detector.set_camera_pitch_deg(assumed)
            self.assertFalse(detector._square_on_ground(quad), f"{assumed}° 该拒")


if __name__ == "__main__":
    unittest.main()
