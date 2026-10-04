"""Hardware-free controller and real UDP bridge tests."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import contextlib
import io

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
sys.path.insert(0, str(ROOT / "humanoid_jetson_deploy"))
from connector import CommandSmoother
from policy_bridge import ConnectorClient, SteeringController
from command_source import UdpCommandSource, clamp_command
import policy_runner
from policy_runner import HumanoidPolicy
import config
import run_policy_vision
from camera_config import to_model_z
from shape_detector import WORK_H, WORK_W, ShapeDetector


def detection(error=10.0, angle=0.0, lost=0, curve=False, curve_px=0.0,
              lateral=0.0):
    return dict(fused_err=error / 52.8, fused_err_cm=error, angle_err_deg=angle,
                lost_frames=lost, curve_mode=curve, bottom_lock_valid=True,
                curve_px=curve_px, base_err_cm=lateral)


# Tests of sign, units and clamping pin the standing trim off: its defaults (+1 cm on a
# straight) would otherwise shift every expected value.
NO_TRIM = dict(bias_cm=0.0, bias_straight_cm=0.0)


class SteeringTests(unittest.TestCase):
    def test_left_point_one_requires_heading_cli_switch(self):
        with patch("sys.argv", ["run_policy_vision.py"]):
            self.assertFalse(run_policy_vision.parse_args().heading_left_0p1)
        with patch("sys.argv", ["run_policy_vision.py", "--heading-left-0p1"]):
            self.assertTrue(run_policy_vision.parse_args().heading_left_0p1)

    def test_sign_units_clamping_and_preview(self):
        # yaw_sign, preview_gain and the centre dead band pinned so this tests the
        # maths, not the defaults: the band would scale the zero-lateral-error frame
        # below by zero and hide the preview term it is here to measure.
        controller = SteeringController(**NO_TRIM, straight_gains=(1, 0, 0),
                                        steer_full_scale_cm=50, yaw_sign=-1,
                                        preview_gain=4, center_dead_cm=0.0)
        np.testing.assert_allclose(controller.command(detection(), 0.8, 0.02), [0.2, -0.1])
        self.assertGreater(controller.command(detection(-10), 0.8, 0.02)[1], 0)
        # Saturated negative is a right turn. The two sides are symmetric unless a
        # caller asks for a separate right limit.
        self.assertEqual(controller.command(detection(1000), 0.8, 0.02)[1], -0.5)
        self.assertEqual(controller.command(detection(-1000), 0.8, 0.02)[1], 0.5)
        capped = SteeringController(**NO_TRIM, straight_gains=(1, 0, 0),
                                    steer_full_scale_cm=50, yaw_sign=-1,
                                    max_wz_right=0.25)
        self.assertEqual(capped.command(detection(1000), 0.8, 0.02)[1], -0.25)
        self.assertEqual(capped.command(detection(-1000), 0.8, 0.02)[1], 0.5)
        self.assertLess(controller.command(detection(0, 30), 0.8, 0.02)[1], 0)
        reverse = SteeringController(**NO_TRIM, yaw_sign=1)
        self.assertGreater(reverse.command(detection(), 0.8, 0.02)[1], 0)

    def test_default_yaw_sign_matches_the_real_robot(self):
        """-1 turned the robot the wrong way, so the default is 1."""
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50)
        self.assertGreater(controller.command(detection(), 0.8, 0.02)[1], 0)

    def test_default_preview_is_off(self):
        """A centred robot must not be steered by the heading angle alone.

        On a curve the measured angle sits near +22 deg. At the old default
        preview_gain 4 that is 4*8*sin(22 deg) = 12 cm of steer - past the 10 cm
        full scale, with the lateral error at zero.
        """
        controller = SteeringController(**NO_TRIM)
        self.assertEqual(controller.command(detection(0.0, 22.0), 0.8, 0.02)[1], 0.0)

    def test_the_centre_dead_band_fades_yaw_authority_near_zero_error(self):
        """A centimetre or two on a 35 cm lane is inside what the near band resolves
        at all, but a small constant wz still integrates into a drift over a lap."""
        gains = dict(NO_TRIM, straight_gains=(1, 0, 0), steer_full_scale_cm=10)
        band = SteeringController(center_dead_cm=4.0, **gains)
        plain = SteeringController(center_dead_cm=0.0, **gains)
        for error, factor in ((4.0, 1.0), (2.0, 0.5), (1.0, 0.25), (0.0, 0.0)):
            with self.subTest(error=error):
                self.assertAlmostEqual(
                    band.command(detection(error), 0.8, 0.02)[1],
                    plain.command(detection(error), 0.8, 0.02)[1] * factor)
        self.assertAlmostEqual(band.command(detection(9.0), 0.8, 0.02)[1],
                               plain.command(detection(9.0), 0.8, 0.02)[1])
        self.assertLess(band.command(detection(-2.0), 0.8, 0.02)[1], 0.0)

    def test_bias_fades_in_with_curve_px_and_moves_the_zero_point(self):
        # bias_straight_cm pinned to 0 so this measures the shape of the fade, not the
        # straight end of it; the two ends are covered by the trim tests below.
        trimmed = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50,
                                     bias_cm=5.0, bias_gate_px=10.0,
                                     bias_straight_cm=0.0)
        trimmed.command(detection(0.0, curve_px=0.0), 0.8, 0.02)
        self.assertEqual(trimmed.last_err_eff, 0.0)        # straight: straights untouched
        trimmed.command(detection(0.0, curve_px=5.0), 0.8, 0.02)
        self.assertEqual(trimmed.last_err_eff, 0.0)        # under the 6 px dead band
        trimmed.command(detection(0.0, curve_px=8.0), 0.8, 0.02)
        self.assertAlmostEqual(trimmed.last_err_eff, 2.5)  # half way up the ramp
        trimmed.command(detection(0.0, curve_px=-50.0), 0.8, 0.02)
        self.assertAlmostEqual(trimmed.last_err_eff, 5.0)  # gate saturated
        self.assertGreater(trimmed.command(detection(0.0, curve_px=-50.0), 0.8, 0.02)[1], 0.0)
        # With the trim open, the loop now settles where the raw reading is -5 cm.
        self.assertEqual(trimmed.command(detection(-5.0, curve_px=-50.0), 0.8, 0.02)[1], 0.0)
        # A missing curve_px (older debug dict) must not open the gate.
        self.assertEqual(SteeringController(bias_cm=5.0, bias_straight_cm=0.0)
                         .command({**detection(0.0), "curve_px": None}, 0.8, 0.02)[1], 0.0)

    def test_the_standing_trim_defaults_to_three_and_can_be_turned_off(self):
        """It is one track's measured offset, not a property of the loop, so it stays a
        knob - 3 cm by default and 0 to disable it."""
        gains = dict(straight_gains=(1, 0, 0), curve_gains=(1, 0, 0),
                     steer_full_scale_cm=50)
        curved = detection(10.0, curve=True, curve_px=-50.0)    # gate fully open
        default = SteeringController(**gains)
        default.command(curved, 0.8, 0.02)
        self.assertAlmostEqual(default.last_err_eff, 13.0)      # the reading plus 3 cm
        off = SteeringController(bias_cm=0.0, **gains)
        off.command(curved, 0.8, 0.02)
        self.assertAlmostEqual(off.last_err_eff, 10.0)          # the reading alone

    def test_the_trim_reads_the_smoothed_curve_and_fades_between_the_two_ends(self):
        """A straight's one-frame curve_px jitters past what a real curve reads, so the
        gate reads the smoothed value. Under the dead band the trim sits exactly on the
        straight end instead of wherever the ramp happened to be, and between the dead
        band and full scale it fades from one end to the other."""
        controller = SteeringController(straight_gains=(1, 0, 0),
                                        curve_gains=(1, 0, 0), steer_full_scale_cm=50)

        def trim(raw, smooth):
            frame = detection(0.0, curve_px=raw)
            frame["curve_px_smooth"] = smooth
            controller.command(frame, 0.8, 0.02)
            return controller.last_err_eff

        # A one-frame spike that the smoothed reading calls straight takes the straight
        # end, where the raw 30 px on its own would have taken the curve end.
        self.assertAlmostEqual(trim(30.0, 2.0), controller.bias_straight_cm)
        self.assertAlmostEqual(
            trim(30.0, 9.0),
            0.5 * (controller.bias_straight_cm + controller.bias_cm))
        self.assertAlmostEqual(trim(30.0, 20.0), controller.bias_cm)
        self.assertAlmostEqual(controller.bias_straight_cm, 1.0)
        self.assertAlmostEqual(controller.bias_cm, 3.0)

    def test_dropping_the_held_command_keeps_the_loop_state(self):
        """The card stop drops the stored command and nothing else. Left set, it is
        republished as the lost-line fallback on the first invalid frame after the stop
        - the replay that was already rejected. Zeroing the loop as well would restart
        the walking process from scratch, which is what the paused-not-reset handover
        exists to avoid."""
        controller = SteeringController(steer_full_scale_cm=50)
        for _ in range(3):
            controller.command(detection(), 0.8, 0.1)
        self.assertNotEqual(controller.integral, 0.0)
        self.assertTrue(controller.err_window)

        controller.drop_held_command()
        self.assertEqual(controller.hold, (0.0, 0.0))
        self.assertEqual(controller.lost_s, 0.0)
        self.assertNotEqual(controller.integral, 0.0)
        self.assertTrue(controller.err_window)

        controller.reset(clear_hold=True)          # still the full wipe
        self.assertEqual(controller.integral, 0.0)
        self.assertEqual(controller.err_window, [])

    def test_invalid_or_lost_detection_keeps_the_speed_and_resets(self):
        """丢线只丢转向：速度取上一条好命令的速度（走路时就是 --vx）。停车的
        路径已经 drop_held_command 把 hold 清成 0，那里无效帧仍然是停。"""
        controller = SteeringController(lost_hold_s=0.0)
        controller.command(detection(), 0.8, 0.02)
        for dbg, confidence in ((detection(lost=1), 0.8), ({}, 0.8),
                                (detection(float("nan")), 0.8), (detection(), 0),
                                (detection(), float("nan"))):
            self.assertEqual(controller.command(dbg, confidence, 0.02), (0.2, 0))
            self.assertIsNone(controller.last_median)
            self.assertEqual(controller.err_window, [])
        fresh = SteeringController().command(detection(), 0.8, 0.02)
        self.assertEqual(controller.command(detection(), 0.8, 0.02), fresh)

    def test_derivative_filter_rejects_a_single_frame_spike(self):
        controller = SteeringController(curve_gains=(0, 0, 1), straight_gains=(0, 0, 1),
                                        deriv_pole=0.0, max_wz=0.5, steer_full_scale_cm=1)
        for _ in range(6):
            controller.command(detection(0.0), 0.8, 0.05)
        settled = controller.command(detection(0.0), 0.8, 0.05)[1]
        spike = controller.command(detection(40.0), 0.8, 0.05)[1]
        after = controller.command(detection(0.0), 0.8, 0.05)[1]
        # Median-of-3 discards the spike entirely, so nothing propagates into D.
        self.assertEqual(settled, 0.0)
        self.assertEqual(spike, 0.0)
        self.assertEqual(after, 0.0)

    def test_first_frames_have_no_derivative_kick(self):
        controller = SteeringController(curve_gains=(0, 0, 1), straight_gains=(0, 0, 1),
                                        steer_full_scale_cm=1)
        for _ in range(2):  # window not yet full
            self.assertEqual(controller.command(detection(30.0), 0.8, 0.05)[1], 0.0)

    def test_line_loss_fades_the_steering_but_never_the_walking_speed(self):
        """Faded, not replayed flat: the last good steering before a loss is often a
        saturated turn computed on a frame already down to one boundary, and holding
        that at full authority for the whole window is what carries the robot off.

        But the **speed** is not part of "the last command": 2026-10-04 实车，无效帧
        占四成，丢线一多就把车钉在 vx=0（C 侧的最小保持再钉半秒），整趟走不起来。
        走路时速度不掉，要停走的是门控/卡窗口那条显式 publish 0 的路。"""
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50)
        held = controller.command(detection(), 0.8, 0.02)
        self.assertEqual(held[0], 0.2)
        # Inside the default 0.2 s window: the yaw fades 0.75, 0.50, 0.25, 0.
        for expected in (0.75, 0.50, 0.25, 0.0):
            got = controller.command(detection(lost=1), 0.8, 0.05)
            self.assertAlmostEqual(got[0], held[0])          # 速度不掉
            self.assertAlmostEqual(got[1], held[1] * expected)
        for _ in range(6):  # window exhausted
            final = controller.command(detection(lost=1), 0.8, 0.05)
        self.assertEqual(final, (0.2, 0.0))
        # A NaN dt must not stall the accumulator and latch the hold for ever.
        self.assertEqual(controller.command(detection(lost=1), 0.8, float("nan")), (0.2, 0.0))

    def test_an_impossible_lateral_offset_counts_as_line_loss(self):
        """Measured on the robot: standing still with no card in view, err sat at
        -35.9 cm for twenty seconds, rock steady. The lane is 35 cm wide, so the
        near band was reporting a line half a metre off - it had locked onto
        something else, and steering on it is a hard turn the wrong way."""
        # Off unless asked for: defaulting it on changes line following.
        self.assertNotEqual(
            SteeringController().command(detection(lateral=-35.9), 0.8, 0.05), (0.0, 0.0))
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50,
                                        max_lateral_cm=17.5)
        for lateral in (17.4, -17.4):
            self.assertNotEqual(
                controller.command(detection(lateral=lateral), 0.8, 0.02), (0.0, 0.0))
        self.assertIsNone(controller.rejected_lateral)
        # Past the lane half-width it is loss: the steering fades out, the walking
        # speed stays（丢线不停车，见上一条）。
        held = controller.command(detection(), 0.8, 0.02)
        self.assertEqual(held[0], 0.2)
        first = controller.command(detection(lateral=-35.9), 0.8, 0.05)
        self.assertAlmostEqual(first[0], held[0])
        self.assertAlmostEqual(first[1], held[1] * 0.75)
        self.assertAlmostEqual(controller.rejected_lateral, -35.9)
        for _ in range(6):
            final = controller.command(detection(lateral=-35.9), 0.8, 0.05)
        self.assertEqual(final, (0.2, 0.0))

    def test_right_turns_are_capped_lower_than_left(self):
        """Negative wz is a right turn on the wire, and right is limited to half."""
        controller = SteeringController(**NO_TRIM, straight_gains=(1, 0, 0),
                                        steer_full_scale_cm=1, max_wz=0.5,
                                        max_wz_right=0.25)
        self.assertEqual(controller.command(detection(1000), 0.8, 0.02)[1], 0.5)     # left
        self.assertEqual(controller.command(detection(-1000), 0.8, 0.02)[1], -0.25)  # right
        # A command inside the cap is untouched on both sides.
        left = controller.command(detection(0.2), 0.8, 0.02)[1]
        right = controller.command(detection(-0.2), 0.8, 0.02)[1]
        self.assertAlmostEqual(left, -right)

    def test_the_stop_does_not_leave_a_command_to_replay(self):
        """reset() keeps `hold` for lost-line patience, so while the controller is
        idle through a card stop it still holds the pre-stop (vx, wz). If the first
        frame after the stop is invalid, command() would republish exactly the
        pre-stop command - the replay that was already tried and rejected."""
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=10)
        moving = controller.command(detection(20.0), 0.9, 0.05)   # a curve command
        self.assertEqual(moving, (0.2, 0.5))
        # Idle through the stop; clear_hold is what run_policy_vision passes.
        for _ in range(5):
            controller.reset(clear_hold=True)
        self.assertEqual(controller.hold, (0.0, 0.0))
        self.assertEqual(controller.command(detection(lost=1), 0.9, 0.05), (0.0, 0.0))
        # And without it, the lost-line case still gets the held steering back -
        # faded by the 0.05 s spent inside the 0.2 s window, not flat.
        # 速度不掉：走路时 vx 保持上一条好命令的 0.2。
        kept = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=10)
        kept.command(detection(20.0), 0.9, 0.05)
        kept.reset()
        got = kept.command(detection(lost=1), 0.9, 0.05)
        self.assertAlmostEqual(got[0], 0.2)
        self.assertAlmostEqual(got[1], 0.375)

    def test_invalid_settings_are_rejected(self):
        for kwargs in (dict(max_wz=1.5), dict(vx=float("nan")), dict(yaw_sign=0),
                       dict(steer_full_scale_cm=0), dict(step_len_cm=-1),
                       dict(lost_hold_s=-1.0), dict(deriv_pole=1.0),
                       dict(deriv_pole=-0.1), dict(bias_cm=float("nan")),
                       dict(bias_gate_px=0.0), dict(max_lateral_cm=-1.0),
                       dict(max_wz_right=0.0), dict(max_wz_right=-0.1),
                       dict(max_wz_right=0.6),
                       dict(single_line_gain=0.0),
                       dict(single_line_gain=-1.0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SteeringController(**kwargs)

    def test_the_single_line_gain_only_applies_on_a_curve(self):
        """The gain is for the frames where one boundary is all the detector has. It
        must leave a normally-detected frame alone, and 1.0 - the value the robot
        ships with - must leave the loop exactly as it was."""
        gains = dict(straight_gains=(1, 0, 0), curve_gains=(1, 0, 0),
                     steer_full_scale_cm=50)
        doubled = SteeringController(single_line_gain=2.0, **gains)
        plain = SteeringController(**gains)

        both = detection(curve=True)
        single = detection(curve=True)
        single["single_line"] = True
        single_straight = detection(curve=False)
        single_straight["single_line"] = True
        # 2026-10-02 实车跑出去的那一类：不是单线，两条边都看到了，只是偏出去
        # 超过对称容差所以锁无效。这个增益以前在这类帧上一次都没生效过。
        unlocked = detection(curve=True)
        unlocked["bottom_lock_valid"] = False
        unlocked_straight = detection(curve=False)
        unlocked_straight["bottom_lock_valid"] = False

        self.assertAlmostEqual(doubled.command(both, 0.8, 0.02)[1],
                               plain.command(both, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(single, 0.8, 0.02)[1],
                               2.0 * plain.command(single, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(single_straight, 0.8, 0.02)[1],
                               plain.command(single_straight, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(unlocked, 0.8, 0.02)[1],
                               2.0 * plain.command(unlocked, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(unlocked_straight, 0.8, 0.02)[1],
                               plain.command(unlocked_straight, 0.8, 0.02)[1])
        self.assertAlmostEqual(SteeringController(**gains).command(single, 0.8, 0.02)[1],
                               plain.command(both, 0.8, 0.02)[1])


class SmoothedStopReachesThePolicyTests(unittest.TestCase):
    def test_smoothed_zero_is_exactly_zero_after_the_policy_clamp(self):
        """policy_runner picks the stride with np.all(command == 0.0)."""
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        for _ in range(30):
            smoother.update({"vx": 0.4, "vy": 0.0, "wz": -0.5, "qr": -1}, 0.02)
        zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1}
        for _ in range(30):
            output = smoother.update(zero, 0.02)
        command = clamp_command([output["vx"], output["vy"], output["wz"]])
        self.assertEqual(command.dtype, np.float32)
        self.assertTrue(np.all(command == 0.0))
        # Negative control: an asymptotic filter would fail exactly here.
        self.assertFalse(np.all(clamp_command([1e-9, 0.0, 0.0]) == 0.0))

    def test_exact_legacy_wire_format(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            client = ConnectorClient(port=receiver.getsockname()[1])
            try:
                client.publish(0.4, -0.2)
                self.assertEqual(json.loads(receiver.recv(1024)),
                                 dict(vx=0.4, vy=0.0, wz=-0.2, qr=-1))
            finally:
                client.close()
            self.assertEqual(json.loads(receiver.recv(1024)),
                             dict(vx=0.0, vy=0.0, wz=0.0, qr=-1))


class VisionEntryPointTests(unittest.TestCase):
    def test_visible_card_cannot_restart_after_classifier_cooldown(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}
        reads = [0]
        clock = [0.0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 60:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        shape.update.side_effect = lambda *a, **k: (
            3 if reads[0] in (2, 35) else None,
            {"presence": True, "presence_cy_frac": 0.9, "shape": "square"},
        )
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--wz-mode", "continuous",
                               "--shape-every", "1", "--card-hold-ms", "5000",
                               "--card-vote-frames", "1", "--card-tilt-ms", "0"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        published = client_cls.return_value.publish.call_args_list
        self.assertEqual(len({call.kwargs["event_id"] for call in published
                              if "event_id" in call.kwargs}), 1)
        self.assertNotIn("event_id", published[-1].kwargs)
        self.assertEqual(published[-1].args[2], -1)

    def test_the_re_pose_window_uses_the_static_mount_angle(self):
        """重摆期间 STM32 上报的是**故意冻结**的旧姿态（策略不能看见内部加的偏置），
        那不是相机真实所在：重摆把机身扳回安装姿态，所以这一整段该用静态安装角。
        实测 43cm 触发距离，地面方框判据只在假设俯角 [38.6, 59.0] 内认卡 —— 冻结
        的 25° 在外面，喂进去等于把重摆要解决的问题原样搬回来。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        shape.action_map = {}
        # 卡一直在、却一直认不出形状：停车窗口整段开着，正是要测的那一段。
        shape.update.return_value = (None, {"presence": True, "presence_cy_frac": 0.9})
        attitude = Mock()
        attitude.value = 25.0          # 冻结值 = 后仰 20° 时相机只朝下 25°
        reads = [0]
        clock = [0.0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 15:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-every-stopped", "1",
                               "--card-tilt-ms", "500"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("attitude_input.AttitudeInput", return_value=attitude),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)

        pitches = [call.args[0] for call in shape.set_camera_pitch_deg.call_args_list]
        self.assertEqual(pitches[0], 25.0)          # 停车之前：实时姿态照用
        self.assertTrue(pitches[1:], "the stop window never opened")
        self.assertTrue(all(pitch == 45.0 for pitch in pitches[1:]), pitches)

    def test_headless_runner_publishes_detection_and_zero_on_capture_failure(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection(angle=35))
        clock = [0.0]
        def tick():
            clock[0] += 0.03
            return clock[0]
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--no-shape-detect"]),
            patch.object(run_policy_vision.signal, "signal") as signals,
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reads = 0
            def read():
                nonlocal reads
                reads += 1
                if reads <= 3:
                    return True, frame
                signals.call_args.args[1](None, None)
                return False, None
            camera.read.side_effect = read
            self.assertEqual(run_policy_vision.main(), 0)
            calls = client_cls.return_value.publish.call_args_list
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[2].args[0], 0.2)
            # 默认 heading 模式：方向误差触发，并在这三帧期间保持同一命令。
            self.assertTrue(any(c.args[1] != 0.0 for c in calls[:3]), [c.args for c in calls])
            self.assertTrue(all(c.kwargs.get("command_mode") == "held" for c in calls[:3]))
            self.assertEqual(len({c.args[:2] for c in calls[:3]}), 1)
            self.assertEqual(calls[3].args, (0.0, 0.0, -1))
            client_cls.return_value.close.assert_called_once()
            camera.release.assert_called_once()

    def test_no_red_detect_turns_off_both_red_paths(self):
        """Red reaches line following twice - _detect_red_bar, and the row blocker
        that blanks the band scan and the bottom lock - and both hang off the one
        flag, so the switch has to reach the detector the runner actually builds."""
        from line_detector_v1_warp import LineDetector
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        detector.red_detect_enable = True
        clock = [0.0]
        def tick():
            clock[0] += 0.03
            return clock[0]
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--no-shape-detect", "--no-red-detect"]),
            patch.object(run_policy_vision.signal, "signal") as signals,
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reads = 0
            def read():
                nonlocal reads
                reads += 1
                if reads <= 3:
                    return True, frame
                signals.call_args.args[1](None, None)
                return False, None
            camera.read.side_effect = read
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertFalse(detector.red_detect_enable)
        self.assertTrue(LineDetector(1280, 720).red_detect_enable)  # on unless asked

    def test_the_stop_leaves_no_command_for_the_first_frame_to_replay(self):
        """Nothing may publish motion once the stop has opened - least of all the
        pre-stop (vx, wz) coming back as the lost-line hold. Left set, `hold` is exactly
        what the controller returns when the first frame after the card is invalid, and
        that is the replay that was already tried and rejected."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        seen = [0]

        def process(_frame, *, dt=None):
            seen[0] += 1
            if seen[0] <= 2:
                return (0, 0, 0.9, None, detection())
            return (0, 0, 0.0, None, detection(lost=1))    # conf 0 and lost

        detector.process.side_effect = process
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True, "presence_cy_frac": 0.9,
                    "shape": "square"})
            if seen[0] >= 2
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]

        def tick():
            clock[0] += 0.1
            return clock[0]

        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--wz-mode", "continuous",
                               "--shape-every", "1", "--card-vote-frames", "1"]),
            patch.object(run_policy_vision.signal, "signal") as signals,
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            count = [0]

            def read():
                count[0] += 1
                if count[0] <= 45:
                    return True, frame
                signals.call_args.args[1](None, None)
                return False, None

            camera.read.side_effect = read
            self.assertEqual(run_policy_vision.main(), 0)

        published = client_cls.return_value.publish.call_args_list
        self.assertGreater(published[0].args[0], 0.0)      # read 1 drove, on a good frame
        self.assertNotEqual(published[0].args[1], 0.0)
        # Read 2 raises the cue, so from there on nothing may move again - including
        # the frames after the window closes, where the detection is invalid.
        for index, call in enumerate(published[1:], start=1):
            with self.subTest(publish=index):
                self.assertEqual(call.args[:2], (0.0, 0.0))

    def test_card_event_is_held_but_qr_clears_when_card_disappears(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        # 投票制下动作只在停车窗口里出，所以这一帧要同时满足停车闸（presence +
        # cy 过线）。触发帧不投票，下一次检测定案；将等待显式置0，把这条
        # 测的"event 生命周期"和投票分开。
        shape.update.side_effect = [
            (3, {"presence": True, "presence_cy_frac": 0.9, "shape": "square"})
        ] * 2 + [(None, {})]
        shape.action_map = {"square": 3}
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-hold-ms", "3000",
                               "--card-vote-frames", "1", "--card-every-stopped", "1",
                               "--card-tilt-ms", "0", "--card-settle-ms", "0"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reads = 0
            def read():
                nonlocal reads
                reads += 1
                if reads <= 3:
                    return True, frame
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            camera.read.side_effect = read
            self.assertEqual(run_policy_vision.main(), 0)
            published = [c.args[2] for c in client_cls.return_value.publish.call_args_list]
            # qr reports the current recognition; the event is retained for delivery.
            self.assertEqual(published, [-1, 3, -1, -1])
            events = [call.kwargs for call in client_cls.return_value.publish.call_args_list]
            self.assertNotIn("event_id", events[0])
            events = events[1:]
            self.assertGreater(events[0]["event_id"], 0)
            self.assertTrue(all(event["event_id"] == events[0]["event_id"] for event in events))
            self.assertTrue(all(event["event_action"] == 3 for event in events))
            # Identifying starts the action window, so it stands from that frame on.
            for call in client_cls.return_value.publish.call_args_list:
                self.assertEqual(call.args[:2], (0.0, 0.0))
            self.assertEqual(shape.update.call_args_list[0].kwargs["lane_offset_cm"], 0.0)

    def test_speed_resumes_only_after_the_hold_window(self):
        """A box stops it; naming the shape opens --card-hold-ms; then it drives again."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        seen = {"card": False}
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (3, {"presence": True, "presence_cy_frac": 0.9, "shape": "square"})
            if seen["card"]
            else (None, {"presence": False, "presence_cy_frac": None}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1                      # 10 Hz vision
            seen["card"] = 1 < reads[0] < 4      # present on frames 2 and 3 only
            if reads[0] > 60:                    # 6 s of mocked time; stop the loop
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--wz-mode", "continuous",
                               "--shape-every", "1", "--card-vote-frames", "1",
                               "--card-hold-ms", "5000", "--card-stop-ms", "3000",
                               "--card-tilt-ms", "0", "--card-settle-ms", "0",
                               "--card-every-stopped", "1"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        published = client_cls.return_value.publish.call_args_list
        # read N lands at t = N * 0.1, so publish[i] is read i+1.
        # read 1 drives; read 2 triggers the stop; read 3 casts the first vote.
        self.assertGreater(published[0].args[0], 0.0)
        self.assertEqual(published[0].args[2], -1)
        self.assertEqual(published[1].args[:2], (0.0, 0.0))
        self.assertEqual(published[1].args[2], -1)
        self.assertEqual(published[2].args[2], 3)
        # The window is 5000 ms at 10 Hz, so it lifts around publish 52 (read 53).
        # Asserted as a window rather than an exact index: the mocked clock
        # accumulates 0.1 fifty-odd times and lands either side of the boundary.
        resumed = next(i for i, call in enumerate(published)
                       if i > 1 and call.args[0] > 0.0)
        self.assertIn(resumed, (52, 53, 54))
        self.assertEqual(published[resumed - 1].args[:2], (0.0, 0.0))
        self.assertEqual(published[resumed - 1].kwargs["event_action"], 3)
        self.assertEqual(published[resumed].args[2], -1)      # released with the resume
        # The whole stop asks the robot to hold its legs straight: that is the pose the
        # card geometry is calibrated for, and the policy's own stopped pose is pitched
        # back about 20 degrees. Nothing before or after the window asks for it.
        # The mechanism in use: main.py turns the two edges of this into the two
        # action requests the STM32 re-poses the body on. The re-pose is only for
        # reading, so it comes off the moment the shape is named — the action then
        # runs after the tilt flag is released. Only the trigger frame requests tilt;
        # the next frame names the shape in this zero-wait lifecycle fixture.
        self.assertIn("event_id", published[2].kwargs)
        self.assertTrue(published[1].kwargs["card_tilt"])
        for index, call in enumerate(published):
            if index != 1:
                self.assertFalse(call.kwargs.get("card_tilt", False))
        # The superseded one is off unless --hold-upright is passed: it drives the same
        # joints, so exactly one of the two may be on.
        for call in published:
            self.assertFalse(call.kwargs.get("hold_upright", False))
        # It drives again the moment the window closes - there is no blind clearance
        # stage. The controller was idle all through the stop and its loop state is kept
        # (the stop is a pause), so this lands near a fresh controller's value for the
        # same frame rather than exactly on it. What has to hold is that nothing from
        # the card-corrupted frames survived: command() was never called while stopped.
        fresh = SteeringController().command(detection(), 0.8, 0.1)
        self.assertAlmostEqual(published[resumed].args[1], fresh[1], delta=1e-3)
        # 起步不再有缓冲段：--card-slow-vx / --card-resume-ms 已经拿掉了，所以
        # 释放的那一帧就是全速。
        self.assertAlmostEqual(published[resumed].args[0], fresh[0])

    def test_a_distant_card_does_not_stop_and_a_flicker_does_not_re_trigger(self):
        """The cue drops out while walking. One absent call used to re-arm the stop,
        so the robot crept forward and stopped again, then sat there for good.
        卡还在远处时速度一动不动（--card-slow-vx 已经拿掉，全程就是 --vx）。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        # reads 2-5: a card, far away (centroid above the trigger line); read 4 drops out
        plan = {2: 0.40, 3: 0.40, 5: 0.40, 6: 0.40, 7: 0.40, 8: 0.90, 9: 0.90}
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True,
                    "presence_cy_frac": plan[reads[0]]})
            if reads[0] in plan
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 30:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-trigger-frac", "0.75",
                               "--card-clear-calls", "4"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        published = client_cls.return_value.publish.call_args_list
        self.assertAlmostEqual(published[0].args[0], 0.2)      # read 1: no card
        self.assertAlmostEqual(published[1].args[0], 0.2)      # read 2: card seen, 不减速
        self.assertAlmostEqual(published[4].args[0], 0.2)      # read 5: 闪断之后照样走
        self.assertAlmostEqual(published[6].args[0], 0.2)      # read 7: centroid still high
        self.assertAlmostEqual(published[7].args[0], 0.0)      # read 8: centroid low -> stops
        self.assertEqual(out.getvalue().count("stand still"), 1)  # triggered exactly once
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        _, _, confidence, _, debug = detector.process(np.full((720, 1280, 3), 255, np.uint8))
        self.assertEqual(SteeringController().command(debug, confidence, 0.03), (0, 0))

    def test_the_default_no_longer_slows_down_or_steers_for_a_card(self):
        """--card-slow-vx / --card-slow-wz 默认都是 0 = 关。0 要是照样进
        `min(vx, 0)`，车会直接停在卡片前面 —— 这是把默认值改成 0 时最容易踩的坑。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}
        # 卡一直在视野里，但质心在高处（还远），够不着 0.75 那条停车线
        shape.update.return_value = (None, {"presence": True, "card_found": True,
                                            "presence_cy_frac": 0.40})
        clock, reads = [0.0], [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 8:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-trigger-frac", "0.75"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        published = client_cls.return_value.publish.call_args_list
        for index, call in enumerate(published):
            if call.args[0] <= 0.0:      # 收尾那个相机读失败的零包
                break
            with self.subTest(publish=index):
                self.assertAlmostEqual(call.args[0], 0.2)     # 不再减速
        self.assertNotAlmostEqual(published[1].args[1], -0.2)  # 不再固定转角

    def test_the_card_is_decided_by_voting_after_the_stop(self):
        """停下之后按票数定案：哪一类票多就是哪一类。不是看哪一帧先"确认" ——
        单帧靠不住（模糊、步态抖动、半张卡出画面），多数票才靠得住。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3, "triangle": 6}
        plan = {2: "triangle", 3: "square", 4: "square"}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "presence_cy_frac": 0.95,
                    "shape": plan.get(reads[0])}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 10:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-every-stopped", "1",
                               "--card-tilt-ms", "0", "--card-settle-ms", "0",
                               "--card-vote-frames", "3"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        qr = [call.args[2] for call in client_cls.return_value.publish.call_args_list]
        # read 1 触发不投票；read 2 投 triangle；read 3/4 投 square，定案 square
        self.assertEqual(qr[0], -1)
        self.assertEqual(qr[1], -1)
        self.assertEqual(qr[3], 3)
        self.assertIn("square", out.getvalue())

    def test_one_detection_casts_one_vote(self):
        """票数必须等于检测次数。检测隔帧跑（--card-every-stopped 2），
        而投票那段在检测块外面 —— card_dbg 不清空就会拿上一帧的结果再投一次，
        票数正好翻倍，把"要 20 票"变成其实只看了 10 帧。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.return_value = (
            None, {"presence": True, "presence_cy_frac": 0.95, "shape": "square"})
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 9:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-every-stopped", "2",
                               "--card-tilt-ms", "0", "--card-settle-ms", "0", "--card-stop-ms", "500",
                               "--card-vote-frames", "99"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)

        # 触发在第 1 帧（t=0.1），窗口 500ms 到 t=0.6 关。窗口内检测只跑
        # 第 1 帧仅触发不投票，第 2/4 帧共两票。
        # 不清 card_dbg 会在第 3/5 帧重复计票。
        self.assertIn("票 2 张", out.getvalue())

    def test_the_box_width_decides_the_stop_when_there_is_a_box(self):
        """cy is an angle, so the body pitching moves it without the card moving at
        all. A 2026-10-01 lap pitched from +16 to -24 degrees; one card read cy=0.877
        where the configured geometry says 13 cm, while its 121 px box says 27 cm.
        The box width survives that: fx*10/zc, and zc moves only +-6% across 20..60
        degrees of pitch, against cy swinging threefold. So when a box exists, the
        width decides and cy is only logged.

        Both wrong-way cases are driven here: cy past the line with a far box must
        NOT stop, and cy short of the line with a close box MUST.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}

        def quad_of_width(w):
            cx, cy = 480.0, 400.0
            return np.array([[cx - w / 2, cy - 30], [cx + w / 2, cy - 30],
                             [cx + w / 2, cy + 30], [cx - w / 2, cy + 30]], np.float32)

        def update(*_a, **_k):
            # cy is 0.90 the whole way -- past the 0.559 line from the very first call.
            # Only the box width moves, from far (40 px) to close (140 px); the 30 cm
            # threshold is 117.7 px.
            w = 40.0 if reads[0] < 7 else 140.0
            return None, {"presence": True, "card_found": True,
                          "presence_cy_frac": 0.90, "quad_work": quad_of_width(w)}

        shape.update.side_effect = update
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 12:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-trigger-dist-cm", "30"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        log = out.getvalue()
        # cy never moves and is past its own line from read 2 on, so a cy-driven stop
        # would have fired early. It must fire only once the box is actually close.
        self.assertEqual(log.count("stand still"), 1)
        fire = [ln for ln in log.splitlines() if "stand still" in ln][0]
        self.assertIn("px", fire)
        self.assertIn("cy=0.90", fire)

    def test_a_strong_presence_cue_stops_it_even_with_no_box(self):
        """Phase one, on the board: quad=512g0v0 for the whole approach while the cue
        read 4.2, 5.5, 7.6 - plainly a card, and it was driven straight past."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": False, "presence_cy_frac": 0.30,
                    "presence_cue": 4.0})
            if reads[0] == 2 else
            (None, {"presence": True, "card_found": False, "presence_cy_frac": 0.80,
                    "presence_cue": 5.5})
            if reads[0] >= 3 else
            (None, {"presence": False, "card_found": False, "presence_cy_frac": None,
                    "presence_cue": 0.0}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 20:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-trigger-frac", "0.5"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertEqual(out.getvalue().count("stand still"), 1)

    def test_a_shape_is_not_acted_on_until_the_card_reaches_the_trigger_line(self):
        """Phase two waits for phase one. Acting on a card 50 cm away classified it
        from a small warp: rules called it diamond then triangle while hu read circle
        at distance 0.01-0.04 throughout."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        plan = {2: 0.20, 3: 0.30, 4: 0.40, 5: 0.55, 6: 0.70}
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (3, {"presence": True, "card_found": True,
                 "presence_cy_frac": plan[reads[0]], "shape": "square"})
            if reads[0] in plan
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 20:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-trigger-frac", "0.5",
                               "--card-vote-frames", "1", "--card-tilt-ms", "0",
                               "--card-settle-ms", "0", "--card-every-stopped", "1"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        # reads 2-4 are short of the line and must be ignored; read 5 reaches it.
        # The trigger frame cannot vote; the next frame settles the one-vote fixture.
        # The default 20
        # would need a stop window this mock does not have.
        qr = [call.args[2] for call in client_cls.return_value.publish.call_args_list]
        self.assertEqual(qr[1:4], [-1, -1, -1])
        self.assertEqual(qr[4], -1)
        self.assertEqual(qr[5], 3)
        self.assertIn("qr=3", out.getvalue())

    def test_a_card_already_driven_past_cannot_trigger_a_second_stop(self):
        """The card just handled is still in frame, low and below the trigger line,
        and presence is armed-gated so it reads false right after the action - which
        cleared the one-shot latch and stopped the robot a second time mid-curve."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        # 2-3 far -> arms; 4 low -> triggers; 5-8 presence gone -> latch clears;
        # 9-10 the same card at the bottom of the frame, which is not an approach.
        plan = {2: 0.30, 3: 0.30, 4: 0.90, 9: 0.84, 10: 0.86}
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True,
                    "presence_cy_frac": plan[reads[0]]})
            if reads[0] in plan
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 30:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertEqual(out.getvalue().count("stand still"), 1)


class SingleLineTrackingTests(unittest.TestCase):
    """On a curve one boundary can shrink out of the field of view, and the other one
    then has to place the centre on its own - which means it has to know which
    boundary it is. That is what the left/right flags report."""

    @staticmethod
    def _detector():
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        detector.red_detect_enable = False
        return detector

    def _band(self, detector, stripes, hint_width=140.0):
        gray = np.zeros((detector.bird_h, detector.bird_w), np.uint8)
        for x in stripes:
            gray[:, x:x + 10] = 255
        bgr = np.zeros((detector.bird_h, detector.bird_w, 3), np.uint8)
        return detector._scan_band_midline(
            gray, bgr, 128, False, 160.0, hint_width,
            detector.band_low_y0 / float(detector.bird_h),
            detector.band_low_y1 / float(detector.bird_h), 10, 5)

    @staticmethod
    def _arc_lane(k):
        """A two-line lane whose far end is k*419^2 px left of its near end, so a
        positive k is a left curve by the code's own convention (curve_px < 0)."""
        import cv2
        image = np.full((720, 1280, 3), 255, np.uint8)
        for x0 in (560, 720):
            points = np.array([[x0 - int(k * (719 - y) ** 2), y]
                               for y in range(280, 720)], np.int32)
            cv2.polylines(image, [points], False, (0, 0, 0), 20)
        return image

    def test_legacy_angle_gain_cannot_make_rejected_geometry_actionable(self):
        """Keep the old camera-space arc as a rejection regression: adding an
        angle weight cannot promote an unsupported pair into a measurement."""
        for angle_gain in (self._detector().pix_angle_gain, 0.0, 10.0):
            detector = self._detector()
            detector.pix_angle_gain = angle_gain
            image = self._arc_lane(0.00057)
            for _ in range(detector.startup_settle_frames + 4):
                _, _, confidence, _, debug = detector.process(image, dt=.1)
            self.assertFalse(debug["measurement_valid"])
            self.assertFalse(debug["heading_valid"])
            self.assertEqual(confidence, 0.0)

    def test_a_zero_width_hint_does_not_shrink_the_inferred_centre(self):
        """The startup window hands the scan a width of 0, and 0 used to become
        max(0, min_track_width) = 24 px. Sizing the inferred half-lane off that put the
        centre 12 px from the line instead of 70 - 19 cm right of the lane, which
        steers right - and the 24 was written back as the next frame's width, after
        which the pair gate rejected the real span and it never grew back."""
        detector = self._detector()
        result = self._band(detector, [200], hint_width=0.0)
        self.assertAlmostEqual(result["center_px"], 204.5 - 70.0)
        self.assertAlmostEqual(result["lane_width_px"], detector.lane_width_init_px)
        # Feeding the reported width back, as process() does, is a fixed point.
        again = self._band(detector, [200], hint_width=result["lane_width_px"])
        self.assertAlmostEqual(again["center_px"], result["center_px"])
        self.assertAlmostEqual(again["lane_width_px"], detector.lane_width_init_px)

    def test_the_bands_report_which_boundary_they_saw(self):
        detector = self._detector()
        one = self._band(detector, [200])            # run centre 204.5, band hint 160
        self.assertFalse(one["left_seen"])
        self.assertTrue(one["right_seen"])
        self.assertEqual(one["single_side"], "right")
        self.assertEqual(one["pair_ratio"], 0.0)

        both = self._band(detector, [100, 240])
        self.assertTrue(both["left_seen"])
        self.assertTrue(both["right_seen"])
        self.assertIsNone(both["single_side"])

    def test_only_a_paired_row_may_report_a_width(self):
        """The runs here are 140 px apart. A row that saw one boundary is quoting the
        memory, and the memory in turn may only take a measurement."""
        detector = self._detector()
        both = self._band(detector, [100, 240])
        self.assertAlmostEqual(both["lane_width_px"], 140.0)
        self.assertEqual(both["pair_ratio"], 1.0)

    def test_the_inferred_centre_cannot_run_away_from_the_frame_middle(self):
        """The side is read off the constant frame middle, so the centre lands w/2 to
        that side of it - within half a lane (<= 150 px) wherever the run is. Siding
        off the tracked centre, which can sit anywhere, is what let the old clamp report
        the frame edge as a measurement: one row handing the fusion +/-160 px."""
        detector = self._detector()
        for start in (0, 40, 100, 159, 160, 220, 300, 310):
            with self.subTest(start=start):
                result = detector._infer_center_from_single_run(
                    (start, start + 9), detector.lane_width_init_px, 0, detector.bird_w - 1)
                run_center = start + 4.5
                side = "left" if run_center < detector.center_x else "right"
                self.assertEqual(result["side"], side)
                self.assertAlmostEqual(
                    result["center_px"],
                    run_center + (70.0 if side == "left" else -70.0))
                self.assertLessEqual(
                    abs(result["center_px"] - detector.center_x),
                    detector.max_track_width / 2.0)

    def test_single_line_without_width_support_cannot_supply_curve_heading(self):
        """P1 does not infer a lane heading from an unassociated single edge."""
        import cv2
        for dx in (0, 160):
            detector = self._detector()
            image = np.full((720, 1280, 3), 255, np.uint8)
            cv2.line(image, (640, 719), (640 + dx, 300), (0, 0, 0), 20)
            for _ in range(detector.startup_settle_frames + 3):
                _, _, confidence, _, debug = detector.process(image, dt=.1)
            self.assertFalse(debug["measurement_valid"])
            self.assertFalse(debug["heading_valid"])
            self.assertFalse(debug["preview_valid"])
            self.assertFalse(debug["curve_mode"])
            # 单边兜底同样要宽度支撑：没有近期配对帧，连它也拿不到方向。
            self.assertFalse(debug["single_edge_valid"])
            self.assertEqual(confidence, 0.0)

    def test_the_single_edge_fallback_reads_which_way_the_one_line_goes(self):
        """丢线兜底用的是单线方向：远处偏左 → 正 heading（左转），镜像同理。"""
        detector = self._detector()
        for slope, sign in ((0.6, 1), (-0.6, -1)):
            with self.subTest(slope=slope):
                gray = np.zeros((detector.bird_h, detector.bird_w), np.uint8)
                for y in range(detector.band_low_y0, detector.band_low_y1):
                    x = int(100 + (y - detector.band_low_y0) * slope)
                    gray[y, x:x + 10] = 255
                bgr = np.zeros((detector.bird_h, detector.bird_w, 3), np.uint8)
                near = detector._scan_band_midline(
                    gray, bgr, 128, False, 160.0, 140.0,
                    detector.band_low_y0 / float(detector.bird_h),
                    detector.band_low_y1 / float(detector.bird_h), 10, 5)
                self.assertIsNotNone(near)
                self.assertEqual(near["single_side"], "left")
                out = detector._fit_single_edge_heading(near)
                self.assertTrue(out["single_edge_valid"])
                self.assertEqual(out["single_edge_side"], "left")
                self.assertEqual(out["single_edge_near_cm"], near["center_cm"])
                self.assertEqual(out["single_edge_z_cm"], near["dist_cm"])
                if sign > 0:
                    self.assertGreater(out["single_edge_heading_deg"], 0.0)
                else:
                    self.assertLess(out["single_edge_heading_deg"], 0.0)

    def test_a_one_frame_unsupported_edge_does_not_reach_the_bias_gate(self):
        import cv2
        image = np.full((720, 1280, 3), 255, np.uint8)
        cv2.line(image, (640, 719), (760, 300), (0, 0, 0), 20)
        _, _, confidence, _, debug = self._detector().process(image, dt=.1)
        self.assertFalse(debug["measurement_valid"])
        self.assertEqual(debug["curve_px_smooth"], 0.)
        self.assertEqual(confidence, 0.)

    def test_a_single_boundary_band_keeps_most_of_its_weight(self):
        """A band that saw one boundary cannot pair, and pair_ratio cannot tell that
        apart from a pair that failed to lock - pair_ratio + single_ratio is 1 by
        construction, so the old pair_ratio penalty fired on exactly these frames. The
        weight it left behind is also the rate the error EMA is updated at."""
        detector = self._detector()
        one = self._band(detector, [200])
        self.assertGreater(one["conf"], 0.6)
        self.assertGreater(detector._result_quality_weight(one), 0.8)
        # Both boundaries in frame still weigh the most: a measured midpoint.
        both = self._band(detector, [100, 240])
        self.assertAlmostEqual(detector._result_quality_weight(both), 1.0)

    def test_repeated_single_edges_cannot_invent_a_trusted_lane_width(self):
        import cv2
        image = np.full((720, 1280, 3), 255, np.uint8)
        cv2.line(image, (640, 719), (760, 300), (0, 0, 0), 20)
        detector = self._detector()
        for _ in range(detector.startup_settle_frames + 3):
            _, _, confidence, _, debug = detector.process(image, dt=.1)
        self.assertFalse(debug["measurement_valid"])
        self.assertEqual(confidence, 0.)
        self.assertGreater(debug["lost_frames"], 0)


class LaneFitTests(unittest.TestCase):
    """--lane-fit 的几何：往合成鸟瞰图里画一条形状已知的车道。

    这一层**不经过相机模型** —— _detect_lane_fit 吃进去的就是鸟瞰图，所以这里只
    依赖纯几何 x(z) = R - sqrt(R^2 - z^2)（车在车道正中、朝向对准切线）。相机和
    IPM 对不对是另一回事；这里问的是"逐段扫描 + 二次拟合"这一段本身能不能把弯道
    读出来。
    """

    HALF_LANE_CM = 17.5
    ARC_CM = 77.6          # 场地基线：中心线 R = 770 mm

    @staticmethod
    def _detector(top_cm=70.0, far_cm=50.0):
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        detector.red_detect_enable = False
        detector.lane_fit_top_cm = top_cm
        detector.lane_fit_far_cm = far_cm
        return detector

    @staticmethod
    def _centre_px(detector, z_cm, radius_cm):
        """左弯 -> 中心往左 -> 负偏移（和代码约定一致）。直道是 0。"""
        if radius_cm is None:
            return 0.0
        off_cm = radius_cm - math.sqrt(radius_cm ** 2 - z_cm ** 2)
        return -off_cm / detector.cm_per_px_at(detector._row_at_cm(z_cm))

    def _birdseye(self, detector, radius_cm):
        """两张图，和 process() 喂给 _detect_lane_fit 的那两张对应：

        detect 是黑帽+阈值之后的（亮线压在黑底上），_scan_band_midline 读的是它；
        raw 是**没处理过的**鸟瞰灰度（亮地板上的暗线），配对失败时质心兜底
        （_centroid_pair_center）找的是它的凹陷。只画一张会让合成夹具走不到真帧
        走的那条路 —— 真帧一直是两张都传的。
        """
        detect = np.zeros((detector.bird_h, detector.bird_w), np.uint8)
        raw = np.full((detector.bird_h, detector.bird_w), 200, np.uint8)
        for y in range(detector.bird_h):
            z_cm = detector.z_cm_at(y)
            if radius_cm is not None and z_cm >= radius_cm:
                continue      # 半径 R 的圆在前方 R 处已经转过 90°，再远没有点
            cpp = detector.cm_per_px_at(y)
            cx = detector.center_x + self._centre_px(detector, z_cm, radius_cm)
            half = self.HALF_LANE_CM / cpp
            line = max(1, int(round(2.0 / cpp)))
            for edge in (-1.0, 1.0):
                x0 = int(round(cx + edge * half - (line if edge < 0 else 0)))
                # **两端都要钳**。只钳起点会造出一条假线：线整个跑出鸟瞰图时，
                # 起点被拉到 0 而长度没变，于是画面边上留了一条完整的 7px 亮段 ——
                # 真帧里那条线是彻底消失的（_collect_track_runs_on_row 连
                # min_line_width 都够不到）。R=77.6 的弯上远端外侧线就是这个下场。
                a, b = max(0, x0), min(detector.bird_w, x0 + line)
                if b <= a:
                    continue
                detect[y, a:b] = 255
                raw[y, a:b] = 60
        return detect, raw

    def _fit(self, detector, radius_cm):
        gray, raw = self._birdseye(detector, radius_cm)
        z0 = detector.z_cm_at(detector.bird_h - 1)
        hint_x = detector.center_x + self._centre_px(detector, z0, radius_cm)
        hint_w = 2 * self.HALF_LANE_CM / detector.cm_per_px_at(detector.bird_h - 1)
        return detector._detect_lane_fit(
            gray, np.zeros((detector.bird_h, detector.bird_w, 3), np.uint8),
            128, False, hint_x, hint_w, gray_raw=raw)

    def test_a_straight_reads_zero_at_both_ends(self):
        detector = self._detector()
        fit = self._fit(detector, None)
        self.assertTrue(fit["fit_ok"])
        self.assertAlmostEqual(fit["fit_near_px"], 0.0, delta=2.0)
        self.assertAlmostEqual(fit["fit_far_px"], 0.0, delta=2.0)
        self.assertAlmostEqual(fit["fit_curve_px"], 0.0, delta=2.0)

    def test_the_arc_separation_survives_the_fit(self):
        """分离度是拟合出来的，不是几何里算出来的：R=77.6 上前视 50cm 处车道中心
        离切线 18.3cm、25cm 处只有 4.1cm，换成像素是 -68.0 和 -18.2，"远-近" 该有
        -49.8px，而直道是 0。拟合要真能把这条弧读回来，这个数就得对得上。"""
        detector = self._detector()
        fit = self._fit(detector, self.ARC_CM)
        truth_near = self._centre_px(detector, 25.0, self.ARC_CM)
        truth_far = self._centre_px(detector, 50.0, self.ARC_CM)
        self.assertTrue(fit["fit_ok"])
        self.assertAlmostEqual(fit["fit_near_px"], truth_near, delta=3.0)
        self.assertAlmostEqual(fit["fit_far_px"], truth_far, delta=3.0)
        self.assertAlmostEqual(fit["fit_curve_px"], truth_far - truth_near, delta=5.0)
        self.assertLess(fit["fit_curve_px"], -50.0)      # 直道那一侧是 0，不会混

    def test_a_single_line_point_never_enters_the_fit(self):
        """弯道远端外侧线跑出鸟瞰图之后只剩一条线，单线盲推（_infer_center_from_
        single_run）是按画面正中判边再横挪半个车道得出来的，能差 130px —— 一个这样
        的点就够把整条二次曲线拖歪。所以拟合只认成对的点，扫不到就停：top_cm 给到
        70 也没用，它自己停在 56cm 左右（约 row 155）。

        这一条是真退过货的：不过滤时同一帧 far=55 的误差是 +49.5px、far=65 是
        +115px 而且符号反了。
        """
        detector = self._detector(top_cm=70.0)
        fit = self._fit(detector, self.ARC_CM)
        self.assertTrue(fit["fit_ok"])
        self.assertLess(fit["fit_top_cm"], 58.0)
        self.assertGreater(fit["fit_top_cm"], 50.0)
        truth_far = self._centre_px(detector, 50.0, self.ARC_CM)
        self.assertAlmostEqual(fit["fit_far_px"], truth_far, delta=5.0)

    def test_the_arc_hides_its_outer_line_before_the_far_point_can_be_read(self):
        """R=77.6 上车道外沿在 z≈59cm 处就够到鸟瞰图的 ±45cm 边界，再远的行只有
        一条线。所以 65cm 这一点在这条赛道的最紧弯上根本读不到 —— 不是拟合不准，
        是没有像素。阈值保持 55cm 的理由就是这个。"""
        detector = self._detector(far_cm=65.0)
        fit = self._fit(detector, self.ARC_CM)
        self.assertNotIn("fit_far_px", fit)
        self.assertFalse(fit.get("fit_ok", False))
        # 同样一帧，55cm 读得到。
        ok = self._fit(self._detector(far_cm=55.0), self.ARC_CM)
        self.assertTrue(ok["fit_ok"])


class LineDetectorStateTests(unittest.TestCase):
    """The detector's cross-frame memory is what keeps a bad lock alive: the scan
    hint feeds the next frame's search, and smoothed_err is a long EMA."""

    def test_setting_the_camera_pitch_rebuilds_the_geometry(self):
        """巡线原来完全不知道机身俯角（姿态只喂了图卡）。台架对照：车一步不动、
        只把站姿换成后仰，读数当场垮（ang 22→45、curve 0→-24、far -33→-70、
        近带锁失效）。俯角被烤进两处 —— 鸟瞰单应 M 和逐行地面 LUT —— 换它必须
        重建这两样，err_scale_cm 跟着走。"""
        import numpy as np
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        M0 = detector.M.copy()
        lut0 = detector._lut_cm_per_px.copy()
        scale0 = detector.err_scale_cm

        # 安装角本身必须是逐位 no-op —— 走路时每帧都在调它
        detector.set_camera_pitch_deg(45.0)
        np.testing.assert_array_equal(detector.M, M0)
        np.testing.assert_array_equal(detector._lut_cm_per_px, lut0)
        self.assertEqual(detector.err_scale_cm, scale0)

        # 换成后仰姿态，几何必须真的变
        detector.set_camera_pitch_deg(30.0)
        self.assertFalse(np.array_equal(detector.M, M0))
        self.assertFalse(np.array_equal(detector._lut_cm_per_px, lut0))
        self.assertNotAlmostEqual(detector.err_scale_cm, scale0)
        # 横向比例尺随俯角变小：视线更平，同样像素跨的横向距离更远
        self.assertLess(detector.err_scale_cm, scale0)

        # 换回安装角要精确还原
        detector.set_camera_pitch_deg(45.0)
        np.testing.assert_allclose(detector.M, M0)
        np.testing.assert_allclose(detector._lut_cm_per_px, lut0)
        self.assertAlmostEqual(detector.err_scale_cm, scale0)

    def test_reset_state_restores_every_key_to_its_starting_value(self):
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        fresh = {k: (list(v) if isinstance(v, list) else v)
                 for k, v in detector._state.items()}
        detector._state["last_lane_center_x"] = 7.0
        detector._state["smoothed_err"] = 0.9
        detector._state["lost_frames"] = 12
        detector._state["near_err_history"].append(3.0)
        detector.reset_state()
        self.assertEqual(detector._state, fresh)
        # And the values have to be independent, not a shared mutable default.
        detector._state["near_err_history"].append(1.0)
        self.assertEqual(detector._initial_state()["near_err_history"], [])

    def test_the_handover_keeps_the_state_unless_cold_start_is_asked_for(self):
        """Stopping is a pause, not a fresh start. The loop state and the detector's
        memory are what the walking process had built, and the robot stops at a point
        where it had just recognised a card, i.e. where it was tracking the lane well -
        so its estimate of where the lane is beats restarting from the image middle.
        --card-cold-start is the wipe, and it is off unless asked for."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.return_value = (None, {"presence": True, "card_found": True,
                                            "presence_cy_frac": 0.9})
        clock = [0.0]

        def tick():
            clock[0] += 0.1
            return clock[0]

        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1"]),
            patch.object(run_policy_vision.signal, "signal") as signals,
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reads = 0

            def read():
                nonlocal reads
                reads += 1
                if reads <= 40:          # long enough for the stop window to close
                    return True, frame
                signals.call_args.args[1](None, None)
                return False, None

            camera.read.side_effect = read
            self.assertEqual(run_policy_vision.main(), 0)
        detector.reset_state.assert_not_called()

    def test_stop_frames_do_not_change_the_first_resumed_walking_seed(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.get.side_effect = [1280, 720]
        clock, reads, seed, inputs = [0.0], [0], [0], []
        detector = Mock()
        detector.snapshot_tracking_state.side_effect = lambda: seed[0]
        detector.restore_tracking_state.side_effect = lambda saved: seed.__setitem__(0, saved)

        def process(_frame, *, dt=None):
            inputs.append((reads[0], seed[0]))
            # Stopped/card frames offer a very different apparent lane.
            seed[0] = 100 if 2 <= reads[0] < 8 else seed[0] + 1
            return (0, 0, 0.9, None, detection())

        detector.process.side_effect = process
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *_a, **_k: (None, {
            "presence": reads[0] == 2,
            "presence_cy_frac": 0.9 if reads[0] == 2 else None})

        def read():
            reads[0] += 1
            clock[0] = reads[0] * 0.1
            if reads[0] > 10:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, None
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-stop-ms", "600", "--card-tilt-ms", "1000"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("attitude_input.AttitudeInput", side_effect=OSError),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertEqual(inputs[0], (1, 0))
        self.assertTrue(all(value == 1 for index, value in inputs if 2 <= index <= 8))
        self.assertEqual(dict(inputs)[9], 2)
        detector.reset_state.assert_not_called()

    def test_a_low_confidence_frame_barely_moves_the_error(self):
        """conf is the detector's own verdict on a reading, and the fusion has to
        obey it. Before this, a 0.07-confidence frame moved smoothed_err exactly as
        hard as a 0.99 one - which is how one garbage frame became full lock."""
        from line_detector_v1_warp import confidence_weighted_ema
        self.assertAlmostEqual(confidence_weighted_ema(0.0, 1.0, 0.72, 0.99), 0.2772)
        self.assertAlmostEqual(confidence_weighted_ema(0.0, 1.0, 0.72, 0.07), 0.0196)
        # A frame it gives no confidence to cannot move the error at all.
        self.assertEqual(confidence_weighted_ema(0.5, -1.0, 0.72, 0.0), 0.5)

    def test_fusion_uses_elapsed_seconds_and_explicit_tau_not_confidence_ema(self):
        import line_detector_v1_warp as ld
        detector = ld.LineDetector(1280, 720)
        detector.bottom_lock_enable = False
        detector.robust_enable = False
        band = SingleLineTrackingTests()._band(detector, [100, 240])
        band.update(band_name="low", weight=1.)
        frame = np.full((720, 1280, 3), 255, np.uint8)
        with patch.object(detector, "_detect_two_band_lanes", return_value=[band]), \
             patch.object(ld, "confidence_weighted_ema", side_effect=AssertionError("legacy EMA")), \
             patch.object(ld, "time_constant_ema", wraps=ld.time_constant_ema) as update:
            first = detector.process(frame, dt=.1)[-1]
            second = detector.process(frame, dt=.07)[-1]
        self.assertTrue(first["measurement_valid"])
        self.assertTrue(second["measurement_valid"])
        # Curve diagnostics may also use the time-based helper; locate the main tau.
        self.assertTrue(any(abs(call.args[2] - .07) < 1e-9 and
                            abs(call.args[3] - detector.filter_tau_s) < 1e-9
                            for call in update.call_args_list))

    def test_shape_dump_leaves_the_frame_and_its_detection_dict_behind(self):
        """Twice now the classifier has been called wrong on the field with nothing left
        to look at but the tallies. This is what leaves a frame to look at."""
        with tempfile.TemporaryDirectory() as folder:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            camera = Mock()
            camera.isOpened.return_value = True
            camera.get.side_effect = [1280, 720]
            detector = Mock()
            detector.process.return_value = (0, 0, 0.9, None, detection())
            shape = Mock()
            shape.action_map = {"square": 3}
            shape.update.return_value = (None, {
                "card_found": True, "presence": True, "shape": "diamond",
                "rules": "diamond", "hu": "circle:0.01", "top": 130,
                "presence_cy_frac": 0.29})
            clock = [0.0]

            def tick():
                clock[0] += 0.03
                return clock[0]

            with (
                patch("sys.argv", ["run_policy_vision.py", "--headless",
                                   "--shape-every", "1", "--shape-dump", folder]),
                patch.object(run_policy_vision.signal, "signal") as signals,
                patch.object(run_policy_vision, "ConnectorClient"),
                patch("utils.open_camera", return_value=camera),
                patch("line_detector_v1_warp.LineDetector", return_value=detector),
                patch("shape_detector.ShapeDetector", return_value=shape),
                patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                reads = 0

                def read():
                    nonlocal reads
                    reads += 1
                    if reads <= 3:
                        return True, frame
                    signals.call_args.args[1](None, None)
                    return False, None

                camera.read.side_effect = read
                self.assertEqual(run_policy_vision.main(), 0)

            written = sorted(p for p in Path(folder).rglob("*")
                             if p.is_file() and p.name != "run_manifest.json")
            self.assertGreaterEqual(len(written), 4)      # a frame and a dict per call
            self.assertEqual(len([p for p in written if p.suffix == ".jpg"]),
                             len([p for p in written if p.suffix == ".json"]))
            with open(written[-1], encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertTrue(saved["card_found"])
            self.assertEqual(saved["rules"], "diamond")
            self.assertEqual(saved["hu"], "circle:0.01")

    def test_dump_on_loss_writes_the_frames_that_straddle_the_lock_dropout(self):
        """On 2026-09-29 a lap left a clean dividing line in the log - bottom_pair_ratio
        1.00 to 0.00, confidence 0.86 to 0.39 - and nothing else to look at. The
        useful frame is the last one BEFORE the drop, so the ring has to hold those
        and flush them when the lock goes, not start recording after it."""
        with tempfile.TemporaryDirectory() as folder:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            vis = np.ones((720, 1280, 3), dtype=np.uint8)
            camera = Mock()
            camera.isOpened.return_value = True
            camera.get.side_effect = [1280, 720]
            # Locked for four frames, then gone: the trip is on the fourth->fifth.
            pairs = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

            def process(_frame, *, dt=None):
                index = min(calls[0], len(pairs) - 1)
                calls[0] += 1
                return (0, 0, 0.9, vis, {**detection(),
                                         "bottom_pair_ratio": pairs[index]})

            calls = [0]
            detector = Mock()
            detector.process.side_effect = process
            clock = [0.0]

            def tick():
                clock[0] += 0.03
                return clock[0]

            with (
                patch("sys.argv", ["run_policy_vision.py", "--headless",
                                   "--no-shape-detect",
                                   "--dump-on-loss", folder]),
                patch.object(run_policy_vision.signal, "signal") as signals,
                patch.object(run_policy_vision, "ConnectorClient"),
                patch("utils.open_camera", return_value=camera),
                patch("line_detector_v1_warp.LineDetector", return_value=detector),
                patch.object(run_policy_vision.time, "monotonic", side_effect=tick),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                reads = 0

                def read():
                    nonlocal reads
                    reads += 1
                    if reads <= 6:
                        return True, frame
                    signals.call_args.args[1](None, None)
                    return False, None

                camera.read.side_effect = read
                self.assertEqual(run_policy_vision.main(), 0)

            written = sorted(p for p in Path(folder).rglob("*")
                             if p.is_file() and p.name != "run_manifest.json")
            frames = [p for p in written if p.name.endswith("_frame.jpg")]
            views = [p for p in written if p.name.endswith("_vis.jpg")]
            self.assertEqual(len(frames), len(views))
            self.assertGreaterEqual(len(frames), 2)          # before and after
            # The last locked frame is in the dump; the box also caught the drop.
            self.assertTrue(any("pair1.00" in p.name for p in frames))
            self.assertTrue(any("pair0.00" in p.name for p in frames))
            self.assertTrue(all(p.suffix == ".json" for p in written
                                if p.suffix == ".json"))
            with open(next(p for p in written if p.suffix == ".json"),
                      encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertIn("bottom_pair_ratio", saved)
            # Arrays are left out so the dict stays readable.
            self.assertTrue(all(isinstance(v, (int, float, str, bool))
                                or v is None for v in saved.values()))

    def test_the_handover_resets_the_detector_before_the_walking_frame(self):
        """A fresh process tracks the same curve, so the frame after a card stop
        should start from a fresh state too - and it has to be the FIRST walking
        frame. Resetting after detector.process() left that frame computed from the
        polluted state, and the first frame is the one that decides where it goes."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        entries = []

        def record(kind, value):
            entries.append((kind, reads[0]))
            return value

        detector = Mock()
        detector.process.side_effect = lambda _f, **_kwargs: record(
            "process", (0, 0, 0.9, None, detection()))
        detector.reset_state.side_effect = lambda: record("reset", None)
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True, "presence_cy_frac": 0.9})
            if reads[0] == 2
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 60:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-cold-start"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        detector.reset_state.assert_called_once()
        # publish index i belongs to read i+1, so the first frame it walks again
        # after the stop is the one whose process call carries that number.
        published = client_cls.return_value.publish.call_args_list
        resumed = next(i for i, call in enumerate(published)
                       if i > 2 and call.args[0] > 0.0)
        walking_read = resumed + 1
        first_walk_process = next(
            i for i, entry in enumerate(entries) if entry == ("process", walking_read))
        reset_at = [k for k, _ in entries].index("reset")
        # Before that process call. Placed below detector.process() instead, the
        # reset lands after it and only the second walking frame would be clean.
        self.assertLess(reset_at, first_walk_process)


class ObservationDumpTests(unittest.TestCase):
    """The stop->restart transient is invisible at the vision log's 2 Hz sampling,
    so policy_runner can dump all 49 observation components at 50 Hz instead."""

    def test_it_names_every_component_and_stays_off_by_default(self):
        self.assertIsNone(policy_runner.open_observation_dump())
        names = policy_runner.observation_columns()
        self.assertEqual(len(names), config.OBS_DIM)
        self.assertEqual(len(set(names)), config.OBS_DIM)
        self.assertEqual(names[9], "cmd_vx")
        self.assertEqual(names[11], "step_distance")

    def test_a_row_records_the_observation_and_the_zero_command_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "obs.csv"
            with patch.dict(os.environ, {"POLICY_OBS_CSV": str(path)}):
                dump = policy_runner.open_observation_dump()
                self.assertIsNotNone(dump)
                obs = np.arange(config.OBS_DIM, dtype=np.float32)
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
                dump.append(obs, action, np.zeros(3, dtype=np.float32))
                dump.append(obs, action, np.array([0.4, 0.0, 0.5], dtype=np.float32))
                dump.file.close()
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 3)                       # header + two ticks
        self.assertEqual(rows[0][3:3 + config.OBS_DIM],
                         policy_runner.observation_columns())
        self.assertEqual(rows[1][2], "1")                    # command exactly zero
        self.assertEqual(rows[2][2], "0")
        self.assertEqual(rows[1][3 + 11], "11")              # step_distance column


class ShapeDetectorReportingTests(unittest.TestCase):
    """_cue_box is whatever the last hit left there and is never cleared, so a miss
    frame used to report a position up to three calls old. The caller gates the
    stop on that centroid, and a stale one walked 0.51 -> 0.78 while the robot was
    standing still."""

    def test_a_cue_miss_reports_no_centroid(self):
        detector = ShapeDetector()
        calls = []

        def cue(_gray):
            calls.append(1)
            return ((40, 300, 120, 90), 4.0) if len(calls) == 1 else (None, 0.0)

        detector._presence_cue = cue
        detector._cue_hist.extend([1, 1, 1])          # window already confirmed
        blank = np.zeros((720, 1280, 3), np.uint8)

        _, first = detector.update(blank)
        self.assertIsNotNone(first.get("presence_cy_frac"))
        self.assertEqual(first.get("presence_cue"), 4.0)

        _, second = detector.update(blank)
        self.assertIsNone(second.get("presence_cy_frac"))   # nothing to gate on
        self.assertEqual(second.get("presence_cue"), 0.0)   # and the log is honest
        self.assertTrue(second.get("presence"))             # window still holds

    def test_presence_survives_the_detector_disarming_itself(self):
        """presence 说的是"画面里有没有卡"，不是"我还要不要开火"。

        2026-10-02 实车：车停在五角星前面，分类器逐帧都认得出来，但检测器内部
        那套动作闸（调用方早就不用了）自己开了一次火 → armed=0 → dbg["presence"]
        被门控成假 → 调用方数满 4 次漏检，把 card_flag / card_triggered 和
        **投出来的票**一起复位：那次停车一张票都没剩下（什么都没做），
        5 秒后同一张卡又触发了一次停车。"""
        detector = ShapeDetector()
        detector._presence_cue = lambda _gray: ((40, 300, 120, 90), 4.0)
        detector._cue_hist.extend([1, 1, 1])          # window already confirmed
        detector.armed = False                        # 内部动作闸开过一次火
        blank = np.zeros((720, 1280, 3), np.uint8)

        _, dbg = detector.update(blank)

        self.assertFalse(detector.armed)
        self.assertTrue(dbg.get("presence"))

    def test_a_low_score_cue_is_not_a_card(self):
        """2026-10-03 实车：白地板 + 黑线 + 反光也满足"细环 + 亮孔"，_presence_cue
        给了一帧 cue=0.077 的结构命中，时间窗攒满就把车停在了一张不存在的卡前面
        （shape_dump 抠出来的框里只有地板）。量出来的分界：地板/反光 ≤0.78，
        真卡（运动模糊、走着看）4.2~8.5 —— 所以出口分数要单独一道闸。"""
        detector = ShapeDetector()
        blank = np.zeros((720, 1280, 3), np.uint8)
        detector._presence_cue = lambda _gray: ((40, 300, 120, 90), 0.4)
        for _ in range(6):                      # 连续命中，时间窗早该攒满
            _, dbg = detector.update(blank)
        self.assertFalse(dbg.get("presence"))
        self.assertIsNone(dbg.get("presence_cy_frac"))   # 没有 cy 可以喂停车闸
        self.assertEqual(dbg.get("presence_cue"), 0.4)   # 诊断列照旧写
        detector._presence_cue = lambda _gray: ((40, 300, 120, 90), 4.0)
        for _ in range(6):
            _, dbg = detector.update(blank)
        self.assertTrue(dbg.get("presence"))

    def test_the_line_only_sees_the_pitch_while_the_robot_is_not_driving(self):
        """--line-pitch 默认关：巡线的几何完全不动（老行为）。

        开着的时候判的是"车在不在走"：vx<=0 的帧（台架 --hold-still、停车窗口、
        窗口之后的 hold）吃实时俯角；走路那一段回静态安装角 —— 步态以 1.7Hz 摆
        30~40°，低通的值描述不了当前这一帧（原注释里试过，反而更糟）。

        ⚠️ 一开始只写了"停车窗口 + hold"，台架测不到 —— 台架上没有卡，窗口从没
        开过，每帧都喂静态安装角，等于开关没开。这一条两次都踩在上面。"""
        def run(extra, pitch):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            camera = Mock()
            camera.isOpened.return_value = True
            camera.get.side_effect = [1280, 720]
            detector = Mock()
            detector.process.return_value = (0, 0, 0.8, None, detection())
            shape = Mock()
            shape.action_map = {"square": 3}
            shape.update.return_value = (None, {"presence": False,
                                                "card_found": False})
            clock = [0.0]
            reads = [0]

            def read():
                reads[0] += 1
                clock[0] += 0.1
                if reads[0] > 6:
                    run_policy_vision.signal.signal.call_args.args[1](None, None)
                    return False, frame
                return True, frame

            camera.read.side_effect = read
            attitude = Mock()
            attitude.value = 30.0        # 后仰着的机身：实时俯角 ≠ 安装角
            with (
                patch("sys.argv", ["run_policy_vision.py", "--headless",
                                   "--camera-pitch-deg", str(pitch)] + extra),
                patch.object(run_policy_vision.signal, "signal"),
                patch.object(run_policy_vision, "ConnectorClient"),
                patch("utils.open_camera", return_value=camera),
                patch("line_detector_v1_warp.LineDetector", return_value=detector),
                patch("shape_detector.ShapeDetector", return_value=shape),
                patch("attitude_input.AttitudeInput", return_value=attitude),
                patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
                patch("cv2.imshow", side_effect=AssertionError("headless")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_policy_vision.main(), 0)
            return detector

        off = run([], 45.0)
        self.assertEqual(off.set_camera_pitch_deg.call_count, 0)

        # 走起来（没开 --hold-still，控制器发 vx=0.2）：回静态安装角。
        # 只看最后一帧 —— 第一帧还没有"上一帧发了多少"，按停着算。
        driving = run(["--line-pitch"], 45.0)
        self.assertGreater(driving.set_camera_pitch_deg.call_count, 0)
        self.assertEqual(driving.set_camera_pitch_deg.call_args_list[-1].args[0], 45.0)

        # 停着不动（--hold-still 把 vx 压成 0）：吃实时俯角
        held = run(["--line-pitch", "--hold-still"], 45.0)
        self.assertGreater(held.set_camera_pitch_deg.call_count, 0)
        self.assertEqual(held.set_camera_pitch_deg.call_args_list[-1].args[0], 30.0)

    def test_hold_still_publishes_zeros_whatever_the_controller_decides(self):
        """台架测机身姿态时得开 C（姿态从它来），但 C 一使能电机、B 一发
        vx=0.2 车就走了。这个开关把发出去的 vx/wz 压成 0，检测和日志照跑。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection(error=9.0))
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.return_value = (None, {"presence": False, "card_found": False})

        def run(extra):
            camera = Mock()
            camera.isOpened.return_value = True
            camera.get.side_effect = [1280, 720]
            clock = [0.0]
            reads = [0]

            def read():
                reads[0] += 1
                clock[0] += 0.1
                if reads[0] > 5:
                    run_policy_vision.signal.signal.call_args.args[1](None, None)
                    return False, frame
                return True, frame

            camera.read.side_effect = read
            with (
                patch("sys.argv", ["run_policy_vision.py", "--headless"] + extra),
                patch.object(run_policy_vision.signal, "signal"),
                patch.object(run_policy_vision, "ConnectorClient") as client_cls,
                patch("utils.open_camera", return_value=camera),
                patch("line_detector_v1_warp.LineDetector", return_value=detector),
                patch("shape_detector.ShapeDetector", return_value=shape),
                patch("attitude_input.AttitudeInput", side_effect=OSError),
                patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
                patch("cv2.imshow", side_effect=AssertionError("headless")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_policy_vision.main(), 0)
            return client_cls.return_value.publish.call_args_list

        driving = run([])
        self.assertTrue(any(call.args[0] > 0 for call in driving))
        held = run(["--hold-still"])
        self.assertTrue(held)
        for call in held:
            self.assertEqual(call.args[0], 0.0)
            self.assertEqual(call.args[1], 0.0)

    def test_card_detection_runs_every_other_frame_while_stopped(self):
        """Every frame costs 94-122 ms, which drops the loop to 6-8 Hz and makes the
        policy see one held packet for 6-8 of its 50 Hz ticks instead of 3."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        seen = []
        shape = Mock()
        shape.action_map = {"square": 3}

        def update(*_a, **_k):
            seen.append(reads[0])
            if reads[0] in (2, 3):
                return None, {"presence": True, "card_found": True,
                              "presence_cy_frac": 0.9}
            return None, {"presence": False, "card_found": False,
                          "presence_cy_frac": None}

        shape.update.side_effect = update
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 40:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            # --card-tilt-ms 0: this test is about the stopped cadence, and the tilt
            # delay would push the first looked-at frame ten frames later.
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-every-stopped", "2", "--card-tilt-ms", "0",
                               "--card-settle-ms", "0"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        # Read 3 triggers the stop; from read 4 the detector runs every second frame.
        inside = [i for i in seen if i > 3]
        self.assertEqual(inside[:10], [4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        # And once the window closes it is back to --shape-every 1, every frame.
        self.assertLess(shape.update.call_count, reads[0] * 0.75)

    def test_the_card_is_not_looked_at_while_the_body_is_being_re_posed(self):
        """--card-tilt-ms holds off every look at the card at the start of the stop.

        A frame taken mid-tilt is a frame of a body in motion, and the classifier reads
        the card's shape off that geometry. The card is stationary and the stop window
        is seconds long, so the wait costs nothing. Same harness as the cadence test,
        with the default tilt instead of 0.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        seen = []
        shape = Mock()
        shape.action_map = {"square": 3}

        def update(*_a, **_k):
            seen.append(reads[0])
            if reads[0] in (2, 3):
                return None, {"presence": True, "card_found": True,
                              "presence_cy_frac": 0.9}
            return None, {"presence": False, "card_found": False,
                          "presence_cy_frac": None}

        shape.update.side_effect = update
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 40:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-every-stopped", "2", "--card-tilt-ms", "1000"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        # The shape is never named in this harness, so the re-pose stays requested for
        # the whole window - exactly the frames the detector is held off for.
        published = client_cls.return_value.publish.call_args_list
        self.assertFalse(published[0].kwargs["card_tilt"])
        self.assertTrue(published[1].kwargs["card_tilt"])
        # The trigger lands on read 2. The legacy path waits 1000ms plus the new
        # 300ms settling margin before looking at a card again.
        # and the every-other-frame cadence picks up from the first read at or after
        # the window. Asserted against the tilt being ten reads long rather than a
        # fixed index: the mocked clock accumulates 0.1 and lands either side of the
        # boundary. With --card-tilt-ms 0 the same harness looks from read 4.
        inside = [i for i in seen if i > 3]
        self.assertEqual(inside[:5], [16, 18, 20, 22, 24])

    def test_votes_start_only_after_stm32_done_plus_settle_time(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        clock = [0.0]
        reads = [0]
        seen = []
        shape = Mock()
        shape.action_map = {"square": 3}

        def update(*_args, **_kwargs):
            seen.append(reads[0])
            return None, {"presence": reads[0] >= 2,
                          "presence_cy_frac": 0.9 if reads[0] >= 2 else None,
                          "shape": "square"}

        shape.update.side_effect = update
        attitude = Mock()
        attitude.value = 45.0
        attitude.card_tilt_status_seen = True
        attitude.card_tilt_event_id = 0
        attitude.card_tilt_done = False

        def read():
            reads[0] += 1
            clock[0] = reads[0] * 0.1
            if reads[0] > 23:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        def poll():
            if reads[0] >= 3:
                attitude.card_tilt_event_id = 1
                attitude.card_tilt_done = True  # delayed DONE from a previous card
            if reads[0] >= 7:
                attitude.card_tilt_event_id = 2
                attitude.card_tilt_done = False
            if reads[0] >= 12:
                attitude.card_tilt_done = True

        camera.read.side_effect = read
        attitude.poll.side_effect = poll
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-every-stopped", "1",
                               "--card-vote-frames", "1", "--card-tilt-ms", "0",
                               "--card-settle-ms", "300"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("attitude_input.AttitudeInput", return_value=attitude),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)

        # Trigger-frame classification is not a vote. Even with the old 0ms
        # blind timer, a stale DONE from the previous card cannot start voting.
        # The current event reaches DONE at read 12; then another 300ms must pass.
        self.assertEqual([i for i in seen if 2 < i < 15], [])
        events = [i + 1 for i, call in enumerate(client_cls.return_value.publish.call_args_list)
                  if "event_id" in call.kwargs]
        self.assertTrue(events)
        self.assertGreaterEqual(events[0], 15)

    def test_a_card_in_view_speeds_the_detection_up_before_the_stop(self):
        """The stop fires on the first cy at or above the trigger line, so
        --shape-every 6 samples that line about every 0.3 s - most of a 12 cm step at
        0.4 m/s. On the 2026-09-30 run the robot sailed from 43 cm to ~22 cm, and at
        22 cm the card's bottom edge was below the frame: no quad could close and every
        frame read g0. The fine reading has to exist before the decision, not after it.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        seen = []
        shape = Mock()
        shape.action_map = {"square": 3}

        def update(*_a, **_k):
            seen.append(reads[0])
            # Below the trigger line throughout, so the robot never stops and the
            # approach is the only phase under test.
            return None, {"presence": True, "card_found": True,
                          "presence_cy_frac": 0.3}

        shape.update.side_effect = update
        clock = [0.0]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 30:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "6", "--card-every-stopped", "2",
                               "--card-trigger-frac", "0.5"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        # The first look rides the --shape-every beat. Every look after it is on the
        # stopped beat, because the card is in view from then on.
        self.assertEqual(seen[0] % 6, 0)
        self.assertGreater(len(seen), 3, "the detector barely ran")
        gaps = [b - a for a, b in zip(seen, seen[1:])]
        self.assertEqual(set(gaps), {2})


class DumpRunIsolationTests(unittest.TestCase):
    def test_two_runs_keep_same_named_frames_and_their_own_commands(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "0001_None_cy0.5_g0.jpg"
            old.write_bytes(b"previous flat dump")
            folders = []
            for tag, fire in (("run_a", 5), ("run_b", 10)):
                folder = Path(run_policy_vision._new_dump_run(
                    d, tag, {"run_id": tag, "arguments": {"wz_fire_cm": fire}}))
                (folder / "0001_square_cy0.5_g1.jpg").write_bytes(tag.encode())
                folders.append(folder)
            self.assertNotEqual(*folders)
            self.assertEqual(old.read_bytes(), b"previous flat dump")
            self.assertEqual((folders[0] / "0001_square_cy0.5_g1.jpg").read_bytes(), b"run_a")
            self.assertEqual(json.loads((folders[1] / "run_manifest.json").read_text())
                             ["arguments"]["wz_fire_cm"], 10)

    def test_shape_and_loss_can_share_one_run_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            first = run_policy_vision._new_dump_run(d, "run_a", {"argv": ["original"]})
            Path(first, "0001_square_cy0.5_g1.jpg").write_bytes(b"shape")
            second = run_policy_vision._new_dump_run(d, "run_a", {"argv": ["replacement"]})
            self.assertEqual(first, second)
            self.assertEqual(json.loads(Path(second, "run_manifest.json").read_text()),
                             {"argv": ["original"]})
            self.assertEqual(Path(second, "0001_square_cy0.5_g1.jpg").read_bytes(), b"shape")


class CardGeometryGateTests(unittest.TestCase):
    """_geom_ok must accept the card everywhere the card stop puts the robot.

    The gate measures the quad's bounding rectangle, which grows with yaw: an
    ideal 10 cm card at 27 cm is 13328 px2 head-on but 22646 at 30 degrees. With
    area_max at 20000 that card was rejected - at exactly the distance
    --card-trigger-frac 0.5 stops at and the 16 cm clearance leaves it.
    """

    def ideal_card(self, detector, z_true, yaw_deg):
        c = detector.cfg
        h = c["cam_height_cm"]
        th = math.radians(c["cam_pitch_deg"])
        vfov = math.radians(c["cam_vfov_deg"])
        fy = WORK_H / (2.0 * math.tan(vfov / 2.0))
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * WORK_W / WORK_H)
        fx = WORK_W / (2.0 * math.tan(hfov / 2.0))
        z0 = to_model_z(z_true)
        yaw = math.radians(yaw_deg)
        corners = []
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            px, pz = sx * 5.0, sz * 5.0
            x = px * math.cos(yaw) - pz * math.sin(yaw)
            z = z0 + px * math.sin(yaw) + pz * math.cos(yaw)
            zc = h * math.sin(th) + z * math.cos(th)
            yc = h * math.cos(th) - z * math.sin(th)
            corners.append((WORK_W / 2.0 + fx * x / zc,
                            WORK_H / 2.0 + fy * yc / zc))
        return np.array(corners, np.float32)

    def test_a_card_in_the_stopping_zone_passes_the_geometry_gate(self):
        detector = ShapeDetector()
        for z in (14, 20, 27, 30, 40, 60, 80):
            for yaw in (0, 30, 45):
                self.assertTrue(
                    detector._geom_ok(self.ideal_card(detector, z, yaw)),
                    f"10cm card rejected at {z} cm, yaw {yaw} deg")


class UdpIntegrationTests(unittest.TestCase):
    def test_controller_connector_receiver_observation_and_both_watchdogs(self):
        source = UdpCommandSource(0, timeout_s=0.15)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            vision_port = probe.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "connector.py"),
             "--vision-port", str(vision_port), "--policy-port", str(source.sock.getsockname()[1]),
             "--vision-timeout", "0.15"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        client = ConnectorClient(port=vision_port)
        controller = SteeringController(**NO_TRIM, straight_gains=(1, 0, 0),
                                        steer_full_scale_cm=50, yaw_sign=-1)
        expected = [0.2, 0.0, -0.1]

        def wait_for(target, publish=False):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if publish:
                    client.publish(*controller.command(detection(), 0.8, 0.02))
                if np.allclose(source.get(), target):
                    return
                time.sleep(0.01)
            self.fail(f"Expected {target}, got {source.get()}; connector={process.poll()}")

        try:
            np.testing.assert_array_equal(source.get(), [0, 0, 0])
            wait_for(expected, publish=True)
            # Use the actual observation builder without loading an ONNX model.
            policy = object.__new__(HumanoidPolicy)
            policy.last_action = np.zeros(12, dtype=np.float32)
            policy.step_distance_m = config.STEP_LENGTH_CM / 100.0
            def observation():
                return policy.build_observation(
                    accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
                    projected_gravity=np.array([0, 0, -1]), velocity_command=source.get(),
                    joint_position_policy=config.Q_DEFAULT, joint_velocity_policy=np.zeros(12))
            np.testing.assert_allclose(
                observation()[9:13],
                [0.2, -0.1, config.STEP_LENGTH_CM / 100.0, 0], rtol=1e-5)
            # A confirmed event crosses both UDP hops even when qr has returned to -1.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                client.publish(0.0, 0.0, -1, event_id=12345, event_action=5)
                if source.get_snapshot().event_id == 12345:
                    break
                time.sleep(0.01)
            snapshot = source.get_snapshot()
            self.assertEqual((snapshot.qr, snapshot.event_id, snapshot.event_action), (-1, 12345, 5))
            wait_for(expected, publish=True)
            # Malformed traffic must neither kill the connector nor renew vision freshness.
            client.socket.sendto(b'{"vx":0.4,"wz":0.2,"qr":Infinity}', client.address)
            wait_for([0, 0, 0])
            self.assertIsNone(process.poll())
            np.testing.assert_array_equal(observation()[9:13], [0, 0, 0, 0])
            wait_for(expected, publish=True)
            # Abrupt death: no graceful zero packet; receiver watchdog must act.
            process.kill()
            process.wait(timeout=2)
            wait_for([0, 0, 0])
        finally:
            client.close()
            source.close()
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=2)


class GroundScaleTests(unittest.TestCase):
    """鸟瞰不是等距的：横向比例尺和纵向距离都随行变化，全图一个常数会把低带放大 51%。"""

    def test_the_camera_config_actually_loads(self):
        # camera_config.load 把异常全吞了、回退到 _DEFAULTS，所以 cameras.json 里
        # 一个多余的逗号会让整套几何静默换成默认值 —— 这个测试就是为了在 CI 里
        # 把它变成红的，而不是到了赛场上才发现。
        import camera_config
        profile = camera_config.load()
        self.assertEqual(profile["profile"], "usb_main")
        self.assertAlmostEqual(profile["distance_calib"]["a"], 1.13233, places=4)

    def setUp(self):
        from line_detector_v1_warp import LineDetector
        self.detector = LineDetector()

    def test_the_ground_lut_is_monotonic_and_spans_the_warped_range(self):
        detector = self.detector
        rows = np.arange(0, detector.bird_h, 25.0)
        depths = np.array([detector.z_cm_at(y) for y in rows])
        widths = np.array([detector.cm_per_px_at(y) for y in rows])
        self.assertTrue(np.all(np.diff(depths) < 0.0), "越往上越远")
        self.assertTrue(np.all(np.diff(widths) < 0.0), "越往上每像素代表越多厘米")
        self.assertAlmostEqual(detector.z_cm_at(detector.bird_h - 1), 20.2, delta=0.5)
        self.assertGreater(detector.z_cm_at(0), 80.0)

    def test_the_near_band_is_not_wider_per_pixel_than_the_far_end(self):
        detector = self.detector
        near = detector.cm_per_px_at(detector.NEAR_BAND_ROW)
        far = detector.cm_per_px_at(0.0)
        self.assertAlmostEqual(near, 0.2238, delta=0.002)
        self.assertAlmostEqual(far, 0.3310, delta=0.003)
        # 旧代码在整幅图上用 0.330 —— 那是远端的值，低带会大 51%。
        self.assertLess(near / far, 0.70)

    def test_the_error_scale_follows_the_near_band_not_the_whole_image(self):
        detector = self.detector
        self.assertAlmostEqual(
            detector.err_scale_cm,
            0.5 * detector.bird_w * detector.cm_per_px_at(detector.NEAR_BAND_ROW),
            places=6,
        )
        self.assertLess(detector.err_scale_cm, 40.0)

    def test_the_depth_is_not_linear_in_the_row(self):
        # 旧的 z = 20 + Δy·0.1504 假设透视是线性的，中段会偏 11cm。
        detector = self.detector
        mid = detector.z_cm_at(200.0) - detector.z_cm_at(201.0)
        far = detector.z_cm_at(50.0) - detector.z_cm_at(51.0)
        self.assertGreater(far, 1.3 * mid, "远端每行代表的距离明显超过中段")

    def test_the_ground_projection_round_trips_through_the_warp(self):
        import cv2
        detector = self.detector
        for model_z in (25.0, 40.0, 60.0):
            h, theta = detector.cam_height, detector.cam_pitch
            fy, cy = detector.fy_px, detector.cy_px
            v = fy * (h * math.cos(theta) - model_z * math.sin(theta)) / (
                h * math.sin(theta) + model_z * math.cos(theta)) + cy
            bird_y = cv2.perspectiveTransform(
                np.float32([[[detector.cx_px, v]]]), detector.M)[0, 0][1]
            self.assertAlmostEqual(
                detector.z_cm_at(bird_y), detector._to_true_z(model_z), delta=0.2)


class LaneWidthAnchorTests(unittest.TestCase):
    """横向比例尺的绝对值靠赛道自身宽度锚定 —— 相机姿势的残差不靠几何能修干净。"""

    def setUp(self):
        from line_detector_v1_warp import LineDetector
        self.detector = LineDetector()

    def _feed(self, width_px, pair_ratio=0.9, frames=200):
        near = {"lane_width_px": width_px, "pair_ratio": pair_ratio}
        for _ in range(frames):
            self.detector._update_lateral_scale(near)

    def test_a_too_wide_reading_shrinks_the_scale_until_the_lane_reads_true(self):
        detector = self.detector
        width_px = 146.5   # 标定照片里量到的
        self._feed(width_px)
        self.assertAlmostEqual(
            width_px * detector.cm_per_px_at(detector.NEAR_BAND_ROW),
            detector.lane_width_true_cm,
            delta=0.1,
        )

    def test_a_single_boundary_frame_carries_no_width_evidence(self):
        detector = self.detector
        self._feed(60.0, pair_ratio=0.2)
        self.assertEqual(detector.lateral_scale, 1.0)

    def test_the_narrow_gate_cannot_drag_the_scale_along(self):
        detector = self.detector
        # 窄门 240mm，只有标准赛道的一半多一点；真跟进去会把比例尺抬 46%。
        self._feed(90.0)
        self.assertLessEqual(detector.lateral_scale, detector.lateral_scale_max)
        self.assertLessEqual(detector.lateral_scale_max, 1.25)

    def test_the_scale_never_leaves_the_plausible_band(self):
        detector = self.detector
        self._feed(400.0)
        self.assertGreaterEqual(detector.lateral_scale, detector.lateral_scale_min)
        self._feed(20.0)
        self.assertLessEqual(detector.lateral_scale, detector.lateral_scale_max)


class AnticipationClipTests(unittest.TestCase):
    """工程性截断：推断项不许把近带的直接测量翻掉。"""

    def test_the_inferred_terms_cannot_outvote_the_near_band(self):
        from line_detector_v1_warp import LineDetector
        det = LineDetector(1280, 720)
        self.assertEqual(det.anticipation_clip, 0.5)
        # 近带 +1.0，推断不管给多大，最多只到一半、且符号跟着近带
        self.assertAlmostEqual(det._clip_anticipation(1.0, -3.0), -0.5)
        self.assertAlmostEqual(det._clip_anticipation(1.0, +3.0), +0.5)
        self.assertAlmostEqual(det._clip_anticipation(-1.0, -3.0), -0.5)
        self.assertAlmostEqual(det._clip_anticipation(-1.0, +3.0), +0.5)
        # 限内的原样通过
        self.assertAlmostEqual(det._clip_anticipation(1.0, 0.2), 0.2)
        # 近带为 0（车就在线上）→ 推断整个被切掉，不再凭空注入误差
        self.assertEqual(det._clip_anticipation(0.0, 3.0), 0.0)

    def test_zero_restores_the_old_fusion(self):
        from line_detector_v1_warp import LineDetector
        det = LineDetector(1280, 720)
        det.anticipation_clip = 0.0
        self.assertAlmostEqual(det._clip_anticipation(1.0, -3.0), -3.0)
        self.assertAlmostEqual(det._clip_anticipation(0.0, 3.0), 3.0)

    def test_the_argument_reaches_the_detector(self):
        with patch("sys.argv", ["run_policy_vision.py", "--anticipation-clip", "0.25"]):
            self.assertAlmostEqual(run_policy_vision.parse_args().anticipation_clip, 0.25)
        with patch("sys.argv", ["run_policy_vision.py"]):
            self.assertAlmostEqual(run_policy_vision.parse_args().anticipation_clip, 0.5)


class DiscreteSteeringTests(unittest.TestCase):
    """离散航向：只换发出去的 wz，SteeringController 本身一个字不动。"""

    def controller(self, yaw_sign=1, **kw):
        from discrete_steering import DiscreteSteeringController
        return DiscreteSteeringController(
            SteeringController(**NO_TRIM, yaw_sign=yaw_sign), **kw)

    def step(self, controller, err, dt=0.05):
        return controller.command(detection(error=err), 0.8, dt)[1]

    def test_a_small_error_is_left_alone(self):
        controller = self.controller()
        for err in (0.0, 1.0, 2.9, 4.9, -4.9):
            with self.subTest(err=err):
                self.assertEqual(self.step(controller, err), 0.0)

    def test_one_threshold_one_amplitude(self):
        """只有一个幅度。曾经按 |err| 分两档（0.4 / 0.5），那是在一个不确定的底层上
        多叠了一层判断；而且幅度挑的是「大且稳」那一端，不是「刚好够用」那一端。"""
        self.assertEqual(self.step(self.controller(), 4.9), 0.0)
        self.assertEqual(self.step(self.controller(), 6.0), 0.5)
        self.assertEqual(self.step(self.controller(), 20.0), 0.5)   # 不会换档
        # 默认不发负的 wz：赛道按行进方向只有左弯，右转永远是转过头之后的过冲
        for err in (-6.0, -12.0):
            with self.subTest(err=err):
                self.assertEqual(self.step(self.controller(), err), 0.0)
        self.assertEqual(self.step(self.controller(allow_right=True), -6.0), -0.5)

    def test_only_a_body_right_of_centre_may_turn_left(self):
        """单边闸看的是 err 的符号（车在车道中心右边 = err > 0）。偏左就滑行 ——
        正是这一条把"转一段"和"滑行"分开，折线靠它。"""
        controller = self.controller(turn_s=0.1, gap_s=2.05)
        self.assertEqual(self.step(controller, 6.0), 0.5)     # 偏右，开转
        self.assertEqual(self.step(controller, -6.0), 0.0)    # 转的途中偏左 → 提前收手
        self.assertEqual(self.step(controller, -6.0), 0.0)    # 收手之后进强制滑行
        self.assertEqual(self.step(controller, -0.001), 0.0)  # 贴着中心也是 0
        # 间隔期间 err 在右边也不开，管的是间隔不是 err
        coasted = 0
        for _ in range(60):
            if self.step(controller, 6.0) == 0.5:
                break
            coasted += 1
        self.assertEqual(coasted, 39)         # 加上上面那一步才是 2.05s / 0.05s = 41

    def test_the_yaw_sign_is_applied(self):
        # yaw_sign 照旧乘上去（真车上是 +1）。单边闸看的是 err 的符号，所以
        # yaw_sign=-1 时同样是"车身偏右才动"，只是动的方向镜像过来。
        self.assertEqual(self.step(self.controller(yaw_sign=-1), 6.0), -0.5)
        self.assertEqual(self.step(self.controller(yaw_sign=-1), -6.0), 0.0)
        both = self.controller(yaw_sign=-1, allow_right=True)
        self.assertEqual(self.step(both, -12.0), 0.5)   # yaw_sign 和 err 两个负号相消

    def test_a_turn_holds_for_its_whole_duration(self):
        """err 一直在带外时，--wz-turn-s 仍然是那段转向的宽度，之后回 0。
        时长必须够长到机械真的做出来 —— 0.15s 在 3:8 的步频下只有两三步。
        （它是上限，不是定长：err 翻到另一侧的 --wz-stop-cm 就收手，另一条测。）"""
        controller = self.controller(turn_s=0.2)
        self.assertEqual(self.step(controller, 6.0), 0.5)      # 起转
        for _ in range(3):
            self.assertEqual(self.step(controller, 6.0), 0.5)  # 0.15s，还在这一段里
        self.assertEqual(self.step(controller, 0.0), 0.0)      # 0.20s，这一段结束
        self.assertEqual(self.step(controller, 0.0), 0.0)

    def test_a_turn_stops_when_the_error_reaches_the_mirror_line(self):
        """收手线在**另一侧**。2026-10-03 实车走了两步：先是定长 2.5s 谁也叫不停
        （出弯那一下把车带出直道）；改成"同侧 2cm 就收"又收得太早 —— 每转完都还
        在中心右边，直道上再攒新的右偏，最后从**右**边出去。所以收手线改成开火线
        的镜像位置：+fire 触发 → 转到 −fire 才停。"""
        controller = self.controller(turn_s=2.5, gap_s=1.0, stop_cm=2.0)
        self.assertEqual(self.step(controller, 6.0), 0.5)      # 起转
        for _ in range(10):
            self.assertEqual(self.step(controller, 4.6), 0.5)  # 同侧，一直转
        self.assertEqual(self.step(controller, 1.0), 0.5)      # 掉到开火线以下也不收
        self.assertEqual(self.step(controller, -1.9), 0.5)     # 翻过去了，还没到 −2
        self.assertEqual(self.step(controller, -2.1), 0.0)     # 到了镜像位置 → 收手
        self.assertEqual(self.step(controller, 9.0), 0.0)      # 收手后照旧强制空
        # --wz-stop-cm 0 = 只翻过中心就收，最早的一种
        crossover = self.controller(turn_s=2.5, gap_s=1.0, stop_cm=0.0)
        self.assertEqual(self.step(crossover, 6.0), 0.5)
        self.assertEqual(self.step(crossover, 0.5), 0.5)
        self.assertEqual(self.step(crossover, -0.1), 0.0)

    def test_the_default_stop_is_the_mirror_of_the_fire_line(self):
        """不写 --wz-stop-cm 就是开火线的镜像：+5 触发 → 转到 −5 才收（从右到左）。"""
        controller = self.controller(turn_s=2.5, gap_s=1.0)
        self.assertEqual(self.step(controller, 6.0), 0.5)
        for _ in range(10):
            self.assertEqual(self.step(controller, 0.5), 0.5)  # 同侧不收
        self.assertEqual(self.step(controller, -4.9), 0.5)     # 还差一点到 −5
        self.assertEqual(self.step(controller, -5.0), 0.0)     # 到了镜像位置（含等号）
        # 负数 = 同侧提前收（上一版的 2cm 语义，现在随手可调）
        same_side = self.controller(turn_s=2.5, gap_s=1.0, stop_cm=-2.0)
        self.assertEqual(self.step(same_side, 6.0), 0.5)
        self.assertEqual(self.step(same_side, 1.9), 0.0)

    def test_a_high_error_still_waits_out_the_gap(self):
        """曾经是"一段跑完误差还在就立刻再开"——那出来的是**连续转弯**。
        形状要求两段之间必须空 2 秒以上，弯道上才是"转一下、滑一段"的多边形。"""
        controller = self.controller(turn_s=0.1, gap_s=2.05)
        fired = [self.step(controller, 6.0) for _ in range(6)]
        self.assertEqual(fired, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.step(controller, 1.0), 0.0)

    def test_the_five_numbers_are_free(self):
        """1.0 / 2.5 只是默认值，不是上下限 —— 曾经把"单段 ≤1s、间隔 >2s"做成
        硬约束，那是拿形状要求去锁调参。现在只查"是不是个能用的数"。"""
        for kw in ({"turn_s": 1.5}, {"turn_s": 3.0}, {"gap_s": 2.0},
                   {"gap_s": 0.5}, {"gap_s": 0.0}, {"step": 0.3},
                   {"stop_cm": 0.0}, {"stop_cm": 5.0}, {"stop_cm": None},
                   {"stop_cm": -2.0}, {"stop_cm": 8.0}):
            with self.subTest(**kw):
                self.controller(**kw)
        for kw in ({"turn_s": 0.0}, {"turn_s": -1.0}, {"gap_s": -0.5},
                   {"step": 0.0}, {"step": 0.9}, {"fire_cm": 0.0},
                   {"turn_s": float("nan")}, {"stop_cm": float("nan")}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                self.controller(**kw)

    def test_a_zero_gap_lets_the_next_turn_start_at_once(self):
        """--wz-gap-s 0 = 不强制滑行：err 还在阈值上就接着开。放开限制之后这条
        路径要能走通。收尾那一帧仍然是 0（那一段真的走完了，这一帧没有指令），
        所以序列是 两帧 0.5 + 一帧 0 循环，不是连续不断。"""
        controller = self.controller(turn_s=0.1, gap_s=0.0)
        fired = [self.step(controller, 6.0) for _ in range(6)]
        self.assertEqual(fired, [0.5, 0.5, 0.0, 0.5, 0.5, 0.0])

    def test_a_lost_frame_is_passed_through_and_fires_nothing(self):
        """丢线那一帧走内层自己的淡出，不脉冲 —— 它发出来的既不是 0 也不是离散
        档，而是内层按 --lost-hold-s 算的中间值，那就证明这一帧没被离散化。"""
        from discrete_steering import DiscreteSteeringController
        inner = SteeringController(**NO_TRIM)
        controller = DiscreteSteeringController(inner)
        self.assertEqual(self.step(controller, 6.0), 0.5)
        lost = dict(detection(error=4.0, lost=3))
        got = controller.command(lost, 0.8, 0.05)[1]
        self.assertAlmostEqual(got, 0.5 * 0.75)
        self.assertNotIn(round(got, 6), (0.0, 0.5, -0.5))
        for _ in range(5):                       # --lost-hold-s 用完就归零
            got = controller.command(lost, 0.8, 0.05)[1]
        self.assertEqual(got, 0.0)

    def test_a_non_finite_error_is_a_lost_frame_too(self):
        controller = self.controller()
        self.assertEqual(self.step(controller, 6.0), 0.5)
        got = controller.command(dict(detection(error=float("nan"))), 0.8, 0.05)[1]
        self.assertNotIn(round(got, 6), (0.0, 0.5, -0.5))

    def test_the_held_command_is_the_pulse_not_the_pid(self):
        """丢线淡出回放的是 hold，所以它必须是脉冲值。"""
        from discrete_steering import DiscreteSteeringController
        inner = SteeringController(**NO_TRIM)
        controller = DiscreteSteeringController(inner)
        self.assertEqual(self.step(controller, 9.0), 0.5)
        self.assertEqual(controller.hold[1], 0.5)
        self.assertEqual(inner.hold[1], 0.5)
        self.assertEqual(inner.last_steer, 0.5)      # 离散日志记录实际输出，不运行隐藏 PID

    def test_the_bias_does_not_move_the_trigger(self):
        """触发看原始 err，不看加过 --bias-cm 的 eff。默认 bias 3.0 而 --wz-fire-cm
        也是 3.0 —— 拿 eff 当触发变量的话，弯道上车正对着中心也会一直打脉冲。"""
        from discrete_steering import DiscreteSteeringController
        inner = SteeringController(bias_cm=10.0, bias_straight_cm=10.0,
                                   bias_dead_px=0.0, bias_gate_px=12.0)
        controller = DiscreteSteeringController(inner)
        self.assertEqual(self.step(controller, 1.0), 0.0)      # err 小于阈值
        self.assertAlmostEqual(inner.last_err_eff, 1.0)         # eff 就是离散阈值输入
        self.assertEqual(self.step(controller, 6.0), 0.5)

    def test_the_amplitude_is_a_fixed_constant(self):
        """0.5 是**实车量出来的好值**，而且挑的是「大且稳」那一端
        （小角度下关节比大角度还不稳）。不跟着 --vx 或任何推导走 ——
        2026-10-03 出过一版按 ω = vx/R 推的（--vx 0.2 下推成 0.258），车上是错的。"""
        for vx in ("0.2", "0.3"):
            with self.subTest(vx=vx), patch("sys.argv",
                                            ["run_policy_vision.py", "--vx", vx]):
                self.assertEqual(run_policy_vision.parse_args().wz_step, 0.5)
        with patch("sys.argv", ["run_policy_vision.py", "--wz-mode", "discrete", "--wz-step", "0.3"]):
            self.assertAlmostEqual(run_policy_vision.parse_args().wz_step, 0.3)

    def test_the_cli_stop_line_has_no_range_limit(self):
        """收手线只是个数：正数 = 另一侧、0 = 中心、负数 = 同侧提前收 —— 不设范围。
        2026-10-03：原来卡 0 ≤ stop ≤ fire，"同侧 3~4cm 就收"和"翻到另一侧更深"
        两种都被拒；限制去掉，只要求是个能用的数。"""
        for value, expected in (("0", 0.0), ("-3.5", -3.5), ("8", 8.0),
                                ("2.5", 2.5)):
            with self.subTest(value=value), \
                    patch("sys.argv", ["run_policy_vision.py",
                                       "--wz-stop-cm", value]):
                self.assertAlmostEqual(
                    run_policy_vision.parse_args().wz_stop_cm, expected)
        with patch("sys.argv", ["run_policy_vision.py", "--wz-stop-cm", "nan"]), \
                self.assertRaises(SystemExit):
            run_policy_vision.parse_args()
        # 不写 = None → 控制器把它解析成 fire 的镜像
        with patch("sys.argv", ["run_policy_vision.py"]):
            self.assertIsNone(run_policy_vision.parse_args().wz_stop_cm)

    def test_reset_drops_a_running_turn(self):
        controller = self.controller(turn_s=1.0)
        self.assertEqual(self.step(controller, 9.0), 0.5)
        controller.reset()
        self.assertEqual(self.step(controller, 0.0), 0.0)
        self.assertEqual(self.step(controller, 9.0), 0.5)

    def test_the_settings_are_validated(self):
        from discrete_steering import DiscreteSteeringController
        inner = SteeringController(**NO_TRIM)
        for kw in (dict(fire_cm=0.0), dict(fire_cm=float("nan")),
                   dict(step=0.0), dict(step=0.9), dict(turn_s=0.0),
                   dict(turn_s=-1.0), dict(gap_s=-0.5),
                   dict(stop_cm=float("nan")),
                   dict(turn_s=float("nan"))):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                DiscreteSteeringController(inner, **kw)


class DiscreteSteeringIntegrationTests(unittest.TestCase):
    """--wz-mode discrete 在真主循环里：发出去的 wz 只会有那 5 个值。"""

    def test_only_the_discrete_levels_reach_the_connector(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        # 直道 → 弯道 → 大偏差，循环喂，让三个档都出现
        errors = [0.3, 0.3, 5.0, 5.0, 5.0, 12.0, 12.0, 12.0, 0.3, 0.3]
        detector.process.side_effect = lambda _f, **_kwargs: (
            0, 0, 0.8, None, detection(error=errors[(reads[0] - 1) % len(errors)]))
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.return_value = (None, {"presence": False, "card_found": False,
                                            "presence_cy_frac": None})
        clock, reads = [0.0], [0]

        def read():
            reads[0] += 1
            clock[0] += 0.05
            if reads[0] > 20:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--wz-mode", "discrete", "--shape-every", "4",
                               "--wz-step", "0.5"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        published = [call.args[1] for call in client_cls.return_value.publish.call_args_list]
        self.assertTrue(published)
        for wz in published:
            with self.subTest(wz=wz):
                self.assertIn(round(wz, 6), (0.0, 0.5, -0.5))
        # 直道那几帧（err=0.3）必须是 0
        self.assertEqual(published[0], 0.0)
        self.assertEqual(published[1], 0.0)


class StartGateTests(unittest.TestCase):
    """两阀是"与"，而且只锁存不撤销。"""

    def test_two_valves_are_anded(self):
        from start_gate import StartGate
        gate = StartGate(mode="both")
        gate.observe_shape("circle")
        gate.observe_shape("circle")
        self.assertTrue(gate.shape_passed)
        self.assertFalse(gate.passed)          # 阀2 过了，阀1 还没
        gate.observe_qr("1")
        self.assertTrue(gate.passed)

    def test_latching_survives_frames_that_report_nothing(self):
        from start_gate import StartGate
        gate = StartGate(mode="both")
        gate.observe_qr("1")
        gate.observe_shape("square")
        gate.observe_shape("square")
        for _ in range(20):
            gate.observe_qr(None)
            gate.observe_shape(None)
        self.assertTrue(gate.qr_passed)
        self.assertTrue(gate.shape_passed)

    def test_the_payload_has_to_match_exactly(self):
        from start_gate import StartGate
        gate = StartGate(mode="qr", expected_qr="1")
        for wrong in ("2", "01", "", " 1 2 "):
            self.assertFalse(gate.observe_qr(wrong))
            self.assertEqual(gate.last_qr, wrong.strip())
            self.assertFalse(gate.qr_passed)
        self.assertTrue(gate.observe_qr(" 1 "))
        self.assertTrue(gate.passed)

    def test_the_shape_needs_consecutive_same_name(self):
        from start_gate import StartGate
        gate = StartGate(mode="shape", shape_confirm=2)
        self.assertFalse(gate.observe_shape("circle"))
        self.assertFalse(gate.observe_shape("square"))   # 换了名字，从头数
        self.assertEqual(gate.shape_streak, 1)
        self.assertTrue(gate.observe_shape("square"))
        self.assertTrue(gate.shape_passed)

    def test_a_none_between_two_readings_does_not_reset_the_streak(self):
        from start_gate import StartGate
        gate = StartGate(mode="shape", shape_confirm=2)
        gate.observe_shape("diamond")
        gate.observe_shape(None)        # 这一帧没跑检测，不该把证据抹掉
        self.assertFalse(gate.shape_passed)
        self.assertTrue(gate.observe_shape("diamond"))

    def test_either_valve_can_be_asked_for_alone(self):
        from start_gate import StartGate
        self.assertTrue(StartGate(mode="qr").observe_qr("1"))
        self.assertTrue(StartGate(mode="qr").passed is False)
        self.assertTrue(StartGate(mode="shape", shape_confirm=1)
                        .observe_shape("cross"))
        self.assertTrue(StartGate(mode="off").passed)
        with self.assertRaises(ValueError):
            StartGate(mode="也许")

    def test_the_status_line_names_both_valves(self):
        from start_gate import StartGate
        gate = StartGate(mode="both", shape_confirm=2)
        gate.observe_shape("triangle")
        text = gate.status()
        self.assertIn("阀1", text)
        self.assertIn("阀2", text)
        self.assertIn("triangle", text)
        self.assertIn("1/2", text)


class QrReaderTests(unittest.TestCase):
    """二维码 round-trip。码是现场生成的，不需要图片素材。"""

    @staticmethod
    def _frame(payload, side_px=110, canvas=(720, 1280)):
        import cv2
        code = cv2.QRCodeEncoder_create().encode(payload)
        scale = max(1, side_px // max(code.shape))
        code = cv2.resize(code, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        gray = np.full(canvas, 255, np.uint8)
        height, width = code.shape
        y = (canvas[0] - height) // 2
        x = (canvas[1] - width) // 2
        gray[y:y + height, x:x + width] = code
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_round_trip_payload_one(self):
        from qr_reader import QrReader
        reader = QrReader()
        reading = reader.decode(self._frame("1"))
        self.assertIsNotNone(reading)
        self.assertEqual(reading.payload, "1")
        self.assertGreater(reading.edge_px, 40.0)
        self.assertEqual(reader.hits, 1)

    def test_a_small_code_needs_the_upscale_pass(self):
        """5cm 的码站在 30~40cm 外只有几十像素，raw 那一遍会漏。"""
        from qr_reader import QrReader
        frame = self._frame("1", side_px=50)
        self.assertIsNotNone(QrReader(upscale=2.0).decode(frame))

    def test_other_payloads_come_back_for_the_gate_to_reject(self):
        """白名单在 StartGate 不在 Reader —— 扫到别的码要能打日志说明。"""
        from qr_reader import QrReader
        reading = QrReader().decode(self._frame("2"))
        self.assertIsNotNone(reading)
        self.assertEqual(reading.payload, "2")

    def test_the_edge_gate_rejects_codes_outside_the_band(self):
        from qr_reader import QrReader
        frame = self._frame("1")
        self.assertIsNone(QrReader(upscale=1.0, min_edge_px=120.0).decode(frame))
        self.assertIsNone(QrReader(upscale=1.0, max_edge_px=50.0).decode(frame))

    def test_a_blank_frame_returns_none_and_still_counts_the_scan(self):
        from qr_reader import QrReader
        reader = QrReader()
        self.assertIsNone(reader.decode(np.full((720, 1280, 3), 255, np.uint8)))
        self.assertEqual((reader.scans, reader.hits), (1, 0))
        self.assertGreaterEqual(reader.last_cost_ms, 0.0)

    def test_a_frame_larger_than_max_side_is_shrunk_first(self):
        """2560x1440 直接放大到 5120x2880 是几百毫秒一次；先缩回 1280
        再解，5cm 的码在 35cm 处还有 ~97px，够用。"""
        from qr_reader import QrReader
        frame = self._frame("1", side_px=220, canvas=(1440, 2560))
        self.assertEqual(QrReader(max_side=1280).decode(frame).payload, "1")


class StartGateIntegrationTests(unittest.TestCase):
    """门控在真主循环里：两阀没全过之前，一个字都不许动。"""

    def _camera(self, frame, clock, reads_out):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        reads = [0]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > reads_out:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        return camera, reads

    def test_the_gate_holds_the_robot_then_hands_the_card_logic_back(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        clock = [0.0]
        camera, reads = self._camera(frame, clock, 16)
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        # 宽 200px 的框：比 43cm 那条停车线（~99px）近得多，所以门控一放开就该
        # 立刻进停车读卡 —— 起点那张卡本来就在线内。
        quad = np.array([[500., 200.], [700., 200.], [700., 300.], [500., 300.]])
        shape = Mock()
        shape.action_map = {"circle": 1}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True, "presence_cy_frac": 0.9,
                    "shape": "circle", "quad_work": quad})
            if reads[0] >= 2
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        reading = SimpleNamespace(payload="1", strategy="raw", edge_px=83.0,
                                  cost_ms=70.0)
        out = io.StringIO()
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--start-gate", "both", "--qr-every", "1",
                               "--shape-every", "1", "--card-every-stopped", "1",
                               "--card-vote-frames", "1", "--card-stop-ms", "500",
                               "--card-hold-ms", "500", "--card-tilt-ms", "0",
                               "--card-settle-ms", "0"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient") as client_cls,
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("qr_reader.QrReader") as qr_cls,
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(out),
        ):
            qr_cls.return_value.decode.return_value = reading
            qr_cls.return_value.scans = 1
            qr_cls.return_value.geom_rejects = 0
            qr_cls.return_value.last_cost_ms = 70.0
            self.assertEqual(run_policy_vision.main(), 0)
        published = client_cls.return_value.publish.call_args_list
        # 读 1：阀1 锁存（二维码）。读 2：图卡连 1/2。读 3：连 2/2 → 释放。
        # 这三帧谁也不许动，而且机身必须是被按住直立的。
        self.assertGreaterEqual(len(published), 6)
        for index in range(3):
            with self.subTest(publish=index):
                self.assertEqual(published[index].args[:2], (0.0, 0.0))
                self.assertTrue(published[index].kwargs["hold_upright"])
                self.assertFalse(published[index].kwargs["card_tilt"])
                self.assertNotIn("event_id", published[index].kwargs)
        # 释放之后，原来那套停车/投票逻辑原样复活，并真的出 event。
        # 同一条 event 会在 --card-hold-ms 内逐帧重发（收端按 id 去重），所以数的是
        # 有几个不同的 id，不是有几帧带着 id。
        events = [call for call in published if call.kwargs.get("event_id")]
        self.assertEqual(len({call.kwargs["event_id"] for call in events}), 1)
        self.assertEqual(events[0].kwargs["event_action"], 1)      # circle
        self.assertEqual(events[0].args[2], 1)                     # qr = 形状号
        # 扫到就锁存：之后一次解码都不该再发生。
        self.assertEqual(qr_cls.return_value.decode.call_count, 1)
        self.assertIn("阀1 通过", out.getvalue())
        self.assertIn("阀2 通过", out.getvalue())
        self.assertIn("start gate released", out.getvalue())

    def test_the_qr_is_only_read_every_n_frames(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        clock = [0.0]
        camera, reads = self._camera(frame, clock, 30)
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"circle": 1}
        shape.update.return_value = (None, {"presence": False, "card_found": False,
                                            "presence_cy_frac": None})
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--start-gate", "qr", "--qr-every", "5"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("qr_reader.QrReader") as qr_cls,
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            qr_cls.return_value.decode.return_value = None
            qr_cls.return_value.scans = 0
            qr_cls.return_value.geom_rejects = 0
            qr_cls.return_value.last_cost_ms = 160.0
            self.assertEqual(run_policy_vision.main(), 0)
        self.assertLessEqual(qr_cls.return_value.decode.call_count, 7)

    def test_a_shape_only_gate_never_builds_a_qr_reader(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        clock = [0.0]
        camera, reads = self._camera(frame, clock, 8)
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, detection())
        shape = Mock()
        shape.action_map = {"circle": 1}
        shape.update.return_value = (None, {"presence": False, "card_found": False,
                                            "presence_cy_frac": None})
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--start-gate", "shape"]),
            patch.object(run_policy_vision.signal, "signal"),
            patch.object(run_policy_vision, "ConnectorClient"),
            patch("utils.open_camera", return_value=camera),
            patch("line_detector_v1_warp.LineDetector", return_value=detector),
            patch("shape_detector.ShapeDetector", return_value=shape),
            patch("qr_reader.QrReader") as qr_cls,
            patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
            patch("cv2.imshow", side_effect=AssertionError("headless must not open windows")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(), 0)
        qr_cls.assert_not_called()

    def test_the_gate_options_are_validated(self):
        for extra in (["--start-gate", "shape", "--no-shape-detect"],
                      ["--start-gate", "qr", "--qr-every", "0"],
                      ["--start-gate", "qr", "--start-gate-qr-payload", " "],
                      ["--start-gate", "both", "--qr-upscale", "0.5"],
                      ["--start-gate", "both", "--qr-min-edge-px", "900"]):
            with self.subTest(extra=extra):
                with patch("sys.argv", ["run_policy_vision.py", *extra]):
                    with self.assertRaises(SystemExit):
                        run_policy_vision.parse_args()
        # 不带门控开关时，一切照旧。
        with patch("sys.argv", ["run_policy_vision.py"]):
            self.assertEqual(run_policy_vision.parse_args().start_gate, "off")


if __name__ == "__main__":
    unittest.main()
