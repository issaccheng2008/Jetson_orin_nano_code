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
    def test_sign_units_clamping_and_preview(self):
        # yaw_sign and preview_gain pinned so this tests the maths, not the defaults.
        controller = SteeringController(**NO_TRIM, straight_gains=(1, 0, 0),
                                        steer_full_scale_cm=50, yaw_sign=-1,
                                        preview_gain=4)
        np.testing.assert_allclose(controller.command(detection(), 0.8, 0.02), [0.4, -0.1])
        self.assertGreater(controller.command(detection(-10), 0.8, 0.02)[1], 0)
        # Saturated negative is a right turn, and right is capped at max_wz_right.
        self.assertEqual(controller.command(detection(1000), 0.8, 0.02)[1], -0.25)
        self.assertEqual(controller.command(detection(-1000), 0.8, 0.02)[1], 0.5)
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

    def test_invalid_or_lost_detection_stops_and_resets(self):
        controller = SteeringController(lost_hold_s=0.0)
        controller.command(detection(), 0.8, 0.02)
        for dbg, confidence in ((detection(lost=1), 0.8), ({}, 0.8),
                                (detection(float("nan")), 0.8), (detection(), 0),
                                (detection(), float("nan"))):
            self.assertEqual(controller.command(dbg, confidence, 0.02), (0, 0))
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

    def test_line_loss_holds_briefly_then_goes_zero(self):
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50)
        held = controller.command(detection(), 0.8, 0.02)
        self.assertEqual(held[0], 0.4)
        for _ in range(3):  # still inside the default 0.2 s window
            self.assertEqual(controller.command(detection(lost=1), 0.8, 0.05), held)
        for _ in range(6):  # window exhausted
            final = controller.command(detection(lost=1), 0.8, 0.05)
        self.assertEqual(final, (0.0, 0.0))
        # A NaN dt must not stall the accumulator and latch the hold for ever.
        self.assertEqual(controller.command(detection(lost=1), 0.8, float("nan")), (0.0, 0.0))

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
        # Past the lane half-width it is loss: hold the last command, then stop.
        held = controller.command(detection(), 0.8, 0.02)
        self.assertEqual(held[0], 0.4)
        self.assertEqual(
            controller.command(detection(lateral=-35.9), 0.8, 0.05), held)
        self.assertAlmostEqual(controller.rejected_lateral, -35.9)
        for _ in range(6):
            final = controller.command(detection(lateral=-35.9), 0.8, 0.05)
        self.assertEqual(final, (0.0, 0.0))

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
        self.assertEqual(moving, (0.4, 0.5))
        # Idle through the stop; clear_hold is what run_policy_vision passes.
        for _ in range(5):
            controller.reset(clear_hold=True)
        self.assertEqual(controller.hold, (0.0, 0.0))
        self.assertEqual(controller.command(detection(lost=1), 0.9, 0.05), (0.0, 0.0))
        # And without it, the old behaviour is still there for the lost-line case.
        kept = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=10)
        kept.command(detection(20.0), 0.9, 0.05)
        kept.reset()
        self.assertEqual(kept.command(detection(lost=1), 0.9, 0.05), (0.4, 0.5))

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

        self.assertAlmostEqual(doubled.command(both, 0.8, 0.02)[1],
                               plain.command(both, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(single, 0.8, 0.02)[1],
                               2.0 * plain.command(single, 0.8, 0.02)[1])
        self.assertAlmostEqual(doubled.command(single_straight, 0.8, 0.02)[1],
                               plain.command(single_straight, 0.8, 0.02)[1])
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
            {"presence": True, "presence_cy_frac": 0.9},
        )
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-hold-ms", "5000"]),
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

    def test_headless_runner_publishes_detection_and_zero_on_capture_failure(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
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
            self.assertEqual(calls[2].args[0], 0.4)
            self.assertNotEqual(calls[2].args[1], 0.0)  # steering published after median warmup
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

        def process(_frame):
            seen[0] += 1
            if seen[0] <= 2:
                return (0, 0, 0.9, None, detection())
            return (0, 0, 0.0, None, detection(lost=1))    # conf 0 and lost

        detector.process.side_effect = process
        shape = Mock()
        shape.action_map = {"square": 3}
        shape.update.side_effect = lambda *a, **k: (
            (None, {"presence": True, "card_found": True, "presence_cy_frac": 0.9})
            if seen[0] >= 2
            else (None, {"presence": False, "card_found": False,
                         "presence_cy_frac": None}))
        clock = [0.0]

        def tick():
            clock[0] += 0.1
            return clock[0]

        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1"]),
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
        # A shape is only acted on once the card has reached the trigger line; the
        # centroid has to be reported for that to be knowable.
        shape.update.side_effect = [(3, {"presence_cy_frac": 0.9})] + [(None, {})] * 2
        shape.action_map = {"square": 3}
        with (
            patch("sys.argv", ["run_policy_vision.py", "--headless",
                               "--shape-every", "1", "--card-hold-ms", "3000"]),
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
            self.assertEqual(published, [3, -1, -1, -1])
            events = [call.kwargs for call in client_cls.return_value.publish.call_args_list]
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
            (3, {"presence": True, "presence_cy_frac": 0.9}) if seen["card"]
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
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-hold-ms", "5000", "--card-stop-ms", "3000"]),
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
        # read 1 drives; read 2 sees the box and names it, opening the window at t=0.2
        self.assertGreater(published[0].args[0], 0.0)
        self.assertEqual(published[0].args[2], -1)
        self.assertEqual(published[1].args[:2], (0.0, 0.0))
        self.assertEqual(published[1].args[2], 3)
        # The window is 5000 ms at 10 Hz, so it lifts around publish 52 (read 53).
        # Asserted as a window rather than an exact index: the mocked clock
        # accumulates 0.1 fifty-odd times and lands either side of the boundary.
        resumed = next(i for i, call in enumerate(published)
                       if i > 1 and call.args[0] > 0.0)
        self.assertIn(resumed, (51, 52, 53))
        self.assertEqual(published[resumed - 1].args[:2], (0.0, 0.0))
        self.assertEqual(published[resumed - 1].kwargs["event_action"], 3)
        self.assertEqual(published[resumed].args[2], -1)      # released with the resume
        # It drives again the moment the window closes - there is no blind clearance
        # stage. The controller was idle all through the stop and its loop state is kept
        # (the stop is a pause), so this lands near a fresh controller's value for the
        # same frame rather than exactly on it. What has to hold is that nothing from
        # the card-corrupted frames survived: command() was never called while stopped.
        fresh = SteeringController().command(detection(), 0.8, 0.1)
        self.assertAlmostEqual(published[resumed].args[0], fresh[0])
        self.assertAlmostEqual(published[resumed].args[1], fresh[1], delta=1e-3)
        self.assertGreater(published[resumed].args[0], 0.3)

    def test_a_distant_card_only_slows_down_and_a_flicker_does_not_re_trigger(self):
        """The cue drops out while walking. One absent call used to re-arm the stop,
        so the robot crept forward and stopped again, then sat there for good."""
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
                               "--card-slow-vx", "0.2", "--card-trigger-frac", "0.75",
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
        self.assertAlmostEqual(published[0].args[0], 0.4)      # read 1: no card, full speed
        self.assertAlmostEqual(published[1].args[0], 0.2)      # read 2: card seen, slows
        self.assertAlmostEqual(published[4].args[0], 0.2)      # read 5: still slow after the blip
        self.assertAlmostEqual(published[6].args[0], 0.2)      # read 7: centroid still high
        self.assertAlmostEqual(published[7].args[0], 0.0)      # read 8: centroid low -> stops
        self.assertEqual(out.getvalue().count("stand still"), 1)  # triggered exactly once
        from line_detector_v1_warp import LineDetector
        detector = LineDetector(1280, 720)
        _, _, confidence, _, debug = detector.process(np.full((720, 1280, 3), 255, np.uint8))
        self.assertEqual(SteeringController().command(debug, confidence, 0.03), (0, 0))

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
                 "presence_cy_frac": plan[reads[0]]})
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
                               "--card-trigger-frac", "0.5"]),
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
        qr = [call.args[2] for call in client_cls.return_value.publish.call_args_list]
        self.assertEqual(qr[1:4], [-1, -1, -1])
        self.assertEqual(qr[4], 3)
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

    def test_the_angle_term_pushes_the_way_the_curve_goes(self):
        """Three encodings of one direction: fused_err > 0 is the left correction,
        angle_err > 0 is "the lane ahead goes left", curve_px < 0 is a left curve. On
        one measured left-curve fixture all three hold together, so the angle term has
        to add to the correction. It was negated, which steered right on a left curve -
        against the curve term and against run_robot.py:257, which carries the same
        heading into its preview term with a plus."""
        def steady(angle_gain):
            detector = self._detector()
            detector.pix_angle_gain = angle_gain
            image = self._arc_lane(0.00057)
            for _ in range(detector.startup_settle_frames + 4):
                _, _, _, _, debug = detector.process(image)
            return debug

        curved = steady(self._detector().pix_angle_gain)
        flat = steady(0.0)                                  # the same frame, no angle term
        self.assertLess(curved["curve_px"], 0.0)
        self.assertGreater(curved["angle_err_deg"], 0.0)
        self.assertGreater(curved["fused_err_cm"], flat["fused_err_cm"])

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

    def test_a_single_line_curve_reaches_curve_mode(self):
        """The single-line gain is gated on curve_mode, and one visible boundary makes
        curve_px read ~0 - both bands place the same half-width offset off the same run.
        Without the heading clause the gain could never fire on the frames it exists
        for; without the straight case it would fire on straights."""
        import cv2
        def frame(dx):
            image = np.full((720, 1280, 3), 255, np.uint8)
            cv2.line(image, (640, 719), (640 + dx, 300), (0, 0, 0), 20)
            return image

        def steady(dx):
            detector = self._detector()
            for _ in range(detector.startup_settle_frames + 3):
                _, _, _, _, debug = detector.process(frame(dx))
            return debug

        straight = steady(0)
        self.assertTrue(straight["single_line"])
        self.assertFalse(straight["curve_mode"])
        self.assertTrue(steady(160)["curve_mode"])

    def test_a_one_frame_curve_spike_does_not_reach_the_bias_gate(self):
        """The gate for the standing trim reads this, not the one-frame curve_px. A
        straight's jitter reaches past any threshold a real curve (9-14 px) also
        reaches, so only the average tells the two apart - which is why curve_mode as
        a one-frame test came out anti-correlated with curvature."""
        import cv2
        image = np.full((720, 1280, 3), 255, np.uint8)
        cv2.line(image, (640, 719), (760, 300), (0, 0, 0), 20)
        _, _, _, _, debug = self._detector().process(image)
        self.assertGreater(abs(debug["curve_px"]), 12.0)        # the spike
        self.assertLess(abs(debug["curve_px_smooth"]), 4.0)     # held back by the EMA

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

    def test_a_one_line_frame_reaches_the_loop_quickly(self):
        """The frame's confidence is the weight of its step into the error EMA, so a
        low value here is latency. At 0.127 against 0.99 for a paired frame it put the
        single-line correction about 1.75 s behind instead of 0.23 s."""
        import cv2
        image = np.full((720, 1280, 3), 255, np.uint8)
        cv2.line(image, (640, 719), (760, 300), (0, 0, 0), 20)
        detector = self._detector()
        for _ in range(detector.startup_settle_frames + 3):
            _, _, confidence, _, debug = detector.process(image)
        self.assertTrue(debug["single_line"])
        self.assertGreater(confidence, 0.5)
        # And the step it actually takes, against a fully-paired frame's 0.28.
        self.assertGreaterEqual((1.0 - detector.smooth_alpha) * confidence, 0.15)


class LineDetectorStateTests(unittest.TestCase):
    """The detector's cross-frame memory is what keeps a bad lock alive: the scan
    hint feeds the next frame's search, and smoothed_err is a long EMA."""

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

    def test_a_low_confidence_frame_barely_moves_the_error(self):
        """conf is the detector's own verdict on a reading, and the fusion has to
        obey it. Before this, a 0.07-confidence frame moved smoothed_err exactly as
        hard as a 0.99 one - which is how one garbage frame became full lock."""
        from line_detector_v1_warp import confidence_weighted_ema
        self.assertAlmostEqual(confidence_weighted_ema(0.0, 1.0, 0.72, 0.99), 0.2772)
        self.assertAlmostEqual(confidence_weighted_ema(0.0, 1.0, 0.72, 0.07), 0.0196)
        # A frame it gives no confidence to cannot move the error at all.
        self.assertEqual(confidence_weighted_ema(0.5, -1.0, 0.72, 0.0), 0.5)

    def test_the_fusion_update_goes_through_the_confidence_weight(self):
        """Guards the call site, not just the helper - the inline EMA that used to
        be there ignored confidence, and the helper test above would not notice it
        coming back."""
        import cv2
        import line_detector_v1_warp as ld
        frame = np.full((720, 1280, 3), 255, np.uint8)
        cv2.line(frame, (610, 719), (670, 300), (0, 0, 0), 24)
        detector = ld.LineDetector(1280, 720)
        real = ld.confidence_weighted_ema
        seen = []

        def spy(previous, fused, alpha, confidence):
            seen.append(confidence)
            return real(previous, fused, alpha, confidence)

        with patch.object(ld, "confidence_weighted_ema", side_effect=spy):
            detector.process(frame)
        self.assertEqual(len(seen), 1)
        self.assertGreater(seen[0], 0.0)

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
        detector.process.side_effect = lambda _f: record(
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
            patch("sys.argv", ["run_policy_vision.py", "--headless", "--shape-every", "1",
                               "--card-every-stopped", "2"]),
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
        expected = [0.4, 0.0, -0.1]

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
            def observation():
                return policy.build_observation(
                    accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
                    projected_gravity=np.array([0, 0, -1]), velocity_command=source.get(),
                    joint_position_policy=config.Q_DEFAULT, joint_velocity_policy=np.zeros(12))
            np.testing.assert_allclose(observation()[9:13], [0.4, -0.1, config.DEFAULT_STEP_DISTANCE, 0])
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


if __name__ == "__main__":
    unittest.main()
