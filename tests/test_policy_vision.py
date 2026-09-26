"""Hardware-free controller and real UDP bridge tests."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
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
from policy_runner import HumanoidPolicy
import config
import run_policy_vision


def detection(error=10.0, angle=0.0, lost=0, curve=False, curve_px=0.0):
    return dict(fused_err=error / 52.8, fused_err_cm=error, angle_err_deg=angle,
                lost_frames=lost, curve_mode=curve, bottom_lock_valid=True,
                curve_px=curve_px)


class SteeringTests(unittest.TestCase):
    def test_sign_units_clamping_and_preview(self):
        # yaw_sign and preview_gain pinned so this tests the maths, not the defaults.
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50,
                                        yaw_sign=-1, preview_gain=4)
        np.testing.assert_allclose(controller.command(detection(), 0.8, 0.02), [0.4, -0.1])
        self.assertGreater(controller.command(detection(-10), 0.8, 0.02)[1], 0)
        self.assertEqual(controller.command(detection(1000), 0.8, 0.02)[1], -0.5)
        self.assertEqual(controller.command(detection(-1000), 0.8, 0.02)[1], 0.5)
        self.assertLess(controller.command(detection(0, 30), 0.8, 0.02)[1], 0)
        reverse = SteeringController(yaw_sign=1)
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
        controller = SteeringController()
        self.assertEqual(controller.command(detection(0.0, 22.0), 0.8, 0.02)[1], 0.0)

    def test_bias_fades_in_with_curve_px_and_moves_the_zero_point(self):
        trimmed = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50,
                                     bias_cm=5.0, bias_gate_px=10.0)
        trimmed.command(detection(0.0, curve_px=0.0), 0.8, 0.02)
        self.assertEqual(trimmed.last_err_eff, 0.0)        # straight: straights untouched
        trimmed.command(detection(0.0, curve_px=5.0), 0.8, 0.02)
        self.assertAlmostEqual(trimmed.last_err_eff, 2.5)  # half way to the gate
        trimmed.command(detection(0.0, curve_px=-50.0), 0.8, 0.02)
        self.assertAlmostEqual(trimmed.last_err_eff, 5.0)  # gate saturated
        self.assertGreater(trimmed.command(detection(0.0, curve_px=-50.0), 0.8, 0.02)[1], 0.0)
        # With the trim open, the loop now settles where the raw reading is -5 cm.
        self.assertEqual(trimmed.command(detection(-5.0, curve_px=-50.0), 0.8, 0.02)[1], 0.0)
        # A missing curve_px (older debug dict) must not open the gate.
        self.assertEqual(SteeringController(bias_cm=5.0)
                         .command({**detection(0.0), "curve_px": None}, 0.8, 0.02)[1], 0.0)

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

    def test_invalid_settings_are_rejected(self):
        for kwargs in (dict(max_wz=1.5), dict(vx=float("nan")), dict(yaw_sign=0),
                       dict(steer_full_scale_cm=0), dict(step_len_cm=-1),
                       dict(lost_hold_s=-1.0), dict(deriv_pole=1.0),
                       dict(deriv_pole=-0.1), dict(bias_cm=float("nan")),
                       dict(bias_gate_px=0.0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SteeringController(**kwargs)


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

    def test_card_event_is_held_but_qr_clears_when_card_disappears(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]
        detector = Mock()
        detector.process.return_value = (0, 0, 0.8, None, detection())
        shape = Mock()
        shape.update.side_effect = [(3, {})] + [(None, {})] * 2  # fires once, like the cooldown
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
        # Then it drives past the card on the replayed command, at the slow cap,
        # until confidence is back: the mock reports 0.8, which is --card-clear-conf.
        # wz must be held constant - it is the average from before the stop, not
        # anything re-derived from the frames where the card sits in the near band.
        self.assertEqual(len({published[resumed + k].args[1] for k in (0, 1)}), 1)
        for offset in (0, 1):
            self.assertAlmostEqual(published[resumed + offset].args[0], 0.2)
        self.assertGreater(published[resumed + 2].args[0], 0.3)

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
            (None, {"presence": True, "presence_cy_frac": plan[reads[0]]})
            if reads[0] in plan else (None, {"presence": False, "presence_cy_frac": None}))
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
        controller = SteeringController(straight_gains=(1, 0, 0), steer_full_scale_cm=50,
                                        yaw_sign=-1)
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
