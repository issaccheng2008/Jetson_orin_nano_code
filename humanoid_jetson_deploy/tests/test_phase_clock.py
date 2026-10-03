from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
from models.phase_clock_model_850.deployment.phase_clock import PhaseClockConfig
from models.phase_clock_model_850.deployment.policy_interface import PolicyController
from phase_clock_adapter import DEFAULT_BUNDLE, load_crossing_controller
import phase_clock_main
from phase_clock_main import StartCueSocket, initial_pose_ready
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID, StatePacket


class PhaseClockIntegrationTests(unittest.TestCase):
    def test_bundle_contract_and_copied_files(self):
        with patch("phase_clock_adapter.PolicyController.from_onnx") as loader:
            load_crossing_controller()
        path, clock = loader.call_args.args
        self.assertEqual(Path(path), DEFAULT_BUNDLE / "policy.onnx")
        self.assertEqual(clock.boundary_ticks, (7, 22, 38))
        self.assertEqual(clock.command_at_tick(7), (0.2, 0.0, 0.23, 1.0))

    def test_one_start_49_inputs_phase_boundaries_and_finish(self):
        observations = []

        def inference(obs):
            observations.append(obs.copy())
            return np.full((1, 12), len(observations), dtype=np.float32)

        controller = PolicyController(inference, PhaseClockConfig())
        controller.reset()
        self.assertFalse(controller.tick(
            np.array([0, 0, 9.81]), np.zeros(3), np.array([0, 0, -1]),
            config.Q_DEFAULT, np.zeros(12), now=100.0).active)
        self.assertTrue(controller.start(100.0))
        self.assertFalse(controller.start(100.1))
        values = (np.array([0, 0, 9.81]), np.zeros(3), np.array([0, 0, -1]),
                  config.Q_DEFAULT, np.zeros(12))
        phases = []
        for tick in (0, 6, 7, 21, 22, 37, 38):
            sample = controller.tick(*values, now=100.0 + tick * 0.02)
            phases.append(sample.phase)
            if tick == 38:
                self.assertTrue(sample.sequence_finished)
                self.assertFalse(sample.active)
                self.assertIsNone(sample.joint_targets)
            else:
                self.assertEqual(sample.joint_targets.shape, (12,))
        self.assertEqual(phases, [0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(len(observations), 6)
        np.testing.assert_allclose(observations[0][0, :13],
                                   [0, 0, .981, 0, 0, 0, 0, 0, -1,
                                    .2, 0, .10, 0], atol=1e-6)
        np.testing.assert_allclose(observations[2][0, 9:13], [.2, 0, .23, 1])
        np.testing.assert_allclose(observations[4][0, 9:13], [.2, 0, 0, 1])
        np.testing.assert_allclose(observations[1][0, 37:49], np.ones(12))

    def test_start_signal_is_explicit_loopback_json(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as finder:
            finder.bind(("127.0.0.1", 0))
            port = finder.getsockname()[1]
        cue = StartCueSocket("127.0.0.1", port)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(b'{"start": false}', ("127.0.0.1", port))
                time.sleep(0.01)
                self.assertFalse(cue.poll())
                sender.sendto(b'{"start": true}', ("127.0.0.1", port))
                deadline = time.monotonic() + 0.2
                while not cue.poll() and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertLess(time.monotonic(), deadline)
                self.assertFalse(cue.poll())
        finally:
            cue.close()
        with self.assertRaisesRegex(ValueError, "loopback"):
            StartCueSocket("0.0.0.0", port)

    def test_initial_pose_and_speed_gate(self):
        self.assertTrue(initial_pose_ready(config.Q_DEFAULT, np.zeros(12), 10, .2))
        q = config.Q_DEFAULT.copy()
        q[0] += np.deg2rad(11)
        self.assertFalse(initial_pose_ready(q, np.zeros(12), 10, .2))
        q[0] = config.Q_DEFAULT[0]
        qd = np.zeros(12)
        qd[3] = .21
        self.assertFalse(initial_pose_ready(q, qd, 10, .2))

    def test_control_loop_disables_after_done_without_restarting(self):
        clock = PhaseClockConfig(walk_end_s=.02, lead_end_s=.04,
                                 sequence_end_s=.06)
        inference = Mock(return_value=np.zeros((1, 12), dtype=np.float32))
        controller = PolicyController(inference, clock)
        state = StatePacket(
            sequence=1, timestamp_us=123456,
            joint_position=config.policy_to_motor_position(config.Q_DEFAULT),
            joint_velocity=np.zeros(12, dtype=np.float32),
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
        )
        link = Mock()
        link.wait_for_state.return_value = state
        link.get_latest_state.return_value = state
        cue = Mock()
        cue.poll.side_effect = lambda: True
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(
                bundle=DEFAULT_BUNDLE, start_bind="127.0.0.1", start_port=5008,
                port="fake", baud=921600, enable_motors=False,
                kp_scale=1., kd_scale=1., pose_tolerance_deg=10.,
                max_joint_speed=.2, log=Path(folder) / "phase.csv")
            with patch("phase_clock_main.parse_args", return_value=args), \
                    patch("phase_clock_main.load_crossing_controller",
                          return_value=controller), \
                    patch("phase_clock_main.StartCueSocket", return_value=cue), \
                    patch("phase_clock_main.SerialLink", return_value=link), \
                    patch("phase_clock_main.send_disable") as disable, \
                    redirect_stdout(io.StringIO()):
                result = phase_clock_main.main()
            self.assertEqual(result, 0)
            self.assertTrue(link.send_command.called)
            disable.assert_called_once()
            self.assertFalse(disable.call_args.kwargs["estop"])
            self.assertEqual(controller.clock.read().phase, 3)
            self.assertLessEqual(inference.call_count, 3)
            self.assertIn("3,", (Path(folder) / "phase.csv").read_text())


if __name__ == "__main__":
    unittest.main()
