from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import main
from command_source import CommandSnapshot
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class OneFootAngleLogTests(unittest.TestCase):
    def test_two_files_share_sample_keys_and_keep_angles_separate(self):
        from one_foot_angle_log import OneFootAngleCsvLogger

        names = ("right_knee", "left_knee")
        with tempfile.TemporaryDirectory() as folder:
            logger = OneFootAngleCsvLogger(folder, names)
            other = OneFootAngleCsvLogger(folder, names)
            self.assertNotEqual(logger.target_path, other.target_path)
            self.assertEqual(logger.target_path.name.replace("targets", "feedback"),
                             logger.feedback_path.name)
            logger.write(.02, 1, 7, [.3, -.4], [.2, -.25])
            # Samples should already be available while the run is active.
            target = read_rows(logger.target_path)[0]
            actual = read_rows(logger.feedback_path)[0]
            for key in ("host_time_iso", "elapsed_s", "step", "state_sequence"):
                self.assertEqual(target[key], actual[key])
            self.assertEqual(target["step"], "1")
            self.assertEqual(target["state_sequence"], "7")
            self.assertAlmostEqual(float(target["target_right_knee_rad"]), .3)
            self.assertAlmostEqual(float(actual["actual_left_knee_rad"]), -.25)
            self.assertFalse(any(key.startswith("actual_") for key in target))
            self.assertFalse(any(key.startswith("target_") for key in actual))
            with self.assertRaises(ValueError):
                logger.write(.04, 2, 8, [.4, -.5], [.2])
            self.assertEqual(len(read_rows(logger.target_path)), 1)
            self.assertEqual(len(read_rows(logger.feedback_path)), 1)
            logger.close()
            logger.close()
            other.close()

    def run_mode(self, folder, mode, results=(True, True, True)):
        argv = ["main.py", "--model", "fake.onnx", "--no-plot",
                "--no-control-diagnostics", "--attitude-port", "0",
                "--position-log-dir", folder]
        if mode == "standalone":
            argv += ["--policy", "one-foot", "--support-foot", "left"]
        elif mode == "vision":
            argv += ["--command-source", "vision", "--one-foot-model", "foot.onnx"]
        with patch("sys.argv", argv):
            args = main.parse_args()
        states = [
            SimpleNamespace(
                status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
                accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
                gyro_rad_s=np.zeros(3, dtype=np.float32),
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                joint_position=config.Q_DEFAULT.copy() + i * .01,
                joint_velocity=np.zeros(12), sequence=11 + i,
            )
            for i in range(3)
        ]
        walk, foot, link, source = Mock(), Mock(), Mock(), Mock()
        # Large raw targets exercise the safety limits before transmission.
        foot.step.return_value = (config.Q_DEFAULT.copy() + .8,
                                  np.ones(12), np.zeros(46), 0.)
        walk.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.)
        source.get_snapshot.side_effect = [
            CommandSnapshot(np.zeros(3), -1, 0, -1),
            CommandSnapshot(np.zeros(3), -1, 765, 3),
            CommandSnapshot(np.zeros(3), -1, 765, 3),
        ]
        link.wait_for_state.return_value = states[0]
        link.get_latest_state.side_effect = states + [RuntimeError("end test")]
        link.send_command.side_effect = list(results) + [True] * 10
        with (
            patch.object(main, "parse_args", return_value=args),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy", return_value=walk),
            patch.object(main, "OneFootPolicy", return_value=foot),
            patch.object(main, "UdpCommandSource", return_value=source),
            patch.object(main, "SerialLink", return_value=link),
            patch.object(main.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(), 1)
        return states, link

    def test_standalone_exports_sent_motor_targets_and_received_motor_angles(self):
        with tempfile.TemporaryDirectory() as folder:
            states, link = self.run_mode(folder, "standalone")
            targets = read_rows(next(Path(folder).glob("onefoot_targets_*.csv")))
            feedback = read_rows(next(Path(folder).glob("onefoot_feedback_*.csv")))
            self.assertEqual(len(targets), 3)
            self.assertEqual(len(feedback), 3)
            for i, (target, actual) in enumerate(zip(targets, feedback)):
                self.assertEqual(target["state_sequence"], str(states[i].sequence))
                np.testing.assert_allclose(
                    [float(target[f"target_{name}_rad"]) for name in config.JOINT_NAMES],
                    link.send_command.call_args_list[i].args[1], atol=1e-8)
                np.testing.assert_allclose(
                    [float(actual[f"actual_{name}_rad"]) for name in config.JOINT_NAMES],
                    states[i].joint_position, atol=1e-8)
            self.assertEqual(len(list(Path(folder).glob("motor_positions_*.csv"))), 1)

    def test_vision_creates_no_one_foot_policy_log(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_mode(folder, "vision")
            self.assertEqual(list(Path(folder).glob("onefoot_*.csv")), [])

    def test_walking_creates_no_one_foot_files(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_mode(folder, "walking")
            self.assertEqual(list(Path(folder).glob("onefoot_*.csv")), [])

    def test_unsent_and_failed_commands_are_not_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_mode(folder, "standalone", (True, False, RuntimeError("write failed")))
            for prefix in ("targets", "feedback"):
                rows = read_rows(next(Path(folder).glob(f"onefoot_{prefix}_*.csv")))
                self.assertEqual([row["step"] for row in rows], ["0"])


if __name__ == "__main__":
    unittest.main()
