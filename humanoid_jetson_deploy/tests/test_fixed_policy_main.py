from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


class FixedPolicyMainTests(unittest.TestCase):
    def test_second_fixed_frame_waits_for_new_state_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.json"
            path.write_text(json.dumps({
                "format": "joint_frames_v1", "hz": 50,
                "joint_names": list(config.JOINT_NAMES),
                "frames": [config.Q_DEFAULT.tolist(), config.Q_DEFAULT.tolist()],
            }), encoding="utf-8")
            with patch("sys.argv", ["main.py", "--fixed-policy", str(path), "--no-plot"]):
                args = main.parse_args()
            base = dict(
                status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
                accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
                gyro_rad_s=np.zeros(3, dtype=np.float32),
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                joint_position=config.Q_DEFAULT.copy(),
                joint_velocity=np.zeros(12, dtype=np.float32),
            )
            first = SimpleNamespace(**base, sequence=1)
            second = SimpleNamespace(**base, sequence=2)
            third = SimpleNamespace(**base, sequence=3)
            with (
                patch.object(main, "parse_args", return_value=args),
                patch.object(main.signal, "signal"),
                patch.object(main, "HumanoidPolicy"),
                patch.object(main, "SerialLink") as link_cls,
                patch.object(main, "PositionCsvLogger") as logger_cls,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                link = link_cls.return_value
                link.wait_for_state.return_value = first
                link.get_latest_state.side_effect = [first, first, second, second, second, third]
                self.assertEqual(main.main(), 0)
                self.assertGreaterEqual(link.get_latest_state.call_count, 3)
                self.assertEqual(logger_cls.return_value.write.call_args_list[1].args[2], 2)
                self.assertEqual(link.send_command.call_args_list[1].args[-1], 0)

    def test_fixed_mode_runs_without_onnx_and_finishes_after_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.json"
            path.write_text(json.dumps({
                "format": "joint_frames_v1", "hz": 50,
                "joint_names": list(config.JOINT_NAMES),
                "frames": [config.Q_DEFAULT.tolist()],
            }), encoding="utf-8")
            with patch("sys.argv", ["main.py", "--fixed-policy", str(path), "--no-plot"]):
                args = main.parse_args()
            state = SimpleNamespace(
                status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
                accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
                gyro_rad_s=np.zeros(3, dtype=np.float32),
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                joint_position=config.Q_DEFAULT.copy(),
                joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
            )
            response = SimpleNamespace(**{**vars(state), "sequence": 2})
            with (
                patch.object(main, "parse_args", return_value=args),
                patch.object(main.signal, "signal"),
                patch.object(main, "HumanoidPolicy") as onnx_cls,
                patch.object(main, "SerialLink") as link_cls,
                patch.object(main, "PositionCsvLogger"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                link = link_cls.return_value
                link.wait_for_state.return_value = state
                link.get_latest_state.side_effect = [state, response]
                self.assertEqual(main.main(), 0)
                onnx_cls.assert_not_called()
                np.testing.assert_allclose(link.send_command.call_args_list[0].args[1], config.Q_DEFAULT)
                self.assertEqual(link.send_command.call_args_list[0].args[-1], 0)


if __name__ == "__main__":
    unittest.main()
