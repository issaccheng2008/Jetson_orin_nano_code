from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from command_source import CommandSnapshot
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


class ShapeMainTests(unittest.TestCase):
    def run_one_tick(self, card):
        with patch("sys.argv", ["main.py", "--model", "walk.onnx", "--no-plot",
                                 "--command-source", "vision", "--one-foot-model", "foot.onnx"]):
            args = main.parse_args()
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        with (
            patch.object(main, "parse_args", return_value=args),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as walk_cls,
            patch.object(main, "OneFootPolicy") as foot_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "PositionCsvLogger"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            walk = walk_cls.return_value
            foot = foot_cls.return_value
            walk.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            foot.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(46), 0.0)
            source_cls.return_value.get_snapshot.return_value = CommandSnapshot(
                np.zeros(3, dtype=np.float32), -1, 765, card)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, RuntimeError("end test")]
            link.get_action_status.return_value = 0
            self.assertEqual(main.main(), 1)
            return walk, foot, link

    def test_arm_card_sends_only_upper_body_request(self):
        walk, foot, link = self.run_one_tick(1)
        link.send_action.assert_called_once_with(765, 1)
        walk.step.assert_called_once()
        foot.step.assert_not_called()

    def test_square_uses_right_support_one_foot_model_without_mcu_action(self):
        walk, foot, link = self.run_one_tick(3)
        link.send_action.assert_not_called()
        foot.select_support_foot.assert_called_once_with("right")
        self.assertEqual(foot.step.call_args.kwargs["lift_command"], 0.0)
        walk.step.assert_not_called()

    def test_diamond_uses_left_support(self):
        _walk, foot, link = self.run_one_tick(4)
        link.send_action.assert_not_called()
        foot.select_support_foot.assert_called_once_with("left")


if __name__ == "__main__":
    unittest.main()
