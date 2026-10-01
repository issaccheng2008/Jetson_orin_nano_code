from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


class WalkingModeTests(unittest.TestCase):
    def args(self, *extra):
        with patch("sys.argv", ["main.py", "--model", "test.onnx", "--no-plot", *extra]):
            return main.parse_args()

    def test_defaults_and_vision_or_turn_options(self):
        args = self.args()
        self.assertEqual((args.command_source, args.vx, args.wz), ("fixed", 0.4, 0.0))
        self.assertEqual(self.args("--command-source", "vision").udp_command_port, 5005)
        self.assertEqual(self.args("--wz", "0.5").wz, 0.5)

    def test_rejects_stop_and_invalid_forward_commands(self):
        for vx in ("0", "-0.1", "nan", "inf", "1.1"):
            with self.subTest(vx=vx), patch.object(main, "parse_args", return_value=self.args("--vx", vx)):
                with self.assertRaisesRegex(SystemExit, "vx must"):
                    main.main()

    def test_runtime_keeps_walking_without_vision_and_disables_on_state_fault(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        with (
            patch.object(main, "parse_args", return_value=self.args()),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "PositionCsvLogger"),
            patch("command_source.socket.socket", side_effect=AssertionError("UDP must stay disconnected")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            self.assertEqual(policy.step.call_count, 2)
            for call in policy.step.call_args_list:
                np.testing.assert_allclose(call.kwargs["velocity_command"], [0.4, 0, 0])
            self.assertEqual(link.send_command.call_args.args[-1], 0)
            link.close.assert_called_once()


    def test_live_vision_commands_are_not_overridden_after_five_seconds(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        with (
            patch.object(main, "parse_args", return_value=self.args("--command-source", "vision")),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "PositionCsvLogger"),
            patch.object(main.time, "monotonic", side_effect=[0, 10, 10, 10, 11, 11, 11, 12]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            source = source_cls.return_value
            source.get.side_effect = [np.array([0.4, 0, -0.2]), np.array([0.4, 0, 0.3])]
            source.status.return_value = {
                "fresh": True, "age_s": 0.01,
                "valid_packets": 10, "invalid_packets": 0,
            }
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            source_cls.assert_called_once_with(5005, timeout_s=0.25, bind="127.0.0.1")
            self.assertEqual(policy.step.call_count, 2)
            for call, expected in zip(policy.step.call_args_list, ([0.4, 0, -0.2], [0.4, 0, 0.3])):
                np.testing.assert_allclose(call.kwargs["velocity_command"], expected)
            source.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
