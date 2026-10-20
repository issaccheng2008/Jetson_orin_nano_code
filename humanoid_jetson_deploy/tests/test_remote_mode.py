import contextlib
import io
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import config
import main
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


class RemoteModeTests(unittest.TestCase):
    def args(self, *extra):
        with patch("sys.argv", ["main.py", "--model", "walk.onnx", "--no-plot",
                                "--command-source", "remote", *extra]):
            return main.parse_args()

    def test_remote_default_port_and_explicit_override(self):
        self.assertEqual(self.args().udp_command_port, 5007)
        self.assertEqual(self.args("--udp-command-port", "5500").udp_command_port, 5500)
        self.assertEqual(self.args("--command-source", "vision").udp_command_port, 5005)

    def test_remote_rejects_non_walking_and_shape_models(self):
        for options in (("--policy", "one-foot"), ("--one-foot-model", "foot.onnx")):
            with self.subTest(options=options), patch.object(main, "parse_args", return_value=self.args(*options)):
                with self.assertRaises(SystemExit):
                    main.main()

    def test_remote_rejects_invalid_udp_settings(self):
        for options in (("--udp-command-port", "0"), ("--command-timeout", "nan")):
            with self.subTest(options=options), patch.object(main, "parse_args", return_value=self.args(*options)):
                with self.assertRaises(SystemExit):
                    main.main()

    def test_remote_receives_live_commands_beyond_fixed_walk_timer(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(), joint_velocity=np.zeros(12), sequence=1)
        with (
            patch.object(main, "parse_args", return_value=self.args()),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "ShapeActionController", side_effect=AssertionError("remote must not run card actions")),
            patch.object(main, "AttitudeBroadcaster", side_effect=AssertionError("remote must not broadcast on its command port")),
            patch.object(main, "PositionCsvLogger"),
            patch.object(main.time, "monotonic", side_effect=[0, 10, 10, 10, 11]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            source_cls.return_value.get.return_value = np.array([.2, 0, -.3])
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, RuntimeError("end test")]
            self.assertEqual(main.main(), 1)
            source_cls.assert_called_once_with(5007, timeout_s=.25, bind="127.0.0.1")
            np.testing.assert_allclose(policy.step.call_args.kwargs["velocity_command"], [.2, 0, -.3])
            source_cls.return_value.close.assert_called_once()

    def test_stop_preempts_minimum_walking_hold(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(), joint_velocity=np.zeros(12), sequence=1)
        with (
            patch.object(main, "parse_args", return_value=self.args("--command-min-hold-s", "0.5")),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "PositionCsvLogger"),
            patch.object(main.time, "monotonic", side_effect=itertools.chain([0.], itertools.repeat(10.))),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            source_cls.return_value.get.side_effect = [np.array([.2, 0, -.3]), np.zeros(3)]
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, RuntimeError("end test")]
            self.assertEqual(main.main(), 1)
            self.assertEqual(policy.step.call_count, 2)
            np.testing.assert_allclose(policy.step.call_args_list[0].kwargs["velocity_command"], [.2, 0, -.3])
            np.testing.assert_allclose(policy.step.call_args_list[1].kwargs["velocity_command"], [0, 0, 0])

    def test_remote_rejects_fixed_frames_before_loading_file(self):
        with patch("sys.argv", ["main.py", "--fixed-policy", "unused.json", "--command-source", "remote"]):
            args = main.parse_args()
        with patch.object(main, "parse_args", return_value=args), self.assertRaises(SystemExit):
            main.main()


if __name__ == "__main__":
    unittest.main()
