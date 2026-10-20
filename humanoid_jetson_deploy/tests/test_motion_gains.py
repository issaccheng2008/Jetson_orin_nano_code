"""Select independent PD profiles from executed velocity commands."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import main
from protocol import COMMAND_ENABLE, STATE_ENCODERS_VALID, STATE_IMU_VALID
import test_actuator_gains as gain_tests


class MotionGainTests(unittest.TestCase):
    def profiles(self):
        self.assertTrue(hasattr(config, "MOTION_GAIN_SCALES"), "three PD profiles are required")
        return {"standing": (1.5, 2.0), "straight": (1.2, 1.4), "turning": (0.8, 1.8)}

    def test_three_modes_and_reverse_or_in_place_motion(self):
        with patch.object(config, "MOTION_GAIN_SCALES", self.profiles()):
            for command, expected in ((None, (1.5, 2)), ([0, 0, 0], (1.5, 2)),
                                      ([.2, 0, 0], (1.2, 1.4)), ([-.2, 0, 0], (1.2, 1.4)),
                                      ([0, .1, 0], (1.2, 1.4)), ([.2, 0, .3], (.8, 1.8)),
                                      ([.2, 0, -.3], (.8, 1.8)), ([0, 0, .3], (.8, 1.8)),
                                      ([1e-8, 0, -1e-8], (1.5, 2))):
                with self.subTest(command=command):
                    self.assertEqual(config.motion_gain_scales(command), expected)

    def test_defaults_keep_standing_at_baseline_and_raise_walking_gains(self):
        self.assertTrue(hasattr(config, "motion_gain_scales"), "motion gain selection is required")
        sender = gain_tests.ActuatorGainTests()
        for command, expected in (([0, 0, 0], (1, 1)), ([.2, 0, 0], (1.5, 2)),
                                  ([.2, 0, .3], (1.5, 2))):
            scales = config.motion_gain_scales(command)
            self.assertEqual(scales, expected)
            packet, _, _ = sender.send(*scales)
            np.testing.assert_allclose(packet.kp, config.JOINT_KP * expected[0])
            np.testing.assert_allclose(packet.kd, config.JOINT_KD * expected[1])

    def test_each_profile_reaches_absolute_wire_gains_without_extra_multiplier(self):
        sender = gain_tests.ActuatorGainTests()
        with patch.object(config, "MOTION_GAIN_SCALES", self.profiles()):
            for command, scales in (([0, 0, 0], (1.5, 2)), ([.2, 0, 0], (1.2, 1.4)),
                                     ([.2, 0, .3], (.8, 1.8))):
                packet, _, _ = sender.send(*config.motion_gain_scales(command))
                np.testing.assert_allclose(packet.kp, config.JOINT_KP * scales[0], rtol=1e-6)
                np.testing.assert_allclose(packet.kd, config.JOINT_KD * scales[1], rtol=1e-6)

    def test_bad_selected_profile_and_invalid_commands_are_rejected(self):
        profiles = self.profiles()
        for value in (-1, np.nan, np.inf):
            profiles["standing"] = (value, 1)
            with patch.object(config, "MOTION_GAIN_SCALES", profiles), self.assertRaises(ValueError):
                config.motion_gain_scales([0, 0, 0])
        for command in ([0, 0], [0, 0, np.nan], [np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                config.motion_gain_scales(command)

    def test_main_accepts_separate_global_scales_greater_than_one(self):
        with patch("sys.argv", ["main.py", "--model", "test.onnx", "--no-plot",
                                "--kp-scale", "1.5", "--kd-scale", "2"]):
            args = main.parse_args()
        robot = SimpleNamespace(sequence=1, status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
            joint_position=config.Q_DEFAULT.copy(), joint_velocity=np.zeros(12),
            accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
            orientation_wxyz=np.array([1, 0, 0, 0]))
        policy, link = Mock(), Mock()
        policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.)
        link.wait_for_state.return_value = robot
        link.get_latest_state.side_effect = [robot, RuntimeError("end test")]
        with patch.object(main, "parse_args", return_value=args), patch.object(main.signal, "signal"), \
                patch.object(main, "HumanoidPolicy", return_value=policy), \
                patch.object(main, "SerialLink", return_value=link), patch.object(main, "PositionCsvLogger"), \
                patch.object(config, "MOTION_GAIN_SCALES", {mode: (1, 1) for mode in config.MOTION_GAIN_SCALES}), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                result = main.main()
            except SystemExit as exc:
                self.fail(f"independent scales above one must be allowed: {exc}")
            self.assertEqual(result, 1)
        self.assertEqual(link.send_command.call_args_list[0].args[2:4], (1.5, 2.0))


if __name__ == "__main__":
    unittest.main()
