"""Walking-only ankle trim at the Nano-to-STM32 target boundary."""

import unittest

import numpy as np

from main import apply_forward_ankle_bias
from target_safety import TargetSafety


class ForwardAnkleBiasTests(unittest.TestCase):
    def test_forward_command_offsets_only_the_two_ankle_pitch_targets(self):
        target = np.zeros(12, dtype=np.float32)
        result = apply_forward_ankle_bias(target, np.array([0.2, 0.0, 0.0]), True, 0.08)
        expected = target.copy()
        expected[4], expected[10] = 0.08, -0.08
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(target, np.zeros(12, dtype=np.float32))

    def test_zero_speed_and_policy_takeover_remove_the_bias(self):
        target = np.zeros(12, dtype=np.float32)
        for velocity, active in (
            (np.array([0.0, 0.0, 0.0]), True),
            (np.array([0.0, 0.0, 0.4]), True),
            (np.array([-0.1, 0.0, 0.0]), True),
            (np.array([0.2, 0.0, 0.0]), False),
            (None, False),
        ):
            with self.subTest(velocity=velocity, active=active):
                np.testing.assert_array_equal(
                    apply_forward_ankle_bias(target, velocity, active, 0.08), target)

    def test_bias_passes_through_existing_target_safety(self):
        target = np.zeros(12, dtype=np.float32)
        biased = apply_forward_ankle_bias(target, np.array([0.2, 0.0, 0.0]), True, 0.08)
        safe, trace = TargetSafety(max_speed_rad_s=3.0).apply_with_trace(
            biased, target, target, 0.02)
        self.assertAlmostEqual(float(trace.raw_target[4]), 0.08, places=6)
        self.assertAlmostEqual(float(safe[4]), 0.06, places=6)
        self.assertAlmostEqual(float(safe[10]), -0.06, places=6)
        stopped, stopped_trace = TargetSafety(max_speed_rad_s=3.0).apply_with_trace(
            apply_forward_ankle_bias(target, np.zeros(3), True, 0.08),
            safe, target, 0.02)
        np.testing.assert_array_equal(stopped_trace.raw_target, target)
        np.testing.assert_allclose(stopped, target)


if __name__ == "__main__":
    unittest.main()
