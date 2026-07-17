from __future__ import annotations

import unittest

import numpy as np

import config


class TargetSafetyTests(unittest.TestCase):
    def test_clamps_each_target_to_measured_position_window(self) -> None:
        current = np.zeros(config.NUM_JOINTS, dtype=np.float32)
        requested = np.linspace(-1.0, 1.0, config.NUM_JOINTS, dtype=np.float32)

        limited = config.clamp_policy_target_to_current(requested, current)

        self.assertTrue(
            np.all(np.abs(limited - current) <= config.MAX_TARGET_DEVIATION_RAD + 1e-7)
        )

    def test_preserves_targets_already_inside_window(self) -> None:
        current = np.zeros(config.NUM_JOINTS, dtype=np.float32)
        requested = np.linspace(-0.05, 0.05, config.NUM_JOINTS, dtype=np.float32)

        limited = config.clamp_policy_target_to_current(requested, current)

        np.testing.assert_allclose(limited, requested)

    def test_also_respects_absolute_joint_limits(self) -> None:
        safe_upper = config.Q_UPPER - config.JOINT_LIMIT_MARGIN_RAD
        current = safe_upper - 0.01
        requested = config.Q_UPPER + 1.0

        limited = config.clamp_policy_target_to_current(requested, current)

        np.testing.assert_allclose(limited, safe_upper)

    def test_rejects_non_finite_encoder_position(self) -> None:
        current = np.zeros(config.NUM_JOINTS, dtype=np.float32)
        current[3] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            config.clamp_policy_target_to_current(np.zeros_like(current), current)


if __name__ == "__main__":
    unittest.main()
