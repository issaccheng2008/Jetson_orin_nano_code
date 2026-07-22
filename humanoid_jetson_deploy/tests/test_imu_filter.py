from __future__ import annotations

import unittest

import numpy as np

from imu_filter import (
    normalize_quaternion_wxyz,
    projected_gravity_from_quaternion,
    validate_stationary_imu_sample,
)


class ImuOrientationTests(unittest.TestCase):
    def test_upright_sensor_to_world(self):
        gravity = projected_gravity_from_quaternion(
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.eye(3),
            sensor_to_world=True,
        )
        np.testing.assert_allclose(gravity, [0.0, 0.0, -1.0], atol=1e-6)

    def test_positive_pitch_sensor_to_world(self):
        angle = np.deg2rad(30.0)
        quaternion = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])
        gravity = projected_gravity_from_quaternion(
            quaternion,
            np.eye(3),
            sensor_to_world=True,
        )
        np.testing.assert_allclose(
            gravity,
            [np.sin(angle), 0.0, -np.cos(angle)],
            atol=1e-6,
        )

    def test_mounting_rotation_is_applied(self):
        sensor_to_policy = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        gravity = projected_gravity_from_quaternion(
            np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]),
            sensor_to_policy,
            sensor_to_world=True,
        )
        np.testing.assert_allclose(gravity, [1.0, 0.0, 0.0], atol=1e-6)

    def test_quaternion_is_normalized(self):
        np.testing.assert_allclose(
            normalize_quaternion_wxyz(np.array([1.1, 0.0, 0.0, 0.0])),
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_stationary_consistency_check(self):
        validate_stationary_imu_sample(
            np.array([0.0, 0.0, 9.81]),
            np.zeros(3),
            np.array([0.0, 0.0, -1.0]),
        )
        with self.assertRaisesRegex(ValueError, "convention"):
            validate_stationary_imu_sample(
                np.array([0.0, 0.0, 9.81]),
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
            )


if __name__ == "__main__":
    unittest.main()
