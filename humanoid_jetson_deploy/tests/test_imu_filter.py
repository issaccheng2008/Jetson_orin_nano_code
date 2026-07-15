from __future__ import annotations

import unittest

import numpy as np

from imu_filter import ProjectedGravityFilter


class ImuFilterTests(unittest.TestCase):
    def test_upright_stationary(self):
        filt = ProjectedGravityFilter()
        for _ in range(100):
            gravity = filt.update(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.005)
        np.testing.assert_allclose(gravity, [0.0, 0.0, -1.0], atol=1e-5)

    def test_normalized(self):
        filt = ProjectedGravityFilter()
        gravity = filt.update(np.array([0.0, 6.936, 6.936]), np.zeros(3), 0.005)
        self.assertAlmostEqual(float(np.linalg.norm(gravity)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
