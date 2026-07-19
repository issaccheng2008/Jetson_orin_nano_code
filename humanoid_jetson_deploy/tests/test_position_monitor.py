from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from position_monitor import PositionCsvLogger, load_position_log


class PositionCsvLoggerTests(unittest.TestCase):
    def test_writes_all_target_and_actual_positions(self):
        names = ("right_knee", "left_knee")
        with tempfile.TemporaryDirectory() as temporary_directory:
            logger = PositionCsvLogger(temporary_directory, names)
            path = Path(logger.path)
            logger.write(
                elapsed_s=0.02,
                step=1,
                state_sequence=7,
                target_motor_rad=np.array([0.3, -0.3]),
                actual_motor_rad=np.array([0.2, -0.2]),
            )
            logger.close()

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], "1")
        self.assertEqual(rows[0]["state_sequence"], "7")
        self.assertAlmostEqual(float(rows[0]["target_right_knee_rad"]), 0.3)
        self.assertAlmostEqual(float(rows[0]["actual_left_knee_rad"]), -0.2)

    def test_loads_logged_target_and_actual_positions(self):
        names = ("right_knee", "left_knee")
        with tempfile.TemporaryDirectory() as temporary_directory:
            logger = PositionCsvLogger(temporary_directory, names)
            path = Path(logger.path)
            logger.write(0.02, 1, 7, np.array([0.3, -0.3]), np.array([0.2, -0.2]))
            logger.write(0.04, 2, 11, np.array([0.4, -0.4]), np.array([0.25, -0.25]))
            logger.close()

            elapsed_s, targets, actuals = load_position_log(path, names)

        np.testing.assert_allclose(elapsed_s, [0.02, 0.04])
        np.testing.assert_allclose(targets, [[0.3, -0.3], [0.4, -0.4]])
        np.testing.assert_allclose(actuals, [[0.2, -0.2], [0.25, -0.25]])


if __name__ == "__main__":
    unittest.main()
