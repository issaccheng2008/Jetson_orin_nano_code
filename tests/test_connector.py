from __future__ import annotations

import unittest

from connector import process_vision_output


class ProcessVisionOutputTests(unittest.TestCase):
    def test_forces_lateral_velocity_to_zero(self) -> None:
        result = process_vision_output({"vx": 0.25, "vy": 0.9, "wz": -0.2, "qr": 3})
        self.assertEqual(result, {"vx": 0.25, "vy": 0.0, "wz": -0.2, "qr": 3})

    def test_clamps_policy_command_ranges(self) -> None:
        result = process_vision_output({"vx": 2.0, "wz": -2.0, "qr": 99})
        self.assertEqual(result, {"vx": 1.0, "vy": 0.0, "wz": -0.5, "qr": -1})

    def test_requires_velocity_fields(self) -> None:
        with self.assertRaises(KeyError):
            process_vision_output({"vx": 0.2})


if __name__ == "__main__":
    unittest.main()
