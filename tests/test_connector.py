from __future__ import annotations

import unittest

from connector import process_vision_output, select_output


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

    def test_holds_slow_vision_command_across_50_hz_ticks(self) -> None:
        latest = process_vision_output({"vx": 0.3, "wz": 0.2, "qr": -1})
        outputs = [
            select_output(latest, last_vision_update=1.0, now=1.0 + tick * 0.02, timeout_s=0.25)
            for tick in range(6)
        ]
        self.assertTrue(all(output == latest and fresh for output, fresh, _age in outputs))

    def test_stale_vision_command_becomes_zero(self) -> None:
        latest = process_vision_output({"vx": 0.3, "wz": 0.2, "qr": 2})
        output, fresh, _age = select_output(
            latest,
            last_vision_update=1.0,
            now=1.251,
            timeout_s=0.25,
        )
        self.assertFalse(fresh)
        self.assertEqual(output, {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1})


if __name__ == "__main__":
    unittest.main()
