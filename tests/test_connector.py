from __future__ import annotations

import math
import unittest

from connector import CommandSmoother, process_vision_output, select_output, slew_toward


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


class SlewTowardTests(unittest.TestCase):
    def test_lands_on_exactly_zero(self) -> None:
        value = 0.4
        for _ in range(100):
            value = slew_toward(value, 0.0, 0.02)
        self.assertEqual(value, 0.0)

    def test_ramp_is_monotone_and_rate_limited(self) -> None:
        value, seen = 0.0, []
        for _ in range(50):
            value = slew_toward(value, 0.4, 0.02)
            seen.append(value)
        self.assertEqual(seen, sorted(seen))
        self.assertLessEqual(max(b - a for a, b in zip(seen, seen[1:])), 0.02 + 1e-12)
        self.assertAlmostEqual(seen[19], 0.4, places=12)  # 19 x 0.02 = 0.38, 20th lands exact
        self.assertEqual(seen[-1], 0.4)

    def test_zero_step_holds_without_nan(self) -> None:
        self.assertEqual(slew_toward(0.3, 0.4, 0.0), 0.3)
        self.assertTrue(math.isfinite(slew_toward(0.3, 0.4, 0.0)))


class CommandSmootherTests(unittest.TestCase):
    TARGET = {"vx": 0.4, "vy": 0.0, "wz": -0.5, "qr": 7}

    def test_step_target_becomes_a_monotone_ramp(self) -> None:
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        vxs, wzs = [], []
        for _ in range(40):
            output = smoother.update(self.TARGET, 0.02)
            vxs.append(output["vx"])
            wzs.append(output["wz"])
        self.assertEqual(vxs, sorted(vxs))
        self.assertLessEqual(max(b - a for a, b in zip(vxs, vxs[1:])), 0.02 + 1e-12)
        self.assertAlmostEqual(vxs[19], 0.4, places=12)  # 0.4 m/s at 1.0 m/s^2
        self.assertEqual(wzs[24], -0.5)  # 0.5 rad/s at 2.0 rad/s^2
        self.assertEqual(smoother.update(self.TARGET, 0.02)["vx"], 0.4)

    def test_watchdog_zero_is_ramped_to_exact_zero(self) -> None:
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        running = {"vx": 0.4, "vy": 0.0, "wz": -0.5, "qr": 2}
        for _ in range(30):
            smoother.update(running, 0.02)
        stale, fresh, _age = select_output(
            running, last_vision_update=1.0, now=1.30, timeout_s=0.25
        )
        self.assertFalse(fresh)
        first = smoother.update(stale, 0.02)
        self.assertNotEqual(first["vx"], 0.0)  # ramped, not stepped
        self.assertLess(first["vx"], 0.4)
        for _ in range(30):
            last = smoother.update(stale, 0.02)
        self.assertEqual(last, {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1})

    def test_qr_passes_through_untouched(self) -> None:
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        self.assertEqual(smoother.update({"vx": 0.4, "wz": 0.0, "qr": 3}, 0.02)["qr"], 3)
        self.assertEqual(smoother.update({"vx": 0.4, "wz": 0.0, "qr": -1}, 0.02)["qr"], -1)

    def test_sweeping_between_extremes_stays_inside_the_policy_ranges(self) -> None:
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        for target in ({"vx": 0.0, "wz": -0.5, "qr": 1}, {"vx": 1.0, "wz": 0.5, "qr": 1}):
            for _ in range(80):
                output = smoother.update(target, 0.02)
                self.assertGreaterEqual(output["vx"], 0.0)
                self.assertLessEqual(output["vx"], 1.0)
                self.assertGreaterEqual(output["wz"], -0.5)
                self.assertLessEqual(output["wz"], 0.5)

    def test_rejects_non_positive_or_non_finite_limits(self) -> None:
        for kwargs in (
            dict(max_vx_accel=0.0, max_wz_accel=2.0),
            dict(max_vx_accel=1.0, max_wz_accel=-1.0),
            dict(max_vx_accel=float("nan"), max_wz_accel=2.0),
            dict(max_vx_accel=float("inf"), max_wz_accel=2.0),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CommandSmoother(**kwargs)


if __name__ == "__main__":
    unittest.main()
