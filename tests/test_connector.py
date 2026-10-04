from __future__ import annotations

import math
import unittest

from connector import CommandSmoother, process_vision_output, select_output, slew_toward


class ProcessVisionOutputTests(unittest.TestCase):
    def test_event_metadata_survives_connector_and_smoothing(self) -> None:
        message = {"vx": 0.0, "wz": 0.0, "qr": -1, "event_id": 77, "event_action": 3}
        result = process_vision_output(message)
        self.assertEqual((result["event_id"], result["event_action"]), (77, 3))
        self.assertEqual(CommandSmoother(1.0, 2.0).update(result, 0.02)["event_id"], 77)

    def test_validated_command_mode_survives_with_shape_event(self) -> None:
        for mode in ("held", "continuous"):
            with self.subTest(mode=mode):
                result = process_vision_output({"vx": 0.5, "wz": 0.5, "qr": 3,
                                                "command_mode": mode,
                                                "event_id": 77, "event_action": 3})
                forwarded = CommandSmoother(1.0, 2.0).update(result, 0.02)
                self.assertEqual(forwarded["command_mode"], mode)
                self.assertEqual((forwarded["event_id"], forwarded["event_action"]), (77, 3))

    def test_unknown_or_non_string_modes_are_rejected(self) -> None:
        for mode in ("", "HOLD", "unknown", None, 1, True, [], {}):
            with self.subTest(mode=mode):
                message = {"vx": 0.5, "wz": 0.5, "qr": -1, "command_mode": mode}
                with self.assertRaises(ValueError):
                    process_vision_output(message)
                smoother = CommandSmoother(1.0, 2.0)
                with self.assertRaises(ValueError):
                    smoother.update(message, 0.02)
                self.assertEqual((smoother.vx, smoother.wz), (0.0, 0.0))

    def test_forces_lateral_velocity_to_zero(self) -> None:
        result = process_vision_output({"vx": 0.25, "vy": 0.9, "wz": -0.2, "qr": 3})
        self.assertEqual(result, {"vx": 0.25, "vy": 0.0, "wz": -0.2, "qr": 3,
                                  "hold_upright": False, "card_tilt": False})

    def test_clamps_policy_command_ranges(self) -> None:
        result = process_vision_output({"vx": 2.0, "wz": -2.0, "qr": 99})
        self.assertEqual(result, {"vx": 1.0, "vy": 0.0, "wz": -0.5, "qr": -1,
                                  "hold_upright": False, "card_tilt": False})

    def test_hold_upright_passes_through_and_defaults_off(self) -> None:
        """Vision asks for it while it stands still to read a card; an older vision
        that never sends the field must keep behaving exactly as before."""
        asked = process_vision_output({"vx": 0.0, "wz": 0.0, "hold_upright": True})
        self.assertTrue(asked["hold_upright"])
        self.assertTrue(CommandSmoother(1.0, 2.0).update(asked, 0.02)["hold_upright"])
        self.assertFalse(process_vision_output({"vx": 0.0, "wz": 0.0})["hold_upright"])

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

    def test_watchdog_zero_stops_on_first_tick_and_clears_state(self) -> None:
        smoother = CommandSmoother(max_vx_accel=1.0, max_wz_accel=2.0)
        running = process_vision_output({"vx": 0.5, "wz": -0.5, "qr": 2,
                                        "command_mode": "held", "event_id": 77,
                                        "event_action": 3})
        smoother.update(running, 0.02)
        stale, fresh, _age = select_output(
            running, last_vision_update=1.0, now=1.30, timeout_s=0.25
        )
        self.assertFalse(fresh)
        first = smoother.update(stale, 0.0)
        self.assertEqual(first, {"vx": 0.0, "vy": 0.0, "wz": 0.0, "qr": -1,
                                 "hold_upright": False, "card_tilt": False})
        self.assertEqual((smoother.vx, smoother.wz), (0.0, 0.0))
        # A continuous restart must start from zero, not the previous left turn.
        restarted = smoother.update({"vx": 0.5, "wz": 0.5, "qr": -1}, 0.02)
        self.assertAlmostEqual(restarted["vx"], 0.02)
        self.assertAlmostEqual(restarted["wz"], 0.04)

    def test_held_commands_keep_full_values_and_publish_on_every_tick(self) -> None:
        smoother = CommandSmoother(1.0, 2.0)
        for wz in (0.5, 0.0, -0.5, 0.5):
            with self.subTest(wz=wz):
                latest = process_vision_output({"vx": 0.5, "wz": wz, "qr": -1,
                                                "command_mode": "held"})
                for tick in range(10):
                    target, fresh, _ = select_output(latest, 1.0, 1.0+tick*0.02, 0.25)
                    self.assertTrue(fresh)
                    result = smoother.update(target, 0.02)
                    self.assertEqual((result["vx"], result["wz"]), (0.5, wz))
                    self.assertEqual(result["command_mode"], "held")
                    self.assertEqual((smoother.vx, smoother.wz), (0.5, wz))
        # The tag does not extend the ordinary vision watchdog to 0.5 seconds.
        target, fresh, _ = select_output(latest, 1.0, 1.251, 0.25)
        self.assertFalse(fresh)
        self.assertEqual(smoother.update(target, 0.02)["wz"], 0.0)

    def test_held_state_is_used_when_switching_to_continuous_or_legacy(self) -> None:
        for mode in (None, "continuous"):
            with self.subTest(mode=mode):
                smoother = CommandSmoother(1.0, 2.0)
                smoother.update({"vx": 0.5, "wz": -0.5, "qr": -1,
                                 "command_mode": "held"}, 0.02)
                target = {"vx": 0.6, "wz": 0.5, "qr": -1}
                if mode is not None:
                    target["command_mode"] = mode
                result = smoother.update(target, 0.02)
                self.assertAlmostEqual(result["vx"], 0.52)
                self.assertAlmostEqual(result["wz"], -0.46)
                self.assertEqual(result.get("command_mode"), mode)

    def test_continuous_tag_preserves_legacy_acceleration_limits(self) -> None:
        for mode in (None, "continuous"):
            with self.subTest(mode=mode):
                target = {"vx": 0.5, "wz": 0.5, "qr": -1}
                if mode is not None:
                    target["command_mode"] = mode
                result = CommandSmoother(1.0, 2.0).update(target, 0.02)
                self.assertAlmostEqual(result["vx"], 0.02)
                self.assertAlmostEqual(result["wz"], 0.04)
                self.assertEqual(result.get("command_mode"), mode)

    def test_full_stop_and_posture_requests_clear_motion_but_preserve_metadata(self) -> None:
        for mode in (None, "held", "continuous"):
            for flag in (None, "hold_upright", "card_tilt"):
                with self.subTest(mode=mode, flag=flag):
                    smoother = CommandSmoother(1.0, 2.0)
                    smoother.update({"vx": 0.5, "wz": -0.5, "qr": -1,
                                     "command_mode": "held"}, 0.02)
                    target = {"vx": 0.5 if flag else 0.0,
                              "wz": 0.5 if flag else 0.0,
                              "qr": 3, "event_id": 77, "event_action": 3}
                    if mode is not None:
                        target["command_mode"] = mode
                    if flag:
                        target[flag] = True
                    result = smoother.update(process_vision_output(target), 0.0)
                    self.assertEqual((result["vx"], result["wz"]), (0.0, 0.0))
                    self.assertEqual((smoother.vx, smoother.wz), (0.0, 0.0))
                    self.assertEqual((result["qr"], result["event_id"], result["event_action"]),
                                     (3, 77, 3))
                    self.assertEqual(result.get("command_mode"), mode)
                    if flag:
                        self.assertTrue(result[flag])
                    restarted = smoother.update({"vx": 0.5, "wz": 0.5, "qr": -1,
                                                 "command_mode": "held"}, 0.02)
                    self.assertEqual((restarted["vx"], restarted["wz"]), (0.5, 0.5))

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

    def test_a_zero_yaw_limit_passes_the_command_through_on_the_same_tick(self) -> None:
        """--wz-mode discrete 要的是方波传递。默认 2.0 会把 0→0.4 拉成 0.2 秒的
        斜坡，比这短的脉冲过完就剩个三角形，峰值也到不了目标。0 = 完全不限。"""
        ramped = CommandSmoother(1.0, 2.0)
        self.assertLess(ramped.update({"vx": 0.0, "wz": 0.4, "qr": -1}, 0.02)["wz"],
                        0.4)
        direct = CommandSmoother(1.0, 0.0)
        for target in (0.4, -0.5, 0.0, 0.5):
            with self.subTest(target=target):
                self.assertAlmostEqual(
                    direct.update({"vx": 0.0, "wz": target, "qr": -1}, 0.02)["wz"],
                    target)
        # vx 那条照旧有斜率限制，关掉 wz 的没有连坐
        self.assertLess(direct.update({"vx": 0.9, "wz": 0.0, "qr": -1}, 0.02)["vx"],
                        0.9)


if __name__ == "__main__":
    unittest.main()
