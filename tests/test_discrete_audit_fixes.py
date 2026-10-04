"""Regression tests for elapsed-time and coordinate semantics in discrete steering."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision" / "jetson"))
from discrete_steering import DiscreteSteeringController
from policy_bridge import SteeringController


def measurement(error=6.0, lost=0, lateral=0.0, curve=False):
    return {"fused_err_cm": error, "angle_err_deg": 0.0, "lost_frames": lost,
            "base_err_cm": lateral, "curve_mode": curve, "bottom_lock_valid": True}


def controller(**kwargs):
    inner_options = kwargs.pop("inner_options", {})
    return DiscreteSteeringController(SteeringController(**inner_options), **kwargs)


class DiscreteAuditFixTests(unittest.TestCase):
    def test_both_output_signs_use_the_trigger_error_coordinate_to_stop(self):
        for yaw_sign in (-1, 1):
            for trigger_sign in (-1, 1):
                with self.subTest(yaw_sign=yaw_sign, trigger_sign=trigger_sign):
                    c = controller(allow_right=True, turn_s=2.0, stop_cm=2.0,
                                   inner_options={"yaw_sign": yaw_sign})
                    expected = yaw_sign * trigger_sign * 0.5
                    self.assertEqual(c.command(measurement(6 * trigger_sign), 1.0, .05)[1], expected)
                    self.assertEqual(c.command(measurement(6 * trigger_sign), 1.0, .05)[1], expected)
                    self.assertEqual(c.command(measurement(-1.9 * trigger_sign), 1.0, .05)[1], expected)
                    self.assertEqual(c.command(measurement(-2.0 * trigger_sign), 1.0, .05)[1], 0.0)

    def test_long_frame_expires_turn_and_carries_elapsed_time_into_gap(self):
        c = controller(turn_s=1.0, gap_s=2.5)
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], .5)
        self.assertEqual(c.command(measurement(), 1.0, 1.7)[1], 0.0)
        self.assertEqual(c.turn_left, 0.0)
        self.assertAlmostEqual(c.gap_left, 1.8)
        self.assertEqual(c.command(measurement(), 1.0, 1.8)[1], 0.0)
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], .5)

    def test_lost_time_expires_turn_even_when_hold_window_is_long(self):
        c = controller(turn_s=.3, gap_s=1.0,
                       inner_options={"lost_hold_s": 5.0})
        c.command(measurement(), 1.0, .05)
        vx, wz = c.command(measurement(lost=1), 0.0, .8)
        self.assertGreater(vx, 0.0)
        self.assertEqual(wz, 0.0)
        self.assertEqual(c.turn_left, 0.0)
        self.assertAlmostEqual(c.gap_left, .5)

    def test_loss_stop_cancels_old_turn_and_waits_before_reacquisition(self):
        c = controller(turn_s=3.0, gap_s=.5)
        c.command(measurement(), 1.0, .05)
        self.assertEqual(c.command(measurement(lost=1), 0.0, .25), (0.2, 0.0))
        self.assertEqual(c.turn_left, 0.0)
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], 0.0)
        self.assertAlmostEqual(c.gap_left, .40)

    def test_long_loss_credits_gap_time_after_the_loss_stop_deadline(self):
        c = controller(turn_s=3.0, gap_s=.5)
        c.command(measurement(), 1.0, .05)
        self.assertEqual(c.command(measurement(lost=1), 0.0, .8), (0.2, 0.0))
        self.assertEqual(c.turn_left, 0.0)
        self.assertEqual(c.gap_left, 0.0)
        # This is a fresh trigger, with a new complete duration, not a resumed turn.
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], .5)
        self.assertEqual(c.turn_left, 3.0)

    def test_gap_clock_also_runs_during_loss(self):
        c = controller(turn_s=1.0, stop_cm=0.0, gap_s=.5)
        c.command(measurement(), 1.0, .05)
        c.command(measurement(-1.0), 1.0, .05)
        c.command(measurement(lost=1), 0.0, .6)
        self.assertEqual(c.gap_left, 0.0)
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], .5)

    def test_short_loss_counts_real_time_and_preserves_fade(self):
        c = controller(turn_s=1.0)
        c.command(measurement(), 1.0, .05)
        self.assertAlmostEqual(c.command(measurement(lost=1), 0.0, .05)[1], .375)
        self.assertAlmostEqual(c.turn_left, .95)
        self.assertEqual(c.command(measurement(), 1.0, .05)[1], .5)
        self.assertAlmostEqual(c.turn_left, .90)

    def test_legacy_good_frame_duration_and_zero_gap_are_preserved(self):
        c = controller(turn_s=.1, gap_s=0.0)
        self.assertEqual([c.command(measurement(), 1.0, .05)[1] for _ in range(6)],
                         [.5, .5, 0.0, .5, .5, 0.0])
        self.assertEqual(controller().command(measurement(-20.0), 1.0, .05)[1], 0.0)

    def test_discrete_does_not_run_pid_or_apply_continuous_trims(self):
        c = controller(inner_options={"bias_cm": 100.0, "bias_straight_cm": 100.0,
                                      "straight_gains": (1e308, 1e308, 1e308)})
        with patch.object(c.inner, "command", side_effect=AssertionError("hidden PID")), \
             patch.object(c.inner, "filtered_derivative", side_effect=AssertionError("hidden D")):
            self.assertEqual(c.command(measurement(1.0), 1.0, .05), (.2, 0.0))
            self.assertEqual(c.command(measurement(6.0), 1.0, .05), (.2, .5))
        self.assertEqual(c.inner.integral, 0.0)
        self.assertEqual(c.inner.err_window, [])
        self.assertEqual(c.last_err_eff, 6.0)
        self.assertEqual(c.last_steer, .5)

    def test_shared_lateral_validation_rejects_nonfinite_and_excessive_input(self):
        for lateral in (math.nan, math.inf, 11.0):
            with self.subTest(lateral=lateral):
                c = controller(inner_options={"max_lateral_cm": 10.0})
                self.assertEqual(c.command(measurement(lateral=lateral), 1.0, .05), (0.0, 0.0))
                self.assertEqual(c.turn_left, 0.0)

    def test_invalid_clock_stops_instead_of_preserving_active_turn(self):
        for dt in (math.nan, math.inf, 0.0, -.1, None):
            with self.subTest(dt=dt):
                c = controller()
                c.command(measurement(), 1.0, .05)
                self.assertEqual(c.command(measurement(), 1.0, dt), (0.2, 0.0))
                self.assertEqual(c.turn_left, 0.0)

    def test_continuous_proportional_curve_and_loss_outputs_are_preserved(self):
        c = SteeringController(straight_gains=(1.0, 0.0, 0.0),
                               curve_gains=(2.0, 0.0, 0.0),
                               bias_cm=0.0, bias_straight_cm=0.0,
                               center_dead_cm=0.0, steer_full_scale_cm=10.0)
        self.assertEqual(c.command(measurement(4.0), 1.0, .05), (.2, .2))
        self.assertEqual(c.command(measurement(4.0, curve=True), 1.0, .05), (.2, .4))
        vx, wz = c.command(measurement(lost=1), 0.0, .05)
        self.assertAlmostEqual(vx, .2)      # 速度不掉（丢线只丢转向）
        self.assertAlmostEqual(wz, .3)
        self.assertEqual(c.command(measurement(1.0), 1.0, .05), (.2, .05))


if __name__ == "__main__":
    unittest.main()
