"""Freshness boundaries between P1 measurement and both steering modes."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from policy_bridge import SteeringController
from discrete_steering import DiscreteSteeringController


def frame(**values):
    return dict(fused_err_cm=6.0, base_err_cm=6.0, angle_err_deg=0.0,
                lost_frames=0, **values)


class MeasurementFreshnessTests(unittest.TestCase):
    def test_current_finite_debug_is_not_enough_when_measurement_is_invalid(self):
        for discrete in (False, True):
            with self.subTest(discrete=discrete):
                c = SteeringController()
                if discrete:
                    c = DiscreteSteeringController(c)
                # Finite diagnostic values must never start movement.
                self.assertEqual(c.command(frame(measurement_valid=False), .9, .1), (0., 0.))

    def test_expired_measurement_discards_yaw_and_preserves_previous_speed(self):
        for discrete in (False, True):
            with self.subTest(discrete=discrete):
                c = SteeringController(lost_hold_s=1.0)
                if discrete:
                    c = DiscreteSteeringController(c)
                self.assertGreater(c.command(frame(measurement_valid=True), .9, .1)[0], 0.)
                self.assertEqual(c.command(frame(measurement_valid=False, measurement_stale=True), .9, .01), (.2, 0.))
                self.assertEqual(c.command(frame(measurement_valid=False), .9, .01), (.2, 0.))
                c.drop_held_command()
                self.assertEqual(c.command(frame(measurement_valid=False, measurement_stale=True), .9, .01), (0., 0.))

    def test_legacy_producer_and_fresh_p1_keep_same_discrete_command(self):
        old = DiscreteSteeringController(SteeringController())
        new = DiscreteSteeringController(SteeringController())
        self.assertEqual(old.command(frame(), .9, .1),
                         new.command(frame(measurement_valid=True, measurement_stale=False), .9, .1))

    def test_expiry_cancels_pulse_even_with_a_long_loss_hold(self):
        c = DiscreteSteeringController(SteeringController(lost_hold_s=10.),
                                       turn_s=5., gap_s=.5)
        self.assertEqual(c.command(frame(measurement_valid=True), .9, .1)[1], .5)
        self.assertEqual(c.command(frame(measurement_valid=False, measurement_stale=True), .9, .1), (.2, 0.))
        self.assertEqual(c.turn_left, 0.)
        self.assertEqual(c.command(frame(measurement_valid=True), .9, .1)[1], 0.)
