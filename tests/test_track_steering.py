"""Opt-in observed-path steering keeps the half-second policy contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from policy_bridge import SteeringController
from track_steering import TrackSteeringController
from tests.test_heading_steering import detection


def observed(near=0., phase="left_bend", **changes):
    value = detection(0., near=near, z=26., track_valid=True,
                      track_near_cm=near, track_near_z_cm=26.,
                      track_heading_deg=20., track_target_z_cm=48.,
                      track_target_x_cm=-17., track_target_bearing_deg=19.5,
                      track_confidence=.8, track_phase=phase)
    value.update(changes)
    return value


class TrackSteeringTests(unittest.TestCase):
    def test_three_fresh_observations_then_measured_target(self):
        controller = TrackSteeringController(SteeringController())
        controller.command(observed(), 1., .1)
        self.assertFalse(controller.diagnostics["track_control_active"])
        controller.command(observed(), 1., .1)
        self.assertFalse(controller.diagnostics["track_control_active"])
        controller.command(observed(), 1., .1)
        self.assertTrue(controller.diagnostics["track_control_active"])
        self.assertEqual(controller.diagnostics["steering_heading_source"], "observed_track")
        self.assertEqual(controller.command(observed(), 1., .1), (.2, 0.))
        self.assertEqual(controller.command(observed(), 1., .201), (.2, .37))

    def test_left_bend_nominal_turn_survives_small_left_offset(self):
        controller = TrackSteeringController(SteeringController())
        for _ in range(6):
            output = controller.command(observed(near=6.), 1., .1)
        self.assertTrue(controller.diagnostics["track_control_active"])
        self.assertEqual(output[1], .37)
        self.assertNotEqual(controller.diagnostics.get("steering_decision"),
                            "left_offset_release")

    def test_one_jittered_straight_phase_does_not_release_bend(self):
        controller = TrackSteeringController(SteeringController())
        for _ in range(6):
            controller.command(observed(near=6.), 1., .1)
        self.assertEqual(controller.diagnostics["track_control_phase"], "left_bend")
        controller.command(observed(near=6., phase="straight_like"), 1., .1)
        self.assertEqual(controller.diagnostics["track_control_phase"], "left_bend")

    def test_track_loss_falls_back_to_heading_and_stale_resets_confirmation(self):
        controller = TrackSteeringController(SteeringController())
        controller.command(observed(), 1., .1)
        controller.command(observed(), 1., .3)
        self.assertFalse(controller.diagnostics["track_control_active"])
        controller.command(detection(0., z=26., track_valid=False), 1., .1)
        self.assertEqual(controller.diagnostics["steering_heading_source"], "ground_x_z")
        self.assertEqual(controller.diagnostics["track_confirm_frames"], 0)

    def test_observed_boundaries_can_recover_when_legacy_pair_is_lost(self):
        controller = TrackSteeringController(SteeringController())
        debug = observed(measurement_valid=False, heading_control_valid=False,
                         fused_err_cm=0., lost_frames=2)
        for _ in range(4):
            output = controller.command(debug, .8, .1)
        self.assertTrue(controller.diagnostics["track_control_active"])
        self.assertEqual(controller.diagnostics["steering_heading_source"], "observed_track")
        self.assertGreater(output[0], 0.)


if __name__ == "__main__":
    unittest.main()
