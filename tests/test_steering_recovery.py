import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from steering_recovery import RecoveryConfig
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from test_heading_steering import detection
from steering_filter import FilterConfig, GeometryFilter
from segment_steering import SegmentSteeringController
from test_segment_steering import detection as segment_detection, bend


class RecoveryTests(unittest.TestCase):
    def controller(self, **kw):
        return HeadingSteeringController(SteeringController(), min_hold_s=0,
            recovery_config=RecoveryConfig(max_loss_s=.8), **kw)

    def test_loss_keeps_recent_turn_then_stops_both_commands(self):
        c = self.controller()
        valid = c.command(detection(30), 1, .1)
        bad = detection(measurement_valid=False, heading_control_valid=False)
        self.assertEqual(c.command(bad, 0, .3), valid)
        self.assertEqual(c.diagnostics['steering_reason'], 'loss_history_turn')
        self.assertEqual(c.command(bad, 0, .51), (0., 0.))
        self.assertEqual(c.diagnostics['steering_reason'], 'loss_timeout_stop')

    def test_no_turn_evidence_and_external_stop_never_start_blind_walking(self):
        c = self.controller()
        c.command(detection(0), 1, .1)
        bad = detection(measurement_valid=False)
        self.assertEqual(c.command(bad, 0, .1), (0., 0.))
        c.command(detection(30), 1, .1)
        c.drop_held_command()
        self.assertEqual(c.command(bad, 0, .1), (0., 0.))

    def test_stale_observation_uses_history_without_trusting_stale_angle(self):
        c = self.controller()
        valid = c.command(detection(30), 1, .1)
        self.assertEqual(c.command(detection(-30, measurement_stale=True), 0, .2), valid)
        self.assertEqual(c.command(detection(-30), 1, .1)[1], -.5)

    def test_valid_held_turn_refreshes_history(self):
        c=HeadingSteeringController(SteeringController(),min_hold_s=.5,recovery_config=RecoveryConfig())
        for _ in range(5):valid=c.command(detection(30),1,.1)
        self.assertEqual(c.command(detection(measurement_valid=False),0,.1),valid)

    def test_offset_safeguard_keeps_raw_two_frame_confirmation(self):
        c=self.controller(filter_config=FilterConfig(algorithm='robust'))
        c.command(detection(30,near=-8),1,.1)
        for _ in range(2):c.command(detection(30,near=4),1,.1)
        self.assertTrue(c.diagnostics['steering_left_offset_confirmed'])

    def test_recovery_does_not_revive_turn_vetoed_by_position(self):
        c=self.controller()
        c.command(detection(60,z=25),1,.1)
        for _ in range(2):c.command(detection(60,near=6,z=25),1,.1)
        self.assertEqual(c.diagnostics['steering_decision'],'left_offset_release')
        self.assertEqual(c.command(detection(measurement_valid=False),0,.1),(0.,0.))

    def test_straight_bias_is_not_turn_evidence(self):
        for sign in (-1,1):
            c=HeadingSteeringController(SteeringController(yaw_sign=sign),straight_wz=.05,
                                        recovery_config=RecoveryConfig(),min_hold_s=0)
            c.command(detection(0),1,.1)
            self.assertEqual(c.command(detection(measurement_valid=False),0,.1),(0.,0.))

    def test_robust_filter_rejects_spike_and_limits_change_after_source_switch(self):
        f = GeometryFilter(FilterConfig(algorithm='robust'))
        f.apply((0, 25, 30, 30, 'ground_x_z'), 0.)
        self.assertEqual(f.apply((0, 25, -50, -50, 'ground_x_z'), .17)[2], 30.)
        out = f.apply((0, 25, -50, -50, 'single_edge'), .34)
        self.assertGreaterEqual(out[2], 30-45*.17)
        for i in range(3, 30): out=f.apply((0,25,-50,-50,'single_edge'),i*.17)
        self.assertLess(out[2], -45)

    def test_qualified_segments_can_supply_direction_without_global_heading(self):
        c = SegmentSteeringController(SteeringController(), segment_fallback=True,
                                      recovery_config=RecoveryConfig(), min_hold_s=0)
        good = segment_detection(bend, heading_control_valid=False,
                                 near_observation_paired=True, near_observation_quality=.9)
        for _ in range(5): command=c.command(good, 1, .17)
        self.assertGreater(command[1], 0)
        self.assertEqual(c.diagnostics['steering_heading_source'], 'measured_segments')
        self.assertTrue(c.diagnostics['segment_control_active'])
        c.drop_held_command()
        good['near_observation_paired']=False
        self.assertEqual(c.command(good, 1, .17), (0.,0.))

    def test_fallback_rejects_bad_quality_stale_or_inconsistent_segments(self):
        for changes in (dict(near_observation_quality=.1),dict(measurement_stale=True),
                        dict(fit_seg_anchored=False),dict(fit_seg0_normal_width_cm=50),
                        dict(fit_seg0_rmse_cm=3),dict(fit_seg0_points=4)):
            c=SegmentSteeringController(SteeringController(),segment_fallback=True,
                                        recovery_config=RecoveryConfig())
            bad=segment_detection(bend,heading_control_valid=False,
                                  near_observation_paired=True,near_observation_quality=.9)
            bad.update(changes)
            self.assertEqual(c.command(bad,1,.17),(0.,0.),changes)
