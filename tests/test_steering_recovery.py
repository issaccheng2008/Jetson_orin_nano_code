import sys
import argparse
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from steering_recovery import RecoveryConfig, TurnHistory, add_arguments, config_from_args
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from test_heading_steering import detection
from steering_filter import FilterConfig, GeometryFilter
from segment_steering import SegmentSteeringController
from test_segment_steering import detection as segment_detection, bend


class RecoveryTests(unittest.TestCase):
    def test_new_default_and_installed_history_stop_alias_enable_same_recovery(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        default = parser.parse_args([])
        self.assertEqual(default.steering_loss_mode, 'history-turn')
        for mode in ('history-turn', 'history-stop'):
            args = parser.parse_args(['--steering-loss-mode', mode])
            self.assertEqual(config_from_args(args), RecoveryConfig())
        self.assertIsNone(config_from_args(parser.parse_args(['--steering-loss-mode', 'legacy'])))

    def test_loss_keeps_last_applied_speed_instead_of_configured_speed(self):
        history = TurnHistory(RecoveryConfig())
        pair, reason = history.command(1., .1, (.1, 0.), walking_vx=.2)
        self.assertEqual(pair, (.1, .3))
        self.assertEqual(reason, 'loss_default_left')

    def controller(self, **kw):
        return HeadingSteeringController(SteeringController(), min_hold_s=0,
            recovery_config=RecoveryConfig(max_loss_s=.8), **kw)

    def test_loss_keeps_recent_turn_then_searches_left_without_stopping(self):
        c = self.controller()
        valid = c.command(detection(30), 1, .1)
        bad = detection(measurement_valid=False, heading_control_valid=False)
        self.assertEqual(c.command(bad, 0, .3), valid)
        self.assertEqual(c.diagnostics['steering_reason'], 'loss_history_turn')
        self.assertEqual(c.command(bad, 0, .51), (.2, .3))
        self.assertEqual(c.diagnostics['steering_reason'], 'loss_default_left')
        for _ in range(20):
            self.assertEqual(c.command(bad, 0, .5), (.2, .3))

    def test_straight_walking_and_new_normal_following_search_left_without_history(self):
        c = self.controller()
        c.command(detection(0), 1, .1)
        bad = detection(measurement_valid=False)
        self.assertEqual(c.command(bad, 0, .1), (.2, .3))
        c.command(detection(30), 1, .1)
        c.drop_held_command()
        self.assertEqual(c.hold, (0., 0.))
        self.assertEqual(list(c._turn_history.samples), [])
        # Normal following resumes only when the outer start/card gate calls us.
        self.assertEqual(c.command(bad, 0, .1), (.2, .3))

    def test_right_turn_history_expires_into_left_search(self):
        c = self.controller()
        valid = c.command(detection(-30), 1, .1)
        self.assertLess(valid[1], 0.)
        bad = detection(measurement_valid=False)
        self.assertEqual(c.command(bad, 0, .2), valid)
        self.assertEqual(c.command(bad, 0, .61), (.2, .3))
        self.assertEqual(c.command(detection(0), 1, .1), (.2, 0.))

    def test_default_left_obeys_yaw_sign_speed_and_model_cap(self):
        for sign in (-1, 1):
            c = HeadingSteeringController(SteeringController(vx=.1, yaw_sign=sign),
                                          recovery_config=RecoveryConfig())
            self.assertEqual(c.command(detection(measurement_valid=False), 0, .1), (.1, .3*sign))
        c = HeadingSteeringController(SteeringController(max_wz=.2), max_step=.2,
            left_levels=(.2,), right_levels=(.2,), recovery_config=RecoveryConfig())
        self.assertEqual(c.command(detection(measurement_valid=False), 0, .1), (.2, .2))

    def test_explicit_zero_speed_and_invalid_clock_still_stop(self):
        c = HeadingSteeringController(SteeringController(vx=0.), recovery_config=RecoveryConfig())
        self.assertEqual(c.command(detection(measurement_valid=False), 0, .1), (0., 0.))
        c = self.controller()
        c.command(detection(30), 1, .1)
        self.assertEqual(c.command(detection(measurement_valid=False), 0, 0.), (0., 0.))

    def test_zero_history_reuse_deadline_immediately_searches_left(self):
        c = HeadingSteeringController(SteeringController(), recovery_config=RecoveryConfig(max_loss_s=0.))
        c.command(detection(-30), 1, .1)
        self.assertEqual(c.command(detection(measurement_valid=False), 0, .1), (.2, .3))

    def test_stale_observation_uses_history_without_trusting_stale_angle(self):
        c = self.controller()
        valid = c.command(detection(30), 1, .1)
        self.assertEqual(c.command(detection(-30, measurement_stale=True), 0, .2), valid)
        self.assertEqual(c.command(detection(-30), 1, .1)[1], -.5)

    def test_valid_held_turn_refreshes_history(self):
        c=HeadingSteeringController(SteeringController(),min_hold_s=.5,recovery_config=RecoveryConfig())
        for _ in range(5):valid=c.command(detection(30),1,.1)
        self.assertEqual(c.command(detection(measurement_valid=False),0,.1),valid)

    def test_modal_recovery_weights_previous_published_command_duration(self):
        c = self.controller(position_gain=0, position_recovery_cm=0)
        left = c.command(detection(30), 1, .01)
        self.assertEqual(left, (.2, .5))
        self.assertEqual(c.command(detection(-80), 1, .3), (.2, -.5))
        self.assertEqual(c.command(detection(80), 1, .01), (.2, 0.))
        # Left was applied for .30 s; right was applied for only .01 s.
        self.assertEqual(c.command(detection(measurement_valid=False), 0, .01), left)

    def test_history_does_not_invent_duration_before_first_observation(self):
        history = TurnHistory(RecoveryConfig())
        history.observe(.1, (.2, .5), .1)
        history.observe(.1, (.2, 0.), 0.)
        self.assertEqual(history.command(.2, .1, (.2, 0.)),
                         ((.2, .3), 'loss_default_left'))

    def test_default_search_duration_is_not_attributed_to_expired_right_turn(self):
        history = TurnHistory(RecoveryConfig())
        history.observe(0., (.2, -.5), 0.)
        self.assertEqual(history.command(.9, .9, (.2, -.5)),
                         ((.2, .3), 'loss_default_left'))
        history.observe(1., (.2, 0.), .1)
        _, pair, duration = history.samples[-1]
        self.assertEqual(pair, (.2, .3))
        self.assertAlmostEqual(duration, .1)

    def test_no_history_search_duration_is_recorded_before_next_valid_observation(self):
        history = TurnHistory(RecoveryConfig())
        history.observe(.1, (.2, 0.), .1)
        self.assertEqual(history.command(.2, .1, (.2, 0.)),
                         ((.2, .3), 'loss_default_left'))
        history.observe(.4, (.2, 0.), .2)
        _, pair, duration = history.samples[-1]
        self.assertEqual(pair, (.2, .3))
        self.assertAlmostEqual(duration, .2)

    def test_history_weights_only_interval_inside_time_window(self):
        history = TurnHistory(RecoveryConfig(history_s=.8))
        history.observe(0., (.2, .5), 0.)
        history.observe(1., (.2, -.5), 1.)
        history.observe(1.3, (.2, 0.), .3)
        # At 1.31, the window contains .49 s of older left and .30 s of
        # newer right. Linear recency weighting makes the right turn stronger.
        self.assertEqual(history.command(1.31, .01, (.2, 0.))[0], (.2, -.5))

    def test_history_stale_turn_uses_default_search_after_long_straight_interval(self):
        history = TurnHistory(RecoveryConfig())
        history.observe(0., (.2, .5), 0.)
        history.observe(.1, (.2, 0.), .1)
        self.assertEqual(history.command(.6, .1, (.2, 0.)),
                         ((.2, .3), 'loss_default_left'))

    def test_history_tied_duration_weight_prefers_newer_turn(self):
        history = TurnHistory(RecoveryConfig(history_s=1.))
        for now, pair in ((0., (.2, 0.)), (.25, (.2, .5)),
                          (.5, (.2, 0.)), (.6875, (.2, -.5)),
                          (.8125, (.2, 0.))):
            history.observe(now, pair, 0.)
        # Each turn's integrated weight is exactly .09375 at t=1.
        self.assertEqual(history.command(1., .1875, (.2, 0.))[0], (.2, -.5))

    def test_current_turn_has_priority_over_longer_opposite_history(self):
        history = TurnHistory(RecoveryConfig())
        history.observe(0., (.2, .5), 0.)
        history.observe(.3, (.2, -.5), .3)
        self.assertEqual(history.command(.31, .01, (.2, -.5))[0], (.2, -.5))

    def test_offset_safeguard_keeps_raw_two_frame_confirmation(self):
        c=self.controller(filter_config=FilterConfig(algorithm='robust'))
        c.command(detection(30,near=-8),1,.1)
        for _ in range(2):c.command(detection(30,near=4),1,.1)
        self.assertTrue(c.diagnostics['steering_left_offset_confirmed'])

    def test_vetoed_turn_is_removed_before_default_left_search(self):
        c=self.controller(position_gain=0, position_recovery_cm=0)
        c.command(detection(60,z=25),1,.1)
        for _ in range(2):c.command(detection(60,near=6,z=25),1,.1)
        self.assertEqual(c.diagnostics['steering_decision'],'left_offset_release')
        self.assertEqual(c.command(detection(measurement_valid=False),0,.1),(.2,.3))
        self.assertEqual(c.diagnostics['steering_reason'], 'loss_default_left')

    def test_straight_bias_is_not_turn_evidence(self):
        for sign in (-1,1):
            c=HeadingSteeringController(SteeringController(yaw_sign=sign),straight_wz=.05,
                                        recovery_config=RecoveryConfig(),min_hold_s=0)
            c.command(detection(0),1,.1)
            self.assertEqual(c.command(detection(measurement_valid=False),0,.1),(.2,.3*sign))

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
        self.assertEqual(c.command(good, 1, .17), (.2,.3))
        self.assertFalse(c.diagnostics['segment_control_active'])

    def test_fallback_rejects_bad_quality_stale_or_inconsistent_segments(self):
        for changes in (dict(near_observation_quality=.1),dict(measurement_stale=True),
                        dict(fit_seg_anchored=False),dict(fit_seg0_normal_width_cm=50),
                        dict(fit_seg0_rmse_cm=3),dict(fit_seg0_points=4)):
            c=SegmentSteeringController(SteeringController(),segment_fallback=True,
                                        recovery_config=RecoveryConfig())
            bad=segment_detection(bend,heading_control_valid=False,
                                  near_observation_paired=True,near_observation_quality=.9)
            bad.update(changes)
            self.assertEqual(c.command(bad,1,.17),(.2,.3),changes)
            self.assertFalse(c.diagnostics['segment_control_active'], changes)
