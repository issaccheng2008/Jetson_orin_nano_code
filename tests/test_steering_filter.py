"""Causal filtering, configurable ladders and control isolation without hardware."""
import importlib
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from test_heading_steering import detection
from segment_steering import SegmentSteeringController
from test_segment_steering import detection as segment_detection, bend


class SteeringFilterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('steering_filter'), 'filter module is not implemented')
        self.module = importlib.import_module('steering_filter')

    def test_constant_signal_and_irregular_time_are_finite(self):
        f = self.module.GeometryFilter(self.module.FilterConfig())
        geometry = (0., 25., 30., 20., 'ground_x_z')
        for now in (0., .03, .21, .48, .7):
            out = f.apply(geometry, now)
            self.assertEqual(out, geometry)
        with self.assertRaises(ValueError):
            f.apply(geometry, .6)

    def test_adaptive_filter_follows_step_faster_than_fixed_filter(self):
        adaptive = self.module.GeometryFilter(self.module.FilterConfig(beta=.1))
        fixed = self.module.GeometryFilter(self.module.FilterConfig(beta=0.))
        initial, step = (0., 25., 0., 0., 'ground_x_z'), (0., 25., 40., 30., 'ground_x_z')
        adaptive.apply(initial, 0.)
        fixed.apply(initial, 0.)
        a, b = adaptive.apply(step, .14), fixed.apply(step, .14)
        self.assertGreater(a[2], b[2])
        self.assertGreater(a[3], b[3])
        self.assertLess(a[2], 40.)
        self.assertGreater(a[2], 0.)

    def test_source_change_and_reset_use_first_valid_value_immediately(self):
        f = self.module.GeometryFilter(self.module.FilterConfig())
        f.apply((0., 25., 30., 20., 'ground_x_z'), 0.)
        other = (3., 25., -20., -10., 'single_edge')
        self.assertEqual(f.apply(other, .1), other)
        f.reset()
        newest = (0., 25., 50., 40., 'single_edge')
        self.assertEqual(f.apply(newest, .2), newest)

    def test_algorithm_can_be_disabled_or_replaced_with_ema(self):
        first, second = (0., 25., 0., 0., 'ground_x_z'), (5., 25., 30., 20., 'ground_x_z')
        for algorithm in ('none', 'ema'):
            f = self.module.GeometryFilter(self.module.FilterConfig(algorithm=algorithm))
            f.apply(first, 0.)
            out = f.apply(second, .1)
            if algorithm == 'none':
                self.assertEqual(out, second)
            else:
                self.assertTrue(0 < out[0] < 5)
                self.assertTrue(0 < out[2] < 30)

    def test_invalid_settings_are_rejected(self):
        for kwargs in ({'beta': -1}, {'min_cutoff_hz': 0}, {'max_cutoff_hz': 1},
                       {'derivative_cutoff_hz': math.nan}, {'position_tau_s': 0},
                       {'hysteresis_deg': -1}, {'enter_deg': 0, 'exit_deg': 1},
                       {'algorithm': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.module.FilterConfig(**kwargs)

    def test_hysteresis_prevents_chatter_and_can_jump_to_highest_level(self):
        select = self.module.select_level
        options = dict(levels=(.3, .37, .5), cap=.5, full_scale=15., gate=0., width=1.)
        current = .3
        for angle in (10.2, 9.9, 10.8, 10.0):
            current = select(angle, current, **options)
            self.assertEqual(current, .3)
        self.assertEqual(select(11.2, .3, **options), .37)
        self.assertEqual(select(9.5, .37, **options), .37)
        self.assertEqual(select(8.9, .37, **options), .3)
        self.assertEqual(select(25., .3, **options), .5)
        self.assertEqual(select(0., .5, **options), .3)

    def make_controller(self, **changes):
        return HeadingSteeringController(SteeringController(), min_hold_s=0.,
            left_levels=(.3, .37, .5), full_scale_deg=15.,
            right_tolerance_deg=0., left_tolerance_deg=0., **changes)

    def test_loss_never_feeds_zero_into_filter_and_stop_resets(self):
        c = self.make_controller(filter_config=self.module.FilterConfig())
        first = c.command(detection(30.), 1., .1)
        invalid = detection(0., heading_control_valid=False, measurement_valid=False)
        c.command(invalid, 0., .05)
        self.assertEqual(c.command(detection(30.), 1., .05), first)
        self.assertAlmostEqual(c.diagnostics['steering_filter_heading_deg'], 30.)
        c.drop_held_command()
        self.assertEqual(c.command(invalid, 0., .1), (0., 0.))
        c.command(detection(-30.), 1., .1)
        self.assertAlmostEqual(c.diagnostics['steering_filter_heading_deg'], -30.)

    def test_command_change_keeps_filter_memory_but_restarts_prediction(self):
        cfg = self.module.FilterConfig(hysteresis_deg=0., enter_deg=0., exit_deg=0.)
        c = self.make_controller(filter_config=cfg)
        c.command(detection(5.), 1., .1)
        c.command(detection(40.), 1., .1)
        value = c.diagnostics['steering_filter_heading_deg']
        self.assertTrue(5 < value < 40)
        c.command(detection(40.), 1., .1)
        self.assertTrue(value < c.diagnostics['steering_filter_heading_deg'] < 40)

    def test_segments_filter_selected_bearing_and_reset_on_heading_fallback(self):
        c = SegmentSteeringController(SteeringController(), filter_config=self.module.FilterConfig())
        for _ in range(3):
            c.command(segment_detection(bend), 1., .15)
        self.assertTrue(c.diagnostics['segment_control_active'])
        self.assertEqual(c.diagnostics['steering_heading_source'], 'measured_segments')
        self.assertAlmostEqual(c.diagnostics['steering_filter_demand_deg'],
                               c.diagnostics['segment_target_bearing_deg'])
        c.command(detection(-30.), 1., .15)
        self.assertEqual(c.diagnostics['steering_heading_source'], 'ground_x_z')
        self.assertAlmostEqual(c.diagnostics['steering_filter_heading_deg'], -30.)

    def test_shadow_output_is_identical_to_legacy_and_states_are_separate(self):
        actual, reference = self.make_controller(), self.make_controller()
        trial = self.make_controller(filter_config=self.module.FilterConfig())
        c = self.module.FilterComparison(actual, trial)
        for angle in (20., 40., -30., 0., 10., 30.):
            self.assertEqual(c.command(detection(angle), 1., .15),
                             reference.command(detection(angle), 1., .15))
            self.assertIn('steering_filter_shadow_wz', c.diagnostics)
        c.drop_held_command()
        self.assertEqual(c.hold, (0., 0.))
        self.assertEqual(trial.hold, (0., 0.))

    def test_disabling_new_filter_keeps_legacy_median_for_hysteresis_only_trial(self):
        c = self.make_controller(filter_config=self.module.FilterConfig(algorithm='none'))
        for _ in range(10):
            c.command(detection(20.), 1., .05)
        self.assertGreater(c.command(detection(-40.), 1., .05)[1], 0.)


if __name__ == '__main__':
    unittest.main()
