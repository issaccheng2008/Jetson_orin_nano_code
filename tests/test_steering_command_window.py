"""Mapped frame commands, non-overlapping windows, and lost-frame exclusion."""
import argparse
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from steering_command_window import SteeringCommandWindow, effective_policy_hold
from steering_recovery import RecoveryConfig, TurnHistory, add_arguments, config_from_args
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from test_heading_steering import detection
import run_policy_vision


class CommandWindowTests(unittest.TestCase):
    def test_all_valid_samples_not_last_three_or_latest(self):
        w = SteeringCommandWindow(.5, 'lower')
        for i, value in enumerate([.1, .1, .1, .1, .25, .3, .5]):
            w.update(.2, value, i*.06, valid=True)
        self.assertEqual(w.update(.2, -.2, .5, valid=True), (.2, .1))
        self.assertEqual(w.diagnostics['steering_window_valid_samples'], 7)

    def test_even_selection_preserves_mapped_levels(self):
        for side, expected in [('lower', .1), ('upper', .25)]:
            with self.subTest(side=side):
                w = SteeringCommandWindow(.5, side)
                w.update(.2, .25, 0., valid=True)
                w.update(.2, .1, .1, valid=True)
                self.assertEqual(w.update(.2, .5, .5, valid=True), (.2, expected))

    def test_negative_values_are_sorted_numerically(self):
        w = SteeringCommandWindow(.5, 'upper')
        for i, wz in enumerate([.3, -.2, -.1, .1]):
            w.update(.2, wz, i*.1, valid=True)
        self.assertEqual(w.update(.2, .3, .5, valid=True), (.2, .1))

    def test_last_lost_frames_are_ignored_even_at_deadline(self):
        w = SteeringCommandWindow(.5, 'lower')
        w.update(.2, -.2, 0., valid=True)
        w.update(.2, .1, .1, valid=True)
        w.update(.2, .25, .2, valid=True)
        for t in [.3, .4]:
            self.assertEqual(w.update(.2, .5, t, valid=False), (.2, -.2))
        self.assertEqual(w.update(.2, .5, .5, valid=False), (.2, .1))
        self.assertEqual(w.diagnostics['steering_window_ignored_samples'], 2)

    def test_no_valid_samples_uses_latest_loss_compensation(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, -.2, 0., valid=False)
        w.update(.2, -.2, .1, valid=False)
        w.update(.2, .3, .4, valid=False)
        self.assertEqual(w.update(.2, .3, .5, valid=False), (.2, .3))
        self.assertEqual(w.diagnostics['steering_window_reason'], 'loss_compensation')
        self.assertEqual(w.diagnostics['steering_window_valid_samples'], 0)

    def test_valid_boundary_frame_does_not_leak_into_empty_previous_window(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, -.2, 0., valid=False)
        self.assertEqual(w.update(.2, .5, .5, valid=True), (.2, -.2))
        self.assertEqual(w.update(.2, .1, 1., valid=True), (.2, .5))

    def test_boundary_belongs_to_next_window_and_windows_do_not_overlap(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, .1, 0., valid=True)
        self.assertEqual(w.update(.2, -.2, .5, valid=True), (.2, .1))
        self.assertEqual(w.update(.2, .25, .9, valid=True), (.2, .1))
        self.assertEqual(w.update(.2, .3, 1., valid=True), (.2, -.2))

    def test_irregular_frame_rate_does_not_drift_deadlines(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, .1, 0., valid=True)
        w.update(.2, -.2, .51, valid=True)
        self.assertEqual(w.update(.2, .3, 1.01, valid=True), (.2, -.2))

    def test_stop_and_explicit_bypass_drop_pending_samples(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, .5, 0., valid=True)
        self.assertEqual(w.update(0., 0., .1, valid=False), (0., 0.))
        self.assertEqual(w.update(.2, -.2, .2, valid=True), (.2, -.2))
        w.reset()
        self.assertEqual(w.update(.2, .1, .3, valid=True), (.2, .1))

    def test_camera_gap_and_backwards_clock_clear_stale_samples(self):
        for resumed in [2., -.1]:
            w = SteeringCommandWindow(.5)
            w.update(.2, .5, 0., valid=True)
            self.assertEqual(w.update(.2, -.2, resumed, valid=True), (.2, -.2))
            self.assertEqual(w.diagnostics['steering_window_reason'], 'initial')

    def test_invalid_command_stops_and_clears_window(self):
        w = SteeringCommandWindow(.5)
        w.update(.2, .5, 0., valid=True)
        self.assertEqual(w.update(.2, math.nan, .1, valid=True), (0., 0.))
        self.assertEqual(w.update(.2, .1, .2, valid=True), (.2, .1))

    def test_zero_window_preserves_legacy_and_policy_hold(self):
        w = SteeringCommandWindow(0.)
        self.assertEqual(w.update(.2, .1, 0., valid=True), (.2, .1))
        self.assertEqual(w.update(.2, .3, .1, valid=False), (.2, .3))
        with patch('sys.argv', ['run_policy_vision.py', '--command-min-hold-s', '.5']):
            args = run_policy_vision.parse_args()
        self.assertEqual(effective_policy_hold(args), 0.)
        args.steering_command_window_s = 0.
        self.assertEqual(effective_policy_hold(args), .5)

    def test_config_validation_before_hardware(self):
        for duration, side in [(math.nan, 'lower'), (-1., 'lower'), (.5, 'mean')]:
            with self.subTest(duration=duration, side=side), self.assertRaises(ValueError):
                SteeringCommandWindow(duration, side)
        for flag in [['--steering-command-window-s', 'nan'],
                     ['--steering-command-window-s', '-1'],
                     ['--steering-command-median', 'mean'],
                     ['--steering-loss-fallback-wz', '.6'],
                     ['--steering-loss-fallback-wz', 'nan']]:
            with patch('sys.argv', ['run_policy_vision.py', *flag]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_fallback_config_reaches_history_and_heading_with_caps(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        cfg = config_from_args(parser.parse_args(['--steering-loss-fallback-wz', '-.25']))
        self.assertEqual(cfg.fallback_wz, -.25)
        for loss_s in [.1, 1.]:
            self.assertEqual(TurnHistory(cfg).command(1., loss_s, (.2, 0.))[0], (.2, -.25))
        for sign in [-1, 1]:
            c = HeadingSteeringController(SteeringController(yaw_sign=sign), recovery_config=cfg)
            self.assertEqual(c.command(detection(measurement_valid=False), 0., .1), (.2, -.25*sign))
        c = HeadingSteeringController(SteeringController(), recovery_config=RecoveryConfig(fallback_wz=0.))
        self.assertEqual(c.command(detection(measurement_valid=False), 0., .1), (.2, 0.))
        c = HeadingSteeringController(SteeringController(max_wz=.2), max_step=.2,
                                     left_levels=(.2,), right_levels=(.2,), recovery_config=cfg)
        self.assertEqual(c.command(detection(measurement_valid=False), 0., .1), (.2, -.2))


if __name__ == '__main__':
    unittest.main()
