"""Independent near position must survive opposite distant target demand."""
import math
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_filter import FilterConfig
from test_heading_steering import detection
from test_segment_steering import detection as segment_detection, settle
from segment_steering import SegmentSteeringController


class PositionCliTests(unittest.TestCase):
    def test_defaults_and_configured_position_values_are_accepted_before_hardware(self):
        import run_policy_vision
        with patch('sys.argv', ['vision']):
            defaults = run_policy_vision.parse_args()
        self.assertEqual(defaults.position_gain, 1.)
        self.assertEqual(defaults.position_recovery_cm, 8.)
        with patch('sys.argv', ['vision', '--position-gain', '2', '--position-dead-cm', '1',
                '--position-lookahead-cm', '30', '--position-max-deg', '9',
                '--position-recovery-cm', '6', '--position-recovery-full-scale-cm', '3',
                '--position-confirm-frames', '3']):
            args = run_policy_vision.parse_args()
        self.assertEqual((args.position_gain, args.position_dead_cm, args.position_lookahead_cm,
                          args.position_max_deg, args.position_recovery_cm,
                          args.position_recovery_full_scale_cm, args.position_confirm_frames),
                         (2., 1., 30., 9., 6., 3., 3))

    def test_invalid_position_cli_settings_are_rejected_before_hardware(self):
        import run_policy_vision
        for name, value in (('gain', '-1'), ('dead-cm', '-1'), ('lookahead-cm', '0'),
                ('max-deg', 'nan'), ('recovery-cm', '-1'), ('recovery-full-scale-cm', '0'),
                ('confirm-frames', '0'), ('confirm-frames', '1.5')):
            with self.subTest(name=name, value=value), patch('sys.argv',
                    ['vision', '--position-'+name, value]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_policy_vision.parse_args()
                self.assertEqual(caught.exception.code, 2)


class PositionSteeringTests(unittest.TestCase):
    def test_symmetric_recovery_overrides_distant_turn_and_hold(self):
        for side in (-1, 1):
            for yaw in (-1, 1):
                c = HeadingSteeringController(SteeringController(yaw_sign=yaw), min_hold_s=.5)
                c.command(detection(side*35, z=25), 1, .01)
                frame = detection(side*35, near=side*12, z=25)
                c.command(frame, 1, .05)
                output = c.command(frame, 1, .05)
                self.assertLess(output[1]*side*yaw, 0)
                self.assertEqual(output[0], .2)
                self.assertTrue(c.diagnostics['steering_position_recovery'])

    def test_position_term_is_separate_from_filtered_direction(self):
        for gain in (0., 1., 2.):
            c = HeadingSteeringController(SteeringController(), position_gain=gain,
                position_recovery_cm=0, position_dead_cm=1)
            # Direction precisely cancels near offset in the old bearing.
            angle = math.degrees(math.atan(6/25))
            output = c.command(detection(angle, near=6, z=25), 1, .1)
            if gain == 0:
                self.assertEqual(output[1], 0)
            else:
                self.assertLess(output[1], 0)
            self.assertAlmostEqual(c.diagnostics['steering_filtered_demand_deg'], 0, places=6)
            self.assertAlmostEqual(c.diagnostics['steering_position_correction_deg'],
                -gain*math.degrees(math.atan2(5, 50)))
            self.assertAlmostEqual(c.diagnostics['steering_combined_demand_deg'],
                c.diagnostics['steering_position_correction_deg'])

    def test_dead_band_and_bounded_correction(self):
        c = HeadingSteeringController(SteeringController(), position_gain=10,
            position_recovery_cm=0, position_max_deg=7, position_dead_cm=2)
        c.command(detection(0, near=1, z=25), 1, .1)
        self.assertEqual(c.diagnostics['steering_position_correction_deg'], 0)
        for _ in range(3):
            c.command(detection(0, near=100, z=25), 1, .1)
        self.assertEqual(c.diagnostics['steering_position_correction_deg'], -7)

    def test_active_filter_cannot_delay_confirmed_position_recovery(self):
        c = HeadingSteeringController(SteeringController(max_wz_right=.3),
            filter_config=FilterConfig(algorithm='robust', position_tau_s=10))
        c.command(detection(35, z=25), 1, .1)
        for _ in range(2):
            output = c.command(detection(35, near=12, z=25), 1, .1)
        self.assertLess(output[1], 0)
        self.assertGreaterEqual(output[1], -.3)

    def test_one_spike_loss_and_external_stop_do_not_confirm_recovery(self):
        for loss in (False, True):
            c = HeadingSteeringController(SteeringController())
            c.command(detection(0, z=25), 1, .1)
            c.command(detection(35, near=12, z=25), 1, .1)
            self.assertFalse(c.diagnostics['steering_position_recovery'])
            if loss:
                c.command(detection(measurement_valid=False), 0, .3)
            else:
                c.drop_held_command()
            c.command(detection(35, near=12, z=25), 1, .1)
            self.assertFalse(c.diagnostics['steering_position_recovery'])

    def test_segment_target_on_opposite_side_cannot_override_near_position(self):
        for side in (-1, 1):
            c = SegmentSteeringController(SteeringController(), min_hold_s=0)
            def path(z):
                return side*(12-.9*(z-25))
            frame = segment_detection(path, heading=side*42)
            output = settle(c, frame)[-1]
            self.assertTrue(c.diagnostics['segment_control_active'])
            self.assertLess(output[1]*side, 0)

    def test_invalid_position_parameters(self):
        for option, value in (('position_gain', -1), ('position_dead_cm', -1),
                ('position_lookahead_cm', 0), ('position_max_deg', float('nan')),
                ('position_recovery_cm', -1), ('position_recovery_full_scale_cm', 0),
                ('position_confirm_frames', 0), ('position_confirm_frames', 1.5)):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    HeadingSteeringController(SteeringController(), **{option:value})

    def test_confirmation_side_switch_resets_and_strength_is_tunable(self):
        c = HeadingSteeringController(SteeringController(), position_confirm_frames=3,
            position_recovery_cm=6, position_recovery_full_scale_cm=2)
        c.command(detection(35, near=9, z=25), 1, .1)
        c.command(detection(-35, near=-9, z=25), 1, .1)
        self.assertEqual(c.diagnostics['steering_position_confirm_frames'], 1)
        c.command(detection(-35, near=-9, z=25), 1, .1)
        self.assertFalse(c.diagnostics['steering_position_recovery'])
        self.assertEqual(c.command(detection(-35, near=-9, z=25), 1, .1)[1], .5)
        self.assertTrue(c.diagnostics['steering_position_recovery'])
        c.command(detection(0, near=0, z=25), 1, .1)
        self.assertFalse(c.diagnostics['steering_position_recovery'])

    def test_disabled_right_recovery_does_not_preserve_an_outward_left_turn(self):
        c = HeadingSteeringController(SteeringController(), allow_right=False, straight_wz=.05)
        frame = detection(60, near=12, z=25)
        c.command(frame, 1, .1)
        self.assertEqual(c.command(frame, 1, .1)[1], .05)

    def test_loss_cannot_revive_turn_vetoed_by_disabled_right_recovery(self):
        from steering_recovery import RecoveryConfig
        for yaw_sign in (-1, 1):
            c = HeadingSteeringController(SteeringController(yaw_sign=yaw_sign),
                allow_right=False, straight_wz=.05, recovery_config=RecoveryConfig())
            frame = detection(60, near=12, z=25)
            c.command(frame, 1, .1)
            self.assertEqual(c.command(frame, 1, .1)[1], .05*yaw_sign)
            self.assertEqual(list(c._turn_history.samples), [])
            self.assertEqual(c.command(detection(measurement_valid=False), 0., .1),
                             (.2, .3*yaw_sign))
            self.assertEqual(c.diagnostics['steering_reason'], 'loss_default_left')
