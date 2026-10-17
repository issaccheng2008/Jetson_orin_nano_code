"""Button operation must share standalone debug perception and steering settings."""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from run_policy_vision import parse_args

DEBUG = ['--camera-device', '/dev/robot_main_camera', '--card-camera', '-1',
         '--start-gate', 'off', '--attitude-port', '0', '--connector-port', '5999']


class VisionDebugProfileTests(unittest.TestCase):
    def parse(self, flags):
        with patch.dict(os.environ, {}, clear=True), patch('sys.argv', ['vision', *flags]):
            return parse_args()

    def test_debug_profile_matches_standalone_with_conflicting_saved_tuning(self):
        debug = self.parse(DEBUG)
        profiled = self.parse(['--vision-profile', 'debug', '--start-gate', 'button',
            '--start-policy-on-gate', '--connector-port', '5006', '--attitude-port', '5007',
            '--line-preprocess', 'contrast', '--heading-regions-cm', '[[20,22],[24,26]]',
            '--steering-filter-mode', 'active', '--no-single-boundary-tracking',
            '--heading-left-wz', '.3', '.37', '.5', '--heading-right-tolerance-deg', '10',
            '--heading-left-tolerance-deg', '0', '--heading-full-scale-deg', '15',
            '--heading-corridor-cm', '5', '--card-trigger-dist-cm', '30',
            '--steering-loss-fallback-wz', '.5', '--camera-exposure-mode', 'manual',
            '--camera-exposure-ms', '30', '--camera-contrast', '10',
            '--steering-angle-wz-table', '[[-90,-15,-0.1],[-15,15,0.1],[15,90,0.5]]'])
        from vision_debug_profile import DEBUG_FIELDS
        for key in DEBUG_FIELDS:
            with self.subTest(parameter=key):
                self.assertEqual(getattr(profiled, key), getattr(debug, key))
        self.assertEqual(profiled.start_gate, 'button')
        self.assertTrue(profiled.start_policy_on_gate)
        self.assertEqual(profiled.connector_port, 5006)
        self.assertEqual(profiled.attitude_port, 5007)
        self.assertFalse(profiled.attitude_pitch_correction)

    def test_configured_profile_preserves_explicit_parameters(self):
        args = self.parse(['--vision-profile', 'configured', '--line-preprocess', 'contrast',
                          '--heading-regions-cm', '[[20,22],[24,26]]', '--steering-filter-mode', 'active'])
        self.assertEqual(args.line_preprocess, 'contrast')
        self.assertEqual(args.heading_regions_cm, ((20., 22.), (24., 26.)))
        self.assertEqual(args.steering_filter_mode, 'active')
        self.assertTrue(args.attitude_pitch_correction)

    def test_debug_profile_preserves_runtime_and_startup_settings(self):
        args = self.parse(['--vision-profile', 'debug', '--start-gate', 'button',
            '--start-policy-on-gate', '--start-policy-port', '/dev/example-stm32',
            '--startup-sequence', '[{"duration_s":1,"vx":0.2,"wz":0.1}]',
            '--recording-root', 'records/example', '--log-items', 'vision', '--headless',
            '--max-seconds', '120', '--hold-still'])
        self.assertTrue(args.hold_still)
        self.assertEqual(args.start_policy_port, '/dev/example-stm32')
        self.assertEqual(args.recording_root, 'records/example')
        self.assertEqual(args.log_items, ('vision',))
        self.assertTrue(args.headless)
        self.assertEqual(args.max_seconds, 120.)
        self.assertIn('duration_s', args.startup_sequence)

    def test_debug_environment_overrides_cannot_reintroduce_installed_tuning(self):
        with patch.dict(os.environ, {'LINE_PREPROCESS': 'contrast',
                'HEADING_REGIONS_CM': '[[20,22],[24,26]]', 'HEADING_FAR_CM': '50',
                'STEERING_ANGLE_WZ_TABLE': '[[-90,0,-.1],[0,90,.3]]',
                'PHOTOMETRIC_MODE': 'raw', 'SHAPE_PREPROCESS': 'canny'}), \
                patch('sys.argv', ['vision', '--vision-profile', 'debug']):
            args = parse_args()
        self.assertEqual(args.line_preprocess, 'legacy')
        self.assertEqual(args.photometric_mode, 'lut')
        self.assertEqual(args.shape_preprocess, 'selective')
        self.assertIsNone(args.heading_regions_cm)
        self.assertIsNone(args.steering_angle_wz_table)
        self.assertEqual(args.heading_far_cm, 29.)

    def test_debug_pins_detector_environment_and_restores_it_after_run(self):
        from camera_config import distance_calib
        from vision_debug_profile import runtime_debug_environment
        with patch.dict(os.environ, {'CAMERA_PROFILE': 'stale_saved_profile',
                'SHAPE_MAX_DIST_CM': '20', 'PHOTOMETRIC_LUT_DIR': '/tmp/old-mapping'}), \
                patch('sys.argv', ['vision', '--vision-profile', 'debug']):
            args = parse_args()
            expected = Path(__file__).resolve().parents[1] / 'new_vision/config/photometric_mapping'
            self.assertEqual(Path(args.photometric_lut_dir).resolve(), expected.resolve())
            with runtime_debug_environment(args):
                from shape_detector import ShapeDetector
                from line_detector_v1_warp import LineDetector
                line, shape = LineDetector(), ShapeDetector()
                self.assertEqual((line.z_a, line.z_b), distance_calib('usb_main'))
                self.assertEqual(shape.cfg['max_dist_cm'], 80.)
                self.assertEqual(shape.cfg['cam_height_cm'], 32.5)
                self.assertEqual(shape.cfg['cam_pitch_deg'], 45.)
            self.assertEqual(os.environ['CAMERA_PROFILE'], 'stale_saved_profile')
            self.assertEqual(os.environ['SHAPE_MAX_DIST_CM'], '20')


if __name__ == '__main__':
    unittest.main()
