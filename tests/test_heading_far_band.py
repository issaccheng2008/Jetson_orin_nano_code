"""Heading's far observation stays in ground centimetres with eight scan rows."""
from pathlib import Path
import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from line_detector_v1_warp import LineDetector
from run_policy_vision import parse_args


class HeadingFarBandTests(unittest.TestCase):
    def scan(self, detector):
        gray = np.full((400, 320), 255, np.uint8)
        gray[:, 87:94] = gray[:, 227:234] = 0
        bands = detector._detect_two_band_lanes(
            gray, np.repeat(gray[:, :, None], 3, axis=2),
            80, True, 160, 140, gray_raw=gray)
        return [qualified for band in bands
                if (qualified := detector._qualified_band(band)) is not None]

    def test_far_observation_moves_to_29_cm_without_moving_near_or_adding_rows(self):
        detector = LineDetector()
        before = self.scan(detector)
        detector.set_heading_far_cm(29)
        after = self.scan(detector)
        self.assertEqual(len(after), 2)
        self.assertEqual(before[0]['ys_list'], after[0]['ys_list'])
        self.assertEqual(before[0]['dist_cm'], after[0]['dist_cm'])
        self.assertEqual(len(after[1]['ys_list']), 8)
        self.assertAlmostEqual(after[1]['dist_cm'], 29, delta=.15)
        self.assertLess(max(after[1]['ys_list']), min(after[0]['ys_list']))
        self.assertTrue(detector._fit_ground_control_heading(after, {})['heading_control_valid'])

    def test_can_retune_and_rebuild_pitch_geometry(self):
        detector = LineDetector()
        for distance in (29, 30, 31.4, 40):
            detector.set_heading_far_cm(distance)
            detector.set_camera_pitch_deg(53 if distance == 30 else 45)
            self.assertAlmostEqual(self.scan(detector)[1]['dist_cm'], distance, delta=.15)
            self.assertEqual(len(self.scan(detector)[1]['ys_list']), 8)

    def test_rejects_nonfinite_or_unobservable_distance_without_changing_band(self):
        detector = LineDetector()
        detector.set_heading_far_cm(29)
        original = (detector.band_mid_y0, detector.band_mid_y1)
        for distance in (float('nan'), float('inf'), 0, 25, 100):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                detector.set_heading_far_cm(distance)
            self.assertEqual((detector.band_mid_y0, detector.band_mid_y1), original)

    def test_cli_defaults_to_29_and_accepts_env_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True), patch('sys.argv', ['vision']):
            self.assertEqual(parse_args().heading_far_cm, 29)
        with patch.dict(os.environ, {'HEADING_FAR_CM': '30'}):
            with patch('sys.argv', ['vision']):
                self.assertEqual(parse_args().heading_far_cm, 30)
            with patch('sys.argv', ['vision', '--heading-far-cm', '28.5']):
                self.assertEqual(parse_args().heading_far_cm, 28.5)

    def test_cli_rejects_nonfinite_and_nonpositive_distance(self):
        for distance in ('nan', 'inf', '0', '-1'):
            with patch('sys.argv', ['vision', '--heading-far-cm', distance]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
