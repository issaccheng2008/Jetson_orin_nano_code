"""Interpolated lane gates and argument validation without camera or motors."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision' / 'jetson'))
from line_detector_v1_warp import LineDetector
from run_policy_vision import parse_args


class LaneStrictnessTests(unittest.TestCase):
    def test_endpoints_and_midpoint_preserve_expected_physical_gates(self):
        expected = {
            0: (2, 1., 12., 200., 100., 15., 2, 1., 12.),
            50: (4, 4.5, 7.25, 126., 64., 9., 4, 4.5, 7.25),
            100: (6, 8., 2.5, 52., 28., 3., 6, 8., 2.5),
        }
        for pct, gates in expected.items():
            with self.subTest(strictness=pct):
                detector = LineDetector()
                detector.set_lane_observation_mode('continuous', pct)
                path, heading = detector._boundary_path_tracker, detector.heading_scan
                actual = (path.min_points, path.min_span_cm, path.max_rmse_cm,
                          path.frame_search_px, path.row_search_px,
                          path.reference_margin_cm, heading.min_observed_rows,
                          heading.min_span_cm, heading.max_residual_cm)
                self.assertEqual(actual, gates)
        strict = LineDetector()
        strict.set_lane_observation_mode('strict', 0)
        self.assertEqual(strict._boundary_path_tracker.min_points, 6)
        self.assertEqual(strict.heading_scan.min_observed_rows, 6)
        endpoint = LineDetector()
        endpoint.set_lane_observation_mode('continuous', 100)
        self.assertEqual(endpoint.lane_observation_mode, 'strict')
        self.assertFalse(endpoint._boundary_path_tracker.continuous)
        self.assertEqual(endpoint.heading_scan, strict.heading_scan)

    def test_bad_percent_is_rejected_before_camera_setup(self):
        for pct in ('-1', '101', 'nan', 'inf'):
            with (self.subTest(pct=pct),
                  patch('sys.argv', ['vision', '--lane-strictness-pct', pct]),
                  contextlib.redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit)):
                parse_args()


if __name__ == '__main__':
    unittest.main()
