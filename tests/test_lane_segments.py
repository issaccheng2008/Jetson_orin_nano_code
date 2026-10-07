"""Ground-space segment diagnostics on measured straight and curved lanes."""

import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision/jetson"))
from lane_segments import describe_lane_segments
from line_detector_v1_warp import LineDetector


class LaneSegmentTests(unittest.TestCase):
    def describe(self, path, outlier=False, width_cm=35.0):
        # Nonuniform row-to-ground scale, as in the actual inverse perspective map.
        z_by_row = np.linspace(88.0, 20.0, 400)
        scale = np.linspace(.29, .22, 400)
        ys = np.arange(190, 399, 2)
        z = z_by_row[ys]
        x = path(z)
        slope = np.gradient(x, z)
        horizontal_width = width_cm * np.sqrt(1 + slope ** 2)
        centres = 160 + x / scale[ys]
        widths = horizontal_width / scale[ys]
        if outlier:
            centres[25] += 160
            centres[60] -= 130
        return describe_lane_segments(ys, centres, widths, z_by_row, scale, 160, 35)

    def test_constant_yaw_is_not_called_a_bend(self):
        for yaw in (0, 40):
            slope = -math.tan(math.radians(yaw))
            result = self.describe(lambda z: slope * (z - 40))
            self.assertTrue(result["fit_seg_valid"], result)
            self.assertEqual(result["fit_seg_count"], 3)
            self.assertEqual(result["fit_seg_pattern"], "constant_heading")
            self.assertAlmostEqual(result["fit_seg0_heading_deg"], yaw, delta=.3)
            self.assertAlmostEqual(result["fit_seg2_normal_width_cm"], 35, delta=.3)

    def test_77cm_left_arc_changes_local_tangent(self):
        radius = 77.0
        result = self.describe(lambda z: -(radius - np.sqrt(radius ** 2 - z ** 2)))
        self.assertTrue(result["fit_seg_valid"], result)
        self.assertEqual(result["fit_seg_pattern"], "left_bend")
        self.assertGreater(result["fit_seg_heading_change_deg"], 15)

    def test_outliers_and_wrong_pair_width(self):
        slope = -math.tan(math.radians(20))
        result = self.describe(lambda z: slope * (z - 40), outlier=True)
        self.assertTrue(result["fit_seg_valid"], result)
        self.assertAlmostEqual(result["fit_seg0_heading_deg"], 20, delta=1)
        wrong = self.describe(lambda z: slope * (z - 40), width_cm=54)
        self.assertFalse(wrong["fit_seg_valid"], wrong)

    def test_missing_middle_segment_does_not_connect_near_to_far(self):
        z_by_row = np.linspace(88.0, 20.0, 400)
        scale = np.full(400, .25)
        ys = np.arange(190, 399, 2)
        ys = ys[(z_by_row[ys] < 32) | (z_by_row[ys] >= 44)]
        result = describe_lane_segments(ys, [160] * len(ys),
                                        [35 / .25] * len(ys), z_by_row,
                                        scale, 160, 35)
        self.assertFalse(result["fit_seg_valid"], result)
        self.assertTrue(result["fit_seg0_valid"])
        self.assertFalse(result.get("fit_seg1_valid", False))

    def test_optional_diagnostic_keeps_published_vision_measurement(self):
        frame = np.full((400, 320, 3), 255, np.uint8)
        for x in (90, 230):
            cv2.line(frame, (x, 160), (x, 399), (0, 0, 0), 7)
        plain = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
        diagnostic = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
        for value in (plain, diagnostic):
            value.M = np.eye(3)
            value.startup_force_simple_bottom = False
        diagnostic.lane_fit_enable = True
        diagnostic.lane_segments_enable = True
        for _ in range(3):
            before = plain.process(frame, dt=.1)
            after = diagnostic.process(frame, dt=.1)
            self.assertEqual(before[:3], after[:3])
        self.assertIn("fit_seg_count", after[-1])


if __name__ == "__main__":
    unittest.main()
