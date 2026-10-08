"""The optional observer must tolerate imperfect lines without moving legacy outputs."""
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from track_geometry import TrackGeometryTracker
from line_detector_v1_warp import LineDetector


class TrackGeometryTests(unittest.TestCase):
    def setUp(self):
        self.z = np.linspace(60., 20., 400)
        self.scale = np.full(400, .25)

    def mask(self, curved=False, only_left=False, gaps=False):
        image = np.zeros((400, 320), np.uint8)
        for y in range(25, 400):
            z = self.z[y]
            centre = -(77.-math.sqrt(77.**2-z**2)) if curved else 0.
            for side, offset in (("left", -17.5), ("right", 17.5+1.5*math.sin(z/8.))):
                if only_left and side == "right":
                    continue
                if gaps and 260 <= y <= 275:
                    continue
                x = round(160+(centre+offset)/.25)
                if 0 <= x < 320:
                    cv2.circle(image, (x, y), 3, 255, -1)
        return image

    def test_nonparallel_curved_boundaries_and_glare_gap(self):
        tracker = TrackGeometryTracker()
        result = tracker.observe(self.mask(curved=True, gaps=True), self.z,
                                 self.scale, 160, (-4., 26., 20., 35.))
        self.assertTrue(result["track_valid"], result)
        self.assertEqual(result["track_mode"], "paired")
        self.assertEqual(result["track_phase"], "left_bend")
        self.assertGreater(result["track_target_bearing_deg"], 10.)
        self.assertLess(result["track_target_z_cm"], 50.1)

    def test_single_edge_requires_recent_paired_width(self):
        tracker = TrackGeometryTracker()
        initial = tracker.observe(self.mask(), self.z, self.scale, 160,
                                  (0., 26., 0., 35.))
        self.assertTrue(initial["track_valid"])
        single = tracker.observe(self.mask(only_left=True), self.z,
                                 self.scale, 160, None, .1)
        self.assertTrue(single["track_valid"], single)
        self.assertEqual(single["track_mode"], "single")
        other = TrackGeometryTracker().observe(self.mask(only_left=True), self.z,
                                                self.scale, 160, None, .1)
        self.assertFalse(other["track_valid"])

    def test_two_visible_boundaries_bootstrap_without_legacy_lock(self):
        tracker = TrackGeometryTracker()
        result = tracker.observe(self.mask(), self.z, self.scale, 160, None, .1)
        self.assertTrue(result["track_valid"], result)
        self.assertEqual(result["track_mode"], "paired")

    def test_optional_detector_observer_does_not_change_legacy_measurement(self):
        frame = np.full((400, 320, 3), 255, np.uint8)
        for x in (90, 230):
            cv2.line(frame, (x, 150), (x, 399), (0, 0, 0), 7)
        old = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
        new = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
        for detector in (old, new):
            detector.M = np.eye(3)
            detector.startup_force_simple_bottom = False
        new.track_geometry_enable = True
        for _ in range(3):
            before = old.process(frame, dt=.1)
            after = new.process(frame, dt=.1)
            self.assertEqual(before[:3], after[:3])
        self.assertTrue(after[-1]["track_valid"])


if __name__ == "__main__":
    unittest.main()
