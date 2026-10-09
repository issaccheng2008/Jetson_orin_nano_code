"""Extraction regressions: dark thin tape, orientation and blank/noisy ground."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from line_preprocess import extract_lane_candidates


class LinePreprocessTests(unittest.TestCase):
    def test_dark_thin_tape_is_not_erased(self):
        gray = np.full((400, 320), 65, np.uint8)
        truth = np.zeros_like(gray)
        cv2.line(gray, (50, 390), (230, 10), 40, 2)
        cv2.line(truth, (50, 390), (230, 10), 255, 1)
        original = gray.copy()
        old = extract_lane_candidates(gray, 'legacy')[1]
        new = extract_lane_candidates(gray, 'contrast')[1]
        self.assertEqual(np.count_nonzero(old), 0)
        self.assertGreater(np.mean(new[truth > 0] > 0), .95)
        np.testing.assert_array_equal(gray, original)

    def test_horizontal_and_oblique_tape_survive(self):
        for points in [((20, 200), (290, 200)), ((20, 350), (290, 30))]:
            gray = np.full((400, 320), 160, np.uint8)
            truth = np.zeros_like(gray)
            cv2.line(gray, *points, 30, 5)
            cv2.line(truth, *points, 255, 1)
            mask = extract_lane_candidates(gray, 'contrast')[1]
            self.assertGreater(np.mean(mask[truth > 0] > 0), .95)

    def test_blank_and_low_noise_ground_do_not_create_tape(self):
        rng = np.random.default_rng(0)
        for brightness in [35, 65, 160, 230]:
            for sigma in [0, 1]:
                gray = np.clip(brightness + rng.normal(0, sigma, (400, 320)),
                               0, 255).astype(np.uint8)
                self.assertEqual(np.count_nonzero(extract_lane_candidates(gray)[1]), 0)

    def test_small_spots_remain_rejected(self):
        gray = np.full((400, 320), 65, np.uint8)
        for x in range(30, 300, 40):
            cv2.circle(gray, (x, 220), 2, 10, -1)
        self.assertEqual(np.count_nonzero(extract_lane_candidates(gray)[1]), 0)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_lane_candidates(np.zeros((400, 320), np.uint8), 'typo')


if __name__ == '__main__':
    unittest.main()
