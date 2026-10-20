"""Canny evidence must recover ink, preserve topology, and reject floor noise."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from canny_candidates import card_canny, lane_canny


class CannyCandidateTests(unittest.TestCase):
    paths = ((lane_canny, (400, 320)), (card_canny, (540, 960)))

    def test_dark_bright_and_orientation_keep_stroke_centers(self):
        for function, shape in self.paths:
            for background, foreground in ((65, 40), (160, 30), (230, 180)):
                for start, end in (((30, 200), (270, 200)),
                                   ((30, 350), (270, 30))):
                    with self.subTest(path=function.__name__, background=background,
                                      start=start):
                        gray = np.full(shape, background, np.uint8)
                        truth = np.zeros_like(gray)
                        cv2.line(gray, start, end, foreground, 3)
                        cv2.line(truth, start, end, 255, 1)
                        original = gray.copy()
                        mask, diagnostics = function(gray)
                        self.assertGreater(np.mean(mask[truth > 0] > 0), 0.97)
                        self.assertEqual(mask.dtype, np.uint8)
                        self.assertEqual(mask.shape, shape)
                        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
                        self.assertGreater(diagnostics["canny_raw_edge_pixels"], 0)
                        np.testing.assert_array_equal(gray, original)

    def test_lane_tape_borders_form_one_stroke_and_two_lanes_stay_two(self):
        for thickness in (2, 5, 15, 30):
            with self.subTest(thickness=thickness):
                gray = np.full((400, 320), 160, np.uint8)
                cv2.line(gray, (70, 30), (70, 370), 30, thickness)
                cv2.line(gray, (245, 30), (245, 370), 30, thickness)
                mask, diagnostics = lane_canny(gray)
                # Raw Canny marks tape borders but does not mark its center.
                self.assertEqual(diagnostics["raw_edges"][200, 70], 0)
                self.assertEqual(mask[200, 70], 255)
                self.assertEqual(mask[200, 245], 255)
                self.assertEqual(np.count_nonzero(mask[100:300, 110:205]), 0)
                self.assertEqual(cv2.connectedComponents(mask, 8)[0] - 1, 2)
                row = mask[200] > 0
                starts = np.count_nonzero(row & ~np.r_[False, row[:-1]])
                self.assertEqual(starts, 2)

    def test_blank_and_sensor_noise_do_not_create_ink(self):
        rng = np.random.default_rng(24)
        for function, shape in self.paths:
            for brightness in (35, 65, 160, 230):
                for sigma in (0, 1, 3, 8, 15):
                    with self.subTest(path=function.__name__, brightness=brightness,
                                      sigma=sigma):
                        gray = np.clip(brightness + rng.normal(0, sigma, shape),
                                       0, 255).astype(np.uint8)
                        self.assertEqual(np.count_nonzero(function(gray)[0]), 0)

    def test_noise_changes_canny_gate_and_tape_remains(self):
        rng = np.random.default_rng(55)
        highs = []
        for sigma in (0, 8):
            gray = np.clip(160 + rng.normal(0, sigma, (400, 320)),
                           0, 255).astype(np.uint8)
            cv2.line(gray, (30, 350), (270, 30), 30, 7)
            mask, diagnostics = lane_canny(gray)
            truth = np.zeros_like(gray)
            cv2.line(truth, (30, 350), (270, 30), 255, 1)
            self.assertGreater(np.mean(mask[truth > 0] > 0), 0.97)
            highs.append(diagnostics["canny_high"])
        self.assertGreater(highs[1], highs[0] * 2)

    def test_single_dark_step_has_canny_edge_but_is_not_a_stroke(self):
        for function, shape in self.paths:
            gray = np.full(shape, 160, np.uint8)
            gray[:, :100] = 30
            mask, diagnostics = function(gray)
            self.assertGreater(np.count_nonzero(diagnostics["raw_edges"]), 0)
            self.assertEqual(np.count_nonzero(mask), 0)

    def test_lane_bend_is_not_removed_by_vertical_span_filter(self):
        gray = np.full((400, 320), 90, np.uint8)
        points = np.array(((40, 370), (75, 200), (230, 200), (280, 80)), np.int32)
        cv2.polylines(gray, [points], False, 40, 5)
        truth = np.zeros_like(gray)
        cv2.polylines(truth, [points], False, 255, 1)
        mask, _ = lane_canny(gray)
        self.assertGreater(np.mean(mask[truth > 0] > 0), 0.97)

    def test_fine_card_ring_retains_hole_and_rejects_broad_lane(self):
        gray = np.full((540, 960), 160, np.uint8)
        cv2.rectangle(gray, (90, 170), (300, 330), 30, 3)
        cv2.circle(gray, (195, 250), 35, 30, 3)
        cv2.line(gray, (500, 60), (790, 480), 30, 24)
        mask, _ = card_canny(gray)
        self.assertEqual(mask[170, 195], 255)
        self.assertEqual(mask[250, 195], 0)
        self.assertEqual(mask[210, 195], 0)
        self.assertEqual(np.count_nonzero(mask[:, 450:]), 0)
        _, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        self.assertIsNotNone(hierarchy)
        self.assertGreaterEqual(np.count_nonzero(hierarchy[0, :, 3] >= 0), 2)

    def test_missing_card_side_is_not_completed(self):
        gray = np.full((540, 960), 160, np.uint8)
        cv2.rectangle(gray, (100, 180), (300, 330), 30, 3)
        gray[174:187, 150:240] = 160
        mask, _ = card_canny(gray)
        self.assertEqual(np.count_nonzero(mask[177:184, 155:235]), 0)
        _, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        self.assertIsNotNone(hierarchy)
        self.assertEqual(np.count_nonzero(hierarchy[0, :, 3] >= 0), 0)

    def test_six_card_outlines_remain_closed_and_hollow(self):
        outlines = {
            "square": [(180, 180), (260, 180), (260, 260), (180, 260)],
            "diamond": [(220, 165), (275, 220), (220, 275), (165, 220)],
            "triangle": [(220, 165), (275, 265), (165, 265)],
            "cross": [(205, 170), (235, 170), (235, 205), (270, 205),
                      (270, 235), (235, 235), (235, 270), (205, 270),
                      (205, 235), (170, 235), (170, 205), (205, 205)],
            "star": [(220, 160), (235, 201), (279, 201), (243, 227),
                     (257, 268), (220, 243), (183, 268), (197, 227),
                     (161, 201), (205, 201)],
        }
        for name in (*outlines, "circle"):
            with self.subTest(shape=name):
                gray = np.full((540, 960), 70, np.uint8)
                if name == "circle":
                    cv2.circle(gray, (220, 220), 50, 35, 3)
                else:
                    cv2.polylines(gray, [np.array(outlines[name], np.int32)],
                                  True, 35, 3)
                mask, _ = card_canny(gray)
                self.assertEqual(mask[220, 220], 0)
                _, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
                self.assertIsNotNone(hierarchy)
                self.assertGreater(np.count_nonzero(hierarchy[0, :, 3] >= 0), 0)

    def test_invalid_input_is_rejected(self):
        for function, shape in self.paths:
            with self.assertRaises(ValueError):
                function(np.zeros(shape, np.float32))
            with self.assertRaises(ValueError):
                function(np.zeros((*shape, 3), np.uint8))
            with self.assertRaises(ValueError):
                function(np.zeros((10, 10), np.uint8))


if __name__ == "__main__":
    unittest.main()
