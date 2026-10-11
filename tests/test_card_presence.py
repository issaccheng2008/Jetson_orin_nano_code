import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision/jetson"))
from card_presence import find_broken_border_presence
from photometric_thresholds import measure


CFG = dict(cue_side_min=30, cue_side_max=340, cue_aspect_max=2.6,
           cue_edge_margin=4, presence_top_frac=1 / 3,
           cue_core_gray_min=100., cue_contrast_min=4., cue_score_min=1.2)


def card(gaps=False, marking=True):
    gray = np.full((540, 960), 180, np.uint8)
    cv2.rectangle(gray, (380, 230), (530, 350), 40, 5)
    if gaps:
        gray[225:237, 438:451] = 180
        gray[287:300, 525:536] = 180
    if marking:
        cv2.circle(gray, (455, 290), 25, 40, 5)
    return gray


class CardPresenceTests(unittest.TestCase):
    def cue(self, gray):
        return find_broken_border_presence(gray, measure(gray), CFG)

    def test_bounded_gaps_keep_outer_card_coordinates_and_scalar_diagnostics(self):
        gray = card(gaps=True)
        box, score, debug = self.cue(gray)
        self.assertIsNotNone(box)
        self.assertGreaterEqual(score, CFG["cue_score_min"])
        x, y, width, height = box
        self.assertLessEqual(abs(x - 377), 5)
        self.assertLessEqual(abs(y - 227), 5)
        self.assertGreater(width, 140)
        self.assertGreater(height, 110)
        self.assertTrue(debug["cue_broken_accepted"])
        self.assertTrue(all(isinstance(v, (bool, float, int)) for v in debug.values()))
        np.testing.assert_array_equal(gray, card(gaps=True))

    def test_affine_brightness_retains_reference_score_and_position(self):
        gray = card(gaps=True)
        original_box, original_score, _ = self.cue(gray)
        for gain, offset in ((.6, 0), (.4, 5)):
            transformed = (gray.astype(np.float32) * gain + offset).astype(np.uint8)
            box, score, _ = self.cue(transformed)
            self.assertEqual(box, original_box)
            self.assertAlmostEqual(score / original_score, 1., delta=.025)

    def test_directional_blur_can_propose_presence_without_a_shape_label(self):
        gray = card(gaps=True)
        for size, axis in ((11, "horizontal"), (11, "vertical")):
            kernel = np.zeros((size, size), np.float32)
            if axis == "horizontal":
                kernel[size // 2] = 1 / size
            else:
                kernel[:, size // 2] = 1 / size
            blurred = cv2.filter2D(gray, -1, kernel)
            box, score, debug = self.cue(blurred)
            self.assertIsNotNone(box, axis)
            self.assertGreaterEqual(score, CFG["cue_score_min"])
            self.assertNotIn("shape", debug)

    def test_rotated_card_uses_card_plane_instead_of_axis_aligned_area(self):
        gray = card(gaps=True)
        for angle in (35., 45., 60.):
            transform = cv2.getRotationMatrix2D((455, 290), angle, 1.)
            rotated = cv2.warpAffine(gray, transform, (960, 540), borderValue=180)
            box, score, _ = self.cue(rotated)
            self.assertIsNotNone(box, angle)
            x, y, width, height = box
            self.assertAlmostEqual(x + width / 2, 455, delta=5)
            self.assertAlmostEqual(y + height / 2, 290, delta=5)
            self.assertGreater(width, 150)
            self.assertGreater(height, 150)
            self.assertGreaterEqual(score, CFG["cue_score_min"])

    def test_missing_whole_side_is_rejected_instead_of_invented(self):
        gray = card()
        gray[225:238, 370:541] = 180
        self.assertIsNone(self.cue(gray)[0])

    def test_unmarked_ring_solid_blob_lanes_cable_and_blank_are_rejected(self):
        images = [card(marking=False), np.full((540, 960), 65, np.uint8)]
        blob = np.full((540, 960), 180, np.uint8)
        cv2.rectangle(blob, (380, 230), (530, 350), 40, -1)
        images.append(blob)
        lanes = np.full((540, 960), 180, np.uint8)
        cv2.line(lanes, (150, 539), (330, 100), 40, 18)
        cv2.line(lanes, (800, 539), (640, 100), 40, 18)
        cv2.polylines(lanes, [np.array([[80, 430], [110, 370], [70, 310],
                                      [130, 240], [140, 160]], np.int32)],
                      False, 40, 5)
        images.append(lanes)
        for i, gray in enumerate(images):
            with self.subTest(scene=i):
                self.assertIsNone(self.cue(gray)[0])

    def test_stateless_far_near_miss_has_no_stale_position(self):
        far = np.full((540, 960), 180, np.uint8)
        far[20:150, 375:536] = card()[225:355, 375:536]
        self.assertIsNone(self.cue(far)[0])
        self.assertIsNotNone(self.cue(card())[0])
        self.assertEqual(self.cue(np.full((540, 960), 180, np.uint8))[:2], (None, 0.))

    def test_archived_cable_and_reflection_failures_remain_negative(self):
        paths = sorted((ROOT / "records/shape_dump_3").glob("*.jpg"))
        if not paths:
            self.skipTest("optional archived real frames are absent")
        self.assertEqual(len(paths), 3)
        for path in paths:
            gray = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (960, 540)),
                                cv2.COLOR_BGR2GRAY)
            with self.subTest(frame=path.name):
                self.assertIsNone(self.cue(gray)[0])


if __name__ == "__main__":
    unittest.main()
