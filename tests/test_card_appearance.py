import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from card_appearance import card_appearance, rectify_card_gray


class CardAppearanceTests(unittest.TestCase):
    def card(self, gain=1.):
        gray = np.full((260, 260), 190, np.uint8)
        cv2.rectangle(gray, (30, 30), (230, 230), 20, 7)
        # Triangle almost reaches the frame's bottom: a fixed inset can clip it.
        cv2.polylines(gray, [np.int32([[130, 58], [55, 211], [205, 211]])],
                      True, 20, 6)
        quad = np.float32([[30, 30], [230, 30], [230, 230], [30, 230]])
        return np.clip(gray.astype(float)*gain, 0, 255).astype(np.uint8), quad

    def test_outline_is_selected_inside_paper_without_cropping_bottom(self):
        gray, quad = self.card()
        result = card_appearance(gray, quad)
        self.assertEqual(result.reason, 'enclosed_symbol')
        self.assertIsNotNone(result.symbol)
        self.assertEqual(len(cv2.approxPolyDP(result.symbol,
            .035*cv2.arcLength(result.symbol, True), True)), 3)
        self.assertGreater(cv2.contourArea(result.symbol), 9000)

    def test_local_threshold_preserves_gain_invariance(self):
        symbols = []
        for gain in (1., .6, .4):
            image, quad = self.card(gain)
            result = card_appearance(image, quad)
            self.assertEqual(result.reason, 'enclosed_symbol')
            symbols.append(result.symbol)
        for symbol in symbols[1:]:
            self.assertLess(cv2.matchShapes(symbols[0], symbol,
                                          cv2.CONTOURS_MATCH_I1, 0), .02)

    def test_empty_outer_ring_is_not_a_symbol(self):
        gray, quad = self.card()
        gray[40:220, 40:220] = 190
        result = card_appearance(gray, quad)
        self.assertIsNone(result.symbol)
        self.assertEqual(result.reason, 'no_separated_symbol')

    def test_connected_outline_can_use_its_central_closed_hole(self):
        gray, quad = self.card()
        cv2.line(gray, (130, 211), (130, 230), 20, 8)
        result = card_appearance(gray, quad)
        self.assertEqual(result.reason, 'enclosed_outline_hole')
        self.assertEqual(len(cv2.approxPolyDP(result.symbol,
            .035*cv2.arcLength(result.symbol, True), True)), 3)

    def test_paper_contour_is_not_counted_as_a_second_ink_frame(self):
        gray, quad = self.card()
        gray[40:220, 40:220] = 190
        cv2.circle(gray, (130, 130), 69, 20, 6)
        result = card_appearance(gray, quad)
        self.assertEqual(result.reason, 'enclosed_symbol')
        self.assertEqual(result.frame_count, 1)
        area = cv2.contourArea(result.symbol)
        self.assertGreater(area, 10000)

    def test_broken_frame_keeps_a_complete_outline_hole(self):
        gray, quad = self.card()
        gray[20:42, 110:150] = 190
        result = card_appearance(gray, quad)
        self.assertEqual(result.reason, 'separate_outline_hole')
        self.assertEqual(len(cv2.approxPolyDP(result.symbol,
            .035*cv2.arcLength(result.symbol, True), True)), 3)

    def test_blank_and_corner_order(self):
        gray, quad = self.card()
        np.testing.assert_array_equal(rectify_card_gray(gray, quad),
                                      rectify_card_gray(gray, quad[[2, 0, 3, 1]]))
        result = card_appearance(np.full_like(gray, 65), quad)
        self.assertEqual(result.reason, 'insufficient_contrast')
        self.assertIsNone(result.symbol)


if __name__ == '__main__':
    unittest.main()
