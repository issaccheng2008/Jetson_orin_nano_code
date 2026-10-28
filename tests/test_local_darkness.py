import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from local_darkness import card_candidates, lane_candidates


class LocalDarknessTests(unittest.TestCase):
    def test_camera_noise_cannot_become_card_strokes(self):
        rng = np.random.default_rng(910)
        gray = np.clip(70 + rng.integers(-2, 3, size=(200, 250)), 0, 255).astype(np.uint8)
        ink, info = card_candidates(gray)
        self.assertEqual(int(np.count_nonzero(ink)), 0)
        self.assertGreaterEqual(info['local_dark_threshold'], 8.)

    def test_dark_card_stroke_survives_without_color_change(self):
        gray = np.full((200, 250), 90, np.uint8)
        cv2.rectangle(gray, (60, 55), (180, 160), 3, 4)
        original = gray.copy()
        card, _ = card_candidates(gray)
        lane, _ = lane_candidates(gray)
        self.assertGreater(int(np.count_nonzero(card)), 0)
        self.assertGreater(int(np.count_nonzero(lane)), 0)
        np.testing.assert_array_equal(gray, original)


if __name__ == '__main__':
    unittest.main()
