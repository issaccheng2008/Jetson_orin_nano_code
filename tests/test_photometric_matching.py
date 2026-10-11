import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from photometric_thresholds import Photometry, measure, GRAY_REFERENCE, MAX_CHANNEL_REFERENCE


class PhotometricMatchingTests(unittest.TestCase):
    def test_line_detector_preserves_color_while_matching_luminance(self):
        from line_detector_v1_warp import LineDetector
        rng = np.random.default_rng(49)
        base = np.clip(rng.normal(84., 6., (720, 1280)), 0, 255).astype(np.uint8)
        frame = np.stack((np.maximum(base.astype(np.int16)-16, 0),
                          np.maximum(base.astype(np.int16)-7, 0), base), axis=2).astype(np.uint8)
        original = frame.copy()
        detector = LineDetector()
        detector.photometric_mode = 'legacy'
        photo = measure(np.max(frame, axis=2), 'legacy', MAX_CHANNEL_REFERENCE)
        raw_bird = cv2.warpPerspective(frame, detector.M,
                                      (detector.bird_w, detector.bird_h))
        expected_gray = photo.match_image(np.max(raw_bird, axis=2))
        debug = detector.process(frame, dt=.1)[-1]
        np.testing.assert_array_equal(debug['bird_color'][detector.ground_valid_mask],
                                      raw_bird[detector.ground_valid_mask])
        np.testing.assert_array_equal(debug['bird'][detector.ground_valid_mask],
                                      expected_gray[detector.ground_valid_mask])
        self.assertFalse(np.any(debug['binary'][~detector.ground_valid_mask]))
        np.testing.assert_array_equal(frame, original)

    def test_gray_match_caps_gain_without_mutating_input(self):
        rng = np.random.default_rng(20261010)
        source = np.clip(rng.normal(72.0, 5.0, (540, 960)), 0, 255).astype(np.uint8)
        original = source.copy()

        photo = measure(source, mode="legacy", reference=GRAY_REFERENCE)
        matched = photo.match_image(source)
        result = measure(matched, mode="legacy", reference=GRAY_REFERENCE)

        self.assertIsNot(matched, source)
        np.testing.assert_array_equal(source, original)
        self.assertAlmostEqual(result.mean, GRAY_REFERENCE[0], delta=0.15)
        self.assertAlmostEqual(result.std, min(photo.match_scale * photo.std,
                                                 GRAY_REFERENCE[1]), delta=0.15)

    def test_color_image_max_channel_matches_line_reference(self):
        rng = np.random.default_rng(17)
        base = np.clip(rng.normal(84.0, 6.0, (540, 960)), 0, 255).astype(np.uint8)
        bgr = np.stack((np.maximum(base.astype(np.int16) - 16, 0),
                        np.maximum(base.astype(np.int16) - 7, 0), base), axis=2).astype(np.uint8)
        photo = measure(np.max(bgr, axis=2), mode="legacy",
                        reference=MAX_CHANNEL_REFERENCE)
        matched = photo.match_image(bgr)
        result = measure(np.max(matched, axis=2), mode="legacy",
                         reference=MAX_CHANNEL_REFERENCE)

        self.assertEqual(matched.shape, bgr.shape)
        self.assertEqual(matched.dtype, bgr.dtype)
        self.assertAlmostEqual(result.mean, MAX_CHANNEL_REFERENCE[0], delta=0.15)
        self.assertAlmostEqual(result.std, min(photo.match_scale * photo.std,
                                                 MAX_CHANNEL_REFERENCE[1]), delta=0.15)

    def test_flat_frame_passes_through_without_inventing_contrast(self):
        photo = Photometry(31.0, 0.0, 150.0, 20.0, "legacy")
        image = np.full((20, 30), 31, dtype=np.uint8)
        matched = photo.match_image(image)

        np.testing.assert_array_equal(matched, image)
        self.assertFalse(photo.match_allowed)
        self.assertEqual(float(matched.std()), 0.0)
        self.assertEqual(photo.diagnostics()["photometric_match_reject_reason"], "low_contrast")

    def test_exclusive_modes_and_legacy_alias(self):
        from photometric_thresholds import resolve_mode
        source = np.tile(np.arange(80, 180, dtype=np.uint8), (200, 1))
        raw = measure(source, "raw")
        threshold = measure(source, "threshold")
        matched = measure(source, "match")
        np.testing.assert_array_equal(raw.apply(source), source)
        np.testing.assert_array_equal(threshold.apply(source), source)
        self.assertNotEqual(threshold.difference(16), 16.)
        self.assertEqual(matched.difference(16), 16.)
        self.assertTrue(matched.match_allowed)
        np.testing.assert_array_equal(matched.apply(source),
                                      measure(source, "legacy").apply(source))
        self.assertEqual(resolve_mode("normalize"), "threshold")

    def test_capped_gain_and_noise_rejection(self):
        source = np.tile(np.array([80, 81, 82, 83], np.uint8), (60, 30))
        p = measure(source, "match")
        self.assertFalse(p.match_allowed)
        np.testing.assert_array_equal(p.apply(source), source)
        richer = np.tile(np.arange(60, 67, dtype=np.uint8), (60, 18))
        q = measure(richer, "match")
        self.assertFalse(q.match_allowed)
        source = np.tile(np.arange(60, 75, dtype=np.uint8), (60, 8))
        q = measure(source, "match")
        self.assertTrue(q.match_allowed)
        self.assertLessEqual(q.match_scale, 2.5)


if __name__ == "__main__":
    unittest.main()
