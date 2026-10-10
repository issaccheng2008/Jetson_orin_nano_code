import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision/jetson"))
from photometric_thresholds import Photometry, measure, GRAY_REFERENCE, MAX_CHANNEL_REFERENCE


class PhotometricMatchingTests(unittest.TestCase):
    def test_gray_image_matches_reference_mean_and_std_without_mutating_input(self):
        rng = np.random.default_rng(20261010)
        source = np.clip(rng.normal(72.0, 5.0, (540, 960)), 0, 255).astype(np.uint8)
        original = source.copy()

        photo = measure(source, mode="legacy", reference=GRAY_REFERENCE)
        matched = photo.match_image(source)
        result = measure(matched, mode="legacy", reference=GRAY_REFERENCE)

        self.assertIsNot(matched, source)
        np.testing.assert_array_equal(source, original)
        self.assertAlmostEqual(result.mean, GRAY_REFERENCE[0], delta=0.15)
        self.assertAlmostEqual(result.std, GRAY_REFERENCE[1], delta=0.15)

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
        self.assertAlmostEqual(result.std, MAX_CHANNEL_REFERENCE[1], delta=0.15)

    def test_flat_frame_maps_to_reference_mean_without_inventing_contrast(self):
        photo = Photometry(31.0, 0.0, 150.0, 20.0, "legacy")
        image = np.full((20, 30), 31, dtype=np.uint8)
        matched = photo.match_image(image)

        self.assertTrue(np.all(matched == 150))
        self.assertEqual(float(matched.std()), 0.0)
        self.assertEqual(photo.diagnostics()["photometric_match_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
