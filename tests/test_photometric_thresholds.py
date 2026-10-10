import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from photometric_thresholds import Photometry, measure, GRAY_REFERENCE, STD_FLOOR
from shape_detector import ShapeDetector


class PhotometricTests(unittest.TestCase):
    def test_threshold_mapping_is_reference_z_score_not_mean_std_weighting(self):
        old_mean, old_std = GRAY_REFERENCE
        p = Photometry(64.59, 15.04, old_mean, old_std, 'normalize')
        for threshold in (50, 100, 180):
            self.assertAlmostEqual((p.intensity(threshold)-p.mean)/p.std,
                                   (threshold-old_mean)/old_std)
        self.assertAlmostEqual(p.difference(12)/p.std, 12/old_std)
        self.assertAlmostEqual(p.difference(-16)/p.std, -16/old_std)

    def test_roi_does_not_include_hud_or_modify_image(self):
        a = np.full((540, 960), 65, np.uint8)
        b = a.copy()
        b[430:] = 255
        self.assertEqual(measure(a), measure(b))
        self.assertEqual(measure(a).mean, 65)

    def test_flat_image_floor_and_legacy(self):
        blank = np.zeros((540, 960), np.uint8)
        p = measure(blank)
        self.assertAlmostEqual(p.scale, STD_FLOOR/GRAY_REFERENCE[1])
        legacy = measure(blank, 'legacy')
        self.assertEqual(legacy.intensity(100), 100)
        self.assertEqual(legacy.difference(-16), -16)
        for mode in ('normalize', 'legacy'):
            detector = ShapeDetector()
            detector.photometric_mode = mode
            for _ in range(3):
                _, dbg = detector.update(blank)
                self.assertFalse(dbg['presence'])
                self.assertIsNone(dbg.get('shape'))

    def test_dark_ring_keeps_cue_in_reference_contrast_units(self):
        # Known structural ring; tests a brightness gate, not six-card geometry.
        gray = np.full((540, 960), 180, np.uint8)
        cv2.rectangle(gray, (400, 260), (500, 360), 40, 5)
        cues = []
        for gain in (1., .6, .4):
            image = (gray.astype(np.float32)*gain).astype(np.uint8)
            detector = ShapeDetector()
            detector.photometric_mode = 'normalize'
            box, score = detector._presence_cue(image)
            self.assertIsNotNone(box)
            cues.append(score)
        np.testing.assert_allclose(cues, cues[0], rtol=.015)
        legacy = ShapeDetector()
        legacy.photometric_mode = 'legacy'
        self.assertIsNone(legacy._presence_cue((gray*.4).astype(np.uint8))[0])

    def test_gain_and_offset_map_thresholds_consistently(self):
        rng = np.random.default_rng(12)
        gray = np.clip(rng.normal(150, 20, (540, 960)), 80, 210).astype(np.uint8)
        p = measure(gray)
        dim = (gray.astype(np.float32)*.5+5).astype(np.uint8)
        q = measure(dim)
        self.assertAlmostEqual(q.intensity(100), .5*p.intensity(100)+5, delta=.6)
        self.assertAlmostEqual(q.difference(16), .5*p.difference(16), delta=.1)

    def test_closeup_card_does_not_raise_selective_rejection_threshold(self):
        frame = np.full((540, 960), 180, np.uint8)
        cv2.rectangle(frame, (250, 200), (600, 400), 20, 20)
        detector = ShapeDetector()
        detector.photometric_mode = 'normalize'
        _, dbg = detector.update(frame)
        self.assertGreater(dbg['photometric_contrast_scale'], 1)
        self.assertEqual(dbg['shape_adaptive_c'], detector.cfg['adaptive_c'])


if __name__ == '__main__':
    unittest.main()
