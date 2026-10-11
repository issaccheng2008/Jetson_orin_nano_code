import json
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from photometric_thresholds import (Photometry, clear_lut_mapping, load_lut_mapping,
                                    measure, resolve_mode, GRAY_REFERENCE, STD_FLOOR)
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
        self.assertEqual(p.scale, 1.0)
        self.assertEqual(p.intensity(100), 100.0)
        self.assertEqual(p.diagnostics()["photometric_threshold_reject_reason"], "low_contrast")
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


class LutModeTests(unittest.TestCase):
    """lut = the calibrated fixed->auto mapping; it replaces, it never stacks."""

    def setUp(self):
        clear_lut_mapping()
        self.addCleanup(clear_lut_mapping)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        levels = np.arange(256, dtype=np.uint8)
        self.lut_gray = (255 - levels).astype(np.uint8)
        self.lut_bgr = np.stack([
            levels,
            np.clip(levels.astype(np.int32) + 30, 0, 255).astype(np.uint8),
            np.clip(levels.astype(np.int32) - 30, 0, 255).astype(np.uint8),
        ])
        np.save(Path(self.tmp.name) / 'lut_bgr.npy', self.lut_bgr, allow_pickle=False)
        np.save(Path(self.tmp.name) / 'lut_gray.npy', self.lut_gray, allow_pickle=False)
        (Path(self.tmp.name) / 'mapping.json').write_text(
            json.dumps({'schema_version': 1}), encoding='utf-8')

    def _photo(self, mode='lut'):
        return Photometry(64.0, 20.0, *GRAY_REFERENCE, mode)

    def test_lut_mode_applies_only_the_calibrated_mapping(self):
        load_lut_mapping(self.tmp.name)
        gray = np.tile(np.arange(256, dtype=np.uint8), (90, 240))[:540, :960]
        photo = self._photo()
        np.testing.assert_array_equal(photo.apply(gray), self.lut_gray[gray])
        self.assertFalse(photo.match_allowed)
        self.assertEqual(photo.scale, 1.0)
        self.assertEqual(photo.difference(16), 16.0)
        self.assertEqual(photo.intensity(100), 100.0)
        debug = photo.diagnostics()
        self.assertTrue(debug['photometric_lut_applied'])
        self.assertFalse(debug['photometric_match_applied'])
        self.assertEqual(debug['photometric_lut_source'], self.tmp.name)

    def test_lut_mode_maps_bgr_per_channel_and_preserves_contract(self):
        load_lut_mapping(self.tmp.name)
        image = np.zeros((4, 5, 3), np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 200
        image[:, :, 2] = 40
        mapped = self._photo().apply(image)
        self.assertEqual(mapped.shape, image.shape)
        self.assertEqual(mapped.dtype, np.uint8)
        np.testing.assert_array_equal(mapped[:, :, 0], self.lut_bgr[0, 10])
        np.testing.assert_array_equal(mapped[:, :, 1], self.lut_bgr[1, 200])
        np.testing.assert_array_equal(mapped[:, :, 2], self.lut_bgr[2, 40])

    def test_lut_mode_without_a_loaded_mapping_fails_loud(self):
        gray = np.full((540, 960), 128, np.uint8)
        with self.assertRaises(RuntimeError):
            self._photo().apply(gray)

    def test_lut_loader_rejects_wrong_shapes_and_missing_files(self):
        np.save(Path(self.tmp.name) / 'lut_gray.npy', np.zeros((255,), np.uint8),
                allow_pickle=False)
        with self.assertRaises(ValueError):
            load_lut_mapping(self.tmp.name)
        clear_lut_mapping()
        with self.assertRaises(OSError):
            load_lut_mapping(Path(self.tmp.name) / 'missing')

    def test_resolve_mode_accepts_lut_and_rejects_unknown(self):
        self.assertEqual(resolve_mode('lut'), 'lut')
        with self.assertRaises(ValueError):
            resolve_mode('bogus')

    def test_cli_defaults_to_lut_and_keeps_legacy_selectable(self):
        from unittest.mock import patch
        import run_policy_vision
        with patch('sys.argv', ['run_policy_vision.py']):
            args = run_policy_vision.parse_args()
        self.assertEqual(resolve_mode(args.photometric_mode), 'lut')
        lut_dir = Path(args.photometric_lut_dir)
        self.assertEqual(lut_dir.name, 'photometric_mapping')
        self.assertEqual(lut_dir.parent.name, 'config')
        with patch('sys.argv', ['run_policy_vision.py', '--photometric-mode', 'legacy']):
            self.assertEqual(resolve_mode(run_policy_vision.parse_args().photometric_mode),
                             'match')

    def test_deployed_mapping_loads_and_is_monotonic(self):
        deployed = (Path(__file__).resolve().parents[1] / 'new_vision' / 'config'
                    / 'photometric_mapping')
        if not deployed.is_dir():
            self.skipTest('deployed photometric mapping not present')
        mapping = load_lut_mapping(deployed)
        for channel in mapping['lut_bgr']:
            self.assertTrue(np.all(np.diff(channel.astype(np.int32)) >= 0))
        self.assertTrue(np.all(np.diff(mapping['lut_gray'].astype(np.int32)) >= 0))


if __name__ == '__main__':
    unittest.main()
