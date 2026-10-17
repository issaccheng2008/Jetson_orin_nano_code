"""Hardware-free checks for moving presence and stopped-card proposal fallback."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
from shape_detector import ShapeDetector


class CardPipelineSplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((540, 960, 3), 160, np.uint8)
        self.quad = np.array([[405, 250], [525, 250], [525, 350], [405, 350]],
                             np.float32)

    def test_moving_presence_never_builds_boxes_or_classifies(self):
        detector = ShapeDetector()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for method in ('_binary_selective', '_detect_segments', '_hough_quads',
                           '_lsd_quads', '_corner_quads', '_cc_quads', '_ring_quads',
                           '_geom_ok', '_verify_quad', '_refine_quad', '_warp_card'):
                stack.enter_context(patch.object(detector, method,
                    side_effect=AssertionError('unused moving-box operation: '+method)))
            stack.enter_context(patch.object(detector, '_presence_cue',
                return_value=((405, 300, 120, 100), 50.)))
            detector.last_quad = self.quad
            for _ in range(8):
                label, debug = detector.update(self.frame, classify=False)
        self.assertIsNone(label)
        self.assertFalse(debug["card_found"])
        self.assertTrue(debug["presence"])
        self.assertIsNone(debug["shape"])
        self.assertEqual(debug["shape_classification_path"], "deferred_until_stop")
        self.assertEqual(debug['shape_presence_mode'], 'cue_only')
        self.assertEqual(debug['quad_total'], 0)
        self.assertEqual(debug['shape_proposals_ms'], 0.)
        self.assertEqual(debug['shape_verification_ms'], 0.)
        self.assertNotIn('quad_work', debug)
        self.assertIsNone(detector.last_quad)

    def test_moving_cue_miss_never_reuses_a_previous_stop_position(self):
        detector = ShapeDetector()
        with patch.object(detector, '_presence_cue',
                          return_value=((405, 300, 120, 100), 50.)):
            for _ in range(8):
                _, debug = detector.update(self.frame, classify=False)
        self.assertIsNotNone(debug['presence_cy_frac'])
        with (patch.object(detector, '_presence_cue', return_value=(None, 0.)),
              patch('card_presence.find_broken_border_presence', return_value=(None, 0., {}))):
            _, missed = detector.update(self.frame, classify=False)
        self.assertFalse(missed['cue_hit'])
        self.assertIsNone(missed['presence_cy_frac'])
        self.assertIsNone(missed['presence_box'])

    def test_staged_box_falls_back_to_corner_before_presence_cue(self):
        detector = ShapeDetector()
        with (patch.object(detector, "_cc_quads", return_value=[]),
              patch.object(detector, "_ring_quads", return_value=[]),
              patch.object(detector, "_detect_segments", return_value=([], [])),
              patch.object(detector, "_hough_quads", return_value=[]),
              patch.object(detector, "_lsd_quads", return_value=[]),
              patch.object(detector, "_corner_quads", return_value=[self.quad]),
              patch.object(detector, "_geom_ok", return_value=True),
              patch.object(detector, "_verify_quad", return_value=(1.0, 1.0)),
              patch.object(detector, "_refine_quad", side_effect=lambda _b, q: q)):
            _, debug = detector.update(self.frame, classify=True,
                                       proposal_mode="staged")
        self.assertTrue(debug["card_found"])
        self.assertTrue(debug["shape_proposal_fallback"])
        self.assertEqual(debug["shape_proposal_source"], "corner")
        self.assertEqual(debug["quad_total"], 1)

    def test_stopped_box_reports_both_readers_without_overriding_local_result(self):
        detector = ShapeDetector()
        appearance = Mock()
        appearance.diagnostics.return_value = {}
        appearance.gray = np.zeros((64, 64), np.uint8)
        appearance.binary = appearance.gray
        appearance.symbol = np.ones((64, 64), np.uint8)
        with (patch.object(detector, "_cc_quads", return_value=[self.quad]),
              patch.object(detector, "_ring_quads", return_value=[]),
              patch.object(detector, "_geom_ok", return_value=True),
              patch.object(detector, "_verify_quad", return_value=(1.0, 1.0)),
              patch.object(detector, "_refine_quad", side_effect=lambda _b, q: q),
              patch.object(detector, "_classify_contour", return_value="square"),
              patch.object(detector, "_classify", return_value="triangle"),
              patch("card_appearance.card_appearance", return_value=appearance)):
            _, debug = detector.update(self.frame, classify=True)
        self.assertTrue(debug["card_found"])
        self.assertEqual(debug["shape_local_candidate"], "square")
        self.assertEqual(debug["shape_legacy_candidate"], "triangle")
        self.assertEqual(debug["shape"], "square")
        self.assertEqual(debug["shape_classification_path"], "local_gray")


if __name__ == "__main__":
    unittest.main()
