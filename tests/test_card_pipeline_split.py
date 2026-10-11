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

    def test_moving_presence_keeps_box_but_never_classifies_or_fires(self):
        detector = ShapeDetector()
        with (patch.object(detector, "_detect_segments", return_value=([], [])),
              patch.object(detector, "_hough_quads", return_value=[]),
              patch.object(detector, "_lsd_quads", return_value=[]),
              patch.object(detector, "_corner_quads", return_value=[]),
              patch.object(detector, "_cc_quads", return_value=[self.quad]),
              patch.object(detector, "_ring_quads", return_value=[]),
              patch.object(detector, "_geom_ok", return_value=True),
              patch.object(detector, "_verify_quad", return_value=(1.0, 1.0)),
              patch.object(detector, "_refine_quad", side_effect=lambda _b, q: q),
              patch.object(detector, "_warp_card", side_effect=AssertionError("classified while walking")),
              patch.object(detector, "_confirm", side_effect=AssertionError("legacy action used"))):
            action, debug = detector.update(self.frame, classify=False, issue_action=False)
        self.assertIsNone(action)
        self.assertTrue(debug["card_found"])
        self.assertTrue(debug["presence"])
        self.assertIsNone(debug["shape"])
        self.assertEqual(debug["shape_classification_path"], "deferred_until_stop")
        self.assertEqual(debug["shape_proposal_source"], "cc")
        self.assertEqual(detector.candidate_count, 0)

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
            _, debug = detector.update(self.frame, classify=True, issue_action=False,
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
            _, debug = detector.update(self.frame, classify=True, issue_action=False)
        self.assertTrue(debug["card_found"])
        self.assertEqual(debug["shape_local_candidate"], "square")
        self.assertEqual(debug["shape_legacy_candidate"], "triangle")
        self.assertEqual(debug["shape"], "square")
        self.assertEqual(debug["shape_classification_path"], "local_gray")


if __name__ == "__main__":
    unittest.main()
