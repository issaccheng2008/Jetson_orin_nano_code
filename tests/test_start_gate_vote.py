"""The competition start card is decided from recent votes after QR release."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision" / "jetson"))
from start_gate import StartGate


class StartGateVoteTests(unittest.TestCase):
    def test_first_matching_pair_does_not_lock_before_qr(self):
        gate = StartGate(mode="both", shape_voting=True,
                         shape_vote_min=8, shape_vote_window=24)
        for name in ["diamond", "diamond"] + ["triangle"] * 10:
            self.assertFalse(gate.observe_shape(name))
        self.assertFalse(gate.shape_passed)
        self.assertEqual(gate.vote_tally(), {"diamond": 2, "triangle": 10})
        gate.observe_qr("1")
        self.assertTrue(gate.observe_shape("triangle"))
        self.assertEqual(gate.last_shape, "triangle")
        self.assertTrue(gate.passed)

    def test_recent_window_replaces_early_bad_votes(self):
        gate = StartGate(mode="both", shape_voting=True,
                         shape_vote_min=8, shape_vote_window=12)
        for _ in range(12):
            gate.observe_shape("diamond")
        for _ in range(12):
            gate.observe_shape("triangle")
        self.assertEqual(gate.vote_tally(), {"triangle": 12})
        gate.observe_qr("1")
        self.assertTrue(gate.observe_shape("triangle"))
        self.assertEqual(gate.last_shape, "triangle")

    def test_qr_early_waits_for_unambiguous_evidence(self):
        gate = StartGate(mode="both", shape_voting=True,
                         shape_vote_min=8, shape_vote_window=24)
        gate.observe_qr("1")
        for name in ["square"] * 4 + ["circle"] * 4:
            self.assertFalse(gate.observe_shape(name))
        self.assertFalse(gate.passed)
        for _ in range(5):
            gate.observe_shape("square")
        self.assertTrue(gate.shape_passed)
        self.assertEqual(gate.last_shape, "square")

    def test_legacy_shape_only_mode_still_uses_consecutive_calls(self):
        gate = StartGate(mode="shape", shape_confirm=2)
        self.assertFalse(gate.observe_shape("cross"))
        self.assertTrue(gate.observe_shape("cross"))
        self.assertTrue(gate.passed)


if __name__ == "__main__":
    unittest.main()
