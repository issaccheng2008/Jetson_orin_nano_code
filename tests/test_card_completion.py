"""Only dispatched, acknowledged card actions consume the three-card budget."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
import card_flow


class CardCompletionTests(unittest.TestCase):
    def test_only_matching_done_counts_once_and_a_new_run_resets(self):
        progress = card_flow.CardCompletion()
        progress.register(42)
        for feedback in ({}, dict(action_id=0, action_done=True),
                         dict(action_id=41, action_done=True),
                         dict(action_id=42, action_done=False)):
            self.assertFalse(progress.observe(feedback))
        self.assertEqual(progress.completed_count, 0)
        self.assertTrue(progress.observe(dict(action_id=42, action_done=True)))
        progress.register(42)
        self.assertFalse(progress.observe(dict(action_id=42, action_done=True)))
        self.assertEqual(progress.completed_count, 1)
        self.assertTrue(progress.enabled)
        self.assertEqual(card_flow.CardCompletion().completed_count, 0)

    def test_delayed_startup_done_and_two_ordinary_actions_disable_checks(self):
        progress = card_flow.CardCompletion()
        progress.register(10)  # startup, still waiting when ordinary card starts
        progress.register(20)
        progress.observe(dict(action_id=20, action_done=True))
        progress.observe(dict(action_id=10, action_done=True))
        self.assertTrue(progress.enabled)
        progress.register(30)
        self.assertTrue(progress.enabled)  # dispatch/classification alone is insufficient
        progress.observe(dict(action_id=30, action_done=True))
        self.assertEqual(progress.completed_count, 3)
        self.assertFalse(progress.enabled)
