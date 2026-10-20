"""Behavioral contract for the replacement ordinary-card lifecycle."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from card_flow import CardFlow


POSE = dict(pose_supported=True, tilt_id=0, tilt_done=False, restore_id=0)


class CardFlowTests(unittest.TestCase):
    def voting(self, flow=None):
        flow = flow or CardFlow(30, settle_s=.1)
        self.assertTrue(flow.trigger(0., True, True, POSE))
        flow.advance(.2, dict(POSE, tilt_id=1, tilt_done=False))
        flow.advance(.4, dict(POSE, tilt_id=1, tilt_done=True))
        flow.advance(.5, dict(POSE, tilt_id=1, tilt_done=True))
        self.assertEqual(flow.phase, 'voting')
        return flow

    def cast(self, flow, names):
        for i, name in enumerate(names):
            now = flow.ready_at + .01*(i+1)
            flow.observe(now, now, name, POSE)

    def test_presence_and_distance_are_both_required(self):
        f = CardFlow(30)
        self.assertFalse(f.trigger(0, False, True, POSE))
        self.assertFalse(f.trigger(0, True, False, POSE))
        self.assertTrue(f.trigger(0, True, True, POSE))
        self.assertFalse(f.trigger(.1, True, True, POSE))

    def test_six_votes_for_one_class_not_six_total_or_two_agreements(self):
        f = self.voting()
        self.cast(f, ['circle', 'square']*3)
        self.assertEqual(f.phase, 'voting')
        for t in (.7, .8, .9):
            f.observe(t, t, 'circle', POSE)
        self.assertEqual(f.winner, 'circle')
        self.assertEqual(f.decision['counts'], {'circle': 6, 'square': 3})
        self.assertEqual(f.decision['reason'], 'six_votes')

    def test_one_vote_is_failure_even_with_unique_leader(self):
        f = self.voting()
        self.cast(f, ['circle', None])
        f.advance(1.5, POSE)
        self.assertEqual(f.phase, 'restore')
        self.assertIsNone(f.winner)
        self.assertEqual(f.retry, 0)

    def test_two_votes_win_without_majority_at_timeout(self):
        f = self.voting()
        self.cast(f, ['circle', 'circle', 'square', 'diamond', 'cross'])
        f.advance(1.5, POSE)
        self.assertEqual(f.phase, 'dispatch')
        self.assertEqual(f.winner, 'circle')

    def test_no_votes_and_all_single_votes_fail_without_retry(self):
        for names in ([], [None]*5, ['circle', 'square', 'cross']):
            with self.subTest(names=names):
                f = self.voting()
                self.cast(f, names)
                f.advance(1.5, POSE)
                self.assertEqual(f.phase, 'restore')
                self.assertEqual(f.retry, 0)

    def test_tie_restores_before_one_new_repose_and_discards_old_votes(self):
        f = self.voting()
        self.cast(f, ['circle', 'square']*2)
        f.advance(1.5, POSE)
        self.assertEqual(f.phase, 'restore_retry')
        self.assertFalse(f.tilt_requested)
        f.advance(2., dict(POSE, restore_id=2, restore_done=True))
        self.assertEqual(f.phase, 'restore_retry')
        f.advance(2.2, dict(POSE, restore_id=2, restore_done=True, tilt_id=0))
        self.assertEqual(f.phase, 'repose')
        self.assertTrue(f.tilt_requested)
        self.assertEqual(f.votes, {})
        f.advance(2.4, dict(POSE, tilt_id=3, tilt_done=True))
        f.advance(2.5, dict(POSE, tilt_id=3, tilt_done=True))
        self.cast(f, ['cross']*2)
        f.advance(3.5, POSE)
        self.assertEqual(f.winner, 'cross')
        self.assertEqual(f.retry, 1)

    def test_second_tie_fails_without_third_attempt(self):
        f = self.voting()
        self.cast(f, ['circle', 'square']*2)
        f.advance(1.5, POSE)
        f.advance(2.2, dict(POSE, restore_id=2, restore_done=True))
        f.advance(2.4, dict(POSE, tilt_id=3, tilt_done=True))
        f.advance(2.5, dict(POSE, tilt_id=3, tilt_done=True))
        self.cast(f, ['cross', 'diamond']*2)
        f.advance(3.5, dict(POSE, restore_id=2))
        self.assertEqual(f.phase, 'restore')
        f.advance(4.2, dict(POSE, restore_id=4, restore_done=True))
        self.assertEqual(f.phase, 'walking')
        self.assertAlmostEqual(f.cooldown_until, 4.7)

    def test_success_cooldown_starts_only_after_matching_action_and_restore(self):
        f = self.voting()
        self.cast(f, ['circle']*6)
        f.dispatch(.6, 42, 1, 3., POSE)
        self.assertFalse(f.tilt_requested)
        f.advance(5, dict(POSE, action_id=41, action_done=True,
                          restore_id=2, restore_done=True))
        self.assertEqual(f.phase, 'action')
        self.assertEqual(f.cooldown_until, 0)
        f.advance(6, dict(POSE, action_id=42, action_done=True,
                          restore_id=2, restore_done=False))
        self.assertEqual(f.phase, 'restore')
        f.advance(7, dict(POSE, restore_id=2, restore_done=True))
        self.assertEqual(f.phase, 'walking')
        self.assertEqual(f.cooldown_until, 37)
        self.assertFalse(f.trigger(36.99, True, True, POSE))
        # Same card can stop again after cooldown, with no absence/far-range latch.
        self.assertTrue(f.trigger(37, True, True, POSE))

    def test_action_minimum_hold_remains_in_force(self):
        f = self.voting()
        self.cast(f, ['square']*6)
        f.dispatch(.6, 42, 3, 6., POSE)
        feedback = dict(POSE, action_id=42, action_done=True,
                        restore_id=2, restore_done=True)
        f.advance(6.5, feedback)
        self.assertEqual(f.phase, 'action')
        f.advance(6.6, feedback)
        self.assertEqual(f.phase, 'walking')

    def test_failure_cooldown_is_half_second_from_resumption(self):
        f = self.voting()
        f.advance(1.5, POSE)
        f.advance(2.2, dict(POSE, restore_id=2, restore_done=True))
        self.assertAlmostEqual(f.cooldown_until, 2.7)
        self.assertFalse(f.trigger(2.69, True, True, POSE))
        self.assertTrue(f.trigger(2.7, True, True, POSE))

    def test_failed_repose_gets_bounded_failure_recovery(self):
        f = CardFlow(30)
        f.trigger(0, True, True, POSE)
        f.advance(3, POSE)
        self.assertEqual(f.decision['reason'], 'repose_timeout')
        self.assertFalse(f.tilt_requested)
        f.advance(6, POSE)
        self.assertEqual(f.phase, 'walking')
        self.assertEqual(f.cooldown_until, 6.5)

    def test_delayed_start_extends_completion_budget_only_once(self):
        f = CardFlow(30)
        f.trigger(0, True, True, POSE)
        f.advance(2, dict(POSE, tilt_id=1, tilt_done=False))
        f.advance(4, dict(POSE, tilt_id=1, tilt_done=False))
        self.assertEqual(f.phase, 'repose')
        f.advance(5, dict(POSE, tilt_id=1, tilt_done=False))
        self.assertEqual(f.phase, 'restore')

    def test_stale_done_and_pre_settle_frames_never_vote(self):
        f = CardFlow(30)
        f.trigger(0, True, True, dict(POSE, tilt_id=5))
        f.advance(.1, dict(POSE, tilt_id=5, tilt_done=True))
        self.assertEqual(f.phase, 'repose')
        f.advance(.2, dict(POSE, tilt_id=6, tilt_done=True))
        f.advance(.31, dict(POSE, tilt_id=6, tilt_done=True))
        f.observe(.4, .29, 'circle', POSE)
        self.assertEqual(f.attempts, 0)
        f.observe(.4, .4, None, POSE)
        self.assertEqual(f.abstentions, 1)

    def test_deadline_excludes_late_capture_or_slow_classification(self):
        f = self.voting()
        f.observe(1.5, 1.4, 'circle', POSE)
        f.observe(1.6, 1.6, 'square', POSE)
        self.assertEqual(f.attempts, 0)

    def test_legacy_repose_wait_is_retained(self):
        f = CardFlow(30, legacy_tilt_s=1.8, settle_s=.1)
        f.trigger(0, True, True, {})
        f.advance(1.8, {})
        self.assertEqual(f.phase, 'repose')
        f.advance(1.91, {})
        self.assertEqual(f.phase, 'voting')

    def test_invalid_times_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CardFlow(30, failure_cooldown_s=value)
