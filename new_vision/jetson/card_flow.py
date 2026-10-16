"""Ordinary walking cards: proximity stop, re-pose, vote, action, restore.

The startup card is owned by StartGate and never enters this state machine.
Presence and camera geometry remain in ShapeDetector/the caller.
"""
from collections import Counter
import json
import math


class CardFlow:
    VOTE_SECONDS = 1.0
    VOTE_TARGET = 6
    RESTORE_SECONDS = 0.65  # Existing STM32 restore ramp / CARD_UNTILT_HOLD_S.

    def __init__(self, success_cooldown_s, failure_cooldown_s=0.5,
                 repose_timeout_s=3.0, settle_s=0.1, legacy_tilt_s=1.8):
        values = (success_cooldown_s, failure_cooldown_s, repose_timeout_s,
                  settle_s, legacy_tilt_s)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('card times must be finite and nonnegative')
        self.success_cooldown_s = success_cooldown_s
        self.failure_cooldown_s = failure_cooldown_s
        self.repose_timeout_s = repose_timeout_s
        self.settle_s = settle_s
        self.legacy_tilt_s = legacy_tilt_s
        self.phase = 'walking'
        self.cooldown_until = 0.0
        self.retry = 0
        self.votes = Counter()
        self.attempts = 0
        self.abstentions = 0
        self.decision = None
        self.winner = None
        self.event_id = 0
        self.action = -1
        self.ready_at = None
        self.tilt_baseline = 0
        self.tilt_pending = 0
        self.restore_baseline = 0
        self.feedback_pose = False
        self.success = False

    @property
    def active(self):
        return self.phase != 'walking'

    @property
    def tilt_requested(self):
        return self.phase in ('repose', 'voting')

    def _repose(self, now, feedback):
        self.phase = 'repose'
        self.feedback_pose = feedback.get('pose_supported', False)
        self.tilt_baseline = feedback.get('tilt_id', 0)
        self.tilt_pending = 0
        self.ready_at = None
        self.deadline = now + self.repose_timeout_s
        self.legacy_ready = now + self.legacy_tilt_s + self.settle_s
        self.votes.clear()
        self.attempts = self.abstentions = 0

    def trigger(self, now, present, close, feedback):
        if self.active or now < self.cooldown_until or not present or not close:
            return False
        self.retry = 0
        self.winner = None
        self.event_id = 0
        self.action = -1
        self.success = False
        self.decision = None
        self._repose(now, feedback)
        return True

    def _restore(self, now, feedback, retry=False):
        self.phase = 'restore_retry' if retry else 'restore'
        self.restore_baseline = feedback.get('restore_id', 0)
        self.restore_not_before = now + self.RESTORE_SECONDS
        self.deadline = now + max(self.repose_timeout_s, self.RESTORE_SECONDS)

    def _resume(self, now):
        self.phase = 'walking'
        self.cooldown_until = now + (self.success_cooldown_s if self.success
                                     else self.failure_cooldown_s)
        self.event_id = 0
        self.action = -1

    def advance(self, now, feedback):
        """Advance deadlines/feedback before deciding whether to classify a frame."""
        if self.phase == 'repose':
            event = feedback.get('tilt_id', 0)
            if self.feedback_pose:
                if event > self.tilt_baseline:
                    # A completed fresh event can arrive without its START packet.
                    if self.tilt_pending == 0:
                        self.tilt_pending = event
                        self.deadline = max(self.deadline, now + self.repose_timeout_s)
                    if event == self.tilt_pending and feedback.get('tilt_done', False):
                        if self.ready_at is None:
                            self.ready_at = now + self.settle_s
                if self.ready_at is not None and now >= self.ready_at:
                    self.phase = 'voting'
                    self.deadline = self.ready_at + self.VOTE_SECONDS
                elif now >= self.deadline:
                    self.decision = dict(reason='repose_timeout', winner=None,
                                         counts={}, attempts=0, abstentions=0)
                    self._restore(now, feedback)
            elif now >= self.legacy_ready:
                self.ready_at = self.legacy_ready
                self.phase = 'voting'
                self.deadline = self.ready_at + self.VOTE_SECONDS
        if self.phase == 'voting' and now >= self.deadline:
            self._decide(now, feedback, 'timeout')
        elif self.phase == 'action':
            done = (feedback.get('action_id') == self.event_id
                    and feedback.get('action_done', False))
            if done and now >= self.action_not_before:
                self.success = True
                # Restoration was requested when the action event was emitted.
                self.phase = 'restore'
            # Policy action failures retain the existing fault/disable path.
        if self.phase in ('restore', 'restore_retry'):
            restored = (now >= self.restore_not_before and
                        (not self.feedback_pose or
                         (feedback.get('restore_id', 0) > self.restore_baseline
                          and feedback.get('restore_done', False))))
            if restored:
                if self.phase == 'restore_retry':
                    self._repose(now, feedback)
                else:
                    self._resume(now)
            elif now >= self.deadline and self.phase != 'restore_retry':
                # Same bounded failure recovery as a missing re-pose DONE.
                self.success = False
                self._resume(now)
            elif now >= self.deadline:
                self.decision = dict(reason='restore_timeout', winner=None,
                                     counts=dict(self.votes), attempts=self.attempts,
                                     abstentions=self.abstentions)
                self.success = False
                self._resume(now)

    def can_vote(self, captured_at):
        return (self.phase == 'voting' and self.ready_at <= captured_at < self.deadline)

    def observe(self, now, captured_at, name, feedback):
        # A detection finishing outside the one-second window gets no vote.
        if not self.can_vote(captured_at) or now >= self.deadline:
            return
        self.attempts += 1
        if name:
            self.votes[name] += 1
        else:
            self.abstentions += 1
        if name and self.votes[name] >= self.VOTE_TARGET:
            self._decide(now, feedback, 'six_votes')

    def _decide(self, now, feedback, reason):
        highest = max(self.votes.values(), default=0)
        leaders = [name for name, count in self.votes.items() if count == highest]
        self.decision = dict(reason=reason, winner=None, counts=dict(self.votes),
                             attempts=self.attempts, abstentions=self.abstentions,
                             retry=self.retry)
        if highest < 2:
            self.decision['reason'] = 'insufficient_votes'
            self._restore(now, feedback)
        elif len(leaders) > 1:
            self.decision['reason'] = 'tie'
            if self.retry == 0:
                self.retry = 1
                self._restore(now, feedback, retry=True)
            else:
                self._restore(now, feedback)
        else:
            self.winner = leaders[0]
            self.decision['winner'] = self.winner
            self.phase = 'dispatch'

    def dispatch(self, now, event_id, action, hold_s, feedback):
        if self.phase != 'dispatch':
            raise RuntimeError('no successful card vote to dispatch')
        self.event_id, self.action = event_id, action
        self._restore(now, feedback)
        self.phase = 'action'
        self.action_not_before = now + hold_s
        # Action completion can take longer than pose restoration.
        self.deadline = math.inf

    def diagnostics(self, now):
        return dict(card_phase=self.phase, card_retry=self.retry,
                    card_vote_total=sum(self.votes.values()),
                    card_vote_attempts=self.attempts,
                    card_vote_abstentions=self.abstentions,
                    card_vote_target=self.VOTE_TARGET,
                    card_vote_counts=json.dumps(dict(self.votes), sort_keys=True),
                    card_cooldown_remaining_s=max(0., self.cooldown_until-now))
