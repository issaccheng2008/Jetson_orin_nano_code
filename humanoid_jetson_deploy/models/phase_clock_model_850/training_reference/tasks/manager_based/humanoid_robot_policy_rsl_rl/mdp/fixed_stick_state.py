"""Vectorized moving crossing; obstacle geometry never enters policy commands."""
import math
from collections import deque

import torch
from deployment.phase_clock import PhaseClockConfig


class FixedStickState:
    WALK, LEAD, FOLLOW, DONE = range(4)

    def __init__(self, num_envs, device, dt, initial_gap=.08, walk_step=.10,
                 crossing_step=.23, bar_width=.03, step_tolerance=.01,
                 far_margin=.005, minimum_air_time=.04, collisionless_mode=False,
                 command_mode='touchdown', phase_clock=None):
        if min(dt, initial_gap, walk_step, crossing_step, bar_width, step_tolerance) <= 0:
            raise ValueError('Distances, tolerance and control period must be positive')
        if command_mode not in ('touchdown', 'phase_clock'):
            raise ValueError('Unknown command mode')
        self.command_mode = command_mode
        self.clock = phase_clock or PhaseClockConfig(control_dt=dt, walk_step=walk_step, crossing_step=crossing_step)
        if abs(self.clock.control_dt - dt) > 1e-8 or abs(self.clock.walk_step - walk_step) > 1e-8 or abs(self.clock.crossing_step - crossing_step) > 1e-8:
            raise ValueError('Clock and task commands/control period must match')
        self.collisionless_mode = bool(collisionless_mode)
        self.initial_gap, self.walk_step = initial_gap, walk_step
        self.crossing_step, self.bar_width = crossing_step, bar_width
        self.step_tolerance, self.far_margin = step_tolerance, far_margin
        self.minimum_air_steps = max(1, math.ceil(minimum_air_time / dt))
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.command_stage = torch.zeros_like(self.stage)
        self.control_steps = torch.zeros_like(self.stage)
        self.sequence_finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.forward_command = torch.full((num_envs,), self.clock.forward_velocity, device=device)
        self.expected_foot = torch.full_like(self.stage, -1)
        self.last_step = torch.full_like(self.stage, -1)
        self.air_steps = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.support_steps = torch.zeros_like(self.stage)
        self.bar_near = torch.zeros(num_envs, device=device)
        self.initial_front = torch.zeros_like(self.bar_near)
        self.step_distance = torch.full_like(self.bar_near, walk_step)
        self.crossing_command = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.walk_completed = torch.zeros_like(self.crossing_command)
        self.walk_completed_event = torch.zeros_like(self.crossing_command)
        self.lead_completed_event = torch.zeros_like(self.crossing_command)
        self.success = torch.zeros_like(self.crossing_command)
        self.success_event = torch.zeros_like(self.crossing_command)
        self.hit = torch.zeros_like(self.crossing_command)
        self.hit_event = torch.zeros_like(self.crossing_command)
        self.current_hit = torch.zeros_like(self.hit)
        self.physics_hit = torch.zeros_like(self.hit)
        self.entry_seen = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.entry_score = torch.zeros_like(self.bar_near)
        self.crossing_progress_score = torch.zeros_like(self.bar_near)
        self.failed = torch.zeros_like(self.crossing_command)
        self.touchdown = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.touchdown_error = torch.zeros(num_envs, 2, device=device)
        self.touchdown_eligible = torch.zeros_like(self.touchdown)

    def control_settings(self):
        return {'command_mode': self.command_mode, 'phase_clock': self.clock.to_dict()}

    @property
    def bar_far(self):
        return self.bar_near + self.bar_width

    @property
    def clean_success(self):
        return self.success & ~self.hit

    def reset(self, env_ids, sole_front, step=-1):
        for value in (self.stage, self.command_stage, self.control_steps, self.sequence_finished, self.air_steps, self.support_steps,
                      self.crossing_command, self.walk_completed, self.walk_completed_event,
                      self.lead_completed_event, self.success, self.success_event,
                      self.hit, self.hit_event, self.current_hit, self.physics_hit, self.entry_seen,
                      self.entry_score, self.crossing_progress_score, self.failed, self.touchdown,
                      self.touchdown_error, self.touchdown_eligible):
            value[env_ids] = 0
        self.expected_foot[env_ids] = -1
        self.last_step[env_ids] = step
        self.initial_front[env_ids] = sole_front[env_ids].amax(dim=1)
        self.bar_near[env_ids] = self.initial_front[env_ids] + self.initial_gap
        self.step_distance[env_ids] = self.walk_step
        self.forward_command[env_ids] = self.clock.forward_velocity

    def record_physics_hit(self, hit):
        self.physics_hit |= hit

    def update(self, step, front, rear, contact, hit, failed=None,
               foot_clearance=None, foot_velocity=None):
        update = self.last_step != step
        if not update.any():
            return self
        for event in (self.walk_completed_event, self.lead_completed_event,
                      self.success_event, self.hit_event, self.touchdown_eligible):
            event[update] = False
        hit = hit | self.physics_hit
        self.physics_hit[update] = False
        self.hit_event[update] = hit[update] & ~self.hit[update]
        self.current_hit[update] = hit[update]
        self.hit[update] |= hit[update]
        if not self.collisionless_mode:
            self.failed[update] |= self.hit[update]
        if failed is not None:
            self.failed[update] |= failed[update]
        touchdown = contact & (self.air_steps >= self.minimum_air_steps)
        # Natural walking can transfer support between sampled control frames.
        # A completed swing does not require an intervening double-support frame.
        touchdown &= update[:, None]
        touchdown &= touchdown.sum(dim=1, keepdim=True) == 1
        self.touchdown[update] = touchdown[update]
        self.air_steps[update] = torch.where(contact[update], 0, self.air_steps[update] + 1)
        self.control_steps[update] += 1
        if self.command_mode == "phase_clock":
            self.command_stage[update] = self.clock.phase_at_tick(self.control_steps)[update]
            self.sequence_finished[update] = (self.command_stage == self.DONE)[update]
        active = update & ~self.failed & ~self.success
        before = self.stage.clone()
        expected_before = self.expected_foot.clone()
        actual_step = front - front.flip(dims=(1,))
        scoring_step = self.step_distance
        if self.command_mode == "phase_clock":
            scoring_step = torch.where(before == self.WALK, self.walk_step,
                                       torch.where(before == self.LEAD, self.crossing_step, 0.))
        self.touchdown_error[update] = (actual_step - scoring_step[:, None])[update]
        foot_ids = torch.arange(2, device=front.device).expand_as(contact)
        expected = foot_ids == self.expected_foot[:, None]
        self.touchdown_eligible |= touchdown & active[:, None] & (
            (before == self.WALK)[:, None] | expected)
        # The first completed swing issues the crossing command immediately.
        # Stride accuracy remains a reward target; it must not block the command.
        walk_done = (before == self.WALK) & active & touchdown.any(dim=1)
        landed_foot = touchdown.long().argmax(dim=1)
        self.walk_completed_event |= walk_done
        self.walk_completed |= walk_done
        self.stage[walk_done] = self.LEAD
        self.expected_foot[walk_done] = 1 - landed_foot[walk_done]
        self.crossing_command[walk_done] = True
        self.step_distance[walk_done] = self.crossing_step
        past = rear > self.bar_far[:, None] + self.far_margin
        lead_done = ((before == self.LEAD) & active
                     & (touchdown & expected & past).any(dim=1))
        self.lead_completed_event |= lead_done
        self.stage[lead_done] = self.FOLLOW
        self.expected_foot[lead_done] = 1 - self.expected_foot[lead_done]
        self.step_distance[lead_done] = 0.0
        supported = active & (self.stage == self.FOLLOW) & past.all(dim=1) & contact.all(dim=1)
        self.support_steps[update & ~supported] = 0
        self.support_steps[supported] += 1
        completed = supported & (self.support_steps >= 2)
        if self.command_mode == "phase_clock":
            completed &= self.sequence_finished
        self.success_event |= completed
        self.success |= completed
        self.stage[completed] = self.DONE
        self.crossing_command[completed] = False
        # Original geometric guidance, adapted to the fixed bar and ordered feet.
        # Entry pays once per foot. Progress requires actual forward swing motion.
        self.entry_score[update] = 0.
        self.crossing_progress_score[update] = 0.
        if foot_clearance is not None and foot_velocity is not None:
            overlap = (front >= self.bar_near[:, None]) & (rear <= self.bar_far[:, None])
            guiding = (active[:, None] & ~contact & overlap
                       & ((before == self.LEAD) | (before == self.FOLLOW))[:, None]
                       & (foot_ids == expected_before[:, None]))
            entering = guiding & ~self.entry_seen
            self.entry_seen |= guiding
            height = (foot_clearance / .03).clamp(0., 1.)
            velocity = (foot_velocity / .5).clamp(-1., 1.)
            progress = ((front - self.bar_near[:, None]) / self.crossing_step).clamp(0., 1.)
            self.entry_score[update] = (height * entering).sum(dim=1)[update]
            self.crossing_progress_score[update] = (height * velocity * progress * guiding).sum(dim=1)[update]
        if self.command_mode == "phase_clock":
            # Policy commands depend only on local control age, never geometric stage.
            table = torch.tensor(self.clock.command_table, device=front.device, dtype=front.dtype)
            commands = table[self.command_stage]
            self.forward_command[update] = commands[update, 0]
            self.step_distance[update] = commands[update, 2]
            self.crossing_command[update] = commands[update, 3].bool()
            # Hardware hands off here. Missing real completion counts as failed.
            self.failed[update] |= (self.sequence_finished & ~self.success)[update]
        else:
            self.command_stage[update] = self.stage[update]
            self.sequence_finished[update] = self.success[update]
        self.last_step[update] = step
        return self


class RecentCleanCrossings:
    """Rolling episode outcomes; failed/time-out/dirty episodes count as zero."""
    def __init__(self, threshold=.6, min_episodes=200, window=1000):
        if not 0 < threshold <= 1 or not 1 <= min_episodes <= window:
            raise ValueError('Require 0 < threshold <= 1 and 1 <= min_episodes <= window')
        self.threshold, self.min_episodes = threshold, min_episodes
        self.outcomes = deque(maxlen=window)

    def record(self, clean_success):
        self.outcomes.extend(bool(value) for value in clean_success)

    @property
    def rate(self):
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.

    @property
    def ready(self):
        return len(self.outcomes) >= self.min_episodes and self.rate >= self.threshold

    def summary(self):
        return {'clean_success_rate': self.rate, 'episodes': len(self.outcomes),
                'threshold': self.threshold, 'min_episodes': self.min_episodes,
                'window': self.outcomes.maxlen, 'ready': self.ready,
                'outcomes': list(self.outcomes)}

    def restore(self, saved):
        self.outcomes.clear()
        self.record(saved.get('outcomes', [])[-self.outcomes.maxlen:])
