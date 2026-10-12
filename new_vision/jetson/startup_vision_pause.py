"""Timed first-card experiment. No camera, detector or motor ownership."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PauseCommand:
    phase: str
    vx: float
    wz: float
    qr: int = -1
    event_id: int = 0
    event_action: int = -1


class StartupVisionPause:
    PERIOD_S = .02

    def __init__(self, sequence, action, hold_s, resume_vx, event_factory):
        if action not in range(1, 7) or not math.isfinite(hold_s) or hold_s <= 0:
            raise ValueError('a latched first card and positive action hold are required')
        self.sequence = sequence
        self.action = action
        self.hold_s = hold_s
        self.resume_vx = resume_vx
        self.event_factory = event_factory
        self.next_due_s = None
        self.action_until_s = None
        self.event_id = 0
        self.phase = 'WALK'
        self.message_count = 0
        self.first_publish_s = self.last_publish_s = None
        self.max_publish_gap_s = 0.

    def wait_s(self, now):
        return max(0., (self.next_due_s or now) - now)

    def command(self, now):
        """Advance by monotonic time rather than by visual observations."""
        if not math.isfinite(now) or (self.last_publish_s is not None and now < self.last_publish_s):
            raise ValueError('startup experiment requires a monotonic clock')
        if self.phase == 'RESUME':
            raise RuntimeError('startup experiment has already resumed vision')
        self.next_due_s = max((self.next_due_s or now) + self.PERIOD_S,
                              now + self.PERIOD_S)
        if self.action_until_s is None:
            pair = self.sequence.command(now)
            if pair is not None:
                return PauseCommand('WALK', *pair)
            self.event_id = self.event_factory()
            if not 0 < self.event_id <= 0xFFFFFFFF:
                raise ValueError('invalid first-card event id')
            self.action_until_s = now + self.hold_s
            self.phase = 'ACTION'
            return PauseCommand('ACTION', 0., 0., self.action, self.event_id, self.action)
        if now < self.action_until_s:
            return PauseCommand('ACTION', 0., 0., -1, self.event_id, self.action)
        self.phase = 'RESUME'
        return PauseCommand('RESUME', self.resume_vx, 0.)

    def published(self, now):
        if self.last_publish_s is not None:
            self.max_publish_gap_s = max(self.max_publish_gap_s, now-self.last_publish_s)
        else:
            self.first_publish_s = now
        self.last_publish_s = now
        self.message_count += 1

    def diagnostics(self):
        elapsed = ((self.last_publish_s-self.first_publish_s)
                   if self.message_count > 1 else 0.)
        return dict(startup_vision_paused=self.phase != 'RESUME',
                    observation_processed=False, startup_pause_phase=self.phase,
                    startup_pause_command_tick=self.message_count,
                    startup_pause_elapsed_s=elapsed,
                    startup_pause_publish_hz=((self.message_count-1)/elapsed if elapsed else None),
                    startup_pause_max_publish_gap_ms=self.max_publish_gap_s*1000.)
