"""Small camera-clock sequencer for commands before the latched first card action."""
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Step:
    duration_s: float
    vx: float
    wz: float


def parse_sequence(raw, default_duration, default_vx, max_wz):
    if not math.isfinite(max_wz) or not 0 <= max_wz <= .5:
        raise ValueError('startup sequence max-wz must be in [0, 0.5]')
    try:
        rows = json.loads(raw) if raw.strip() else [dict(duration_s=default_duration, vx=default_vx, wz=0.)]
    except (ValueError, TypeError) as exc:
        raise ValueError('startup-sequence must be a JSON array of duration_s/vx/wz objects') from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError('startup-sequence must be a nonempty JSON array')
    steps = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'duration_s', 'vx', 'wz'}:
            raise ValueError(f'startup sequence step {i} requires exactly duration_s, vx, wz')
        if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v)
               for v in row.values()):
            raise ValueError(f'startup sequence step {i} values must be finite numbers')
        step = Step(float(row['duration_s']), float(row['vx']), float(row['wz']))
        if step.duration_s <= 0 or not 0 <= step.vx <= 1 or abs(step.wz) > max_wz:
            raise ValueError(f'startup sequence step {i}: duration_s >0, vx in [0,1], |wz| <= max-wz required')
        steps.append(step)
    return tuple(steps)


class StartupSequence:
    def __init__(self, steps):
        self.steps = steps
        self.reset()

    def reset(self):
        self.index = -1
        self.deadline = None
        self.last_time = None

    def command(self, now):
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('startup sequence requires a finite monotonic clock')
        self.last_time = now
        if self.index == -1 or (self.deadline is not None and now >= self.deadline):
            self.index += 1
            # Start the next step when it is first selected. A slow camera frame
            # must not skip an entire requested command or shorten its own duration.
            self.deadline = (now+self.steps[self.index].duration_s
                             if self.index < len(self.steps) else None)
        if self.index >= len(self.steps):
            return None
        step = self.steps[self.index]
        return step.vx, step.wz

    def describe(self):
        return ' -> '.join(f'{s.duration_s:g}s(vx={s.vx:g},wz={s.wz:g})' for s in self.steps)
