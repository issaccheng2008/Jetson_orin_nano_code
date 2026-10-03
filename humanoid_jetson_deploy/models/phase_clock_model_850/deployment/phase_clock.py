"""Shared command schedule for simulation and the upper-computer controller."""
from dataclasses import asdict, dataclass
import math
import time


@dataclass(frozen=True)
class PhaseClockConfig:
    control_dt: float = .02
    walk_end_s: float = .14
    lead_end_s: float = .44
    sequence_end_s: float = .76
    forward_velocity: float = .20
    walk_step: float = .10
    crossing_step: float = .23

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Clock parameters must be finite')
        if not (self.control_dt > 0 and 0 < self.walk_end_s < self.lead_end_s < self.sequence_end_s
                and self.forward_velocity > 0 and self.walk_step > 0 and self.crossing_step > 0):
            raise ValueError('Require positive periods/commands and increasing phase boundaries')
        a, b, c = self.boundary_ticks
        if not 0 < a < b < c:
            raise ValueError('Phase boundaries collapse after control-period quantization')

    @property
    def boundary_ticks(self):
        return tuple(math.ceil((t - 1e-10) / self.control_dt)
                     for t in (self.walk_end_s, self.lead_end_s, self.sequence_end_s))

    def phase_at_tick(self, tick):
        # Supports scalar integers and Torch/NumPy integer arrays without imports.
        phase = tick * 0
        for boundary in self.boundary_ticks:
            phase = phase + (tick >= boundary)
        return phase

    def command_at_tick(self, tick):
        return self.command_table[int(self.phase_at_tick(tick))]

    @property
    def command_table(self):
        v = self.forward_velocity
        return ((v, 0., self.walk_step, 0.), (v, 0., self.crossing_step, 1.),
                (v, 0., 0., 1.), (0., 0., 0., 0.))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ClockSample:
    phase: int
    tick: int
    elapsed_s: float
    command: tuple
    active: bool
    sequence_finished: bool


class PhaseClock:
    """One start cue per reset. Time uses a monotonic seconds clock."""
    def __init__(self, config=None):
        self.config = config or PhaseClockConfig()
        self.reset()

    def reset(self):
        self._start = None
        self._last = None

    @staticmethod
    def _time(now):
        value = time.monotonic() if now is None else float(now)
        if not math.isfinite(value):
            raise ValueError('Time must be finite monotonic seconds')
        return value

    def start(self, now=None):
        if self._start is not None:
            return False
        self._start = self._last = self._time(now)
        return True

    def read(self, now=None):
        now = self._time(now)
        if self._start is None:
            return ClockSample(0, 0, 0., (0., 0., 0., 0.), False, False)
        if now < self._last:
            raise ValueError('Monotonic clock moved backwards')
        self._last = now
        elapsed = now - self._start
        # Absolute monotonic origins can be millions of seconds. Account for
        # their subtraction rounding, rather than delaying a boundary one tick.
        tolerance = max(1e-10, math.ulp(now) + math.ulp(self._start))
        if tolerance >= self.config.control_dt * .5:
            raise ValueError('Timestamp resolution is too coarse for the control period')
        tick = max(0, math.floor((elapsed + tolerance) / self.config.control_dt))
        phase = int(self.config.phase_at_tick(tick))
        return ClockSample(phase, tick, elapsed, self.config.command_table[phase],
                           phase < 3, phase == 3)
