"""Keep each normal walking command constant for at least 0.5 host seconds."""

import math

import numpy as np


MIN_WALKING_COMMAND_HOLD_S = 0.5


class WalkingCommandHold:
    def __init__(self):
        self.clear()

    def clear(self):
        self.applied = None
        self.latest_request = None
        self.applied_at = None

    @staticmethod
    def _time(now):
        now = float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("Walking command time must be finite and nonnegative")
        return now

    def remaining(self, now):
        now = self._time(now)
        return (max(0., self.applied_at + MIN_WALKING_COMMAND_HOLD_S - now)
                if self.applied_at is not None else 0.)

    def apply(self, requested, now):
        now = self._time(now)
        requested = np.asarray(requested, dtype=np.float32)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("Walking command must contain three finite values")
        if self.applied_at is not None and now < self.applied_at:
            raise ValueError("Walking command clock moved backwards")
        if np.all(requested == 0.):
            # A stop never waits for a motion contract to expire. A subsequent
            # start is a new contract, independent of the interrupted command.
            self.clear()
            return requested.copy()
        self.latest_request = requested.copy()
        if self.applied is None or (self.remaining(now) == 0. and
                                    not np.array_equal(self.latest_request, self.applied)):
            self.applied = self.latest_request.copy()
            self.applied_at = now
        # Repeated identical packets do not extend the minimum duration.
        return self.applied.copy()
