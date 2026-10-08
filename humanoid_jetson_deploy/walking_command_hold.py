"""Experiment 2: hold increases/reversals, permit early yaw reductions."""

import math

import numpy as np


MIN_WALKING_COMMAND_HOLD_S = 0.5


class WalkingCommandHold:
    def __init__(self, fast_release=True):
        self.fast_release = bool(fast_release)
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
        early_release = (self.fast_release and self.applied is not None
                         and np.array_equal(self.latest_request[:2], self.applied[:2])
                         and self.latest_request[2]*self.applied[2] >= 0.
                         and abs(self.latest_request[2]) < abs(self.applied[2]))
        if self.applied is None or (self.remaining(now) == 0. and
                                    not np.array_equal(self.latest_request, self.applied)) or early_release:
            self.applied = self.latest_request.copy()
            self.applied_at = now
        # Repeated identical packets do not extend the minimum duration.
        return self.applied.copy()
