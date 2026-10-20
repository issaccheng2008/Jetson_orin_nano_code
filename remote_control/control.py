"""Exclusive, expiring velocity lease; independent of HTTP and robot hardware."""
from __future__ import annotations

import math
import secrets
import threading
import time

from connector import slew_toward


class ControlError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ControlError(f"{name} must be a finite number")
    return float(value)


class ControlState:
    def __init__(self, *, max_vx=.4, max_wz=.5, timeout=.3,
                 vx_accel=1., wz_accel=2., yaw_sign=1, clock=time.monotonic):
        for name, value, upper in (("max_vx", max_vx, 1), ("max_wz", max_wz, .5),
                                   ("timeout", timeout, 1), ("vx_accel", vx_accel, math.inf),
                                   ("wz_accel", wz_accel, math.inf)):
            if not 0 < finite_number(value, name) <= upper:
                raise ValueError(f"{name} must be positive and <= {upper}")
        if type(yaw_sign) not in (int, float) or yaw_sign not in (-1, 1):
            raise ValueError("yaw_sign must be -1 or 1")
        self.max_vx, self.max_wz = float(max_vx), float(max_wz)
        self.timeout, self.yaw_sign = float(timeout), int(yaw_sign)
        self.vx_accel, self.wz_accel = vx_accel, wz_accel
        self.clock = clock
        self.lock = threading.RLock()
        self.session = None
        self.last_update = 0.
        self.sequence = 0
        self.target_vx = self.target_wz = self.vx = self.wz = 0.

    def _zero(self):
        self.session = None
        self.target_vx = self.target_wz = self.vx = self.wz = 0.

    def _expire(self):
        if self.session is not None and self.clock() - self.last_update >= self.timeout:
            self._zero()

    def _require_owner(self, session):
        self._expire()
        if not isinstance(session, str) or not self.session or not secrets.compare_digest(session, self.session):
            raise ControlError("Control expired or belongs to another device; enable again", 409)

    def snapshot(self):
        with self.lock:
            self._expire()
            return {"active": self.session is not None,
                    "target_vx": self.target_vx, "target_wz": self.target_wz,
                    "vx": self.vx, "wz": self.wz,
                    "max_vx": self.max_vx, "max_wz": self.max_wz,
                    "yaw_sign": self.yaw_sign, "timeout_ms": self.timeout * 1000,
                    "command_age_ms": max(0., (self.clock() - self.last_update) * 1000)
                    if self.session else None}

    def arm(self):
        with self.lock:
            self._expire()
            if self.session:
                raise ControlError("Another device already has control", 409)
            self._zero()
            self.session = secrets.token_urlsafe(24)
            self.sequence = 0
            self.last_update = self.clock()
            return {**self.snapshot(), "session": self.session}

    def command(self, session, seq, vx, wz):
        # Validate before changing timestamps: malformed traffic never keeps a lease alive.
        vx, wz = finite_number(vx, "vx"), finite_number(wz, "wz")
        if type(seq) is not int or not 0 < seq < 2**53:
            raise ControlError("seq must be a positive safe integer")
        with self.lock:
            self._require_owner(session)
            if seq <= self.sequence:
                raise ControlError("Out-of-order command", 409)
            self.target_vx = min(self.max_vx, max(0., vx))
            self.target_wz = min(self.max_wz, max(-self.max_wz, wz)) * self.yaw_sign
            self.sequence = seq
            self.last_update = self.clock()
            return self.snapshot()

    def stop(self, session):
        with self.lock:
            self._require_owner(session)
            self._zero()
            return self.snapshot()

    def force_stop(self):
        with self.lock:
            self._zero()

    def tick(self, dt):
        with self.lock:
            self._expire()
            dt = min(.04, max(0., dt))
            self.vx = slew_toward(self.vx, self.target_vx, self.vx_accel * dt)
            self.wz = slew_toward(self.wz, self.target_wz, self.wz_accel * dt)
            return self.snapshot()
