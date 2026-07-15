"""Gyro-assisted projected-gravity estimator for a six-axis IMU."""

from __future__ import annotations

import numpy as np


class ProjectedGravityFilter:
    """Estimate world-down expressed in the IMU/body frame.

    The accelerometer correction assumes a stationary upright sensor reports
    approximately +9.81 m/s^2 along its upward axis. Dynamic acceleration is
    rejected when its magnitude is far from gravity.
    """

    def __init__(self, correction_time_constant_s: float = 0.5) -> None:
        self.gravity_body = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.time_constant = float(correction_time_constant_s)
        self.initialized = False

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1.0e-6:
            raise ValueError("Cannot normalize a near-zero vector")
        return vector / norm

    def update(self, accel_m_s2: np.ndarray, gyro_rad_s: np.ndarray, dt: float) -> np.ndarray:
        accel = np.asarray(accel_m_s2, dtype=np.float32).reshape(3)
        gyro = np.asarray(gyro_rad_s, dtype=np.float32).reshape(3)
        dt = float(np.clip(dt, 1.0e-4, 0.1))

        accel_norm = float(np.linalg.norm(accel))
        if not self.initialized and accel_norm > 1.0:
            self.gravity_body = self._normalize(-accel).astype(np.float32)
            self.initialized = True

        # A world-fixed vector expressed in a rotating body frame follows
        # dg_body/dt = -omega_body x g_body.
        predicted = self.gravity_body - np.cross(gyro, self.gravity_body) * dt
        predicted = self._normalize(predicted)

        if 0.65 * 9.81 <= accel_norm <= 1.35 * 9.81:
            measured = self._normalize(-accel)
            alpha = dt / (self.time_constant + dt)
            predicted = self._normalize((1.0 - alpha) * predicted + alpha * measured)

        self.gravity_body = predicted.astype(np.float32)
        return self.gravity_body.copy()
