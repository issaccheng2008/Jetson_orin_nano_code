"""Projected gravity derived from the DM-IMU-L1 fused orientation."""

from __future__ import annotations

import numpy as np


def normalize_quaternion_wxyz(orientation_wxyz: np.ndarray) -> np.ndarray:
    """Validate and normalize a W, X, Y, Z quaternion."""
    quaternion = np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    if not np.isfinite(quaternion).all():
        raise ValueError("IMU quaternion contains a non-finite value")
    norm = float(np.linalg.norm(quaternion))
    if not 0.5 <= norm <= 1.5:
        raise ValueError(f"IMU quaternion norm is invalid: {norm:.6f}")
    return quaternion / norm


def roll_pitch_yaw_from_quaternion(
    orientation_wxyz: np.ndarray,
    imu_to_policy: np.ndarray,
    *,
    sensor_to_world: bool,
) -> np.ndarray:
    """Return policy-frame roll, pitch, and yaw in radians."""
    w, x, y, z = normalize_quaternion_wxyz(orientation_wxyz)
    rotation = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    rotation_world_sensor = rotation if sensor_to_world else rotation.T
    sensor_to_policy = np.asarray(imu_to_policy, dtype=np.float64).reshape(3, 3)
    rotation_world_policy = rotation_world_sensor @ sensor_to_policy.T

    pitch = np.arctan2(
        -rotation_world_policy[2, 0],
        np.hypot(rotation_world_policy[0, 0], rotation_world_policy[1, 0]),
    )
    if np.hypot(rotation_world_policy[0, 0], rotation_world_policy[1, 0]) > 1.0e-8:
        roll = np.arctan2(rotation_world_policy[2, 1], rotation_world_policy[2, 2])
        yaw = np.arctan2(rotation_world_policy[1, 0], rotation_world_policy[0, 0])
    else:
        roll = np.arctan2(-rotation_world_policy[1, 2], rotation_world_policy[1, 1])
        yaw = 0.0
    return np.array((roll, pitch, yaw), dtype=np.float32)


def projected_gravity_from_quaternion(
    orientation_wxyz: np.ndarray,
    imu_to_policy: np.ndarray,
    *,
    sensor_to_world: bool,
) -> np.ndarray:
    """Return world-down expressed in the simulated IMU/policy frame."""
    w, x, y, z = normalize_quaternion_wxyz(orientation_wxyz)

    if sensor_to_world:
        # R_world_sensor.T @ [0, 0, -1].
        gravity_sensor = np.array(
            (
                2.0 * (w * y - x * z),
                -2.0 * (y * z + w * x),
                2.0 * (x * x + y * y) - 1.0,
            ),
            dtype=np.float64,
        )
    else:
        # R_sensor_world @ [0, 0, -1].
        gravity_sensor = np.array(
            (
                -2.0 * (x * z + w * y),
                2.0 * (w * x - y * z),
                2.0 * (x * x + y * y) - 1.0,
            ),
            dtype=np.float64,
        )

    gravity_policy = np.asarray(imu_to_policy, dtype=np.float64) @ gravity_sensor
    norm = float(np.linalg.norm(gravity_policy))
    if not np.isfinite(norm) or norm < 1.0e-6:
        raise ValueError("Projected gravity is invalid")
    return (gravity_policy / norm).astype(np.float32)


def validate_stationary_imu_sample(
    accel_policy_m_s2: np.ndarray,
    gyro_policy_rad_s: np.ndarray,
    projected_gravity: np.ndarray,
) -> None:
    """Reject inconsistent startup orientation data before motors are enabled."""
    accel = np.asarray(accel_policy_m_s2, dtype=np.float64).reshape(3)
    gyro = np.asarray(gyro_policy_rad_s, dtype=np.float64).reshape(3)
    gravity = np.asarray(projected_gravity, dtype=np.float64).reshape(3)
    if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
        raise ValueError("IMU acceleration/gyroscope contains a non-finite value")

    accel_norm = float(np.linalg.norm(accel))
    if not 0.75 * 9.81 <= accel_norm <= 1.25 * 9.81:
        raise ValueError(
            "Keep the robot stationary during startup; accelerometer magnitude "
            f"is {accel_norm:.3f} m/s^2"
        )
    gyro_norm = float(np.linalg.norm(gyro))
    if gyro_norm > 0.35:
        raise ValueError(
            "Keep the robot stationary during startup; angular speed is "
            f"{gyro_norm:.3f} rad/s"
        )

    alignment = float(np.dot(-accel / accel_norm, gravity))
    if alignment < 0.90:
        raise ValueError(
            "IMU quaternion convention is inconsistent with acceleration "
            f"(gravity alignment={alignment:.3f}); check "
            "IMU_QUATERNION_IS_SENSOR_TO_WORLD"
        )
