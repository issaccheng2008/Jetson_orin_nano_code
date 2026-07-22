"""Robot and deployment constants that must match the Isaac Lab task."""

from __future__ import annotations

import numpy as np


NUM_JOINTS = 12
OBS_DIM = 47
ACTION_DIM = 12
POLICY_HZ = 50.0
POLICY_DT = 1.0 / POLICY_HZ
ACTION_SCALE = 0.25
ACCEL_OBS_SCALE = 0.1

JOINT_NAMES = (
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
)

# Isaac Lab default pose, in policy joint coordinates and radians.
Q_DEFAULT = np.array(
    [
        0.15,
        0.0,
        0.0,
        0.30,
        -0.15,
        0.0,
        -0.15,
        0.0,
        0.0,
        -0.30,
        0.15,
        0.0,
    ],
    dtype=np.float32,
)

# Limits from v2.4.1.urdf, in policy joint coordinates and radians.
Q_LOWER = np.array(
    [
        -1.57,
        -1.57,
        -1.57,
        -1.57,
        -0.50,
        -0.50,
        -1.57,
        -0.50,
        -1.57,
        -1.57,
        -0.50,
        -0.50,
    ],
    dtype=np.float32,
)
Q_UPPER = np.array(
    [
        1.57,
        0.50,
        1.57,
        1.57,
        0.50,
        0.50,
        1.57,
        1.57,
        1.57,
        1.57,
        0.50,
        0.50,
    ],
    dtype=np.float32,
)

JOINT_LIMIT_MARGIN_RAD = 0.05
MAX_TARGET_SPEED_RAD_S = 3.0
# Maximum commanded-position error relative to the latest encoder position.
# This is the user-adjustable "x" safety window, in degrees.
MAX_TARGET_DEVIATION_DEG = 10
MAX_TARGET_DEVIATION_RAD = float(np.deg2rad(MAX_TARGET_DEVIATION_DEG))

# These values describe the physical encoder convention, not the URDF convention.
# Calibrate all 12 joints before changing CALIBRATION_CONFIRMED to True.
#
# q_policy = MOTOR_SIGN * (q_motor - MOTOR_ZERO_RAD)
# q_motor  = MOTOR_ZERO_RAD + MOTOR_SIGN * q_policy
MOTOR_SIGN = np.ones(NUM_JOINTS, dtype=np.float32)
MOTOR_ZERO_RAD = np.zeros(NUM_JOINTS, dtype=np.float32)
CALIBRATION_CONFIRMED = True

# Transform real IMU vectors into the simulated IMU/policy frame:
# vector_policy = IMU_TO_POLICY @ vector_sensor
#
# Identity is only correct when the physical IMU axes and mounting direction match
# the simulated ImuCfg. Use a signed permutation rotation matrix after measuring the
# real installation.
IMU_TO_POLICY = np.eye(3, dtype=np.float32)

# The DM-IMU-L1 reports W, X, Y, Z for the sensor orientation in the world
# frame. Set this to False only if a stationary tilt test demonstrates that
# your firmware reports the inverse convention.
IMU_QUATERNION_IS_SENSOR_TO_WORLD = True

# Set this to True after IMU_TO_POLICY has been measured once and stored here.
# This mounting calibration is persistent and does not require a level startup.
IMU_CALIBRATION_CONFIRMED = True


def validate_imu_configuration() -> None:
    rotation = np.asarray(IMU_TO_POLICY, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("IMU_TO_POLICY must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-4):
        raise ValueError("IMU_TO_POLICY must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-4):
        raise ValueError("IMU_TO_POLICY must be a proper rotation with determinant +1")


def motor_to_policy_position(q_motor: np.ndarray) -> np.ndarray:
    q_motor = np.asarray(q_motor, dtype=np.float32)
    return MOTOR_SIGN * (q_motor - MOTOR_ZERO_RAD)


def motor_to_policy_velocity(qd_motor: np.ndarray) -> np.ndarray:
    qd_motor = np.asarray(qd_motor, dtype=np.float32)
    return MOTOR_SIGN * qd_motor


def policy_to_motor_position(q_policy: np.ndarray) -> np.ndarray:
    q_policy = np.asarray(q_policy, dtype=np.float32)
    return MOTOR_ZERO_RAD + MOTOR_SIGN * q_policy


def clamp_policy_target(q_target: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(q_target, dtype=np.float32),
        Q_LOWER + JOINT_LIMIT_MARGIN_RAD,
        Q_UPPER - JOINT_LIMIT_MARGIN_RAD,
    )


def clamp_policy_target_to_current(
    q_target: np.ndarray,
    q_current: np.ndarray,
) -> np.ndarray:
    """Keep each target near its measured position and inside hard joint limits."""
    q_target = np.asarray(q_target, dtype=np.float32)
    q_current = np.asarray(q_current, dtype=np.float32)
    expected_shape = (NUM_JOINTS,)
    if q_target.shape != expected_shape or q_current.shape != expected_shape:
        raise ValueError(
            f"target/current joint arrays must have shape {expected_shape}, "
            f"got {q_target.shape} and {q_current.shape}"
        )
    if not np.all(np.isfinite(q_target)) or not np.all(np.isfinite(q_current)):
        raise ValueError("target/current joint arrays must contain only finite values")

    lower = np.maximum(
        Q_LOWER + JOINT_LIMIT_MARGIN_RAD,
        q_current - MAX_TARGET_DEVIATION_RAD,
    )
    upper = np.minimum(
        Q_UPPER - JOINT_LIMIT_MARGIN_RAD,
        q_current + MAX_TARGET_DEVIATION_RAD,
    )
    if np.any(lower > upper):
        joint_index = int(np.flatnonzero(lower > upper)[0])
        raise ValueError(
            "no safe target window for "
            f"{JOINT_NAMES[joint_index]}: measured position is outside the "
            "configured joint limits"
        )
    return np.clip(q_target, lower, upper)
