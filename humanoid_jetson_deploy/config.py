"""Robot and deployment constants that must match the Isaac Lab task."""

from __future__ import annotations

import numpy as np


NUM_JOINTS = 12
# 电机 PD 基准，直接下发 STM32；不再在下位机乘 1.5/2.0。
# 顺序与 joint_target 的线序一致：左腿六项，再右腿六项；每腿依次为
# 髋 pitch、髋 roll、髋 yaw、膝 pitch、踝 pitch、踝 roll。
JOINT_KP = np.array([
    35.0, 30.0, 20.0, 35.0, 30.0, 12.0,  # 左腿
    35.0, 30.0, 20.0, 35.0, 30.0, 12.0,  # 右腿
], dtype=np.float32)
JOINT_KD = np.array([
    1.5, 1.2, 1.0, 1.5, 1.6, 0.7,  # 左腿
    1.5, 1.2, 1.0, 1.5, 1.6, 0.7,  # 右腿
], dtype=np.float32)
GAIN_SCALE = 1.0  # 全部 KP/KD 的统一倍率，例如 1.2 表示同时增加 20%。


def command_gains(kp_scale: float, kd_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the common multiplier and existing caller-specific P/D scales."""
    scales = np.asarray([GAIN_SCALE, kp_scale, kd_scale], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("gain scales must be finite and non-negative")
    return (np.asarray(JOINT_KP, dtype=np.float64) * GAIN_SCALE * kp_scale,
            np.asarray(JOINT_KD, dtype=np.float64) * GAIN_SCALE * kd_scale)


OBS_DIM = 49
ACTION_DIM = 12
POLICY_HZ = 50.0
POLICY_DT = 1.0 / POLICY_HZ
ACTION_SCALE = 0.25
ACCEL_OBS_SCALE = 0.1

# Humanoid_Robot_RSL_RL main at 4eb3d5b4d72a792c610ad46f0a8c65b931ed3b22.
#
# 速度和步长各是一个数，就这两个。以前它们是"一对比例"（--max-vx / --max-step-cm
# 定出"每 m/s 给多大步距"，实际步距 = vx × 那个比例），结果是改速度就顺手改了
# 步长 —— 2026-10-03 把速度 0.3 改成 0.2 时，所有不显式写那两个参数的跑法步幅
# 都短了 25%，而且没人看得出来。现在步长直接给，不随 vx 变。
MAX_COMMAND_VX = 0.2    # m/s, the speed the vision commands with a valid detection
STEP_LENGTH_CM = 5.0    # cm, the step the policy is asked for, whatever vx is
CROSSING_COMMAND = 0.0  # normal walking only

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
# 2026-10-04：整体放宽 25%。实车转弯时 l_ankle_roll 实测被地面带到 −0.60 rad，
# 越过 ±0.5 限位 + margin 0.05 + 10° 反馈窗 → target safety 以
# "no safe target window" 退出（阈值原本 −0.6245 rad）；放宽后阈值到 −0.75 rad。
Q_LIMIT_SCALE = np.float32(1.25)
Q_LOWER = Q_LIMIT_SCALE * np.array(
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
Q_UPPER = Q_LIMIT_SCALE * np.array(
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

