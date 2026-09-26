"""One-foot ONNX interface, matching training commit c1b4e8c8bdedafc8c7fd4c162a7c3c9e0a28df9f."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import config
from policy_runner import HumanoidPolicy


def mirror_joint_data(value: np.ndarray) -> np.ndarray:
    """Swap right/left six-joint blocks and negate every paired coordinate."""
    return -np.concatenate((value[6:], value[:6]))


@dataclass(frozen=True)
class OneFootCommand:
    """One stand/lift/lower sequence; stay at command zero after lowering."""

    stand_seconds: float = 1.0
    lift_seconds: float = 4.0

    def __post_init__(self) -> None:
        for name, value in (("stand_seconds", self.stand_seconds),
                            ("lift_seconds", self.lift_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def get(self, elapsed_s: float) -> float:
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        return float(self.stand_seconds <= elapsed_s < self.stand_seconds + self.lift_seconds)


class OneFootPolicy(HumanoidPolicy):
    obs_dim = 46
    model_description = "one-foot standing policy (46 observations, 12 actions)"

    def __init__(self, model_path: str, support_foot: str = "right") -> None:
        if support_foot not in ("right", "left"):
            raise ValueError("support_foot must be right or left")
        # Fixed for this run, including command-zero phases, just as in training.
        self.support_is_left = support_foot == "left"
        super().__init__(model_path)

    def select_support_foot(self, support_foot: str) -> None:
        """Select the physical support side at the start of a new card action."""
        if support_foot not in ("right", "left"):
            raise ValueError("support_foot must be right or left")
        self.support_is_left = support_foot == "left"
        self.reset()

    def build_observation(
        self,
        accel_m_s2: np.ndarray,
        gyro_rad_s: np.ndarray,
        projected_gravity: np.ndarray,
        lift_command: float,
        joint_position_policy: np.ndarray,
        joint_velocity_policy: np.ndarray,
    ) -> np.ndarray:
        if lift_command not in (0.0, 1.0):
            raise ValueError("lift_command must be 0 or 1")
        values = []
        for name, value, size in (
            ("accel_m_s2", accel_m_s2, 3),
            ("gyro_rad_s", gyro_rad_s, 3),
            ("projected_gravity", projected_gravity, 3),
            ("joint_position_policy", joint_position_policy, 12),
            ("joint_velocity_policy", joint_velocity_policy, 12),
        ):
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
            values.append(array.copy())
        accel, gyro, gravity, q, qd = values
        q_rel = q - config.Q_DEFAULT
        if self.support_is_left:
            accel *= [1, -1, 1]
            gyro *= [-1, 1, -1]
            gravity *= [1, -1, 1]
            q_rel = mirror_joint_data(q_rel)
            qd = mirror_joint_data(qd)
        obs = np.concatenate((
            accel * config.ACCEL_OBS_SCALE, gyro, gravity, [lift_command],
            q_rel, qd, self.last_action,
        )).astype(np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError("Observation contains a non-finite value")
        return obs

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        # last_action remains the raw canonical ONNX output in the base runner.
        physical_action = mirror_joint_data(action) if self.support_is_left else action
        return config.Q_DEFAULT + config.ACTION_SCALE * physical_action
