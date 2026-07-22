"""ONNX policy loading, observation construction, and action post-processing."""

from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

import config


class HumanoidPolicy:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise RuntimeError("Expected an ONNX policy with one input and one output")
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.input_name = self.input.name
        self.output_name = self.output.name
        self.last_action = np.zeros(config.ACTION_DIM, dtype=np.float32)

        probe = np.zeros((1, config.OBS_DIM), dtype=np.float32)
        result = self.session.run([self.output_name], {self.input_name: probe})[0]
        if result.shape != (1, config.ACTION_DIM):
            raise RuntimeError(f"Expected ONNX output (1, 12), received {result.shape}")

    def reset(self) -> None:
        self.last_action.fill(0.0)

    def build_observation(
        self,
        accel_m_s2: np.ndarray,
        gyro_rad_s: np.ndarray,
        projected_gravity: np.ndarray,
        velocity_command: np.ndarray,
        joint_position_policy: np.ndarray,
        joint_velocity_policy: np.ndarray,
    ) -> np.ndarray:
        q_rel = np.asarray(joint_position_policy, dtype=np.float32) - config.Q_DEFAULT

        velocity_command = np.asarray(velocity_command, dtype=np.float32)
        if velocity_command.shape != (3,):
            raise RuntimeError(
                "Velocity command must have shape (3,) in [vx, vy, wz] order; "
                f"received {velocity_command.shape}"
            )

        # Training observes only [vx, wz]. It does not observe the fixed-zero vy.
        policy_velocity_command = velocity_command[[0, 2]]

        obs = np.concatenate(
            (
                np.asarray(accel_m_s2, dtype=np.float32) * config.ACCEL_OBS_SCALE,
                np.asarray(gyro_rad_s, dtype=np.float32),
                np.asarray(projected_gravity, dtype=np.float32),
                policy_velocity_command,
                q_rel,
                np.asarray(joint_velocity_policy, dtype=np.float32),
                self.last_action,
            )
        ).astype(np.float32)

        if obs.shape != (config.OBS_DIM,):
            raise RuntimeError(
                f"Observation shape is {obs.shape}; expected ({config.OBS_DIM},)"
            )
        if not np.isfinite(obs).all():
            raise RuntimeError("Observation contains a non-finite value")
        return obs

    def step(self, **observation_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        obs = self.build_observation(**observation_values)
        start_ns = time.perf_counter_ns()
        action = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, config.OBS_DIM)}
        )[0][0].astype(np.float32)
        latency_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        if action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid ONNX policy output")
        q_target = config.Q_DEFAULT + config.ACTION_SCALE * action
        self.last_action = action.copy()
        return q_target, action, obs, latency_ms
