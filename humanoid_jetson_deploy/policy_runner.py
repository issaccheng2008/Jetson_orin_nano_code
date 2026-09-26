"""ONNX policy loading, observation construction, and action post-processing."""

from __future__ import annotations

import csv
import os
import time

import numpy as np
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    # Fixed-frame playback does not need ONNX Runtime. Keep the module's
    # inference entry point available for tests and fail only if ONNX is used.
    class _MissingOnnxRuntime:
        @staticmethod
        def InferenceSession(*_args, **_kwargs):
            raise ModuleNotFoundError("onnxruntime is required for --model")

    ort = _MissingOnnxRuntime()

import config


def observation_columns():
    """Name every element of the 49-wide observation, in build_observation order.

    Written out rather than inferred so a dump can be read without counting
    offsets by hand; keep it in step with build_observation.
    """
    names = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
             "grav_x", "grav_y", "grav_z", "cmd_vx", "cmd_wz", "step_distance",
             "crossing"]
    names += [f"q_rel_{j}" for j in config.JOINT_NAMES]
    names += [f"qd_{j}" for j in config.JOINT_NAMES]
    names += [f"last_action_{j}" for j in config.JOINT_NAMES]
    if len(names) != config.OBS_DIM:
        raise RuntimeError(f"{len(names)} observation names for {config.OBS_DIM} values")
    return names


class ObservationDump:
    """Append every policy tick's observation to a CSV, when POLICY_OBS_CSV is set.

    A card stop is invisible in the vision log at 2 Hz and in main.py's `|obs|max`
    summary; this is the only place all 49 components exist at 50 Hz. Off unless the
    env var is set, so a normal run pays nothing.
    """

    def __init__(self, path):
        self.path = path
        self.header_written = os.path.exists(path) and os.path.getsize(path) > 0
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.step = 0
        if not self.header_written:
            self.writer.writerow(
                ["t_host_iso", "step", "cmd_is_exact_zero"]
                + observation_columns()
                + [f"action_{j}" for j in config.JOINT_NAMES])
        self.file.flush()

    def append(self, obs, action, velocity_command) -> None:
        self.step += 1
        # The one-foot policy is stepped with a lift command instead, so this can
        # be absent; that path is not the one being measured.
        exact_zero = velocity_command is not None and bool(
            np.all(np.asarray(velocity_command) == 0.0))
        self.writer.writerow(
            [f"{time.time():.6f}", self.step, int(exact_zero)]
            + [f"{float(v):.6g}" for v in obs]
            + [f"{float(v):.6g}" for v in action])
        if self.step % 25 == 0:
            self.file.flush()


def open_observation_dump():
    path = os.environ.get("POLICY_OBS_CSV")
    return ObservationDump(path) if path else None


class HumanoidPolicy:
    obs_dim = config.OBS_DIM
    model_description = "current walking/stepping policy; legacy walking-test models are incompatible"

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise RuntimeError("Expected an ONNX policy with one input and one output")
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        if (
            self.input.type != "tensor(float)"
            or len(self.input.shape) != 2
            or self.input.shape[1] != self.obs_dim
            or (isinstance(self.input.shape[0], int) and self.input.shape[0] != 1)
        ):
            raise RuntimeError(
                f"Expected float32 ONNX input [1, {self.obs_dim}] "
                f"(dynamic batch allowed), received {self.input.type} {self.input.shape}. "
                f"Export the {self.model_description}."
            )
        self.input_name = self.input.name
        self.output_name = self.output.name
        self.last_action = np.zeros(config.ACTION_DIM, dtype=np.float32)
        self.observation_dump = open_observation_dump()

        probe = np.zeros((1, self.obs_dim), dtype=np.float32)
        result = self.session.run([self.output_name], {self.input_name: probe})[0]
        if result.shape != (1, config.ACTION_DIM) or not np.isfinite(result).all():
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
                np.array(
                    [
                        0.0 if np.all(velocity_command == 0.0)
                        else config.DEFAULT_STEP_DISTANCE,
                        config.CROSSING_COMMAND,
                    ],
                    dtype=np.float32,
                ),
                q_rel,
                np.asarray(joint_velocity_policy, dtype=np.float32),
                self.last_action,
            )
        ).astype(np.float32)

        if obs.shape != (self.obs_dim,):
            raise RuntimeError(
                f"Observation shape is {obs.shape}; expected ({self.obs_dim},)"
            )
        if not np.isfinite(obs).all():
            raise RuntimeError("Observation contains a non-finite value")
        return obs

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        return config.Q_DEFAULT + config.ACTION_SCALE * action

    def step(self, **observation_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        obs = self.build_observation(**observation_values)
        start_ns = time.perf_counter_ns()
        action = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, self.obs_dim)}
        )[0][0].astype(np.float32)
        latency_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        if action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid ONNX policy output")
        q_target = self.action_to_target(action)
        self.last_action = action.copy()
        if self.observation_dump is not None:
            self.observation_dump.append(
                obs, action, observation_values.get("velocity_command"))
        return q_target, action, obs, latency_ms


