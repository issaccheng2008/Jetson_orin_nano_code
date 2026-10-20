"""ONNX policy loading, observation construction, and action post-processing."""

from __future__ import annotations

import csv
import os
from pathlib import Path
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


def observation_columns(obs_dim=config.OBS_DIM):
    """Name the walking49 or onefoot46 observation in build_observation order.

    Written out rather than inferred so a dump can be read without counting
    offsets by hand; keep it in step with build_observation.
    """
    names = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
             "grav_x", "grav_y", "grav_z"]
    if obs_dim == 49:
        names += ["cmd_vx", "cmd_wz", "step_distance", "crossing"]
    elif obs_dim == 46:
        names += ["lift_command"]
    else:
        raise ValueError(f"Unsupported observation dump width: {obs_dim}")
    names += [f"q_rel_{j}" for j in config.JOINT_NAMES]
    names += [f"qd_{j}" for j in config.JOINT_NAMES]
    names += [f"last_action_{j}" for j in config.JOINT_NAMES]
    if len(names) != obs_dim:
        raise RuntimeError(f"{len(names)} observation names for {obs_dim} values")
    return names


class ObservationDump:
    """Append each policy tick to a CSV with its own observation schema.

    A card stop is invisible in the vision log at 2 Hz and in main.py's `|obs|max`
    summary; this is the only place all 49 components exist at 50 Hz. Off unless the
    env var is set, so a normal run pays nothing.
    """

    def __init__(self, path, obs_dim=config.OBS_DIM):
        self.obs_dim = obs_dim
        columns = observation_columns(obs_dim)
        # Preserve the existing walking file name and schema. A second model
        # loading the same environment path must not append 46-wide rows there.
        requested_path = Path(path)
        actual_path = (requested_path.with_name(
            f"{requested_path.stem}_onefoot46{requested_path.suffix}")
            if obs_dim == 46 else requested_path)
        self.path = str(actual_path)
        self.mode = "onefoot46" if obs_dim == 46 else "walking49"
        header = (["t_host_iso", "step", "cmd_is_exact_zero"] + columns
                  + [f"action_{j}" for j in config.JOINT_NAMES])
        self.header_written = actual_path.exists() and actual_path.stat().st_size > 0
        if self.header_written:
            with actual_path.open(newline="", encoding="utf-8") as existing:
                if next(csv.reader(existing), None) != header:
                    raise ValueError(f"Observation dump schema mismatch: {self.path}")
        self.file = actual_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.step = 0
        if not self.header_written:
            self.writer.writerow(header)
        self.file.flush()

    def append(self, obs, action, velocity_command) -> None:
        obs = np.asarray(obs)
        action = np.asarray(action)
        if obs.shape != (self.obs_dim,) or action.shape != (config.ACTION_DIM,):
            raise ValueError(
                f"Observation dump {self.mode} expected obs ({self.obs_dim},) "
                f"and action ({config.ACTION_DIM},), got {obs.shape} and {action.shape}")
        self.step += 1
        # One-foot uses a lift command, not a velocity command. Leave this field
        # empty there; its lift_command observation has the applicable value.
        exact_zero = (int(bool(np.all(np.asarray(velocity_command) == 0.0)))
                      if velocity_command is not None else "")
        self.writer.writerow(
            [f"{time.time():.6f}", self.step, exact_zero]
            + [f"{float(v):.6g}" for v in obs]
            + [f"{float(v):.6g}" for v in action])
        if self.step % 25 == 0:
            self.file.flush()


def open_observation_dump(obs_dim=config.OBS_DIM):
    path = os.environ.get("POLICY_OBS_CSV")
    if not path:
        return None
    dump = ObservationDump(path, obs_dim)
    print(f"Policy observation log: {dump.path} (schema={dump.mode})")
    return dump


class HumanoidPolicy:
    obs_dim = config.OBS_DIM
    model_description = "current walking/stepping policy; legacy walking-test models are incompatible"

    def __init__(self, model_path: str, step_distance_m: float | None = None) -> None:
        # 步长是一个数，直接给，不随 vx 变（main.py --step-cm）。
        # ⚠️ 但步频**不是** vx/步长：它是策略自己的，会跟着 vx 和 step_distance
        # 动（2026-10-03 实测：vx 从 0.3 降到 0.2 反而让步频变快）。所以这里给的
        # 是"要求策略走的步幅"，不是"结果速度"。
        self.step_distance_m = (
            config.STEP_LENGTH_CM / 100.0 if step_distance_m is None
            else float(step_distance_m)
        )
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
        self.observation_dump = open_observation_dump(self.obs_dim)

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
                        # 在走就是步长那个数；停下（vx=0）必须是 0。
                        # 恒定的 5cm 直接发下去会让策略在停车时迈原地步（真漂移）
                        # —— 那正是当初把步长做成比例的原因（vx=0 时比例给 0）。
                        # 训练是 vx=0.20 固定、step_distance 在 0.02~0.12 之间随机的，
                        # 所以"在走"的那一档落在分布里，vx=0 配 5cm 不在。
                        (self.step_distance_m if velocity_command[0] > 0.0 else 0.0),
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


