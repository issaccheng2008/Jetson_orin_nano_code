"""Hardware-independent 49-observation / 12-action adapter.

The caller supplies synchronized base-frame IMU and joint encoder samples.
No motor transport, foot contacts, obstacle sensing or standing controller.
"""
from dataclasses import dataclass
import numpy as np

from .phase_clock import PhaseClock, PhaseClockConfig

JOINT_NAMES = tuple(f'{side}_{name}_joint' for side in ('r', 'l') for name in
                    ('leg_pitch', 'leg_roll', 'leg_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll'))
DEFAULT_JOINT_POS = np.array([.15, 0, 0, .30, -.15, 0, -.15, 0, 0, -.30, .15, 0], dtype=np.float32)
ACTION_SCALE = .25


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of shape ({size},)')
    return result


def projected_gravity(base_to_world_xyzw):
    """R(base->world).T @ [0, 0, -1], unit direction; quaternion XYZW."""
    quat = _vector(base_to_world_xyzw, 4, 'base_to_world_xyzw')
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        raise ValueError('Zero quaternion')
    x, y, z, w = quat / norm
    return np.array([2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))], dtype=np.float32)


def build_observation(acc, gyro, gravity, command, q, qd, last_action):
    """Acceleration includes gravity specific force (stationary upright: +9.81 z)."""
    return np.concatenate((_vector(acc,3,'acc')*.1, _vector(gyro,3,'gyro'),
                           _vector(gravity,3,'gravity'), _vector(command,4,'command'),
                           _vector(q,12,'q')-DEFAULT_JOINT_POS, _vector(qd,12,'qd'),
                           _vector(last_action,12,'last_action'))).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ControlSample:
    phase: int
    elapsed_s: float
    active: bool
    sequence_finished: bool
    action: object = None
    joint_targets: object = None


class PolicyController:
    """Call tick once per 20 ms sensor/control frame; finish hands off upstream."""
    def __init__(self, inference, config=None):
        self.inference = inference
        self.clock = PhaseClock(config)
        self.last_action = np.zeros(12, dtype=np.float32)

    @classmethod
    def from_onnx(cls, model_path, config=None):
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        inp, out = session.get_inputs(), session.get_outputs()
        if len(inp) != 1 or len(out) != 1 or inp[0].shape[-1] != 49 or out[0].shape[-1] != 12:
            raise ValueError('Expected one 49-input / 12-output ONNX actor')
        if inp[0].type != 'tensor(float)' or out[0].type != 'tensor(float)':
            raise ValueError('Expected float32 ONNX tensors')
        return cls(lambda obs: session.run([out[0].name], {inp[0].name: obs})[0], config)

    def reset(self):
        self.clock.reset()
        self.last_action.fill(0.)

    def start(self, now=None):
        if self.clock.start(now):
            self.last_action.fill(0.)
            return True
        return False

    def tick(self, acc, gyro, gravity, q, qd, now=None):
        sample = self.clock.read(now)
        if not sample.active:
            return ControlSample(sample.phase, sample.elapsed_s, False, sample.sequence_finished)
        obs = build_observation(acc, gyro, gravity, sample.command, q, qd, self.last_action)
        result = np.asarray(self.inference(obs[None,:]), dtype=np.float32)
        if result.shape != (1,12) or not np.isfinite(result).all():
            raise ValueError('Actor must return finite float32 actions of shape (1,12)')
        action = result[0].copy()
        self.last_action[:] = action
        return ControlSample(sample.phase, sample.elapsed_s, True, False, action,
                             DEFAULT_JOINT_POS + ACTION_SCALE * action)
