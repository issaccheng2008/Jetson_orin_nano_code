"""Per-run joint target limits shared by the regular and phase-clock loops."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import config


def _vector(value, name):
    try:
        value = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be 12 finite angles") from exc
    if value.shape != (config.NUM_JOINTS,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape ({config.NUM_JOINTS},) and be finite")
    return value


def _nonnegative(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def limit_target_slew(target, previous, dt, max_speed_rad_s):
    target = _vector(target, "target")
    previous = _vector(previous, "previous target")
    dt = _nonnegative(dt, "target dt")
    speed = _nonnegative(max_speed_rad_s, "max-target-speed-rad-s")
    if speed == 0:
        return target.copy()
    maximum_change = speed * dt
    return previous + np.clip(target - previous, -maximum_change, maximum_change)


@dataclass(frozen=True)
class TargetTrace:
    raw_target: np.ndarray
    absolute_target: np.ndarray
    slew_target: np.ndarray
    final_target: np.ndarray
    previous_target: np.ndarray
    reference_q: np.ndarray
    absolute_mask: np.ndarray
    slew_mask: np.ndarray
    window_mask: np.ndarray
    dt: float


@dataclass(frozen=True)
class TargetSafety:
    lower: tuple = tuple(float(x) for x in config.Q_LOWER)
    upper: tuple = tuple(float(x) for x in config.Q_UPPER)
    margin_rad: float = config.JOINT_LIMIT_MARGIN_RAD
    max_speed_rad_s: float = config.MAX_TARGET_SPEED_RAD_S
    max_deviation_deg: float = config.MAX_TARGET_DEVIATION_DEG

    def __post_init__(self):
        lower = _vector(self.lower, "joint lower bounds")
        upper = _vector(self.upper, "joint upper bounds")
        for name in ("margin_rad", "max_speed_rad_s", "max_deviation_deg"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if np.any(lower >= upper):
            raise ValueError("Every joint lower bound must be below its upper bound")
        if np.any(lower + self.margin_rad >= upper - self.margin_rad):
            raise ValueError("joint-limit-margin-rad leaves an empty joint range")
        # Immutable copies ensure one run cannot mutate another run's limits.
        object.__setattr__(self, "lower", tuple(float(x) for x in lower))
        object.__setattr__(self, "upper", tuple(float(x) for x in upper))

    @property
    def safe_lower(self):
        return np.asarray(self.lower, dtype=np.float32) + self.margin_rad

    @property
    def safe_upper(self):
        return np.asarray(self.upper, dtype=np.float32) - self.margin_rad

    def clamp_absolute(self, target):
        return np.clip(_vector(target, "target"), self.safe_lower, self.safe_upper)

    def clamp_relative(self, target, current):
        target = _vector(target, "target")
        current = _vector(current, "current joint position")
        if self.max_deviation_deg == 0:
            return self.clamp_absolute(target)
        deviation = float(np.deg2rad(self.max_deviation_deg))
        lower = np.maximum(self.safe_lower, current - deviation)
        upper = np.minimum(self.safe_upper, current + deviation)
        if np.any(lower > upper):
            index = int(np.flatnonzero(lower > upper)[0])
            raise ValueError(f"no safe target window for {config.JOINT_NAMES[index]}")
        return np.clip(target, lower, upper)

    def apply(self, target, previous, current, dt):
        return self.apply_with_trace(target, previous, current, dt)[0]

    def apply_with_trace(self, target, previous, current, dt):
        # Preserve the existing absolute -> slew -> feedback-window order.
        raw = _vector(target, "target").copy()
        previous = _vector(previous, "previous target").copy()
        current = _vector(current, "current joint position").copy()
        absolute = self.clamp_absolute(raw)
        slew = limit_target_slew(absolute, previous, dt, self.max_speed_rad_s)
        final = self.clamp_relative(slew, current)
        trace = TargetTrace(raw, absolute.copy(), slew.copy(), final.copy(), previous, current,
                            absolute != raw, slew != absolute, final != slew, float(dt))
        return final, trace

    @classmethod
    def from_args(cls, args):
        lower, upper = config.Q_LOWER, config.Q_UPPER
        limits_path = getattr(args, "joint_limits_json", None)
        if limits_path:
            try:
                payload = json.loads(Path(limits_path).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read joint limits JSON {limits_path}: {exc}") from exc
            if (not isinstance(payload, dict)
                    or set(payload) != {"joint_names", "lower_rad", "upper_rad"}
                    or payload["joint_names"] != list(config.JOINT_NAMES)):
                raise ValueError("joint limits JSON must name all 12 config.JOINT_NAMES in order "
                                 "and contain only joint_names/lower_rad/upper_rad")
            for key in ("lower_rad", "upper_rad"):
                if (not isinstance(payload[key], list)
                        or any(type(x) not in (int, float) for x in payload[key])):
                    raise ValueError(f"joint limits {key} must contain numeric angles")
            lower, upper = payload["lower_rad"], payload["upper_rad"]
        return cls(lower=lower, upper=upper,
                   margin_rad=getattr(args, "joint_limit_margin_rad", config.JOINT_LIMIT_MARGIN_RAD),
                   max_speed_rad_s=getattr(args, "max_target_speed_rad_s", config.MAX_TARGET_SPEED_RAD_S),
                   max_deviation_deg=getattr(args, "max_target_deviation_deg", config.MAX_TARGET_DEVIATION_DEG))

    def describe(self):
        return (f"Target limits: margin={self.margin_rad:g} rad; "
                f"slew={self.max_speed_rad_s:g} rad/s (0=off); "
                f"feedback-window={self.max_deviation_deg:g} deg (0=off); "
                f"absolute-lower={list(self.lower)}; absolute-upper={list(self.upper)}")


def add_target_safety_arguments(parser):
    parser.add_argument("--max-target-speed-rad-s", type=float,
                        default=config.MAX_TARGET_SPEED_RAD_S,
                        help="Joint target slew limit in rad/s; 0 disables it")
    parser.add_argument("--max-target-deviation-deg", type=float,
                        default=config.MAX_TARGET_DEVIATION_DEG,
                        help="Joint target window around feedback in degrees; 0 disables it")
    parser.add_argument("--joint-limit-margin-rad", type=float,
                        default=config.JOINT_LIMIT_MARGIN_RAD,
                        help="Margin inside absolute joint bounds, radians; 0 removes only the margin")
    parser.add_argument("--joint-limits-json",
                        help="Optional 12-joint absolute bounds JSON in policy coordinates and radians")
