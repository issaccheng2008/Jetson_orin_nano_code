"""Validated, finite sequence of 50 Hz joint targets in policy coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config


class FixedJointPolicy:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read fixed policy {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != "joint_frames_v1":
            raise ValueError("fixed policy format must be joint_frames_v1")
        if type(payload.get("hz")) not in (int, float) or payload["hz"] != config.POLICY_HZ:
            raise ValueError(f"fixed policy hz must be {config.POLICY_HZ:g}")
        if payload.get("joint_names") != list(config.JOINT_NAMES):
            raise ValueError("fixed policy joint_names must match config.JOINT_NAMES in order")
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("fixed policy frames must be a nonempty list")
        try:
            values = np.asarray(frames, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"fixed policy frames contain invalid numbers: {exc}") from exc
        if values.shape != (len(frames), config.NUM_JOINTS):
            raise ValueError(f"fixed policy frames must each have {config.NUM_JOINTS} angles")
        if not np.isfinite(values).all():
            raise ValueError("fixed policy frames must contain finite angles")
        lower = config.Q_LOWER + config.JOINT_LIMIT_MARGIN_RAD
        upper = config.Q_UPPER - config.JOINT_LIMIT_MARGIN_RAD
        if np.any((values < lower) | (values > upper)):
            raise ValueError("fixed policy frame angle exceeds a joint limit")
        self.frames = values
        self.index = 0

    def next_target(self) -> np.ndarray | None:
        if self.index >= len(self.frames):
            return None
        target = self.frames[self.index].copy()
        self.index += 1
        return target
