"""Separate sent motor-target and received STATE-angle CSVs for one-foot runs."""

from __future__ import annotations

from contextlib import ExitStack
import csv
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


class OneFootAngleCsvLogger:
    """Write paired control-tick samples in STM32 wire order and radians."""

    def __init__(self, log_dir: str | Path, joint_names: Sequence[str]) -> None:
        self.joint_names = tuple(joint_names)
        directory = Path(log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.target_path = directory / f"onefoot_targets_{timestamp}.csv"
        self.feedback_path = directory / f"onefoot_feedback_{timestamp}.csv"
        self._files = ExitStack()
        try:
            target_file = self._files.enter_context(
                self.target_path.open("x", encoding="utf-8", newline="", buffering=1))
            feedback_file = self._files.enter_context(
                self.feedback_path.open("x", encoding="utf-8", newline="", buffering=1))
            self._target_writer = csv.writer(target_file)
            self._feedback_writer = csv.writer(feedback_file)
            keys = ["host_time_iso", "elapsed_s", "step", "state_sequence"]
            self._target_writer.writerow(keys + [f"target_{name}_rad" for name in self.joint_names])
            self._feedback_writer.writerow(keys + [f"actual_{name}_rad" for name in self.joint_names])
        except Exception:
            self.close()
            raise

    def write(self, elapsed_s: float, step: int, state_sequence: int,
              target_motor_rad: np.ndarray, actual_motor_rad: np.ndarray) -> None:
        # Validate both vectors before adding either row.
        target = np.asarray(target_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        actual = np.asarray(actual_motor_rad, dtype=np.float32).reshape(len(self.joint_names))
        keys = [
            datetime.now().astimezone().isoformat(timespec="milliseconds"),
            f"{elapsed_s:.6f}", int(step), int(state_sequence),
        ]
        self._target_writer.writerow(keys + [f"{value:.8f}" for value in target])
        self._feedback_writer.writerow(keys + [f"{value:.8f}" for value in actual])

    def close(self) -> None:
        self._files.close()
