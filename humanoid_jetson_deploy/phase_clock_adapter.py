"""Validate and load the exported crossing actor and its shared command clock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import config
from models.phase_clock_model_850.deployment.phase_clock import PhaseClockConfig
from models.phase_clock_model_850.deployment.policy_interface import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    JOINT_NAMES,
    PolicyController,
)


DEFAULT_BUNDLE = Path(__file__).resolve().parent / "models" / "phase_clock_model_850"


def load_crossing_controller(bundle: str | Path = DEFAULT_BUNDLE) -> PolicyController:
    """Reject a mixed bundle or incompatible robot mapping before opening motors."""
    root = Path(bundle)
    manifest = json.loads((root / "manifest_sha256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Crossing bundle file missing or checksum mismatch: {name}")

    contract = json.loads((root / "policy_contract.json").read_text(encoding="utf-8"))
    clock_values = json.loads((root / "phase_clock.json").read_text(encoding="utf-8"))
    if contract["onnx_sha256"] != manifest["policy.onnx"]:
        raise ValueError("Crossing ONNX digest differs from its policy contract")
    if contract["deployment_control"]["command_mode"] != "phase_clock":
        raise ValueError("Crossing bundle is not configured for phase-clock deployment")
    if clock_values != contract["deployment_control"]["phase_clock"]:
        raise ValueError("Clock JSON differs from the ONNX policy contract")
    if tuple(contract["joint_names"]) != JOINT_NAMES or JOINT_NAMES != config.JOINT_NAMES:
        raise ValueError("Crossing joint order differs from the motor mapping")
    if not np.allclose(DEFAULT_JOINT_POS, config.Q_DEFAULT, atol=1e-7):
        raise ValueError("Crossing default pose differs from the motor mapping")
    if not np.allclose(contract["default_joint_positions_rad"], config.Q_DEFAULT, atol=1e-7):
        raise ValueError("Crossing contract default pose differs from the motor mapping")
    if ACTION_SCALE != config.ACTION_SCALE or contract["action_scale"] != ACTION_SCALE:
        raise ValueError("Crossing action scale differs from the motor mapping")

    clock = PhaseClockConfig(**clock_values)
    if not np.isclose(clock.control_dt, config.POLICY_DT, atol=1e-12):
        raise ValueError("Crossing clock period differs from the 50 Hz motor loop")
    return PolicyController.from_onnx(root / "policy.onnx", clock)
