"""Independent per-run target trace CSV and provenance manifest."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

import config
from walking_command_hold import MIN_WALKING_COMMAND_HOLD_S

SCHEMA = "control_target_trace_v3"
SCALARS = ["schema", "run_id", "row_index", "step", "host_unix_s", "read_monotonic_s", "elapsed_s",
           "policy_mode", "target_source", "phase", "phase_tick", "dt_s", "infer_ms",
           "state_sequence", "state_timestamp_us", "state_flags", "state_receive_monotonic_s",
           "state_receive_age_s", "state_receive_metadata_available", "held_reference",
           "card_tilt_active", "upright_hold", "command_flags", "command_timestamp_us",
           "send_result", "send_error", "send_done_monotonic_s", "obs_dim", "observation_available", "cmd_vx", "cmd_vy",
           "cmd_wz", "lift_command", "command_step_distance_m", "command_crossing", "support_foot"]
SCALARS += ["requested_cmd_vx", "requested_cmd_vy", "requested_cmd_wz", "command_hold_remaining_s", "command_hold_reason"]
JOINT_GROUPS = ["raw_action", "raw_target_policy", "absolute_target_policy", "slew_target_policy",
                "final_target_policy", "motor_send_argument", "previous_final_policy", "reference_q_policy",
                "reference_qd_policy", "received_q_packet", "received_qd_packet", "received_q_policy",
                "received_qd_policy", "absolute_clip_mask", "slew_clip_mask", "window_clip_mask"]
IMU_GROUPS = ["received_accel", "received_gyro", "reference_accel_policy", "reference_gyro_policy",
              "reference_gravity_policy"]
HEADER = (SCALARS + [f"{group}_{name}" for group in JOINT_GROUPS for name in config.JOINT_NAMES]
          + [f"{group}_{axis}" for group in IMU_GROUPS for axis in "xyz"]
          + [f"received_orientation_{axis}" for axis in "wxyz"]
          + [f"observation_{i}" for i in range(49)]
          + ["stm32_command_rx_count", "stm32_system_control_cycle"])


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def source_identity(path, role):
    path = Path(path)
    if not path.is_file():
        return dict(role=role, path=str(path), sha256=None, status="source_file_missing")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(role=role, path=str(path.resolve()), sha256=digest.hexdigest(), status="hashed_at_run_start")


def state_snapshot(state):
    return dict(sequence=int(state.sequence), timestamp_us=getattr(state, "timestamp_us", None),
                flags=int(state.status_flags),
                command_rx_count=getattr(state, "command_rx_count", None),
                system_control_cycle=getattr(state, "system_control_cycle", None),
                q_packet=np.asarray(state.joint_position).tolist(),
                qd_packet=np.asarray(state.joint_velocity).tolist(),
                accel_packet=np.asarray(state.accel_m_s2).tolist(), gyro_packet=np.asarray(state.gyro_rad_s).tolist(),
                orientation_wxyz=np.asarray(state.orientation_wxyz).tolist())


def receive_metadata(link, state, read_monotonic_s):
    # Other transports/test doubles may not provide host receive metadata. This
    # is explicit in every row and must not be mistaken for zero age.
    method = getattr(link, "get_state_receive_info", None)
    if callable(method):
        info = method(state, read_monotonic_s)
        if isinstance(info, dict):
            return info
    return dict(state_receive_monotonic_s=None, state_receive_age_s=None,
                state_receive_metadata_available=False)


def add_diagnostic_arguments(parser):
    parser.add_argument("--diagnostic-log-dir", type=Path,
                        help="Detailed target CSV/manifest directory; default: old log directory/control_diagnostics")
    parser.add_argument("--no-control-diagnostics", action="store_true",
                        help="Explicitly disable the independent detailed target trace log")


class ControlDiagnostics:
    def __init__(self, directory, args, limits, sources, initial_state, entrypoint, associated_log):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex
        self.csv_path = directory / f"control_trace_{self.run_id}.csv"
        self.manifest_path = directory / f"control_trace_{self.run_id}.manifest.json"
        sources = dict(sources)
        bounds_source = getattr(args, "joint_limits_json", None)
        if bounds_source:
            sources["absolute_joint_limits_json"] = bounds_source
        identities = [source_identity(path, role) for role, path in sources.items() if path]
        self.manifest = dict(schema=SCHEMA, run_id=self.run_id, entrypoint=entrypoint,
            argv=list(sys.argv), arguments=_json_value(vars(args)), created_utc=datetime.now(timezone.utc).isoformat(),
            sources=identities, firmware_sha256="unknown_not_reported_by_protocol",
            initial_state=state_snapshot(initial_state), associated_legacy_log=str(associated_log),
            target_limits=dict(lower_rad=limits.lower, upper_rad=limits.upper, margin_rad=limits.margin_rad,
                               max_speed_rad_s=limits.max_speed_rad_s, max_deviation_deg=limits.max_deviation_deg,
                               absolute_bounds_source=str(bounds_source) if bounds_source else "config.Q_LOWER/Q_UPPER"),
            joint_mapping=dict(policy_joint_names=list(config.JOINT_NAMES), motor_sign=config.MOTOR_SIGN.tolist(),
                               motor_zero_rad=config.MOTOR_ZERO_RAD.tolist(), imu_to_policy=config.IMU_TO_POLICY.tolist(),
                               q_default=config.Q_DEFAULT.tolist(), action_scale=config.ACTION_SCALE,
                               physical_mapping="not_verified; packet order follows Nano config"),
            observation_schemas=dict(walking49="acc3*0.1,gyro3,gravity3,vx,wz,step,crossing,qrel12,qd12,last_action12",
                                     onefoot46="acc3*0.1,gyro3,gravity3,lift,qrel12,qd12,last_action12",
                                     phase_clock49="same slices as walking49; phase-clock supplies vx,wz,step,crossing"),
            csv_columns=HEADER, received_data_semantics="STM32-processed packet; not raw physical encoder truth",
            raw_action_semantics="Canonical ONNX output; left-support onefoot maps action to -concat(action[6:],action[:6]) before raw_target_policy",
            walking_command_contract=dict(minimum_hold_s=MIN_WALKING_COMMAND_HOLD_S, timing="first actual walking model use; host monotonic clock",
                exceptions="all-zero stop, safety/shape stop, upright/onefoot takeover, fault/disable/exit clear the hold",
                requested_command="source value before local safety overrides; cmd_vx/vy/wz is actually used walking input"),
            command_semantics="motor_send_argument is the exact 12-angle argument to SerialLink, not an STM32 execution acknowledgement",
            status="running", rows_written=0)
        with self.manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(_json_value(self.manifest), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        self.file = self.csv_path.open("x", encoding="utf-8", newline="", buffering=65536)
        self.writer = csv.writer(self.file)
        self.writer.writerow(HEADER)
        self.file.flush()
        self.rows_written = 0
        self._closed = False
        print(f"Control diagnostic run={self.run_id}: {self.csv_path}; manifest={self.manifest_path}")
        for source in identities:
            if source["sha256"] is None:
                print(f"Control diagnostic SHA unavailable: {source['role']} {source['path']} ({source['status']})")

    def write(self, *, state, trace, motor_target, action, reference_qd, reference_accel, reference_gyro,
              reference_gravity, observation, receive_info=None, velocity_command=None, **scalars):
        record = dict(schema=SCHEMA, run_id=self.run_id, row_index=self.rows_written, host_unix_s=time.time(),
                      state_sequence=int(state.sequence), state_timestamp_us=getattr(state, "timestamp_us", None),
                      state_flags=int(state.status_flags), dt_s=trace.dt if trace is not None else None,
                      stm32_command_rx_count=getattr(state, "command_rx_count", None),
                      stm32_system_control_cycle=getattr(state, "system_control_cycle", None))
        record.update(receive_info or {})
        record.update(scalars)
        if trace is not None and motor_target is not None:
            if not np.array_equal(np.asarray(motor_target), config.policy_to_motor_position(trace.final_target)):
                raise ValueError("Diagnostic final target differs from the motor send argument")
        groups = dict(raw_action=action, motor_send_argument=motor_target, reference_qd_policy=reference_qd,
                      received_q_packet=state.joint_position, received_qd_packet=state.joint_velocity,
                      received_q_policy=config.motor_to_policy_position(state.joint_position),
                      received_qd_policy=config.motor_to_policy_velocity(state.joint_velocity))
        if trace is not None:
            groups.update(raw_target_policy=trace.raw_target, absolute_target_policy=trace.absolute_target,
                          slew_target_policy=trace.slew_target, final_target_policy=trace.final_target,
                          previous_final_policy=trace.previous_target, reference_q_policy=trace.reference_q,
                          absolute_clip_mask=trace.absolute_mask, slew_clip_mask=trace.slew_mask, window_clip_mask=trace.window_mask)
        for group, value in groups.items():
            self._put_vector(record, group, value, config.JOINT_NAMES)
        for group, value in dict(received_accel=state.accel_m_s2, received_gyro=state.gyro_rad_s,
                reference_accel_policy=reference_accel, reference_gyro_policy=reference_gyro,
                reference_gravity_policy=reference_gravity).items():
            self._put_vector(record, group, value, "xyz")
        self._put_vector(record, "received_orientation", state.orientation_wxyz, "wxyz")
        if velocity_command is not None:
            value = np.asarray(velocity_command)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Diagnostic velocity command must be three finite values")
            record.update(zip(("cmd_vx", "cmd_vy", "cmd_wz"), value.tolist()))
        observation = np.asarray(observation) if observation is not None else np.array([])
        if observation.ndim != 1 or observation.size not in (0, 46, 49) or not np.isfinite(observation).all():
            raise ValueError("Diagnostic observation must have actual ONNX width 46/49 or be absent")
        record["obs_dim"] = int(observation.size)
        record["observation_available"] = int(observation.size != 0)
        record.update({f"observation_{i}": float(value) for i, value in enumerate(observation)})
        unexpected = set(record) - set(HEADER)
        if unexpected:
            raise ValueError(f"Unexpected diagnostic fields: {sorted(unexpected)}")
        for name, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f"Diagnostic {name} must be finite or absent")
        row = [record.get(name, "") if record.get(name) is not None else "" for name in HEADER]
        if self._closed:
            raise RuntimeError("Diagnostic log is already closed")
        self.writer.writerow(row)
        self.rows_written += 1
        # Keep the last 100 ms of counters available after a serial write stall.
        # This is a userspace flush, not a hard realtime or disk durability guarantee.
        if self.rows_written % 5 == 0 or record.get("send_result") in ("link_lost", "error"):
            self.file.flush()

    @staticmethod
    def _put_vector(record, prefix, value, labels):
        if value is None:
            return
        value = np.asarray(value)
        if value.shape != (len(labels),) or not np.isfinite(value).all():
            raise ValueError(f"Diagnostic {prefix} must have {len(labels)} finite values")
        for label, number in zip(labels, value):
            record[f"{prefix}_{label}"] = int(number) if value.dtype == bool else float(number)

    def close(self, status="completed", error=None):
        if self._closed:
            return
        try:
            self.file.flush()
        finally:
            self.file.close()
            self._closed = True
        self.manifest.update(status=status, error=error, rows_written=self.rows_written)
        # Only update this run's unique manifest. No earlier run is opened.
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(self.manifest), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
