"""50 Hz encoder teaching and deterministic, linear joint-space playback.

All saved angles use the STM32 *wire* order, never config.JOINT_NAMES (whose
historical policy labels differ). No policy, IMU, or ONNX runtime is required.
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np

import config
from motor_test_common import STM32_BRIDGE_JOINT_NAMES, monotonic_us
from protocol import (COMMAND_ENABLE, COMMAND_ESTOP, COMMAND_TEACHING,
                      STATE_ENCODERS_VALID, STATE_FAULT, STATE_TEACHING_READY)

DT = 0.02
NAMES = list(STM32_BRIDGE_JOINT_NAMES)
FORMAT = "humanoid-teaching-v1"


def vector(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (12,) or not np.isfinite(q).all():
        raise ValueError("需要 12 个有限的关节角度 (rad)")
    return q.copy()


def positive(value, name, minimum=DT):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} 必须为有限数且 >= {minimum:g}")
    return value


def metadata(kind):
    return dict(format=FORMAT, kind=kind, joint_names=NAMES, units="rad",
                coordinate_system="stm32_model_wire", nominal_hz=50,
                created_at=datetime.now(timezone.utc).isoformat())


def validate_metadata(data, kind):
    if (not isinstance(data, dict) or data.get("format") != FORMAT or data.get("kind") != kind
            or data.get("joint_names") != NAMES or data.get("units") != "rad"
            or data.get("coordinate_system") != "stm32_model_wire"):
        raise ValueError("文件格式、关节顺序或坐标系不匹配")


class TeachingStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name, suffix=None):
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]{1,120}", name) or name.startswith("."):
            raise ValueError("文件名只允许字母、数字、中文、下划线、点和短横线")
        path = (self.root / name).resolve()
        if path.parent != self.root or (suffix and path.suffix != suffix):
            raise ValueError("不允许此文件路径或扩展名")
        return path

    def new_name(self, kind, suffix):
        return f"{kind}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}{suffix}"

    def save_checkpoints(self, name, points):
        if not isinstance(points, list) or len(points) > 2000:
            raise ValueError("checkpoint 列表最多 2000 项")
        clean = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("每个 checkpoint 必须为 JSON 对象")
            clean.append(dict(label=str(point.get("label", ""))[:100],
                              position_rad=vector(point["position_rad"]).tolist(),
                              duration_s=positive(point.get("duration_s", 2), "区间时间")))
        result = dict(metadata("checkpoints"), checkpoints=clean)
        path = self.path(name, ".json")
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        return result

    def checkpoints(self, name):
        data = json.loads(self.path(name, ".json").read_text(encoding="utf-8"))
        validate_metadata(data, "checkpoints")
        points = data["checkpoints"]
        if not isinstance(points, list) or len(points) > 2000:
            raise ValueError("checkpoint 列表不合法")
        for p in points:
            if not isinstance(p, dict):
                raise ValueError("每个 checkpoint 必须为 JSON 对象")
            vector(p["position_rad"])
            positive(p["duration_s"], "区间时间")
        return data

    def recording(self, name):
        path = self.path(name, ".jsonl")
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("单次回放文件不能大于 128 MiB；请拆分录制")
        with path.open(encoding="utf-8") as f:
            validate_metadata(json.loads(next(f)), "recording")
            samples = [json.loads(line) for line in f if line.strip()]
        if len(samples) < 2:
            raise ValueError("至少需要两帧记录")
        times = [float(s["t_s"]) for s in samples]
        if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("记录时间必须有限且严格递增")
        if any(b - a > 0.10 + 1e-6 for a, b in zip(times, times[1:])):
            raise ValueError("记录中有超过 100 ms 的反馈缺口，拒绝连续回放")
        return [t - times[0] for t in times], [vector(s["position_rad"]) for s in samples]

    def files(self):
        return [dict(name=p.name, bytes=p.stat().st_size)
                for p in sorted(self.root.iterdir(), reverse=True)
                if p.suffix in (".json", ".jsonl") and p.is_file() and not p.is_symlink()]


class Trajectory:
    def __init__(self, times, positions):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray([vector(q) for q in positions])
        if (len(times) < 2 or len(times) != len(positions) or times[0] != 0
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError("轨迹必须从 0 开始，包含至少两个严格递增的时间点")

    def validate(self, speed):
        # Refuse, rather than silently clip and distort the taught movement.
        if (np.any(self.positions < config.Q_LOWER + config.JOINT_LIMIT_MARGIN_RAD)
                or np.any(self.positions > config.Q_UPPER - config.JOINT_LIMIT_MARGIN_RAD)):
            raise ValueError("轨迹超出 config.py 的关节限位，拒绝回放")
        speeds = np.abs(np.diff(self.positions, axis=0)) / np.diff(self.times)[:, None]
        if np.any(speeds > speed + 1e-6):
            raise ValueError(f"区间速度超过 {speed:g} rad/s；请延长区间时间或降低回放速度")

    def at(self, elapsed):
        if elapsed >= self.times[-1]:
            return self.positions[-1].copy()
        i = max(0, bisect.bisect_right(self.times, elapsed) - 1)
        f = np.clip((elapsed - self.times[i]) / (self.times[i + 1] - self.times[i]), 0, 1)
        return self.positions[i] + f * (self.positions[i + 1] - self.positions[i])


class TeachingController:
    """One owner for all sends; web threads only change intent under the lock."""
    def __init__(self, link, store, *, enable_motors=False, clock=time.monotonic,
                 max_speed=config.MAX_TARGET_SPEED_RAD_S):
        self.link, self.store, self.clock = link, store, clock
        self.enable_motors = enable_motors
        self.max_speed = positive(max_speed, "最大速度", 0.001)
        self.lock = threading.RLock()
        self.mode = "idle"
        self.error = ""
        self.state = None
        self.state_time = None
        self.target = np.zeros(12)
        self.trajectory = None
        self.play_started = 0.0
        self.record_file = None
        self.record_name = ""
        self.record_count = 0
        self.record_missing = 0
        self.last_sample = None
        self.record_time_us = 0
        self.record_started = 0.0
        self.last_flush = 0.0
        self.checkpoint_name = ""
        self.points = []
        self.last_tick = None
        self.last_progress_key = None
        self.last_progress_time = 0.0
        self.session_started = False
        self.stop_event = threading.Event()
        self.thread = None

    def _fresh(self, require_raw=False):
        if self.state is None or self.state_time is None or self.clock() - self.state_time > 0.10:
            raise ValueError("反馈过期或尚未连接 STM32")
        flags = self.state.status_flags
        if not flags & STATE_TEACHING_READY:
            raise ValueError("STM32 尚未烧录支持示教的固件 (缺少 STATE_TEACHING_READY)")
        if flags & STATE_FAULT or not flags & STATE_ENCODERS_VALID:
            raise ValueError("STM32 电机故障或编码器未就绪")
        if self.clock() - self.last_progress_time > 0.10:
            raise ValueError("STM32 序号/时间戳停止更新")
        if require_raw and (not self.session_started or self.clock() - self.session_time < 0.10):
            raise ValueError("请先进入松弛模式并等待真实角度反馈")
        return vector(self.state.joint_position)

    def _can_enable(self):
        if not self.enable_motors:
            raise ValueError("启动参数缺少 --enable-motors，当前仅监控")
        if self.mode == "fault":
            raise ValueError("故障已锁存；检查后先执行 reset")

    def _hold(self):
        q = self._fresh(require_raw=True)
        self._validate_pose(q)
        self.target, self.mode, self.trajectory = q, "hold", None

    @staticmethod
    def _validate_pose(q):
        if (np.any(q < config.Q_LOWER + config.JOINT_LIMIT_MARGIN_RAD)
                or np.any(q > config.Q_UPPER - config.JOINT_LIMIT_MARGIN_RAD)):
            raise ValueError("当前位置超出关节限位，不能恢复 PD；请松弛并调整姿态")

    def execute(self, action, data=None):
        data = data or {}
        # Parse/load large files outside the control lock.
        loaded = None
        if action == "play-log":
            loaded = self.store.recording(data["name"])
        elif action in ("load-checkpoints", "play-checkpoints"):
            loaded = self.store.checkpoints(data["name"])
        with self.lock:
            if action == "release":
                self._can_enable()
                q = self._fresh()
                if self.mode == "playback":
                    raise ValueError("请先停止回放")
                self.target, self.mode, self.trajectory = q, "free", None
                if not self.session_started:
                    self.session_started = True
                    self.session_time = self.clock()
            elif action == "hold":
                self._can_enable()
                self._hold()
            elif action == "stop-playback":
                self._can_enable()
                self._hold()
            elif action in ("disable", "estop"):
                # An explicit stop must also stop the firmware's startup pose,
                # even when this server has so far only been monitoring it.
                # Never send this extension blindly to older reboot-on-stop firmware.
                if (not self.session_started and self.state is not None
                        and self.state.status_flags & STATE_TEACHING_READY):
                    self.session_started, self.session_time = True, self.clock()
                if action == "estop":
                    self.mode, self.error = "fault", "操作员急停；reset 后才能重新使能"
                elif self.mode != "fault":
                    self.mode, self.error = "disabled", ""
                self.trajectory = None
                self._stop_recording()
            elif action == "reset":
                self._fresh()
                self.mode, self.error, self.trajectory = "disabled", "", None
            elif action == "start-recording":
                self._can_enable()
                self._fresh(require_raw=True)
                if self.mode != "free":
                    raise ValueError("请先进入松弛模式，再开始手动示教记录")
                if self.record_file:
                    raise ValueError("正在记录，请先停止")
                name = self.store.new_name("recording", ".jsonl")
                f = self.store.path(name, ".jsonl").open("x", encoding="utf-8")
                try:
                    f.write(json.dumps(metadata("recording"), ensure_ascii=False) + "\n")
                    f.flush()
                except Exception:
                    f.close()
                    raise
                self.record_file, self.record_name = f, name
                self.record_count = self.record_missing = self.record_time_us = 0
                self.last_sample = None
                self.record_started = self.last_flush = self.clock()
            elif action == "stop-recording":
                self._stop_recording()
            elif action == "checkpoint":
                q = self._fresh(require_raw=True)
                points = self.points + [dict(label=str(data.get("label", f"姿态 {len(self.points)+1}")),
                                            position_rad=q.tolist(), duration_s=2.0)]
                name = self.checkpoint_name or self.store.new_name("checkpoints", ".json")
                self.store.save_checkpoints(name, points)
                self.points, self.checkpoint_name = points, name
            elif action == "new-checkpoints":
                self.points, self.checkpoint_name = [], ""
            elif action == "load-checkpoints":
                self.points, self.checkpoint_name = loaded["checkpoints"], data["name"]
            elif action == "save-checkpoints":
                name = data.get("name") or self.checkpoint_name or self.store.new_name("checkpoints", ".json")
                saved = self.store.save_checkpoints(name, data["checkpoints"])
                self.points, self.checkpoint_name = saved["checkpoints"], name
            elif action in ("play-log", "play-checkpoints"):
                self._can_enable()
                current = self._fresh(require_raw=True)
                if self.record_file or self.mode == "playback":
                    raise ValueError("请先停止记录或当前回放")
                if action == "play-log":
                    times, poses = loaded
                    speed = positive(data.get("speed", 1), "回放倍率", 0.05)
                    times = [t / speed for t in times]
                else:
                    points = loaded["checkpoints"]
                    if len(points) < 2:
                        raise ValueError("至少需要两个 checkpoint")
                    times, poses = [0.0], [vector(p["position_rad"]) for p in points]
                    for point in points[:-1]:
                        times.append(times[-1] + positive(point["duration_s"], "区间时间"))
                approach = positive(data.get("approach_s", 2), "到首帧时间")
                approach = max(approach, float(np.max(np.abs(poses[0] - current))) / self.max_speed)
                trajectory = Trajectory([0.0] + [approach + t for t in times], [current] + poses)
                trajectory.validate(self.max_speed)
                self.trajectory, self.play_started = trajectory, self.clock()
                self.mode, self.target = "playback", current
            else:
                raise ValueError(f"未知命令: {action}")
            return self.snapshot()

    def _stop_recording(self):
        if self.record_file:
            f, self.record_file = self.record_file, None
            try:
                f.flush()
                os.fsync(f.fileno())
            finally:
                f.close()

    def _record(self, frames):
        for state, received in frames:
            if not self.record_file or received < self.record_started:
                continue
            if state.status_flags & STATE_FAULT or not state.status_flags & STATE_ENCODERS_VALID:
                raise ValueError("记录中出现无效编码器或电机故障")
            key = state.sequence, state.timestamp_us
            if self.last_sample is not None:
                sequence_delta = (key[0] - self.last_sample[0]) & 0xffff
                delta_us = (key[1] - self.last_sample[1]) & 0xffffffff
                if sequence_delta == 0 or delta_us == 0:
                    continue  # Never invent 50 Hz samples by repeating stale feedback.
                if delta_us > 100000 or sequence_delta > 0x7fff:
                    raise ValueError("记录反馈中断超过 100 ms 或 STM32 重启，记录已停止")
                self.record_time_us += delta_us
                # Estimate missing 20 ms state slots; frame sequences also include action statuses.
                self.record_missing += max(0, round(delta_us / 20000) - 1)
            self.last_sample = key
            row = dict(t_s=self.record_time_us / 1e6, host_monotonic_s=received,
                       sequence=state.sequence, timestamp_us=state.timestamp_us,
                       position_rad=vector(state.joint_position).tolist(), status_flags=state.status_flags)
            self.record_file.write(json.dumps(row, allow_nan=False) + "\n")
            self.record_count += 1
        if self.record_file and self.clock() - self.last_flush >= 1:
            self.record_file.flush()
            self.last_flush = self.clock()

    def _send(self, flags, kp=0, kd=0):
        sent = self.link.send_command(monotonic_us(), self.target, kp, kd,
                                      flags | COMMAND_TEACHING, lock_timeout_s=0.02)
        if sent is False:
            raise RuntimeError("串口写入锁超时")

    def tick(self):
        with self.lock:
            now = self.clock()
            frames = self.link.drain_states()
            for state, received in frames:
                key = state.sequence, state.timestamp_us
                if self.last_progress_key is not None:
                    ds = (key[0] - self.last_progress_key[0]) & 0xffff
                    dt = (key[1] - self.last_progress_key[1]) & 0xffffffff
                    if self.session_started and (ds > 0x7fff or dt > 0x7fffffff):
                        self.state, self.state_time = state, received
                        self.last_progress_key, self.last_progress_time = key, received
                        raise ValueError("STM32 重启或反馈倒序")
                if key != self.last_progress_key and (self.last_progress_key is None or
                        (key[0] != self.last_progress_key[0] and key[1] != self.last_progress_key[1])):
                    self.last_progress_time = received
                    self.last_progress_key = key
                self.state, self.state_time = state, received
            if self.record_file and self.link.state_queue_drops:
                raise ValueError("反馈队列溢出，记录完整性丢失")
            if self.mode in ("free", "hold", "playback"):
                current = self._fresh()
                if self.last_tick is not None and now - self.last_tick > 0.08:
                    raise ValueError("控制循环延迟超过 80 ms，停止执行")
                self._record(frames)
                if self.mode == "free":
                    # User requirement: echo the latest received angles every 20 ms.
                    self.target = current
                    self._send(COMMAND_ENABLE)
                else:
                    if self.mode == "playback":
                        elapsed = max(0, now - self.play_started)
                        self.target = self.trajectory.at(elapsed)
                        if elapsed >= self.trajectory.times[-1]:
                            self.mode, self.trajectory = "hold", None
                    self._validate_pose(self.target)
                    if np.max(np.abs(self.target - current)) > config.MAX_TARGET_DEVIATION_RAD:
                        raise ValueError("跟踪误差超过配置窗口，已停止输出")
                    kp, kd = config.motion_gain_scales()
                    self._send(COMMAND_ENABLE, kp, kd)
            elif self.session_started:
                self._send(COMMAND_ESTOP if self.mode == "fault" else 0)
            self.last_tick = now

    def fail(self, exc):
        with self.lock:
            self.mode, self.error, self.trajectory = "fault", str(exc), None
            try:
                self._stop_recording()
            except OSError as disk_error:
                self.error += f"; 关闭记录失败: {disk_error}"
            if self.session_started:
                try:
                    self._send(COMMAND_ESTOP)
                except Exception:
                    pass

    def snapshot(self):
        with self.lock:
            age = None if self.state_time is None else max(0, self.clock() - self.state_time)
            q = None if self.state is None else np.asarray(self.state.joint_position, dtype=float)
            if q is not None and not np.isfinite(q).all():
                q = None
            return dict(mode=self.mode, error=self.error, enable_motors=self.enable_motors,
                        joint_names=NAMES, position_rad=None if q is None else q.tolist(),
                        target_rad=self.target.tolist(), state_age_s=age,
                        status_flags=0 if self.state is None else self.state.status_flags,
                        firmware_ready=bool(self.state and self.state.status_flags & STATE_TEACHING_READY),
                        recording=bool(self.record_file), record_name=self.record_name,
                        record_count=self.record_count, record_missing=self.record_missing,
                        checkpoint_name=self.checkpoint_name, checkpoints=list(self.points),
                        playback_elapsed_s=max(0, self.clock()-self.play_started) if self.trajectory else 0,
                        playback_total_s=float(self.trajectory.times[-1]) if self.trajectory else 0)

    def start(self):
        def run():
            deadline = self.clock()
            while not self.stop_event.is_set():
                try:
                    self.tick()
                except Exception as exc:
                    self.fail(exc)
                deadline += DT
                delay = deadline - self.clock()
                if delay < 0:
                    deadline = self.clock()
                self.stop_event.wait(max(0, delay))
        self.thread = threading.Thread(target=run, name="teaching-50hz", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        with self.lock:
            try:
                self._stop_recording()
            finally:
                if self.session_started:
                    for _ in range(5):
                        try:
                            self._send(COMMAND_ESTOP)
                        except Exception:
                            pass
                        time.sleep(0.01)
                self.link.close()
