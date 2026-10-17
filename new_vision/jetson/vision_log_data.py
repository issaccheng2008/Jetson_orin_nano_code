"""Small, offline frame summaries; no vision/robot/GUI dependencies."""
from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class VisionFrame:
    frame: int
    host_time_ns: int | None
    elapsed_s: float
    heading_deg: float | None
    filtered_heading_deg: float | None
    line_state: str
    vision_wz: float | None
    robot_wz: float | None
    robot_status: str
    rowhdg_deg: float | None = None
    rowhdg_valid: bool | None = None
    demand_deg: float | None = None


@dataclass(frozen=True, slots=True)
class VisionLog:
    path: Path
    frames: tuple[VisionFrame, ...]
    skipped_rows: int
    clock_adjustments: int


def discover_logs(folder):
    """Find only line telemetry, never unrelated video or control indexes."""
    folder = Path(folder).expanduser().resolve()
    if folder.is_file():
        if folder.suffix.lower() != '.jsonl':
            raise ValueError('请选择 line_frames.jsonl 或包含它的当次日志文件夹')
        return (folder,)
    if not folder.is_dir():
        raise FileNotFoundError(f'日志目录不存在：{folder}')
    paths = tuple(sorted(folder.rglob('line_frames.jsonl')))
    if not paths:
        raise FileNotFoundError(f'没有找到 line_frames.jsonl：{folder}')
    return paths


def line_state(row, measurement):
    if measurement.get('startup_card_only'):
        return '起步等待'
    reason = str(measurement.get('steering_reason', ''))
    valid = measurement.get('measurement_valid')
    if valid is None and finite(row.get('confidence')) is not None:
        valid = float(row['confidence']) > 0
    if (valid is False or measurement.get('measurement_stale')
            or (finite(measurement.get('lost_frames')) or 0) > 0
            or reason.startswith('loss_')
            or reason in ('brief_loss_hold', 'geometry_lost_yaw_zero', 'invalid_clock')):
        return '丢线'
    mode = measurement.get('tracking_mode')
    if mode == 'LEFT_ONLY':
        return '左单线'
    if mode == 'RIGHT_ONLY':
        return '右单线'
    if mode == 'VISIBLE_ONLY' or measurement.get('single_line'):
        return '单线'
    left, right = measurement.get('left_seen'), measurement.get('right_seen')
    if left and right:
        return '双线'
    if left:
        return '左单线'
    if right:
        return '右单线'
    return '双线' if mode in ('BOTH', 'PAIRED') else '未记录'


def _angles(measurement, state):
    if state in ('丢线', '起步等待'):
        return None, None
    heading = finite(measurement.get('steering_heading_deg'))
    if heading is None and measurement.get('heading_control_valid'):
        heading = finite(measurement.get('heading_control_deg'))
    if heading is None and measurement.get('single_edge_valid'):
        heading = finite(measurement.get('single_edge_heading_deg'))
    filtered = finite(measurement.get('steering_filtered_heading_deg'))
    if filtered is None:
        filtered = finite(measurement.get('steering_filter_heading_deg'))
    return heading, filtered


def load_log(path):
    """Stream JSONL into compact summaries, reporting malformed/truncated rows."""
    path = Path(path).expanduser().resolve()
    frames = []
    skipped = adjustments = 0
    origin = clock_key = None
    statuses = {'written': '已采用', 'unknown': '未知/过期',
                'not_written': '未发送', 'motors_off': '电机未使能'}
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get('schema', 'line_frames_v1') != 'line_frames_v1':
                    raise ValueError('not line telemetry')
                frame = row.get('frame')
                if type(frame) is not int:
                    raise ValueError('missing frame number')
                host = finite(row.get('host_time_ns'))
                mono = finite(row.get('process_monotonic_s'))
                key = clock_key or ('process_monotonic_s' if mono is not None else 'host_time_ns')
                stamp = mono if key == 'process_monotonic_s' else host / 1e9 if host is not None else None
                if stamp is None:
                    raise ValueError('missing timestamp')
                measurement = row.get('measurement', {})
                if not isinstance(measurement, dict):
                    raise ValueError('invalid measurement')
            except (ValueError, TypeError, OverflowError):
                skipped += 1
                continue
            if origin is None:
                origin, clock_key = stamp, key
            elapsed = stamp - origin
            if frames and elapsed < frames[-1].elapsed_s:
                elapsed = frames[-1].elapsed_s
                adjustments += 1
            state = line_state(row, measurement)
            heading, filtered = _angles(measurement, state)
            recorded_status = row.get('robot_cmd_status')
            status = statuses.get(recorded_status, '未记录') if isinstance(recorded_status, str) else '未记录'
            robot_wz = finite(row.get('robot_cmd_wz')) if status == '已采用' else None
            if status == '已采用' and robot_wz is None:
                status = '未知/过期'
            host_ns = (row['host_time_ns'] if type(row.get('host_time_ns')) is int
                       else int(host) if host is not None else None)
            frames.append(VisionFrame(frame, host_ns,
                elapsed, heading, filtered, state, finite(row.get('wz')), robot_wz, status,
                finite(measurement.get('rowhdg_deg')),
                measurement.get('rowhdg_valid') if type(measurement.get('rowhdg_valid')) is bool else None,
                finite(measurement.get('steering_demand_deg'))
                if state not in ('丢线', '起步等待') else None))
    if not frames:
        raise ValueError(f'日志没有有效的视觉帧（跳过 {skipped} 行）：{path}')
    return VisionLog(path, tuple(frames), skipped, adjustments)


class FrameCursor:
    """Selection and 20-record window shared by slider, row click and D/F."""
    def __init__(self, frames):
        if not frames:
            raise ValueError('至少需要一帧')
        self.frames = frames
        self.times = tuple(frame.elapsed_s for frame in frames)
        self.index = 0

    def select(self, index):
        self.index = max(0, min(len(self.frames) - 1, int(index)))

    def step(self, delta):
        self.select(self.index + delta)

    def seek_time(self, elapsed_s):
        right = bisect_left(self.times, float(elapsed_s))
        if 0 < right < len(self.times):
            if elapsed_s - self.times[right - 1] <= self.times[right] - elapsed_s:
                right -= 1
        self.select(right)

    def window(self):
        start = max(0, min(self.index - 10, len(self.frames) - 20))
        return start, min(start + 20, len(self.frames))
