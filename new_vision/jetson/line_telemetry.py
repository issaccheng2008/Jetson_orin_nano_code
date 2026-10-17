"""Buffered scalar telemetry for every processed vision frame (no image arrays)."""
import json
import math
import numbers
import os
from pathlib import Path
import time
import numpy as np


COUNT_FIELDS = (
    'steering_paired_count', 'steering_single_count',
    'card_stop_box_hit_count', 'card_stop_cue_hit_count',
)


def execution_log_fields(executed):
    """Flatten fresh policy feedback; this is adopted yaw, not measured body yaw."""
    fields = dict(robot_cmd_wz=None, robot_cmd_status='unknown',
                  robot_cmd_host_unix_s=None, robot_cmd_monotonic_s=None,
                  robot_cmd_step=None, robot_cmd_policy_mode=None,
                  robot_cmd_enabled=None)
    if not isinstance(executed, dict):
        return fields
    result = executed.get('send_result')
    if result not in ('written', 'not_written', 'unknown'):
        return fields
    fields.update(robot_cmd_status=result,
        robot_cmd_host_unix_s=executed.get('host_unix_s'),
        robot_cmd_monotonic_s=executed.get('monotonic_s'),
        robot_cmd_step=executed.get('step'),
        robot_cmd_policy_mode=executed.get('policy_mode'),
        robot_cmd_enabled=executed.get('enabled'))
    if result == 'written':
        if executed.get('enabled') is not True:
            fields['robot_cmd_status'] = 'motors_off'
        else:
            try:
                wz = float(executed['velocity'][2])
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                wz = math.nan
            if math.isfinite(wz):
                fields['robot_cmd_wz'] = wz
            else:
                fields['robot_cmd_status'] = 'unknown'
    return fields


class LineTelemetry:
    def __init__(self, run_directory):
        self.path = Path(run_directory) / 'line_frames.jsonl'
        self.counts_path = self.path.with_name('counts_summary.txt')
        self._file = self.path.open('x', encoding='utf-8', buffering=65536)
        self._pending = 0
        self._previous_write_ms = None
        self._latest_counts = None
        self._last_counts_write_s = -math.inf

    def _write_counts_summary(self):
        if self._latest_counts is None:
            return
        row = self._latest_counts
        paired = row['steering_paired_count']
        single = row['steering_single_count']
        box = row['card_stop_box_hit_count']
        cue = row['card_stop_cue_hit_count']
        steering_total = paired + single
        card_total = box + cue
        stop_box = row.get('card_stop_box_trigger_count')
        stop_cue = row.get('card_stop_cue_trigger_count')
        updated = time.strftime('%Y-%m-%d %H:%M:%S %z',
                                time.localtime(row['host_time_ns'] / 1e9))
        lines = [
            '巡线与图卡停车证据计数（当前运行累计）',
            f'运行目录: {self.path.parent.name}',
            f'最后记录: {updated}，视觉帧 #{row["frame"]}',
            '',
            '巡线指令来源（帧；仅正常行走且控制器使用有效路径）',
            f'  双边路径: {paired}',
            f'  单边路径: {single}',
            f'  单边占已计巡线帧: {100 * single / steering_total:.1f}%'
            if steering_total else '  单边占已计巡线帧: —（尚无有效巡线帧）',
            '',
            '行走中图卡停车候选（检测命中次；不是停车次数或不同图卡数）',
            f'  找到图卡框: {box}',
            f'  无框但 cue 命中: {cue}',
            f'  框占两种命中: {100 * box / card_total:.1f}%'
            if card_total else '  框占两种命中: —（尚无候选）',
            '',
            '每次视觉检测调用最多给框或 cue 其中一项加 1；',
            '这两项说明停车候选证据来源，不表示该帧已经触发停车。',
            *((
                '',
                '实际行走停车触发（事件，不包含起步首卡动作）',
                f'  依据框宽: {stop_box}',
                f'  依据 cue 高度: {stop_cue}',
            ) if stop_box is not None and stop_cue is not None else ()),
            '逐帧触发条件与动作仍以 line_frames.jsonl 和 vision 日志为准。',
            '',
        ]
        temporary = self.counts_path.with_suffix('.txt.tmp')
        temporary.write_text('\n'.join(lines), encoding='utf-8')
        os.replace(temporary, self.counts_path)
        self._last_counts_write_s = time.monotonic()

    @staticmethod
    def scalar(value):
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value) if math.isfinite(value) else None
        # np.bool_ is not a Python bool or numbers.Number.
        if isinstance(value, np.bool_):
            return bool(value.item())
        raise TypeError('not a scalar diagnostic')

    def write(self, debug, **context):
        measurements = {}
        for key, value in debug.items():
            try:
                measurements[key] = self.scalar(value)
            except TypeError:
                pass  # Debug images and fit arrays belong to loss dumps.
        # Context errors are programming errors, never silently discarded.
        # LOG=vision includes the exact lookup demand, even without control/video.
        measurements.setdefault('steering_demand_deg', None)
        row = {key: self.scalar(value) for key, value in context.items()}
        row['schema'] = 'line_frames_v1'
        row['measurement'] = measurements
        # A row cannot contain the time it takes to write itself. Store the previous
        # row's serialization/write/periodic-flush cost, one frame behind.
        row['telemetry_prev_write_ms'] = self._previous_write_ms
        write_start_ns = time.perf_counter_ns()
        self._file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self._pending += 1
        if self._pending >= 25:
            self._file.flush()
            self._pending = 0
        self._previous_write_ms = (time.perf_counter_ns() - write_start_ns) / 1_000_000.0

        if all(key in row for key in COUNT_FIELDS) and row.get('host_time_ns') is not None:
            self._latest_counts = {key: row[key] for key in COUNT_FIELDS}
            for key in ('card_stop_box_trigger_count', 'card_stop_cue_trigger_count'):
                if key in row:
                    self._latest_counts[key] = row[key]
            self._latest_counts.update(frame=row.get('frame', '?'),
                                       host_time_ns=row['host_time_ns'])
            if time.monotonic() - self._last_counts_write_s >= 1.0:
                self._write_counts_summary()

    def close(self):
        try:
            self._write_counts_summary()
        finally:
            self._file.close()
