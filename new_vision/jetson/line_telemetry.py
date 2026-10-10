"""Buffered scalar telemetry for every processed vision frame (no image arrays)."""
import json
import math
import numbers
from pathlib import Path
import time
import numpy as np


class LineTelemetry:
    def __init__(self, run_directory):
        self.path = Path(run_directory) / 'line_frames.jsonl'
        self._file = self.path.open('x', encoding='utf-8', buffering=65536)
        self._pending = 0
        self._previous_write_ms = None

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

    def close(self):
        self._file.close()
