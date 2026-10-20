"""Low-cost wall-clock stages and cached, read-only Jetson runtime counters."""
from pathlib import Path
import time


class StageTimer:
    def __init__(self, prefix):
        self.prefix = prefix
        self.started = self.previous = time.perf_counter_ns()
        self.cpu_started = time.process_time_ns()
        self.thread_started = time.thread_time_ns()
        self.fields = {}

    def lap(self, name):
        now = time.perf_counter_ns()
        self.fields[f'{self.prefix}_stage_{name}_ms'] = (now-self.previous)/1e6
        self.previous = now

    def call(self, name, function, *args):
        started = time.perf_counter_ns()
        try:
            return function(*args)
        finally:
            key = f'{self.prefix}_{name}_ms'
            self.fields[key] = self.fields.get(key, 0.) + (time.perf_counter_ns()-started)/1e6

    def finish(self):
        self.fields[f'{self.prefix}_profile_total_ms'] = (time.perf_counter_ns()-self.started)/1e6
        # Process CPU includes OpenCV workers and capture threads; thread CPU is
        # only the calling thread. Neither is a non-overlapping wall-time stage.
        self.fields[f'{self.prefix}_process_cpu_ms'] = (time.process_time_ns()-self.cpu_started)/1e6
        self.fields[f'{self.prefix}_main_thread_cpu_ms'] = (time.thread_time_ns()-self.thread_started)/1e6
        return self.fields


class RuntimeCounters:
    def __init__(self):
        self.next_sample = 0.
        self.sampled_at = 0.
        self.cached = {}
        self.cpu_paths = sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'))
        self.thermal_paths = []
        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
            try:
                name = (zone/'type').read_text().strip()
                if name in ('cpu-thermal', 'gpu-thermal', 'tj-thermal'):
                    self.thermal_paths.append((name.split('-')[0], zone/'temp'))
            except OSError:
                pass

    def sample(self):
        started = time.perf_counter_ns()
        now = time.monotonic()
        if now >= self.next_sample:
            self.next_sample = now + 1.
            self.sampled_at = now
            values = []
            for path in self.cpu_paths:
                try:
                    values.append(float(path.read_text())/1000.)
                except (OSError, ValueError):
                    pass
            self.cached = {}
            if values:
                self.cached.update(runtime_cpu_min_mhz=min(values), runtime_cpu_max_mhz=max(values))
            for name, path in self.thermal_paths:
                try:
                    self.cached[f'runtime_{name}_temp_c'] = float(path.read_text())/1000.
                except (OSError, ValueError):
                    pass
        return dict(self.cached, runtime_sample_age_ms=(now-self.sampled_at)*1000.,
                    runtime_counters_ms=(time.perf_counter_ns()-started)/1e6)


def environment_metadata():
    import cv2
    return dict(profiling_version=1, opencv_version=cv2.__version__,
                opencv_threads=cv2.getNumThreads(), opencv_optimized=cv2.useOptimized(),
                opencv_opencl=cv2.ocl.useOpenCL(),
                timing_note='line_stage_* are non-overlapping wall stages; shape/QR are inside decision_ms; CPU time is separate.')
