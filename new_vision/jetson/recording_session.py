"""Persistently numbered test sessions, independent of system clock readiness."""
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
import time
from recording_options import LOG_ITEMS


# Advisory metadata only; never wait for the clock before starting a recording.
MIN_RECORDING_YEAR = 2026


@contextmanager
def _counter_lock(root):
    # Use a stable inode: test_counter.txt itself is replaced atomically below.
    with (root / '.test_counter.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b'\0')
                lock.flush()
            lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(f'recording counter busy: {root}; retry after the other allocator exits') from exc
        try:
            yield
        finally:
            release()


def _allocate_directory(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _counter_lock(root):
        counter = root / 'test_counter.txt'
        try:
            text = counter.read_text(encoding='ascii').strip()
        except FileNotFoundError:
            text = '0'
        except UnicodeError as exc:
            raise RuntimeError(f'invalid test_counter.txt: {counter}; restore the last allocated number') from exc
        if not text.isascii() or not text.isdecimal():
            raise RuntimeError(f'invalid test_counter.txt: {counter}; restore the last allocated number')
        last_number = max(int(text), max((int(p.name) for p in root.iterdir()
                          if p.name.isascii() and p.name.isdecimal()), default=0))
        number = last_number + 1
        # Do not merge with an existing directory, even if it has no manifest.
        while True:
            directory = root / str(number)
            try:
                directory.mkdir(exist_ok=False)
                break
            except FileExistsError:
                number += 1
        temporary = root / f'.test_counter.{uuid.uuid4().hex}.tmp'
        try:
            with temporary.open('x', encoding='ascii') as output:
                output.write(f'{number}\n')
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, counter)
            if os.name != 'nt':
                # Persist the new directory and atomic counter rename on Jetson.
                descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return directory, number


def create_session(root, log_items=LOG_ITEMS):
    log_items = tuple(log_items)
    unknown = set(log_items) - set(LOG_ITEMS)
    if unknown:
        raise ValueError(f'unknown recording log items: {sorted(unknown)}')
    directory, number = _allocate_directory(root)
    now = datetime.now().astimezone()
    started_monotonic_ns = time.monotonic_ns()
    manifest = {
        'schema': 'test_recording_v1', 'started_at': now.isoformat(),
        'wall_clock_valid': now.year >= MIN_RECORDING_YEAR,
        'started_monotonic_ns': started_monotonic_ns,
        'test_id': directory.name,
        'test_number': number,
        'enabled_logs': list(log_items),
        'camera_video_semantics': 'VISION is the per-frame candidate before connector median windows; ROBOT CMD is policy feedback after windows/hold/takeovers and serial send; red means line lost; absent/stale execution is UNKNOWN',
        'time_alignment': 'vision host_time_ns / 1e9 and control host_unix_s',
        'execution_semantics': 'cmd_wz is model input; motor_send_argument_* is sent joint target, not measured body yaw',
    }
    paths = {
        'vision': {'vision': 'vision/run_*/line_frames.jsonl', 'vision_counts': 'vision/run_*/counts_summary.txt'},
        'control': {'execution_and_imu': 'control/control_trace_*.csv'},
        'motor': {'motor_positions': 'motor_positions/*.csv'},
        'policy': {'policy_console': 'main_*.log'},
        'video': {'camera_video': 'video/*.avi',
                  'camera_video_index': 'video/frames.jsonl'},
        'shape': {'shape_dumps': 'shape/run_*'}, 'loss': {'loss_dumps': 'loss/run_*'},
    }
    for name in log_items:
        manifest.update(paths[name])
    (directory / 'recording_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory
