"""Exclusive test sessions created only after the board clock has a valid date."""
from datetime import datetime
import json
from pathlib import Path
import uuid
import time


# This deployment dates from 2026. A pre-deployment date is not trustworthy.
MIN_RECORDING_YEAR = 2026


def create_session(root):
    now = datetime.now().astimezone()
    # The Jetson can boot at the epoch and recover its date moments later.
    # Allocate no paths/writers until then: a session cannot change directory
    # once the vision, video and policy processes hold open files inside it.
    if now.year < MIN_RECORDING_YEAR:
        deadline = time.monotonic() + 60.
        print('[recording] Waiting up to 60s for a valid system clock before '
              'creating the test directory.', flush=True)
        while now.year < MIN_RECORDING_YEAR:
            if time.monotonic() >= deadline:
                raise RuntimeError('recording system clock is still unset after 60s; '
                                   'no test directory created; synchronize clock and retry')
            time.sleep(1.)
            now = datetime.now().astimezone()
    started_monotonic_ns = time.monotonic_ns()
    day = now.strftime('%Y-%m-%d')
    stamp = now.strftime('%H%M%S_%f')
    directory = (Path(root).expanduser().resolve() / day /
                 ('test_' + stamp + '_' + uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'test_recording_v1', 'started_at': now.isoformat(),
        'wall_clock_valid': True,
        'started_monotonic_ns': started_monotonic_ns,
        'test_id': directory.name,
        'vision': 'vision/run_*/line_frames.jsonl',
        'execution_and_imu': 'control/control_trace_*.csv',
        'motor_positions': 'motor_positions/*.csv',
        'policy_console': 'main_*.log',
        'camera_video': 'video/camera_commands.avi (when enabled)',
        'camera_video_index': 'video/frames.jsonl',
        'camera_video_semantics': 'VISION is the per-frame candidate before connector median windows; ROBOT CMD is policy feedback after windows/hold/takeovers and serial send; red means line lost; absent/stale execution is UNKNOWN',
        'time_alignment': 'vision host_time_ns / 1e9 and control host_unix_s',
        'execution_semantics': 'cmd_wz is model input; motor_send_argument_* is sent joint target, not measured body yaw',
    }
    (directory / 'recording_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory
