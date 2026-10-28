"""One local-date directory per day, one exclusive directory per vision test."""
from datetime import datetime
import json
from pathlib import Path
import uuid


def create_session(root):
    now = datetime.now().astimezone()
    directory = (Path(root).expanduser().resolve() / now.strftime('%Y-%m-%d') /
                 ('test_' + now.strftime('%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'test_recording_v1', 'started_at': now.isoformat(),
        'test_id': directory.name,
        'vision': 'vision/run_*/line_frames.jsonl',
        'execution_and_imu': 'control/control_trace_*.csv',
        'motor_positions': 'motor_positions/*.csv',
        'policy_console': 'main_*.log',
        'camera_video': 'video/camera_commands.avi (when enabled)',
        'camera_video_index': 'video/frames.jsonl',
        'camera_video_semantics': 'Arrow is vision-published vx/wz, before connector bias/model hold; red means line lost',
        'time_alignment': 'vision host_time_ns / 1e9 and control host_unix_s',
        'execution_semantics': 'cmd_wz is model input; motor_send_argument_* is sent joint target, not measured body yaw',
    }
    (directory / 'recording_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory
