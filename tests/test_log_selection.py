"""LOG selection changes recorded files, never robot control behavior."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
sys.path.insert(0, str(ROOT / 'humanoid_jetson_deploy'))

import main
import policy_gate_launcher
import recording_session
import run_policy_vision
import test_button_start
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID
from recording_options import LOG_ITEMS, parse_log_items


class LogSelectionTests(unittest.TestCase):
    def test_vision_only_records_execution_feedback_without_video(self):
        scenario = test_button_start.ButtonVisionIntegrationTests()
        feedback = dict(velocity=[.2, 0., -.17], enabled=True, send_result='written',
                        host_unix_s=1700000000., monotonic_s=3., step=123, policy_mode='walking49')
        attitude = Mock()
        attitude.value = 45.
        attitude.card_task.return_value = {}
        attitude.card_tilt_status_seen = False
        attitude.executed_command.return_value = feedback
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.CommandVideo') as video, \
                patch('attitude_input.AttitudeInput', return_value=attitude):
            launcher, *_ = scenario.run_scenario(card=False, press_at=6,
                extra_args=('--recording-root', tmp, '--log-items', 'vision', '--attitude-port', '5010'))
            path, = launcher.recording_directory.glob('vision/run_*/line_frames.jsonl')
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(rows)
            self.assertEqual(rows[-1].get('robot_cmd_wz'), -.17)
            self.assertEqual(rows[-1].get('robot_cmd_status'), 'written')
            self.assertEqual(attitude.poll.call_count, 60 + len(rows))
            video.assert_not_called()

    def test_lists_all_none_and_invalid_names(self):
        self.assertEqual(parse_log_items('all'), LOG_ITEMS)
        self.assertEqual(parse_log_items('none'), ())
        self.assertEqual(parse_log_items(''), ())
        self.assertEqual(parse_log_items('video, VISION control vision'), ('vision', 'control', 'video'))
        for text in ('vison', 'all vision', 'none control'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_log_items(text)

    def test_explicit_selection_controls_video_and_legacy_flags_still_work(self):
        for options, enabled in (([], True), (['--no-record-video'], False),
                (['--log-items', 'vision control'], False),
                (['--log-items', 'video', '--no-record-video'], True)):
            with patch('sys.argv', ['run_policy_vision.py', *options]):
                self.assertEqual(run_policy_vision.parse_args().record_video, enabled)
        with patch('sys.argv', ['run_policy_vision.py', '--log-items', 'typo']), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                run_policy_vision.parse_args()

    def test_manifest_lists_only_enabled_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = recording_session.create_session(tmp, log_items=('vision', 'control'))
            manifest = json.loads((session / 'recording_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['enabled_logs'], ['vision', 'control'])
            self.assertIn('vision', manifest)
            self.assertIn('execution_and_imu', manifest)
            for name in ('motor_positions', 'policy_console', 'camera_video', 'camera_video_index'):
                self.assertNotIn(name, manifest)

    def test_button_selection_creates_only_requested_visual_outputs(self):
        scenario = test_button_start.ButtonVisionIntegrationTests()
        for selection, expected in (('vision', {'vision'}), ('video', set()),
                                    ('shape loss', {'shape', 'loss'}), ('control', set())):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as tmp:
                with patch('command_video.CommandVideo') as video:
                    launcher, commands, *_ = scenario.run_scenario(card=True, press_at=6,
                        extra_args=('--recording-root', tmp, '--log-items', selection),
                        on_poll=lambda frame: self.assertEqual(list(Path(tmp).iterdir()), []) if frame <= 6 else None)
                session = launcher.recording_directory
                self.assertEqual({p.name for p in session.iterdir() if p.is_dir()}, expected)
                self.assertEqual(video.call_count, int('video' in selection.split()))
                self.assertTrue(any(call.args[0] > 0 for call in commands))
                if 'vision' in expected:
                    self.assertTrue(list(session.glob('vision/run_*/counts_summary.txt')))

    def test_none_still_runs_control_but_creates_no_test_directory(self):
        scenario = test_button_start.ButtonVisionIntegrationTests()
        with tempfile.TemporaryDirectory() as tmp, patch('command_video.CommandVideo') as video:
            launcher, commands, *_ = scenario.run_scenario(card=False, press_at=6,
                extra_args=('--recording-root', tmp, '--log-items', 'none'))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            video.assert_not_called()
            launcher.start.assert_called_once()
            self.assertTrue(any(call.args[0] > 0 for call in commands))

    def test_policy_launch_disables_unselected_files_and_keeps_console_visible(self):
        for selection in ((), ('control',), ('motor',), ('policy',)):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'walk.onnx').touch()
                process = Mock(pid=12, stdout=io.BytesIO())
                process.poll.return_value = None
                with (patch.object(policy_gate_launcher, 'REPO_ROOT', root),
                      patch.object(policy_gate_launcher.subprocess, 'Popen', side_effect=[process, Mock()]) as popen,
                      redirect_stdout(io.StringIO())):
                    launcher = policy_gate_launcher.PolicyGateLauncher('walk.onnx', 'test-port', 10,
                        recording_directory=root / 'session', log_items=selection)
                    launcher.start()
                    command = popen.call_args_list[0].args[0]
                    self.assertEqual('--no-control-diagnostics' in command, 'control' not in selection)
                    self.assertEqual('--no-position-log' in command, 'motor' not in selection)
                    self.assertEqual(popen.call_count, 2 if 'policy' in selection else 1)
                    self.assertEqual(launcher.log_path is not None, 'policy' in selection)
                    if 'policy' not in selection:
                        self.assertIsNone(popen.call_args_list[0].kwargs.get('stdout'))
                    if not selection:
                        self.assertFalse((root / 'session').exists())
                        self.assertFalse((root / 'records').exists())
                    launcher.close()

    def test_motor_csv_can_be_disabled_without_disabling_control_trace(self):
        import config
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / 'frames.json'
            frames.write_text(json.dumps(dict(format='joint_frames_v1', hz=50,
                joint_names=list(config.JOINT_NAMES), frames=[config.Q_DEFAULT.tolist()])))
            with patch('sys.argv', ['main.py', '--fixed-policy', str(frames), '--no-plot',
                    '--no-position-log', '--diagnostic-log-dir', str(Path(tmp) / 'control')]):
                args = main.parse_args()
            fields = dict(status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
                accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
                gyro_rad_s=np.zeros(3, dtype=np.float32), orientation_wxyz=np.array([1., 0, 0, 0]),
                joint_position=config.Q_DEFAULT.copy(), joint_velocity=np.zeros(12, dtype=np.float32))
            with (patch.object(main, 'parse_args', return_value=args),
                  patch.object(main.signal, 'signal'), patch.object(main, 'SerialLink') as link_cls,
                  patch.object(main, 'PositionCsvLogger') as motor_log,
                  redirect_stdout(io.StringIO())):
                link_cls.return_value.wait_for_state.return_value = SimpleNamespace(sequence=1, **fields)
                link_cls.return_value.get_latest_state.side_effect = [
                    SimpleNamespace(sequence=1, **fields), SimpleNamespace(sequence=2, **fields)]
                self.assertEqual(main.main(), 0)
                motor_log.assert_not_called()
                self.assertTrue(list((Path(tmp) / 'control').glob('control_trace_*.csv')))
                link_cls.return_value.send_command.assert_called()


if __name__ == '__main__':
    unittest.main()
