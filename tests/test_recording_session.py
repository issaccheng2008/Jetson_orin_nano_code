import io
import sys
import tempfile
import unittest
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from recording_session import create_session
import policy_gate_launcher


class RecordingSessionTests(unittest.TestCase):
    def test_clock_sync_is_awaited_before_creating_any_session(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch('recording_session.datetime') as clock,
                  patch('recording_session.time.sleep') as sleep):
                clock.now.side_effect = [datetime(1970, 1, 1, tzinfo=timezone.utc),
                                        datetime(2026, 10, 16, tzinfo=timezone.utc)]
                first = create_session(directory)
            self.assertEqual(first.parent.name, '2026-10-16')
            sleep.assert_called_once()
            self.assertFalse((Path(directory) / 'clock_unsynced').exists())
            self.assertFalse((Path(directory) / '1970-01-01').exists())
            manifest = json.loads((first / 'recording_manifest.json').read_text())
            self.assertTrue(manifest['wall_clock_valid'])
            self.assertGreater(manifest['started_monotonic_ns'], 0)

    def test_clock_timeout_leaves_no_partial_recording_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch('recording_session.datetime') as clock,
                  patch('recording_session.time.monotonic', side_effect=[10., 10., 70.]),
                  patch('recording_session.time.sleep')):
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                with self.assertRaisesRegex(RuntimeError, 'clock'):
                    create_session(directory)
                self.assertEqual(list(Path(directory).iterdir()), [])
                clock.now.return_value = datetime(2026, 10, 16, tzinfo=timezone.utc)
                session = create_session(directory)
                self.assertEqual(session.parent.name, '2026-10-16')

    def test_vision_does_not_initialize_control_when_recording_clock_times_out(self):
        import run_policy_vision
        with tempfile.TemporaryDirectory() as directory:
            with (patch('sys.argv', ['run_policy_vision.py', '--recording-root', directory]),
                  patch('recording_session.datetime') as clock,
                  patch('recording_session.time.monotonic', side_effect=[0., 60.]),
                  patch.object(run_policy_vision, 'ConnectorClient') as client,
                  patch('policy_gate_launcher.PolicyGateLauncher') as launcher):
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                with self.assertRaisesRegex(RuntimeError, 'clock'):
                    run_policy_vision.main()
                client.assert_not_called()
                launcher.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_synchronized_clock_keeps_local_calendar_date(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('recording_session.datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 16, tzinfo=timezone.utc)
                session = create_session(directory)
            self.assertEqual(session.parent.name, '2026-10-16')
            manifest = json.loads((session / 'recording_manifest.json').read_text())
            self.assertTrue(manifest['wall_clock_valid'])

    def test_clock_sync_applies_to_next_session_and_preserves_unsynced_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'clock_unsynced/test_legacy'
            first.mkdir(parents=True)
            (first / 'evidence.txt').write_text('keep')
            with patch('recording_session.datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 16, tzinfo=timezone.utc)
                second = create_session(directory)
            self.assertEqual(first.parent.name, 'clock_unsynced')
            self.assertEqual(second.parent.name, '2026-10-16')
            self.assertEqual((first / 'evidence.txt').read_text(), 'keep')

    def test_daily_sessions_are_unique_and_preserve_previous_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_session(directory)
            (first / 'evidence.txt').write_text('keep')
            second = create_session(directory)
            self.assertEqual(first.parent, second.parent)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'evidence.txt').read_text(), 'keep')
            self.assertTrue((first / 'recording_manifest.json').is_file())

    def test_policy_logs_and_imu_share_the_visual_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'walk.onnx').touch()
            (root / 'foot.onnx').touch()
            session = create_session(root / 'tests')
            with (patch.object(policy_gate_launcher, 'REPO_ROOT', root),
                  patch.object(policy_gate_launcher.subprocess, 'Popen',
                               side_effect=[Mock(pid=12, stdout=io.BytesIO()), Mock()]) as popen):
                launcher = policy_gate_launcher.PolicyGateLauncher('walk.onnx', 'port', 10,
                    one_foot_model='foot.onnx', recording_directory=session)
                launcher.start()
                command = popen.call_args_list[0].args[0]
                for flag in ('--diagnostic-log-dir', '--position-log-dir'):
                    self.assertTrue(Path(command[command.index(flag)+1]).is_relative_to(session))
                self.assertTrue(launcher.log_path.is_relative_to(session))
                launcher.close()
