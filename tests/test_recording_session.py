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
    def test_unsynchronized_clock_does_not_create_epoch_date_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('recording_session.datetime') as clock:
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                first = create_session(directory)
                second = create_session(directory)
            self.assertEqual(first.parent.name, 'clock_unsynced')
            self.assertNotEqual(first, second)
            self.assertFalse((Path(directory) / '1970-01-01').exists())
            manifest = json.loads((first / 'recording_manifest.json').read_text())
            self.assertFalse(manifest['wall_clock_valid'])
            self.assertGreater(manifest['started_monotonic_ns'], 0)

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
            with patch('recording_session.datetime') as clock:
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                first = create_session(directory)
                (first / 'evidence.txt').write_text('keep')
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
