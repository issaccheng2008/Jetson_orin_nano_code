import io
import sys
import tempfile
import unittest
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from recording_session import create_session
import policy_gate_launcher


class RecordingSessionTests(unittest.TestCase):
    def test_unset_clock_creates_numbered_session_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch('recording_session.datetime') as clock,
                  patch('time.sleep', side_effect=AssertionError('must not wait for clock'))):
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                first = create_session(directory)
            self.assertEqual(first, Path(directory).resolve() / '1')
            self.assertEqual((Path(directory) / 'test_counter.txt').read_text().strip(), '1')
            manifest = json.loads((first / 'recording_manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['wall_clock_valid'])
            self.assertEqual(manifest['test_id'], '1')
            self.assertEqual(manifest['test_number'], 1)
            self.assertTrue(manifest['started_at'].startswith('1970-01-01'))
            self.assertGreater(manifest['started_monotonic_ns'], 0)

    def test_counter_persists_across_new_python_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_session(directory)
            code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                    'from recording_session import create_session; '
                    'print(create_session(sys.argv[2]).name)')
            result = subprocess.run([sys.executable, '-B', '-c', code,
                str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'), directory],
                capture_output=True, text=True, check=True)
            self.assertEqual(first.name, '1')
            self.assertEqual(result.stdout.strip(), '2')
            self.assertEqual((Path(directory) / 'test_counter.txt').read_text().strip(), '2')

    def test_deleted_latest_directory_does_not_reset_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_session(directory)
            (first / 'recording_manifest.json').unlink()
            first.rmdir()
            self.assertEqual(create_session(directory).name, '2')

    def test_stale_or_missing_counter_skips_existing_numeric_paths(self):
        for counter in (None, '2\n'):
            with self.subTest(counter=counter), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / '8').mkdir()
                (root / '8/evidence.txt').write_text('keep')
                (root / '9').write_text('also keep')
                (root / '2026-10-16').mkdir()
                if counter is not None:
                    (root / 'test_counter.txt').write_text(counter)
                session = create_session(root)
                self.assertEqual(session.name, '10')
                self.assertEqual((root / '8/evidence.txt').read_text(), 'keep')
                self.assertEqual((root / '9').read_text(), 'also keep')

    def test_invalid_counter_fails_instead_of_restarting_numbering(self):
        for value in ('', '-1', 'oops', '1.5'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                counter = Path(directory) / 'test_counter.txt'
                counter.write_text(value)
                with self.assertRaisesRegex(RuntimeError, 'test_counter'):
                    create_session(directory)
                self.assertEqual(counter.read_text(), value)
                self.assertFalse(any(p.name.isdecimal() for p in Path(directory).iterdir()))

    def test_non_ascii_counter_fails_without_allocating_a_number(self):
        with tempfile.TemporaryDirectory() as directory:
            counter = Path(directory) / 'test_counter.txt'
            counter.write_bytes(b'\xff\xfe')
            with self.assertRaisesRegex(RuntimeError, 'test_counter'):
                create_session(directory)
            self.assertEqual(counter.read_bytes(), b'\xff\xfe')
            self.assertFalse(any(p.name.isdecimal() for p in Path(directory).iterdir()))

    def test_invalid_log_selection_does_not_consume_a_number(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                create_session(directory, log_items=('typo',))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_atomic_counter_update_preserves_old_counter_and_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_session(directory)
            with patch('os.replace', side_effect=OSError('disk error')):
                with self.assertRaisesRegex(OSError, 'disk error'):
                    create_session(directory)
            root = Path(directory)
            self.assertEqual((root / 'test_counter.txt').read_text().strip(), '1')
            self.assertTrue((root / '2').is_dir())
            self.assertTrue((first / 'recording_manifest.json').is_file())
            self.assertEqual(create_session(directory).name, '3')
            self.assertFalse(list(root.glob('.test_counter.*.tmp')))

    def test_simultaneous_processes_allocate_distinct_sessions(self):
        code = '''import sys, time
sys.path.insert(0, sys.argv[1])
from recording_session import create_session
for attempt in range(200):
    try:
        print(create_session(sys.argv[2]).name)
        break
    except RuntimeError as exc:
        if 'busy' not in str(exc): raise
        time.sleep(.01)
else: raise RuntimeError('allocation stayed busy')
'''
        with tempfile.TemporaryDirectory() as directory:
            processes = [subprocess.Popen([sys.executable, '-B', '-c', code,
                str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'), directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(6)]
            numbers = []
            results = [process.communicate(timeout=15) for process in processes]
            for process, (out, err) in zip(processes, results):
                self.assertEqual(process.returncode, 0, err)
                self.assertTrue(out.strip().isdecimal(), out)
                numbers.append(int(out.strip()))
            self.assertEqual(sorted(numbers), list(range(1, 7)))
            self.assertEqual((Path(directory) / 'test_counter.txt').read_text().strip(), '6')

    def test_synchronized_clock_is_advisory_and_does_not_name_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('recording_session.datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 16, tzinfo=timezone.utc)
                session = create_session(directory)
            self.assertEqual(session.name, '1')
            self.assertEqual(session.parent, Path(directory).resolve())
            manifest = json.loads((session / 'recording_manifest.json').read_text())
            self.assertTrue(manifest['wall_clock_valid'])

    def test_clock_sync_does_not_restart_numbering_or_move_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'clock_unsynced/test_legacy'
            first.mkdir(parents=True)
            (first / 'evidence.txt').write_text('keep')
            with (patch('recording_session.datetime') as clock,
                  patch('time.sleep', side_effect=AssertionError('must not wait for clock'))):
                clock.now.return_value = datetime(1970, 1, 1, tzinfo=timezone.utc)
                before_sync = create_session(directory)
                clock.now.return_value = datetime(2026, 10, 16, tzinfo=timezone.utc)
                second = create_session(directory)
            self.assertEqual(first.parent.name, 'clock_unsynced')
            self.assertEqual(before_sync.name, '1')
            self.assertEqual(second.name, '2')
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
