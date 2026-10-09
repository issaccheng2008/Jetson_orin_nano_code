import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from recording_session import create_session
import policy_gate_launcher


class RecordingSessionTests(unittest.TestCase):
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
