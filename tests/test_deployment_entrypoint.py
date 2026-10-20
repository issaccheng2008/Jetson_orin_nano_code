"""Run CLI from files that Git will actually deliver, without local extras."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


class DeploymentEntrypointTests(unittest.TestCase):
    def test_tracked_vision_files_support_help_without_camera_hardware(self):
        root = Path(__file__).resolve().parents[1]
        tracked = subprocess.check_output([
            'git', '-c', 'safe.directory=' + root.as_posix(), 'ls-files', '-z',
            'new_vision/jetson/*.py', 'new_vision/config/*.json'], cwd=root,
            ).decode('utf-8').rstrip('\0').split('\0')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in tracked:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([
                sys.executable, str(target / 'new_vision/jetson/run_policy_vision.py'),
                '--help'], cwd=target, env=env, capture_output=True, text=True,
                timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--camera-exposure-ms', result.stdout)


if __name__ == '__main__':
    unittest.main()
