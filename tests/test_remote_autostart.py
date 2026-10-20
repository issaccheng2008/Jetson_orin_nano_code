"""Exercise startup scripts without systemd, serial hardware, or a policy process."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
BASH = os.environ.get('BUTTON_TEST_BASH') or (shutil.which('bash') if os.name != 'nt' else None)


@unittest.skipUnless(BASH, 'Bash required; set BUTTON_TEST_BASH on Windows')
class RemoteAutostartTests(unittest.TestCase):
    def preflight_validation(self, overrides=None, token='valid-secret'):
        # Execute the actual embedded Python validator without dependency imports,
        # systemd, a policy constructor, or any serial/device access.
        source = (REPO / 'scripts/remote_autostart_common.sh').read_text(encoding='utf-8')
        code = source.split('"$POLICY_PYTHON" -c \'\n', 1)[1].split('\n\' "$REMOTE_PORT"', 1)[0]
        settings = dict(web='8080', policy='5007', startup='20', vx='0.4', wz='0.5',
                        kp='1', kd='1', step='2.5')
        settings.update(overrides or {})
        return subprocess.run([sys.executable, '-c', code, *settings.values()],
                              env={**os.environ, 'ROBOT_REMOTE_TOKEN': token},
                              text=True, capture_output=True)

    def test_preflight_rejects_out_of_range_settings_before_legacy_migration(self):
        installer = (REPO / 'scripts/install_remote_autostart.sh').read_text(encoding='utf-8')
        preflight = installer.index('\npreflight_remote_config\n')
        for mutation in ('systemd-analyze verify "$unit_dir/', 'sudo systemctl disable --now',
                         'sudo mv --', 'sudo systemctl mask'):
            self.assertLess(preflight, installer.index(mutation))
        for name, value in (('vx', '1.01'), ('wz', '0.51'), ('kp', '1.01'), ('kd', '1.01'),
                            ('kp', '-0.1'), ('kd', '-0.1'), ('vx', '0'), ('wz', '0'),
                            ('startup', '0'), ('startup', 'nan'), ('step', '0'),
                            ('step', 'inf'), ('web', '5007'), ('policy', '65536')):
            with self.subTest(name=name, value=value):
                result = self.preflight_validation({name: value})
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_preflight_accepts_zero_gains_and_valid_boundary_speeds(self):
        result = self.preflight_validation({'kp': '0', 'kd': '0', 'vx': '1', 'wz': '0.5'})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_rejects_invalid_tokens_without_printing_them(self):
        for token in ('', 'private token', 'private\nsecret', 'private\tsecret', '\u79c1\u5bc6-token'):
            with self.subTest(token=repr(token)):
                result = self.preflight_validation(token=token)
                self.assertNotEqual(result.returncode, 0)
                if token:
                    self.assertNotIn(token, result.stdout + result.stderr)

    def run_bash(self, checkout, *args):
        return subprocess.run([BASH, *args], cwd=checkout, text=True, capture_output=True)

    def checkout(self, root):
        checkout = Path(root) / 'robot with spaces'
        shutil.copytree(REPO / 'scripts', checkout / 'scripts')
        shutil.copytree(REPO / 'config', checkout / 'config',
                        ignore=shutil.ignore_patterns('remote_start.env', 'button_start.env'))
        return checkout

    def config(self, checkout, extra=''):
        (checkout / 'config/remote_start.env').write_text(
            'source "$REMOTE_REPO_DIR/config/remote_start.env.example"\n'
            'ROBOT_REMOTE_TOKEN=test-secret-not-in-output\n' + extra)

    def test_dry_run_renders_only_remote_service_without_writing_config(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            result = self.run_bash(checkout, 'scripts/install_remote_autostart.sh', '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((checkout / 'config/remote_start.env').exists())
            self.assertEqual(result.stdout.count('ExecStart='), 1)
            for value in ('humanoid-remote.service', 'run_remote_control.sh',
                          'Conflicts=humanoid-button-vision.service humanoid-button-connector.service',
                          'KillMode=control-group', 'KillSignal=SIGINT', 'TimeoutStopSec=20',
                          'Restart=on-failure', 'systemd-analyze verify', 'mask', 'backup'):
                self.assertIn(value, result.stdout)
            self.assertNotIn('run_button_', result.stdout)

    def test_wrapper_has_only_managed_remote_command_and_quotes_model(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            self.config(checkout, 'WALKING_MODEL="$REPO_DIR/model name (13).onnx"\nPOLICY_STEP_CM=2.5\n')
            result = self.run_bash(checkout, 'scripts/run_remote_control.sh', '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = shlex.split(result.stdout)
            self.assertEqual(args[1:4], ['-m', 'remote_control', '--launch-policy'])
            self.assertTrue(args[args.index('--model') + 1].endswith('/model name (13).onnx'))
            self.assertEqual(args[args.index('--max-vx') + 1], '0.4')
            self.assertEqual(args[args.index('--policy-port') + 1], '5007')
            self.assertEqual(args[args.index('--policy-step-cm') + 1], '2.5')
            self.assertNotIn('test-secret-not-in-output', result.stdout)
            self.assertNotIn('connector.py', result.stdout)

    def test_missing_token_fails_without_launch(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            self.config(checkout, 'ROBOT_REMOTE_TOKEN=\n')
            result = self.run_bash(checkout, 'scripts/run_remote_control.sh', '--dry-run')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ROBOT_REMOTE_TOKEN', result.stderr)

    def test_installer_dry_run_preserves_and_hides_existing_private_config(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            self.config(checkout)
            config = checkout / 'config/remote_start.env'
            before = config.read_bytes()
            result = self.run_bash(checkout, 'scripts/install_remote_autostart.sh', '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(config.read_bytes(), before)
            self.assertNotIn('test-secret-not-in-output', result.stdout + result.stderr)

    def test_missing_model_prevents_python_execution(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            self.config(checkout)
            for relative in ('remote_control/__main__.py', 'humanoid_jetson_deploy/main.py'):
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            result = self.run_bash(checkout, 'scripts/run_remote_control.sh')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing required file', result.stderr)
            self.assertIn('policy(13).onnx', result.stderr)

    def test_wrapper_preflights_then_executes_from_repo_and_exports_token(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            fake = checkout / 'fake_python.sh'
            fake.write_text('#!/usr/bin/env bash\nif [[ "$1" = -c ]]; then\n'
                            'printf "PREFLIGHT=%s\\n" "$2"\n'
                            'else printf "CWD=%s TOKEN=%s\\n" "$PWD" "$ROBOT_REMOTE_TOKEN"; fi\n')
            fake.chmod(0o755)
            self.config(checkout, 'POLICY_PYTHON="$REPO_DIR/fake_python.sh"\n')
            for relative in ('remote_control/__main__.py', 'humanoid_jetson_deploy/main.py',
                             'humanoid_jetson_deploy/policy(13).onnx'):
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            result = self.run_bash(REPO, (checkout / 'scripts/run_remote_control.sh').as_posix())
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = self.run_bash(checkout, '-c', 'pwd -P').stdout.strip()
            self.assertIn(f'CWD={expected} TOKEN=test-secret-not-in-output', result.stdout)
            self.assertIn('import numpy, onnxruntime, serial', result.stdout)
            self.assertNotIn('cv2', result.stdout)

    def test_legacy_install_is_refused_before_any_config_write(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = self.checkout(root)
            result = self.run_bash(checkout, 'scripts/install_button_autostart.sh')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('install_remote_autostart.sh', result.stderr)
            self.assertFalse((checkout / 'config/button_start.env').exists())


if __name__ == '__main__':
    unittest.main()
