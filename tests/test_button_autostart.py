"""Hardware-free checks of generated service paths and wrapper working directory."""
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
BASH = os.environ.get('BUTTON_TEST_BASH') or (shutil.which('bash') if os.name != 'nt' else None)


@unittest.skipUnless(BASH, 'Bash required; set BUTTON_TEST_BASH on Windows')
class ButtonAutostartTests(unittest.TestCase):
    def run_bash(self, *args, cwd):
        return subprocess.run([BASH, *args], cwd=cwd, text=True,
                              capture_output=True, check=True).stdout

    def test_connector_wrapper_passes_bias_and_defaults_old_config_to_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot with spaces'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config',
                            ignore=shutil.ignore_patterns('button_start.env'))
            old_config = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                                   if not line.startswith('WZ_BIAS=')) + '\n'
            for bias in (None, '0.1', '-0.1'):
                with self.subTest(bias=bias):
                    (checkout / 'config/button_start.env').write_text(
                        old_config + (f'WZ_BIAS={bias}\n' if bias else ''))
                    args = shlex.split(self.run_bash('scripts/run_button_connector.sh', '--dry-run', cwd=checkout))
                    self.assertIn('--wz-bias', args)
                    self.assertEqual(args[args.index('--wz-bias') + 1], bias or '0')
                    self.assertEqual(args[args.index('--max-wz-accel') + 1], '0')

    def test_rendered_service_working_directory_is_absolute(self):
        # systemd parses WorkingDirectory as a whole path, without shell unquoting.
        for name in ('robot', 'robot with spaces'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                checkout = Path(temporary) / name
                shutil.copytree(REPO / 'scripts', checkout / 'scripts')
                shutil.copytree(REPO / 'config', checkout / 'config',
                                ignore=shutil.ignore_patterns('button_start.env'))
                rendered = self.run_bash('scripts/install_button_autostart.sh', '--dry-run', cwd=checkout)
                for line in rendered.splitlines():
                    if line.startswith('WorkingDirectory='):
                        value = line.split('=', 1)[1]
                        self.assertTrue(PurePosixPath(value).is_absolute(), value)
                self.assertEqual(rendered.count('ExecStart=/bin/bash '), 2)
                self.assertFalse((checkout / 'config/button_start.env').exists())

    def test_both_wrappers_set_repository_directory_before_exec(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot with spaces'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config',
                            ignore=shutil.ignore_patterns('button_start.env'))
            # This stand-in interpreter imports nothing and opens no hardware.
            fake_python = checkout / 'fake_python.sh'
            fake_python.write_text('#!/usr/bin/env bash\n'
                                   'if [[ "$1" != -c ]]; then printf "CWD=%s\\n" "$PWD"; fi\n')
            fake_python.chmod(0o755)
            config = checkout / 'config/button_start.env'
            config.write_text('source "$BUTTON_REPO_DIR/config/button_start.env.example"\n'
                              'VISION_PYTHON="$BUTTON_REPO_DIR/fake_python.sh"\n'
                              'POLICY_PYTHON="$VISION_PYTHON"\n')
            for relative in ('connector.py', 'new_vision/jetson/run_policy_vision.py',
                             'humanoid_jetson_deploy/main.py',
                             'humanoid_jetson_deploy/policy_49_max.onnx',
                             'humanoid_jetson_deploy/policy-one-foot-standing_old.onnx'):
                file = checkout / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            expected = self.run_bash('-c', 'pwd -P', cwd=checkout).strip()
            for wrapper in ('connector', 'vision'):
                output = self.run_bash(f'{checkout.as_posix()}/scripts/run_button_{wrapper}.sh', cwd=REPO)
                self.assertEqual(output.strip(), f'CWD={expected}')

    def test_vision_wrapper_passes_configured_mode_and_defaults_old_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config',
                            ignore=shutil.ignore_patterns('button_start.env'))
            config = checkout / 'config/button_start.env'
            # Copy the real template but omit the new setting to represent an
            # existing installed file, which git pull deliberately preserves.
            old_config = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                                   if not line.startswith('WZ_MODE=')) + '\n'
            for mode in (None, 'heading', 'segments', 'segment'):
                with self.subTest(mode=mode):
                    config.write_text(old_config + (f'WZ_MODE={mode}\n' if mode else ''))
                    result = subprocess.run([BASH, 'scripts/run_button_vision.sh', '--dry-run'],
                                            cwd=checkout, text=True, capture_output=True)
                    if mode == 'segment':
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('WZ_MODE', result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        args = shlex.split(result.stdout)
                        self.assertEqual(args[args.index('--wz-mode') + 1], mode or 'heading')

    def test_vision_wrapper_exposes_loss_hold_without_changing_old_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config',
                            ignore=shutil.ignore_patterns('button_start.env'))
            old_config = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                                   if not line.startswith('LOST_HOLD_S=')) + '\n'
            for hold in (None, '0.5'):
                with self.subTest(hold=hold):
                    (checkout / 'config/button_start.env').write_text(
                        old_config + (f'LOST_HOLD_S={hold}\n' if hold else ''))
                    args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
                    self.assertIn('--lost-hold-s', args)
                    self.assertEqual(args[args.index('--lost-hold-s') + 1], hold or '0.2')


if __name__ == '__main__':
    unittest.main()
