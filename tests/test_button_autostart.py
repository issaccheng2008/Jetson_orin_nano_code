"""Hardware-free checks of generated service paths and wrapper working directory."""
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
BASH = os.environ.get('BUTTON_TEST_BASH') or (shutil.which('bash') if os.name != 'nt' else None)


@unittest.skipUnless(BASH, 'Bash required; set BUTTON_TEST_BASH on Windows')
class ButtonAutostartTests(unittest.TestCase):
    def test_complete_button_command_is_accepted_by_vision_entrypoint(self):
        sys.path.insert(0, str(REPO / 'new_vision/jetson'))
        import run_policy_vision
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config',
                            ignore=shutil.ignore_patterns('button_start.env'))
            original = (REPO / 'config/button_start.env.example').read_text()
            for extra in ('', 'WZ_MODE=segments\nCAMERA_EXPOSURE_MODE=manual\nCAMERA_EXPOSURE_MS=5\n',
                          'POSITION_GAIN=2\nSHAPE_CUE_SCORE_MIN=2\n'):
                with self.subTest(config=extra):
                    (checkout / 'config/button_start.env').write_text(original + '\n' + extra)
                    command = shlex.split(self.run_bash('scripts/run_button_vision.sh',
                                                       '--dry-run', cwd=checkout))
                    with patch('sys.argv', command[2:]):
                        args = run_policy_vision.parse_args()
                    self.assertEqual(args.wz_mode, 'segments' if extra.startswith('WZ_MODE') else 'heading')
                    self.assertEqual(args.camera_exposure_mode, 'manual' if extra.startswith('WZ_MODE') else 'keep')

    def test_camera_controls_defaults_and_configured_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config', ignore=shutil.ignore_patterns('button_start.env'))
            original = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                if not line.startswith(('CAMERA_EXPOSURE_', 'CAMERA_BRIGHTNESS=', 'CAMERA_CONTRAST=',
                    'CAMERA_SATURATION=', 'CAMERA_SHARPNESS=', 'CAMERA_WHITE_BALANCE_', 'CAMERA_POWER_LINE_'))) + '\n'
            for extra in ('', 'CAMERA_EXPOSURE_MODE=manual\nCAMERA_EXPOSURE_MS=5\nCAMERA_BRIGHTNESS=-2\n'
                           'CAMERA_SHARPNESS=3\nCAMERA_WHITE_BALANCE_MODE=manual\nCAMERA_WHITE_BALANCE_K=4600\nCAMERA_POWER_LINE_HZ=50\n'):
                (checkout / 'config/button_start.env').write_text(original+extra)
                args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
                self.assertEqual(args[args.index('--camera-exposure-mode')+1], 'manual' if extra else 'keep')
                if extra:
                    for flag,value in (('--camera-exposure-ms','5'),('--camera-brightness','-2'),
                        ('--camera-sharpness','3'),('--camera-white-balance-k','4600'),('--camera-power-line-hz','50')):
                        self.assertEqual(args[args.index(flag)+1],value)
                else:
                    self.assertNotIn('--camera-exposure-ms',args)
                    self.assertNotIn('--camera-brightness',args)
                    self.assertNotIn('--camera-white-balance-k',args)

    def test_first_card_walk_time_defaults_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config', ignore=shutil.ignore_patterns('button_start.env'))
            original = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                if not line.startswith('STARTUP_FIRST_WALK_S=')) + '\n'
            for value in (None, '0.3', '1.0'):
                (checkout / 'config/button_start.env').write_text(original+(f'STARTUP_FIRST_WALK_S={value}\n' if value else ''))
                args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
                self.assertEqual(args[args.index('--startup-first-walk-s')+1], value or '0.5')
            sequence = '[{"duration_s":0.5,"vx":0.2,"wz":0},{"duration_s":0.5,"vx":0.2,"wz":-0.5}]'
            (checkout / 'config/button_start.env').write_text(original+f"STARTUP_SEQUENCE='{sequence}'\n")
            args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
            self.assertEqual(args[args.index('--startup-sequence')+1], sequence)

    def test_video_defaults_old_config_and_passes_disable_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config', ignore=shutil.ignore_patterns('button_start.env'))
            original = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                if not line.startswith(('RECORD_VIDEO=', 'VIDEO_'))) + '\n'
            for extra, flag, fps, width in (('', '--record-video', '10', '960'),
                    ('RECORD_VIDEO=0\nVIDEO_FPS=5\nVIDEO_WIDTH=640\n', '--no-record-video', '5', '640')):
                (checkout / 'config/button_start.env').write_text(original+extra)
                args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
                self.assertIn(flag, args)
                self.assertEqual(args[args.index('--video-fps')+1], fps)
                self.assertEqual(args[args.index('--video-width')+1], width)

    def test_vision_wrapper_passes_hold_filter_and_defaults_existing_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / 'robot'
            shutil.copytree(REPO / 'scripts', checkout / 'scripts')
            shutil.copytree(REPO / 'config', checkout / 'config', ignore=shutil.ignore_patterns('button_start.env'))
            original = '\n'.join(line for line in (REPO / 'config/button_start.env.example').read_text().splitlines()
                                 if not line.startswith(('COMMAND_MIN_HOLD_S=', 'STEERING_'))) + '\n'
            for extra, expected in (('', ('0', 'legacy')), ('COMMAND_MIN_HOLD_S=0.23\nSTEERING_FILTER_MODE=active\nSTEERING_FILTER_ALGORITHM=ema\nSTEERING_FILTER_MIN_HZ=1\nSTEERING_FILTER_ROBUST_TAU_S=0.7\nSTEERING_LOSS_MAX_S=0.6\nSTEERING_SEGMENT_FALLBACK=0\n', ('0.23','active'))):
                (checkout / 'config/button_start.env').write_text(original+extra)
                args = shlex.split(self.run_bash('scripts/run_button_vision.sh', '--dry-run', cwd=checkout))
                self.assertEqual(args[args.index('--command-min-hold-s')+1], expected[0])
                self.assertEqual(args[args.index('--steering-filter-mode')+1], expected[1])
                if extra:
                    self.assertEqual(args[args.index('--steering-filter-algorithm')+1], 'ema')
                    self.assertEqual(args[args.index('--steering-filter-min-hz')+1], '1')
                    self.assertEqual(args[args.index('--steering-filter-robust-tau-s')+1], '0.7')
                    self.assertEqual(args[args.index('--steering-loss-max-s')+1], '0.6')
                    self.assertIn('--no-steering-segment-fallback',args)

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
