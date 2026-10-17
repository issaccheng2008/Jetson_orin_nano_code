"""Old installed config must not drift from the requested debug profile."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision/jetson'))
from run_policy_vision import parse_args
from test_button_autostart import BASH


@unittest.skipUnless(BASH, 'Bash required')
class ButtonDebugProfileTests(unittest.TestCase):
    def test_old_config_defaults_to_debug_but_can_opt_into_configured(self):
        for mode in (None, 'debug', 'configured'):
            with self.subTest(profile=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                shutil.copytree(ROOT / 'scripts', root / 'scripts')
                shutil.copytree(ROOT / 'config', root / 'config',
                                ignore=shutil.ignore_patterns('button_start.env'))
                text = (ROOT / 'config/button_start.env.example').read_text()
                text = '\n'.join(line for line in text.splitlines() if not line.startswith('VISION_PROFILE='))
                text += '\nLINE_PREPROCESS=contrast\nSINGLE_BOUNDARY_TRACKING=0\n'
                text += "HEADING_REGIONS_CM='[[20,22],[24,26]]'\nSTEERING_FILTER_MODE=active\n"
                text += "STEERING_ANGLE_WZ_TABLE='[[-90,-15,-0.1],[-15,15,0.1],[15,90,0.3]]'\n"
                if mode:
                    text += f'VISION_PROFILE={mode}\n'
                (root / 'config/button_start.env').write_text(text)
                output = subprocess.check_output([BASH, 'scripts/run_button_vision.sh', '--dry-run'],
                    cwd=root, text=True)
                argv = shlex.split(output)
                with patch('sys.argv', argv[2:]):
                    args = parse_args()
                self.assertEqual(args.start_gate, 'button')
                self.assertTrue(args.start_policy_on_gate)
                self.assertEqual(args.connector_port, 5006)
                self.assertEqual(args.attitude_port, 5007)
                if mode == 'configured':
                    self.assertEqual(args.line_preprocess, 'contrast')
                    self.assertIsNotNone(args.steering_angle_wz_table)
                else:
                    self.assertEqual(args.vision_profile, 'debug')
                    self.assertEqual(args.line_preprocess, 'legacy')
                    self.assertTrue(args.single_boundary_tracking)
                    self.assertIsNone(args.heading_regions_cm)
                    self.assertIsNone(args.steering_angle_wz_table)
                    self.assertEqual(args.wz_step, .5)
                    self.assertEqual(args.card_trigger_dist_cm, 43.)
                    self.assertFalse(args.attitude_pitch_correction)
