"""CLI hold settings must reach model use and the per-run manifest."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import main
from control_diagnostics import ControlDiagnostics
from target_safety import TargetSafety
from test_control_diagnostics import state
import test_walking_command_hold as hold_tests
from walking_command_hold import WalkingCommandHold


class CommandHoldInterfaceTests(unittest.TestCase):
    def test_custom_duration_reaches_real_main_model_and_constructor(self):
        harness = hold_tests.WalkingCommandHoldTests()
        seen, _ = harness.run_main([0., .1, .23, .3],
            [[.2,0,.5], [.2,0,.3], [.2,0,.3], [0,0,0]],
            command_hold=WalkingCommandHold(.23), command_min_hold_s=.23)
        self.assertEqual(harness.command_hold_constructor.kwargs, {'min_hold_s': .23})
        self.assertAlmostEqual(float(seen[1][1][10]), .5)
        self.assertAlmostEqual(float(seen[2][1][10]), .3)
        self.assertEqual(float(seen[3][1][10]), 0.)

    def test_cli_accepts_user_duration_and_zero(self):
        for duration in ('0', '.2', '.5', '1.2'):
            with patch('sys.argv', ['main.py', '--model', 'fake.onnx', '--command-min-hold-s', duration]), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.parse_args().command_min_hold_s, float(duration))

    def test_cli_rejects_nonfinite_or_negative_duration(self):
        for duration in ('-1', 'nan', 'inf'):
            with patch('sys.argv', ['main.py', '--model', 'fake.onnx', '--command-min-hold-s', duration]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args()

    def test_manifest_records_actual_user_duration(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            logger = ControlDiagnostics(folder, SimpleNamespace(command_min_hold_s=.23),
                TargetSafety(), {}, state(), 'main', 'old.csv')
            logger.close()
            manifest = json.loads(logger.manifest_path.read_text())
            self.assertEqual(manifest['walking_command_contract']['minimum_hold_s'], .23)


if __name__ == '__main__':
    unittest.main()
