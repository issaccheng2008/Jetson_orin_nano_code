"""Competition start: motor policy starts once and vision waits for STM32 ACK."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
sys.path.insert(0, str(ROOT / "humanoid_jetson_deploy"))

import main
import policy_gate_launcher
import run_policy_vision
from protocol import STATE_COMMAND_FRESH, STATE_MOTORS_ENABLED


class PolicyGateLauncherTests(unittest.TestCase):
    def test_configurable_hold_reaches_child_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'walk.onnx').touch()
            (root / 'foot.onnx').touch()
            policy = Mock(pid=1234, stdout=io.BytesIO())
            with (patch.object(policy_gate_launcher, 'REPO_ROOT', root),
                  patch.object(policy_gate_launcher.subprocess, 'Popen', side_effect=[policy, Mock()]) as popen):
                launcher = policy_gate_launcher.PolicyGateLauncher('walk.onnx', '/dev/ttyACM0', 1200,
                    one_foot_model='foot.onnx', command_min_hold_s=.23)
                launcher.start()
                command = popen.call_args_list[0].args[0]
                self.assertEqual(command[command.index('--command-min-hold-s')+1], '0.23')
                launcher.close()

    def test_model_cli_defaults_and_independent_overrides(self):
        walking = "humanoid_jetson_deploy/policy_49_max.onnx"
        one_foot = "humanoid_jetson_deploy/policy-one-foot-standing_old.onnx"
        for options, expected_walk, expected_foot in (
            ([], walking, one_foot),
            (["--start-policy-model", "walk.onnx"], "walk.onnx", one_foot),
            (["--start-policy-one-foot-model", "foot.onnx"], walking, "foot.onnx"),
        ):
            with self.subTest(options=options), patch("sys.argv", ["run_policy_vision.py", *options]):
                args = run_policy_vision.parse_args()
                self.assertEqual(args.start_policy_model, expected_walk)
                self.assertEqual(args.start_policy_one_foot_model, expected_foot)

    def test_command_matches_the_competition_policy_and_starts_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "humanoid_jetson_deploy").mkdir()
            (root / "humanoid_jetson_deploy/policy_49_max.onnx").touch()
            (root / "humanoid_jetson_deploy/policy-one-foot-standing_old.onnx").touch()
            policy = Mock(pid=1234, stdout=io.BytesIO())
            policy.poll.return_value = None
            tee = Mock()
            with (patch.object(policy_gate_launcher, "REPO_ROOT", root),
                  patch.object(policy_gate_launcher.subprocess, "Popen",
                               side_effect=[policy, tee]) as popen):
                launcher = policy_gate_launcher.PolicyGateLauncher(
                    "humanoid_jetson_deploy/policy_49_max.onnx", "/dev/ttyACM0", 1200)
                launcher.start()
                launcher.start()
                self.assertEqual(popen.call_count, 2)  # policy and tee, once each
                command = popen.call_args_list[0].args[0]
                self.assertIn("--enable-motors", command)
                self.assertEqual(command[command.index("--model") + 1],
                                 "humanoid_jetson_deploy/policy_49_max.onnx")
                self.assertEqual(command[command.index("--one-foot-model") + 1],
                                 "humanoid_jetson_deploy/policy-one-foot-standing_old.onnx")
                self.assertEqual(command[command.index("--max-seconds") + 1], "1200")
                self.assertEqual(command[command.index("--udp-command-port") + 1], "5005")
                self.assertEqual(command[command.index("--startup-ready-file") + 1],
                                 str(launcher.ready_file))
                self.assertEqual(popen.call_args_list[1].args[0],
                                 ["tee", str(launcher.log_path), str(launcher.live_log_path)])
                self.assertFalse(launcher.ready())
                launcher.ready_file.write_text("ready\n")
                self.assertTrue(launcher.ready())
                launcher.close()
                policy.terminate.assert_called_once()
                self.assertFalse(launcher.ready_file.exists())

    def test_custom_one_foot_model_is_checked_and_sent_to_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "walk.onnx").touch()
            (root / "custom foot.onnx").touch()
            policy = Mock(pid=1234, stdout=io.BytesIO())
            with (patch.object(policy_gate_launcher, "REPO_ROOT", root),
                  patch.object(policy_gate_launcher.subprocess, "Popen",
                               side_effect=[policy, Mock()]) as popen):
                launcher = policy_gate_launcher.PolicyGateLauncher(
                    "walk.onnx", "/dev/ttyACM0", 1200,
                    one_foot_model="custom foot.onnx")
                launcher.start()
                command = popen.call_args_list[0].args[0]
                self.assertEqual(command[command.index("--model") + 1], "walk.onnx")
                self.assertEqual(command[command.index("--one-foot-model") + 1], "custom foot.onnx")
                launcher.close()

    def test_missing_selected_one_foot_model_prevents_child_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "walk.onnx").touch()
            with (patch.object(policy_gate_launcher, "REPO_ROOT", root),
                  patch.object(policy_gate_launcher.subprocess, "Popen") as popen):
                launcher = policy_gate_launcher.PolicyGateLauncher(
                    "walk.onnx", "/dev/ttyACM0", 1200,
                    one_foot_model="missing.onnx")
                with self.assertRaisesRegex(RuntimeError, "missing.onnx"):
                    launcher.start()
                popen.assert_not_called()

    def test_policy_failure_reports_recent_child_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "humanoid_jetson_deploy").mkdir()
            (root / "humanoid_jetson_deploy/policy(13).onnx").touch()
            (root / "humanoid_jetson_deploy/policy-one-foot-standing_old.onnx").touch()
            policy = Mock(pid=1234, stdout=io.BytesIO())
            policy.poll.return_value = 1
            tee = Mock()
            with (patch.object(policy_gate_launcher, "REPO_ROOT", root),
                  patch.object(policy_gate_launcher.subprocess, "Popen",
                               side_effect=[policy, tee])):
                launcher = policy_gate_launcher.PolicyGateLauncher(
                    "humanoid_jetson_deploy/policy(13).onnx", "/dev/ttyACM0", 1200)
                launcher.start()
                launcher.log_path.write_text("startup\nFAULT: USB write timeout\n")
                with self.assertRaisesRegex(RuntimeError, "FAULT: USB write timeout") as error:
                    launcher.ready()
                self.assertIn(str(launcher.log_path), str(error.exception))
                tee.wait.assert_called_with(timeout=1.0)
                launcher.close()

    def test_real_child_stream_reaches_both_logs_and_error_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "humanoid_jetson_deploy").mkdir()
            (root / "humanoid_jetson_deploy/policy(13).onnx").touch()
            (root / "humanoid_jetson_deploy/policy-one-foot-standing_old.onnx").touch()
            (root / "humanoid_jetson_deploy/main.py").write_text(
                "import sys\nprint('policy startup', flush=True)\n"
                "print('FAULT: simulated failure', file=sys.stderr, flush=True)\n"
                "raise SystemExit(1)\n")
            with patch.object(policy_gate_launcher, "REPO_ROOT", root):
                launcher = policy_gate_launcher.PolicyGateLauncher(
                    "humanoid_jetson_deploy/policy(13).onnx", "/dev/null", 1200)
                launcher.start()
                launcher.process.wait(timeout=5)
                with self.assertRaisesRegex(RuntimeError, "FAULT: simulated failure"):
                    launcher.ready()
                self.assertEqual(launcher.log_path.read_text(),
                                 launcher.live_log_path.read_text())
                self.assertIn("policy startup", launcher.log_path.read_text())
                launcher.close()

    def test_ready_marker_requires_stm32_fresh_enabled_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "ready"
            write = main.write_startup_ready_if_confirmed
            good = STATE_COMMAND_FRESH | STATE_MOTORS_ENABLED
            self.assertFalse(write(str(marker), good, True, 0))
            self.assertFalse(write(str(marker), STATE_COMMAND_FRESH, True, 1))
            self.assertFalse(write(str(marker), STATE_MOTORS_ENABLED, True, 1))
            self.assertFalse(write(str(marker), good, False, 1))
            self.assertFalse(marker.exists())
            self.assertTrue(write(str(marker), good, True, 1))
            self.assertEqual(marker.read_text(), "ready\n")


if __name__ == "__main__":
    unittest.main()
