"""Transport substitutes validate logging alignment; no hardware execution claims."""

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import main
import phase_clock_main
from command_source import CommandSnapshot
from control_diagnostics import ControlDiagnostics, HEADER, receive_metadata
from models.phase_clock_model_850.deployment.phase_clock import PhaseClockConfig
from models.phase_clock_model_850.deployment.policy_interface import PolicyController
from protocol import STATE_IMU_VALID, STATE_ENCODERS_VALID
from serial_link import SerialLink
from target_safety import TargetSafety


def state(q=None, sequence=1):
    return SimpleNamespace(sequence=sequence, timestamp_us=123456 + sequence,
        status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
        joint_position=config.policy_to_motor_position(config.Q_DEFAULT if q is None else q),
        joint_velocity=np.zeros(12, dtype=np.float32),
        accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32), gyro_rad_s=np.zeros(3, dtype=np.float32),
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32))


def vector(row, prefix):
    return np.array([float(row[f"{prefix}_{name}"]) for name in config.JOINT_NAMES])


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ControlDiagnosticsTests(unittest.TestCase):
    def make_logger(self, folder):
        model = Path(folder) / "model.onnx"
        model.write_bytes(b"test-model-identity")
        return ControlDiagnostics(folder, SimpleNamespace(model=model), TargetSafety(),
                                  {"walking49": model}, state(), "test", "old.csv")

    def values(self):
        current = np.zeros(12, dtype=np.float32)
        previous = current.copy()
        previous[1] = .3
        raw = current.copy()
        raw[0], raw[1] = 2., .5
        target, trace = TargetSafety().apply_with_trace(raw, previous, current, .02)
        return dict(state=state(np.full(12, .05)), trace=trace,
            motor_target=config.policy_to_motor_position(target), action=np.full(12, .4),
            reference_qd=np.zeros(12), reference_accel=np.array([0., 0., 9.81]),
            reference_gyro=np.zeros(3), reference_gravity=np.array([0., 0., -1.]),
            observation=np.zeros(49), held_reference=1, step=0, policy_mode="walking49",
            target_source="onnx", send_result="written")

    def test_trace_is_same_control_result_and_stage_masks_are_exact(self):
        values = self.values()
        trace = values["trace"]
        expected = TargetSafety().apply(trace.raw_target, trace.previous_target, trace.reference_q, trace.dt)
        np.testing.assert_array_equal(expected, trace.final_target)
        for mask, before, after in ((trace.absolute_mask, trace.raw_target, trace.absolute_target),
                                  (trace.slew_mask, trace.absolute_target, trace.slew_target),
                                  (trace.window_mask, trace.slew_target, trace.final_target)):
            np.testing.assert_array_equal(mask, before != after)
        self.assertTrue(trace.absolute_mask[0])
        self.assertTrue(trace.slew_mask[0])
        self.assertTrue(trace.window_mask[1])
        expected[:] = 99
        self.assertFalse(np.any(trace.final_target == 99))

    def test_schema_held_reference_and_unique_manifest_identity(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            first, second = self.make_logger(folder), self.make_logger(folder)
            self.assertNotEqual(first.csv_path, second.csv_path)
            first.write(**self.values())
            first.close()
            second.close()
            rows = read_rows(first.csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), set(HEADER))
            self.assertNotIn(None, rows[0])
            np.testing.assert_array_equal(vector(rows[0], "reference_q_policy"), np.zeros(12))
            np.testing.assert_allclose(vector(rows[0], "received_q_policy"), .05)
            np.testing.assert_allclose(vector(rows[0], "motor_send_argument"),
                                       config.policy_to_motor_position(vector(rows[0], "final_target_policy")))
            self.assertEqual(rows[0]["held_reference"], "1")
            manifest = json.loads(first.manifest_path.read_text())
            self.assertEqual(manifest["rows_written"], 1)
            self.assertEqual(manifest["csv_columns"], HEADER)
            self.assertEqual(manifest["firmware_sha256"], "unknown_not_reported_by_protocol")
            self.assertEqual(manifest["sources"][0]["sha256"], hashlib.sha256(b"test-model-identity").hexdigest())
            self.assertEqual(manifest["target_limits"]["max_deviation_deg"], 10)
            self.assertIn("not raw", manifest["received_data_semantics"])

    def test_absent_fixed_action_and_46_width_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            logger = self.make_logger(folder)
            values = self.values()
            values.update(action=None, observation=None, target_source="fixed_joint_frames", policy_mode="fixed")
            logger.write(**values)
            values.update(action=np.zeros(12), observation=np.arange(46), target_source="onnx", policy_mode="onefoot46")
            logger.write(**values)
            logger.close()
            rows = read_rows(logger.csv_path)
            self.assertEqual(rows[0]["obs_dim"], "0")
            self.assertEqual(rows[0]["observation_available"], "0")
            self.assertEqual(rows[0][f"raw_action_{config.JOINT_NAMES[0]}"], "")
            self.assertEqual(rows[1]["obs_dim"], "46")
            self.assertEqual(rows[1]["observation_45"], "45.0")
            self.assertEqual(rows[1]["observation_46"], "")
            self.assertEqual([row["row_index"] for row in rows], ["0", "1"])

    def test_bad_data_and_write_failures_do_not_silently_pass(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            logger = self.make_logger(folder)
            for changed in ({"motor_target": np.full(12, 99)}, {"observation": np.zeros(48)},
                            {"action": np.full(12, np.nan)}, {"dt_override": 1}, {"infer_ms": np.inf}):
                values = self.values()
                values.update(changed)
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    logger.write(**values)
            self.assertEqual(logger.rows_written, 0)
            writer = logger.writer
            logger.writer = Mock()
            logger.writer.writerow.side_effect = OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                logger.write(**self.values())
            self.assertEqual(logger.rows_written, 0)
            logger.writer = writer
            logger.close("fault", "disk full")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                logger.write(**self.values())

    def test_receive_age_matches_specific_packet_and_missing_is_explicit(self):
        link = SerialLink.__new__(SerialLink)
        link._lock = threading.Lock()
        old, new = state(sequence=1), state(sequence=2)
        link._state_receive_times = {(old.sequence, old.timestamp_us): 10.,
                                     (new.sequence, new.timestamp_us): 10.02}
        result = receive_metadata(link, old, 10.03)
        self.assertAlmostEqual(result["state_receive_age_s"], .03)
        self.assertEqual(result["state_receive_monotonic_s"], 10.)
        missing = receive_metadata(link, state(sequence=3), 10.03)
        self.assertFalse(missing["state_receive_metadata_available"])
        self.assertIsNone(missing["state_receive_age_s"])
        self.assertFalse(receive_metadata(object(), old, 10.03)["state_receive_metadata_available"])

    def test_main_send_and_held_reference_are_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["main.py", "--model", "fake.onnx", "--no-plot",
                    "--command-source", "vision", "--one-foot-model", "foot.onnx", "--attitude-port", "0",
                    "--diagnostic-log-dir", folder]):
                args = main.parse_args()
            frozen, live = state(), state(config.Q_DEFAULT + .35, 2)
            window = CommandSnapshot(np.zeros(3, dtype=np.float32), card_tilt=True)
            gone = CommandSnapshot(np.zeros(3, dtype=np.float32), card_tilt=False)
            policy = Mock()
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.)
            link = Mock()
            link.wait_for_state.return_value = frozen
            link.get_latest_state.side_effect = [frozen, frozen, live, RuntimeError("end test")]
            link.get_action_status.return_value = 0
            link.send_command.return_value = True
            source = Mock()
            source.get_snapshot.side_effect = [window, gone, gone]
            with patch.object(main, "parse_args", return_value=args), patch.object(main.signal, "signal"), \
                    patch.object(main, "HumanoidPolicy", return_value=policy), patch.object(main, "OneFootPolicy"), \
                    patch.object(main, "UdpCommandSource", return_value=source), \
                    patch.object(main, "SerialLink", return_value=link), patch.object(main, "PositionCsvLogger"), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(), 1)
            rows = read_rows(next(Path(folder).glob("*.csv")))
            self.assertEqual(len(rows), 3)
            for row, call in zip(rows, link.send_command.call_args_list):
                np.testing.assert_array_equal(vector(row, "motor_send_argument"), call.args[1])
                self.assertEqual(row["send_result"], "written")
            self.assertEqual(rows[-1]["held_reference"], "1")
            np.testing.assert_allclose(vector(rows[-1], "reference_q_policy"), config.Q_DEFAULT)
            np.testing.assert_allclose(vector(rows[-1], "received_q_policy"), config.Q_DEFAULT + .35)

    def test_phase_exact_observation_send_and_completion_are_aligned(self):
        seen = []
        def inference(obs):
            seen.append(obs.copy())
            return np.full((1, 12), .4, dtype=np.float32)
        controller = PolicyController(inference, PhaseClockConfig(walk_end_s=.02, lead_end_s=.04, sequence_end_s=.06))
        link = Mock()
        link.wait_for_state.return_value = state()
        link.get_latest_state.return_value = state()
        link.send_command.return_value = True
        cue = Mock()
        cue.poll.side_effect = [False, True, True, True, True, True, True]
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["phase_clock_main.py", "--log", str(Path(folder) / "legacy.csv"),
                                    "--diagnostic-log-dir", str(Path(folder) / "diagnostics")]):
                args = phase_clock_main.parse_args()
            with patch.object(phase_clock_main, "parse_args", return_value=args), \
                    patch.object(phase_clock_main, "load_crossing_controller", return_value=controller), \
                    patch.object(phase_clock_main, "StartCueSocket", return_value=cue), \
                    patch.object(phase_clock_main, "SerialLink", return_value=link), \
                    patch.object(phase_clock_main, "send_disable"), redirect_stdout(io.StringIO()):
                self.assertEqual(phase_clock_main.main(), 0)
            rows = read_rows(next((Path(folder) / "diagnostics").glob("*.csv")))
            sent = [row for row in rows if row["send_result"] == "written"]
            self.assertEqual(len(sent), len(link.send_command.call_args_list))
            for row, call in zip(sent, link.send_command.call_args_list):
                np.testing.assert_array_equal(vector(row, "motor_send_argument"), call.args[1])
            model_rows = [row for row in rows if row["target_source"] == "onnx"]
            self.assertEqual(len(model_rows), len(seen))
            for row, actual in zip(model_rows, seen):
                np.testing.assert_array_equal([float(row[f"observation_{i}"]) for i in range(49)], actual[0])
            self.assertEqual(rows[0]["target_source"], "initial_pose_hold")
            self.assertEqual(rows[0][f"raw_action_{config.JOINT_NAMES[0]}"], "")
            self.assertEqual(rows[-1]["target_source"], "sequence_end_no_target")
            self.assertEqual(rows[-1]["send_result"], "not_attempted")
            self.assertEqual(rows[-1][f"final_target_policy_{config.JOINT_NAMES[0]}"], "")

    def test_main_fixed_onefoot_and_upright_log_the_actual_target_source(self):
        for mode in ("fixed", "one-foot", "upright"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                argv = ["main.py", "--no-plot", "--attitude-port", "0", "--diagnostic-log-dir", folder]
                if mode == "fixed":
                    frames = Path(folder) / "frames.json"
                    frames.write_text(json.dumps(dict(format="joint_frames_v1", hz=50,
                        joint_names=list(config.JOINT_NAMES), frames=[config.Q_DEFAULT.tolist()])))
                    argv += ["--fixed-policy", str(frames)]
                elif mode == "one-foot":
                    argv += ["--model", "fake.onnx", "--policy", mode]
                else:
                    argv += ["--model", "fake.onnx", "--command-source", "vision"]
                with patch("sys.argv", argv):
                    args = main.parse_args()
                policy = Mock()
                policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(46), 0.)
                link, source = Mock(), Mock()
                link.wait_for_state.return_value = state()
                link.get_latest_state.side_effect = [state(), RuntimeError("end test")]
                link.send_command.return_value = True
                source.get_snapshot.return_value = CommandSnapshot(np.zeros(3), hold_upright=True)
                with patch.object(main, "parse_args", return_value=args), patch.object(main.signal, "signal"), \
                        patch.object(main, "HumanoidPolicy", return_value=policy), \
                        patch.object(main, "OneFootPolicy", return_value=policy), \
                        patch.object(main, "UdpCommandSource", return_value=source), \
                        patch.object(main, "SerialLink", return_value=link), patch.object(main, "PositionCsvLogger"), \
                        redirect_stdout(io.StringIO()):
                    main.main()
                row = read_rows(next(Path(folder).glob("*.csv")))[0]
                np.testing.assert_array_equal(vector(row, "motor_send_argument"), link.send_command.call_args_list[0].args[1])
                if mode == "one-foot":
                    self.assertEqual((row["target_source"], row["policy_mode"], row["obs_dim"]), ("onnx", "onefoot46", "46"))
                else:
                    self.assertEqual(row["obs_dim"], "0")
                    self.assertEqual(row[f"raw_action_{config.JOINT_NAMES[0]}"], "")
                    self.assertEqual(row["target_source"], "fixed_joint_frames" if mode == "fixed" else "hold_upright_zero_target")


if __name__ == "__main__":
    unittest.main()
