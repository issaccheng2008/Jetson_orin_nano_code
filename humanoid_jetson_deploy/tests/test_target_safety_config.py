from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
from fixed_joint_policy import FixedJointPolicy
import main
import phase_clock_main
from protocol import STATE_IMU_VALID, STATE_ENCODERS_VALID
from target_safety import TargetSafety


class TargetSafetyConfigTests(unittest.TestCase):
    def test_default_chain_matches_legacy_formulas(self):
        limits = TargetSafety()
        rng = np.random.default_rng(7)
        for _ in range(40):
            current = rng.uniform(config.Q_LOWER + .1, config.Q_UPPER - .1).astype(np.float32)
            previous = current + rng.uniform(-.1, .1, 12).astype(np.float32)
            target = rng.uniform(-3, 3, 12).astype(np.float32)
            dt = float(rng.uniform(.005, .05))
            clipped = config.clamp_policy_target(target)
            old_slew = previous + np.clip(clipped - previous, -3 * dt, 3 * dt)
            expected = config.clamp_policy_target_to_current(old_slew, current)
            np.testing.assert_array_equal(limits.apply(target, previous, current, dt), expected)
            np.testing.assert_array_equal(main.slew_limit(clipped, previous, dt), old_slew)

    def test_custom_values_and_zero_mean_disable_the_requested_stage(self):
        previous = np.zeros(12, dtype=np.float32)
        target = np.full(12, .3, dtype=np.float32)
        np.testing.assert_allclose(TargetSafety(max_speed_rad_s=1., max_deviation_deg=0).apply(
            target, previous, previous, .02), .02)
        np.testing.assert_allclose(TargetSafety(max_speed_rad_s=0, max_deviation_deg=5).apply(
            target, previous, previous, .02), np.deg2rad(5))
        np.testing.assert_array_equal(TargetSafety(max_speed_rad_s=0, max_deviation_deg=0).apply(
            target, previous, previous, .02), target)
        np.testing.assert_array_equal(TargetSafety(margin_rad=0, max_speed_rad_s=0,
            max_deviation_deg=0).apply(np.full(12, 9), previous, previous, .02), config.Q_UPPER)

    def test_invalid_limits_and_disabled_checks_still_reject_bad_vectors(self):
        for key in ("margin_rad", "max_speed_rad_s", "max_deviation_deg"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TargetSafety(**{key: value})
        for kwargs in ({"lower": [0] * 11}, {"upper": [float("nan")] * 12},
                       {"lower": [1] * 12, "upper": [0] * 12}, {"margin_rad": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TargetSafety(**kwargs)
        limits = TargetSafety(max_speed_rad_s=0, max_deviation_deg=0)
        vectors = [np.zeros(12), np.zeros(12), np.zeros(12)]
        for index in range(3):
            for invalid in (np.zeros(11), np.full(12, np.nan)):
                values = vectors.copy()
                values[index] = invalid
                with self.subTest(index=index), self.assertRaises(ValueError):
                    limits.apply(*values, .02)
        for dt in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                limits.apply(*vectors, dt)

    def write_bounds(self, directory, **changes):
        payload = dict(joint_names=list(config.JOINT_NAMES), lower_rad=[-.8] * 12,
                       upper_rad=[.8] * 12)
        payload.update(changes)
        path = Path(directory) / "limits.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_json_bounds_validate_names_shape_and_fixed_frames_consistently(self):
        with tempfile.TemporaryDirectory() as folder:
            original = config.Q_UPPER.copy()
            limits = TargetSafety.from_args(SimpleNamespace(joint_limits_json=self.write_bounds(folder),
                joint_limit_margin_rad=0, max_target_speed_rad_s=0, max_target_deviation_deg=0))
            frames_path = Path(folder) / "frames.json"
            frames_path.write_text(json.dumps(dict(format="joint_frames_v1", hz=50,
                joint_names=list(config.JOINT_NAMES), frames=[[.6] * 12])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint limit"):
                FixedJointPolicy(frames_path)
            np.testing.assert_allclose(FixedJointPolicy(frames_path, limits).next_target(), .6)
            np.testing.assert_array_equal(config.Q_UPPER, original)
            for changes in ({"joint_names": list(reversed(config.JOINT_NAMES))},
                            {"lower_rad": [0] * 11}, {"upper_rad": [float("inf")] * 12},
                            {"extra": 1}, {"lower_rad": [False] * 12}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    TargetSafety.from_args(SimpleNamespace(joint_limits_json=self.write_bounds(folder, **changes)))

    def test_both_cli_entries_accept_options_and_reject_invalid_configuration(self):
        for module, base in ((main, ["main.py", "--model", "unused.onnx"]),
                             (phase_clock_main, ["phase_clock_main.py"])):
            with patch("sys.argv", base + ["--max-target-speed-rad-s", "0",
                    "--max-target-deviation-deg", "15", "--joint-limit-margin-rad", "0"]):
                limits = TargetSafety.from_args(module.parse_args())
                self.assertEqual((limits.max_speed_rad_s, limits.max_deviation_deg, limits.margin_rad), (0, 15, 0))
            for option in ("--max-target-speed-rad-s", "--max-target-deviation-deg", "--joint-limit-margin-rad"):
                with patch("sys.argv", base + [option, "nan"]), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    module.parse_args()

    def state(self, sequence=1):
        return SimpleNamespace(sequence=sequence, timestamp_us=123,
            status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
            joint_position=config.Q_DEFAULT.copy(), joint_velocity=np.zeros(12),
            accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3),
            orientation_wxyz=np.array([1, 0, 0, 0]))

    def test_regular_walking_onefoot_and_fixed_send_configured_target(self):
        target = config.Q_DEFAULT + .1
        for mode in ("walking", "one-foot", "fixed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                argv = ["main.py", "--no-plot", "--max-target-speed-rad-s", "0",
                        "--max-target-deviation-deg", "0", "--joint-limit-margin-rad", "0"]
                if mode == "fixed":
                    frames = Path(folder) / "frames.json"
                    frames.write_text(json.dumps(dict(format="joint_frames_v1", hz=50,
                        joint_names=list(config.JOINT_NAMES), frames=[target.tolist()])), encoding="utf-8")
                    argv += ["--fixed-policy", str(frames)]
                else:
                    argv += ["--model", "fake.onnx", "--policy", mode]
                with patch("sys.argv", argv):
                    args = main.parse_args()
                policy = Mock()
                policy.step.return_value = (target, np.zeros(12), np.zeros(49), 0.)
                link = Mock()
                link.wait_for_state.return_value = self.state()
                link.get_latest_state.side_effect = ([self.state(), self.state(2)] if mode == "fixed"
                                                     else [self.state(), RuntimeError("end test")])
                output = io.StringIO()
                with patch.object(main, "parse_args", return_value=args), patch.object(main.signal, "signal"), \
                        patch.object(main, "SerialLink", return_value=link), \
                        patch.object(main, "HumanoidPolicy", return_value=policy), \
                        patch.object(main, "OneFootPolicy", return_value=policy), \
                        patch.object(main, "PositionCsvLogger"), patch.object(main.time, "sleep"), \
                        redirect_stdout(output):
                    main.main()
                np.testing.assert_array_equal(link.send_command.call_args_list[0].args[1], target)
                self.assertIn("slew=0 rad/s", output.getvalue())

    def test_phase_loop_uses_same_configured_chain(self):
        target = config.Q_DEFAULT + .1
        controller = Mock()
        controller.start.return_value = True
        controller.clock.read.return_value.tick = 1
        controller.tick.side_effect = [SimpleNamespace(sequence_finished=False, active=True,
            joint_targets=target, phase=0, elapsed_s=0., action=np.zeros(12)),
            SimpleNamespace(sequence_finished=True, phase=3, elapsed_s=.76)]
        cue = Mock()
        cue.poll.return_value = True
        link = Mock()
        link.wait_for_state.return_value = self.state()
        link.get_latest_state.return_value = self.state()
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["phase_clock_main.py", "--max-target-speed-rad-s", "0",
                    "--max-target-deviation-deg", "0", "--log", str(Path(folder) / "phase.csv")]):
                args = phase_clock_main.parse_args()
            with patch.object(phase_clock_main, "parse_args", return_value=args), \
                    patch.object(phase_clock_main, "load_crossing_controller", return_value=controller), \
                    patch.object(phase_clock_main, "StartCueSocket", return_value=cue), \
                    patch.object(phase_clock_main, "SerialLink", return_value=link), \
                    patch.object(phase_clock_main, "send_disable"), redirect_stdout(io.StringIO()):
                self.assertEqual(phase_clock_main.main(), 0)
        np.testing.assert_array_equal(link.send_command.call_args_list[0].args[1], target)


if __name__ == "__main__":
    unittest.main()
