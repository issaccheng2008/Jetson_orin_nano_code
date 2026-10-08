"""The contract is checked at actual model input, with controllable host time."""

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import main
from command_source import CommandSnapshot, UdpCommandSource
from policy_runner import HumanoidPolicy
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID
from walking_command_hold import MIN_WALKING_COMMAND_HOLD_S, WalkingCommandHold


class WalkingCommandHoldTests(unittest.TestCase):
    def test_fixed_duration_pair_latest_request_and_exact_boundary(self):
        self.assertEqual(MIN_WALKING_COMMAND_HOLD_S, .5)
        hold = WalkingCommandHold()
        first = [.2, 0, .5]
        np.testing.assert_allclose(hold.apply(first, 10), first)
        np.testing.assert_allclose(hold.apply([.4, 0, 0], 10.1), first)
        np.testing.assert_allclose(hold.apply([.6, 0, -.5], 10.499), first)
        latest = [.3, 0, 0]
        np.testing.assert_allclose(hold.apply(latest, 10.5), latest)
        np.testing.assert_allclose(hold.apply(first, 10.99), latest)
        np.testing.assert_allclose(hold.apply(first, 11), first)

    def test_repeat_does_not_extend_hold_and_zero_preempts_and_restart_is_fresh(self):
        hold = WalkingCommandHold()
        hold.apply([.2, 0, .5], 10)
        hold.apply([.2, 0, .5], 10.49)
        np.testing.assert_allclose(hold.apply([.4, 0, -.5], 10.5), [.4, 0, -.5])
        np.testing.assert_array_equal(hold.apply([0, 0, 0], 10.51), np.zeros(3))
        self.assertIsNone(hold.applied)
        np.testing.assert_allclose(hold.apply([.3, 0, 0], 10.52), [.3, 0, 0])
        self.assertAlmostEqual(hold.remaining(10.52), .5)

    def test_clearing_takeover_and_ownership(self):
        hold = WalkingCommandHold()
        requested = np.array([.2, 0, .5], dtype=np.float32)
        returned = hold.apply(requested, 10)
        requested[:] = 99
        returned[:] = 98
        np.testing.assert_allclose(hold.applied, [.2, 0, .5])
        hold.clear()
        np.testing.assert_allclose(hold.apply([.4, 0, -.5], 10.1), [.4, 0, -.5])
        self.assertAlmostEqual(hold.remaining(10.1), .5)

    def test_invalid_command_and_clock_are_not_accepted(self):
        hold = WalkingCommandHold()
        for command in ([.1, 0], [.1, 0, np.nan], [.1, 0, np.inf]):
            with self.assertRaises(ValueError):
                hold.apply(command, 10)
        for now in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                hold.apply([.2, 0, .5], now)
        hold.apply([.2, 0, .5], 10)
        with self.assertRaisesRegex(ValueError, "backwards"):
            hold.apply([.4, 0, -.5], 9)

    def test_udp_watchdog_zero_clears_motion_before_minimum_hold_expires(self):
        source = UdpCommandSource.__new__(UdpCommandSource)
        source.lock = threading.Lock()
        source.command = np.array([.2, 0, .5], dtype=np.float32)
        source.qr, source.event_id, source.event_action = -1, 0, -1
        source.hold_upright = source.card_tilt = False
        source.last_update, source.timeout_s = 10., .25
        hold = WalkingCommandHold()
        with patch.object(main.time, "monotonic", return_value=10.1):
            hold.apply(source.get_snapshot().velocity, 10.1)
        with patch.object(main.time, "monotonic", return_value=10.3):
            stopped = hold.apply(source.get_snapshot().velocity, 10.3)
        np.testing.assert_array_equal(stopped, np.zeros(3))
        self.assertIsNone(hold.applied)

    def run_main(self, times, requests, *, hold_upright=None, onefoot_at=None):
        seen = []
        clock = SimpleNamespace(now=9.)
        hold = WalkingCommandHold()
        robot_state = SimpleNamespace(sequence=1, timestamp_us=123,
            status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
            joint_position=config.policy_to_motor_position(config.Q_DEFAULT), joint_velocity=np.zeros(12),
            accel_m_s2=np.array([0, 0, 9.81]), gyro_rad_s=np.zeros(3), orientation_wxyz=np.array([1, 0, 0, 0]))
        actor = HumanoidPolicy.__new__(HumanoidPolicy)
        actor.last_action = np.zeros(12, dtype=np.float32)
        actor.step_distance_m = .05
        def step(**values):
            obs = actor.build_observation(**values)
            seen.append((clock.now, obs.copy()))
            return config.Q_DEFAULT.copy(), np.zeros(12), obs, 0.
        policy, source, link = Mock(), Mock(), Mock()
        policy.step.side_effect = step
        snapshots = [CommandSnapshot(np.array(request, dtype=np.float32),
                        event_id=1 if index == onefoot_at else 0,
                        event_action=3 if index == onefoot_at else -1,
                        hold_upright=bool(hold_upright and hold_upright[index]))
                     for index, request in enumerate(requests)]
        source.get_snapshot.side_effect = snapshots
        ticks = iter(times)
        def get_state(**unused):
            try:
                clock.now = 10. + next(ticks)
            except StopIteration:
                raise RuntimeError("end test")
            return robot_state
        link.wait_for_state.return_value = robot_state
        link.get_latest_state.side_effect = get_state
        link.get_action_status.return_value = 0
        link.send_command.return_value = True
        with tempfile.TemporaryDirectory() as folder:
            argv = ["main.py", "--model", "fake.onnx", "--no-plot", "--attitude-port", "0",
                    "--command-source", "vision", "--diagnostic-log-dir", folder]
            if onefoot_at is not None:
                argv += ["--one-foot-model", "foot.onnx"]
            with patch("sys.argv", argv):
                args = main.parse_args()
            shape, foot = Mock(), Mock()
            shape.phase = "idle"
            shape.action_id = 3
            shape.advance.side_effect = [SimpleNamespace(busy=i == onefoot_at, send_upper=False,
                policy="one-foot" if i == onefoot_at else "walking", support_foot="right",
                lift_command=1., action_id=3) for i in range(len(times))]
            foot.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(46), 0.)
            with patch.object(main, "parse_args", return_value=args), patch.object(main.signal, "signal"), \
                    patch.object(main, "HumanoidPolicy", return_value=policy), \
                    patch.object(main, "OneFootPolicy", return_value=foot), \
                    patch.object(main, "ShapeActionController", return_value=shape), \
                    patch.object(main, "UdpCommandSource", return_value=source), \
                    patch.object(main, "SerialLink", return_value=link), patch.object(main, "PositionCsvLogger"), \
                    patch.object(main, "WalkingCommandHold", return_value=hold), \
                    patch.object(main.time, "monotonic", side_effect=lambda: clock.now), \
                    patch.object(main.time, "monotonic_ns", side_effect=lambda: round(clock.now * 1e9)), \
                    patch.object(main.time, "sleep"), redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(), 1)
            with next(Path(folder).glob("*.csv")).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIsNone(hold.applied)  # fault/disable never leaves a live contract
            self.sent_gain_scales = [call.args[2:4] for call in link.send_command.call_args_list[:len(times)]]
            return seen, rows

    def test_real_main_policy_observation_pair_stays_constant_until_half_second(self):
        first = [.2, 0, .5]
        latest = [.6, 0, 0]
        seen, rows = self.run_main([0, .1, .49, .5, .51, .99, 1.0],
            [first, [.4, 0, 0], [.3, 0, -.5], latest, first, first, first])
        for (_, obs), expected in zip(seen, [first, first, first, latest, latest, latest, first]):
            np.testing.assert_allclose(obs[9:11], [expected[0], expected[2]])
        self.assertAlmostEqual(float(rows[1]["requested_cmd_vx"]), .4)
        self.assertAlmostEqual(float(rows[1]["cmd_vx"]), .2)
        self.assertAlmostEqual(float(rows[1]["command_hold_remaining_s"]), .4)

    def test_real_main_stop_is_immediate_and_restart_does_not_replay_old_command(self):
        seen, _ = self.run_main([0, .1, .2], [[.2, 0, .5], [0, 0, 0], [.4, 0, -.5]])
        np.testing.assert_array_equal(seen[1][1][9:13], np.zeros(4))
        np.testing.assert_allclose(seen[2][1][9:11], [.4, -.5])

    def test_real_main_upright_takeover_clears_contract(self):
        seen, rows = self.run_main([0, .1, .2], [[.2, 0, .5], [0, 0, 0], [.4, 0, -.5]],
                                   hold_upright=[False, True, False])
        self.assertEqual(len(seen), 2)
        self.assertEqual(rows[1]["command_hold_reason"], "upright_takeover")
        np.testing.assert_allclose(seen[-1][1][9:11], [.4, -.5])

    def test_real_main_onefoot_takeover_clears_contract(self):
        seen, rows = self.run_main([0, .1, .2], [[.2, 0, .5], [.3, 0, 0], [.4, 0, -.5]], onefoot_at=1)
        self.assertEqual(len(seen), 2)
        self.assertEqual(rows[1]["command_hold_reason"], "onefoot_takeover")
        np.testing.assert_allclose(seen[-1][1][9:11], [.4, -.5])

    def test_gains_follow_executed_held_command_and_stop_immediately(self):
        self.assertTrue(hasattr(config, "MOTION_GAIN_SCALES"), "three PD profiles are required")
        profiles = {"standing": (1.5, 2), "straight": (1.2, 1.4), "turning": (.8, 1.8)}
        with patch.object(config, "MOTION_GAIN_SCALES", profiles):
            self.run_main([0, .1, .5, .6, .7],
                          [[.2, 0, .5], [.3, 0, 0], [.3, 0, 0], [0, 0, 0], [.2, 0, .5]])
        self.assertEqual(self.sent_gain_scales, [(.8, 1.8), (.8, 1.8), (1.2, 1.4), (1.5, 2), (.8, 1.8)])

    def test_upright_and_onefoot_use_standing_profile(self):
        self.assertTrue(hasattr(config, "MOTION_GAIN_SCALES"), "three PD profiles are required")
        profiles = {"standing": (1.5, 2), "straight": (1.2, 1.4), "turning": (.8, 1.8)}
        with patch.object(config, "MOTION_GAIN_SCALES", profiles):
            self.run_main([0, .1], [[.2, 0, .5], [0, 0, 0]], hold_upright=[False, True])
            self.assertEqual(self.sent_gain_scales[1], (1.5, 2))
            self.run_main([0, .1], [[.2, 0, .5], [.3, 0, 0]], onefoot_at=1)
            self.assertEqual(self.sent_gain_scales[1], (1.5, 2))


if __name__ == "__main__":
    unittest.main()
