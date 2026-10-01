from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from command_source import CommandSnapshot
from protocol import STATE_ENCODERS_VALID, STATE_IMU_VALID


class WalkingModeTests(unittest.TestCase):
    def args(self, *extra):
        with patch("sys.argv", ["main.py", "--model", "test.onnx", "--no-plot", *extra]):
            return main.parse_args()

    def test_defaults_and_vision_or_turn_options(self):
        args = self.args()
        self.assertEqual((args.command_source, args.vx, args.wz), ("fixed", 0.3, 0.0))
        self.assertEqual(self.args("--command-source", "vision").udp_command_port, 5005)
        self.assertEqual(self.args("--wz", "0.5").wz, 0.5)

    def test_rejects_stop_and_invalid_forward_commands(self):
        for vx in ("0", "-0.1", "nan", "inf", "1.1"):
            with self.subTest(vx=vx), patch.object(main, "parse_args", return_value=self.args("--vx", vx)):
                with self.assertRaisesRegex(SystemExit, "vx must"):
                    main.main()

    def test_runtime_keeps_walking_without_vision_and_disables_on_state_fault(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        with (
            patch.object(main, "parse_args", return_value=self.args()),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "PositionCsvLogger"),
            patch("command_source.socket.socket", side_effect=AssertionError("UDP must stay disconnected")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            self.assertEqual(policy.step.call_count, 2)
            for call in policy.step.call_args_list:
                np.testing.assert_allclose(call.kwargs["velocity_command"], [0.3, 0, 0])
            self.assertEqual(link.send_command.call_args.args[-1], 0)
            link.close.assert_called_once()
            # --max-vx / --max-step-cm reach the observation builder as one ratio.
            self.assertAlmostEqual(policy_cls.call_args.args[1],
                                   config.MAX_STEP_DISTANCE / config.MAX_COMMAND_VX)


    def test_live_vision_commands_are_not_overridden_after_five_seconds(self):
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        with (
            patch.object(main, "parse_args", return_value=self.args("--command-source", "vision")),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "PositionCsvLogger"),
            patch.object(main.time, "monotonic", side_effect=[0, 10, 10, 10, 11, 11, 11, 12]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            source = source_cls.return_value
            source.get_snapshot.side_effect = [
                CommandSnapshot(np.array([0.4, 0.0, -0.2], dtype=np.float32)),
                CommandSnapshot(np.array([0.4, 0.0, 0.3], dtype=np.float32)),
            ]
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            source_cls.assert_called_once_with(5005, timeout_s=0.25, bind="127.0.0.1")
            self.assertEqual(policy.step.call_count, 2)
            for call, expected in zip(policy.step.call_args_list, ([0.4, 0, -0.2], [0.4, 0, 0.3])):
                np.testing.assert_allclose(call.kwargs["velocity_command"], expected)
            source.close.assert_called_once()

    def test_vision_reading_a_card_holds_the_legs_straight(self):
        """The policy's own stopped pose is pitched back about 20 degrees and the card
        geometry is calibrated at the mount angle, so while the vision stands still to
        read a card it asks for straight legs — and the policy is not stepped at all
        for those frames."""
        state = SimpleNamespace(
            status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
            accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            joint_position=config.Q_DEFAULT.copy(),
            joint_velocity=np.zeros(12, dtype=np.float32), sequence=1,
        )
        held = CommandSnapshot(np.zeros(3, dtype=np.float32), hold_upright=True)
        walking = CommandSnapshot(np.array([0.3, 0.0, 0.0], dtype=np.float32))
        output = io.StringIO()
        with (
            patch.object(main, "parse_args", return_value=self.args("--command-source", "vision")),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "PositionCsvLogger"),
            contextlib.redirect_stdout(output),
        ):
            source = source_cls.return_value
            source.get_snapshot.side_effect = [held, held, walking, held]
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = state
            link.get_latest_state.side_effect = [state, state, state, state,
                                                 RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            # Two held frames, then one walking frame, then held again when the state
            # read fails: only the walking one reached the policy.
            self.assertEqual(policy.step.call_count, 1)
            np.testing.assert_allclose(
                policy.step.call_args.kwargs["velocity_command"], [0.3, 0.0, 0.0])
            self.assertIn("hold_upright", output.getvalue())

    def test_the_upright_hold_does_not_flap_once_engaged(self):
        """The legs move while they straighten, so re-testing `stopped` every frame
        would drop the hold the moment it started working, hand the legs back to the
        policy, let them settle, and engage again — the two poses alternating."""
        def state(sequence, joint_velocity):
            return SimpleNamespace(
                status_flags=STATE_ENCODERS_VALID | STATE_IMU_VALID,
                accel_m_s2=np.array([0, 0, 9.81], dtype=np.float32),
                gyro_rad_s=np.zeros(3, dtype=np.float32),
                orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                joint_position=config.Q_DEFAULT.copy(),
                joint_velocity=joint_velocity, sequence=sequence,
            )

        held = CommandSnapshot(np.zeros(3, dtype=np.float32), hold_upright=True)
        settled = state(1, np.zeros(12, dtype=np.float32))
        # Straightening: the joints are moving, so `stopped` is false on this frame.
        moving = state(2, np.full(12, 1.0, dtype=np.float32))
        released = CommandSnapshot(np.array([0.3, 0.0, 0.0], dtype=np.float32))
        with (
            patch.object(main, "parse_args", return_value=self.args("--command-source", "vision")),
            patch.object(main.signal, "signal"),
            patch.object(main, "HumanoidPolicy") as policy_cls,
            patch.object(main, "SerialLink") as link_cls,
            patch.object(main, "UdpCommandSource") as source_cls,
            patch.object(main, "PositionCsvLogger"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            source_cls.return_value.get_snapshot.side_effect = [held, held, released]
            policy = policy_cls.return_value
            policy.step.return_value = (config.Q_DEFAULT.copy(), np.zeros(12), np.zeros(49), 0.0)
            link = link_cls.return_value
            link.wait_for_state.return_value = settled
            link.get_latest_state.side_effect = [settled, moving, settled,
                                                 RuntimeError("stale state")]
            self.assertEqual(main.main(), 1)
            # Engaged on the settled frame and kept through the moving one; only the
            # release frame reached the policy.
            self.assertEqual(policy.step.call_count, 1)


if __name__ == "__main__":
    unittest.main()
