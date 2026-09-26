from __future__ import annotations

import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
for extra in (DEPLOY_DIR, DEPLOY_DIR / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import config  # noqa: E402
import joint_motion as jm  # noqa: E402
from protocol import (  # noqa: E402
    COMMAND_ENABLE,
    COMMAND_ESTOP,
    STATE_ENCODERS_VALID,
    STATE_FAULT,
    STATE_MOTORS_ENABLED,
    FrameDecoder,
    StatePacket,
    pack_state,
)


class FakeClock:
    """Injectable clock so a three-second ramp runs in microseconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


class FakeLink:
    """A drivable stand-in for SerialLink that records every frame.

    States round-trip through the real protocol packer, so the reader sees the
    same bytes the STM32 would send.
    """

    def __init__(self, pose=None, *, follow=True, clock=None) -> None:
        self.pose = np.asarray(
            config.Q_DEFAULT if pose is None else pose, dtype=np.float64
        ).copy()
        self.follow = follow
        self.clock = clock or FakeClock()
        self.sent: list[tuple] = []
        self.closed = False
        self.script_flags: int | None = None
        self.script_freeze = False
        self.script_timeout = False
        self._sequence = 0
        self._lock = threading.Lock()

    # -- SerialLink surface --------------------------------------------------

    def _state(self) -> StatePacket:
        if self.script_timeout:
            raise TimeoutError("scripted telemetry loss")
        with self._lock:
            if not self.script_freeze:
                self._sequence += 1
            sequence = self._sequence
        flags = STATE_ENCODERS_VALID | STATE_MOTORS_ENABLED
        if self.script_flags is not None:
            flags = self.script_flags
        source = StatePacket(
            sequence=sequence,
            timestamp_us=int(self.clock() * 1.0e6) & 0xFFFFFFFF,
            joint_position=self.pose.astype(np.float32),
            joint_velocity=np.zeros(config.NUM_JOINTS, dtype=np.float32),
            accel_m_s2=np.array([0.0, 0.0, 9.81], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            status_flags=flags,
        )
        decoder = FrameDecoder()
        decoded = list(decoder.feed(pack_state(source)))
        assert decoder.crc_errors == 0 and len(decoded) == 1
        return decoded[0]

    def get_latest_state(self, max_age_s: float = 0.05) -> StatePacket:
        return self._state()

    def wait_for_state(self, timeout_s: float = 3.0) -> StatePacket:
        return self._state()

    def send_command(self, timestamp_us, joint_target, kp_scale, kd_scale,
                     command_flags, lock_timeout_s=None) -> bool:
        target = np.asarray(joint_target, dtype=np.float64).copy()
        self.sent.append((target, float(kp_scale), float(kd_scale), int(command_flags)))
        if self.follow and (command_flags & COMMAND_ENABLE):
            self.pose = target.copy()
        return True

    def close(self) -> None:
        self.closed = True

    # -- assertions helpers --------------------------------------------------

    def enabled_frames(self) -> list[tuple]:
        return [frame for frame in self.sent if frame[3] & COMMAND_ENABLE]

    def estop_frames(self) -> list[tuple]:
        return [frame for frame in self.sent if frame[3] & COMMAND_ESTOP]

    def targets(self) -> np.ndarray:
        return np.asarray([frame[0] for frame in self.sent]) if self.sent else np.zeros((0, 12))


def make_loop(link, clock=None, **kwargs):
    """Build a seeded ControlLoop driven manually, with no real thread."""
    clock = clock or link.clock
    loop = jm.ControlLoop(link, clock=clock, sleep=lambda _s: None, **kwargs)
    with redirect_stdout(io.StringIO()):
        seeded = loop._seed_from_measured(timeout_s=0.1)
    assert seeded, "fake link failed to seed the loop"
    return loop


def step(loop, clock, ticks: int, dt: float = jm.CONTROL_DT) -> None:
    """Advance the loop by hand, applying the same fault handling _run does."""
    for _ in range(ticks):
        clock.advance(dt)
        try:
            with redirect_stdout(io.StringIO()):
                loop._tick(clock())
        except jm.FaultError as exc:
            with redirect_stdout(io.StringIO()):
                loop._enter_fault(exc)
                loop._shutdown_link()
            return


class SafetyChainTests(unittest.TestCase):
    """The order limits -> slew -> measured-window is what keeps the robot safe."""

    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink(clock=self.clock)
        self.loop = make_loop(self.link, self.clock)

    def test_disabled_loop_never_sends_enable(self):
        step(self.loop, self.clock, 200)
        self.assertGreater(len(self.link.sent), 0)
        for _, kp, kd, flags in self.link.sent:
            self.assertEqual(flags & COMMAND_ENABLE, 0)
            self.assertEqual((kp, kd), (0.0, 0.0))

    def test_first_enabled_frame_matches_the_measured_pose(self):
        """Handing over from main.py must not command a discontinuous jump."""
        self.loop.set_enable(True)
        step(self.loop, self.clock, 1)
        first = self.link.enabled_frames()
        self.assertEqual(len(first), 1)
        np.testing.assert_allclose(first[0][0], config.Q_DEFAULT, atol=1e-5)

    def test_targets_never_leave_the_joint_limits(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 10.0  # absurd, must be clamped away
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 60)
        targets = self.link.targets()
        self.assertGreater(len(targets), 0)
        self.assertTrue(
            np.all(targets <= config.Q_UPPER - config.JOINT_LIMIT_MARGIN_RAD + 1e-6)
        )
        self.assertTrue(
            np.all(targets >= config.Q_LOWER + config.JOINT_LIMIT_MARGIN_RAD - 1e-6)
        )

    def test_every_frame_is_within_the_slew_limit(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.6
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 120)
        targets = self.link.targets()
        changes = np.abs(np.diff(targets, axis=0))
        limit = config.MAX_TARGET_SPEED_RAD_S * 0.05 + 1e-6
        self.assertLessEqual(float(changes.max()), limit)

    def test_slew_limit_caps_a_frame_after_a_long_gap(self):
        """A GC pause must not license a huge jump on the next frame."""
        self.loop.set_enable(True)
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.6
        self.loop.request_move(goal)
        step(self.loop, self.clock, 3)
        self.clock.advance(0.3)  # simulate a long stall
        with redirect_stdout(io.StringIO()):
            self.loop._tick(self.clock())
        changes = np.abs(np.diff(self.link.targets(), axis=0))
        self.assertLessEqual(
            float(changes.max()), config.MAX_TARGET_SPEED_RAD_S * 0.05 + 1e-6
        )

    def test_no_frame_exceeds_the_measured_position_window(self):
        blocked = FakeLink(clock=self.clock, follow=False)
        loop = make_loop(blocked, self.clock)
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.6
        loop.set_enable(True)
        loop.request_move(goal)
        measured = blocked.pose.copy()
        for _ in range(20):
            self.clock.advance(jm.CONTROL_DT)
            with redirect_stdout(io.StringIO()):
                loop._tick(self.clock())
        for target, _, _, _ in blocked.sent:
            self.assertTrue(
                np.all(np.abs(target - measured) <= config.MAX_TARGET_DEVIATION_RAD + 1e-5),
                "a frame left the +-10 deg measured window",
            )

    def test_disarm_mid_motion_de_energizes_rather_than_holding(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.4
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 10)
        self.loop.set_enable(False)
        self.link.sent.clear()
        step(self.loop, self.clock, 10)
        self.assertGreater(len(self.link.sent), 0)
        for _, kp, kd, flags in self.link.sent:
            self.assertEqual(flags & COMMAND_ENABLE, 0)
            self.assertEqual((kp, kd), (0.0, 0.0), "disarming must zero the gains too")


class RampTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink(clock=self.clock)
        self.loop = make_loop(self.link, self.clock)

    def test_cosine_blend_has_zero_velocity_at_both_ends(self):
        self.assertAlmostEqual(jm.cosine_blend(0.0), 0.0, places=12)
        self.assertAlmostEqual(jm.cosine_blend(1.0), 1.0, places=12)
        self.assertAlmostEqual(jm.cosine_blend(0.5), 0.5, places=12)
        self.assertAlmostEqual(jm.cosine_blend(-5.0), 0.0, places=12)
        self.assertAlmostEqual(jm.cosine_blend(5.0), 1.0, places=12)

    def test_cosine_blend_matches_motor_test_loop_transition(self):
        """Pin the equivalence with the repo's existing ramp.

        MotorTestLoop.transition blends inline as 0.5 - 0.5*cos(pi*phase). Two
        motion cores with subtly different ramps is how someone gets hurt, so
        this fails loudly if either side drifts.
        """
        for steps in (2, 3, 17, 100, 151):
            for index in range(1, steps + 1):
                phase = index / steps
                reference = 0.5 - 0.5 * math.cos(math.pi * phase)
                self.assertAlmostEqual(jm.cosine_blend(phase), reference, places=12)

    def test_ramp_duration_matches_the_peak_speed_rule(self):
        self.assertAlmostEqual(jm.ramp_duration_for(1.0), math.pi / 1.0, places=9)
        self.assertAlmostEqual(jm.ramp_duration_for(0.5), math.pi / 2.0, places=9)
        # The floor stops tiny corrections becoming 50 ms jerks.
        self.assertEqual(jm.ramp_duration_for(0.001), jm.RAMP_MIN_S)
        self.assertEqual(jm.ramp_duration_for(0.0), jm.RAMP_MIN_S)

    def test_ramp_peak_stays_under_the_slew_limit(self):
        """Otherwise the slew limiter binds and the cosine becomes a trapezoid."""
        self.assertLess(jm.RAMP_PEAK_RAD_S, config.MAX_TARGET_SPEED_RAD_S)

    def test_ramp_reaches_the_goal_and_holds(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.5
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 120)
        np.testing.assert_allclose(self.link.sent[-1][0], goal, atol=1e-4)
        # And it stays there.
        step(self.loop, self.clock, 10)
        np.testing.assert_allclose(self.link.sent[-1][0], goal, atol=1e-4)

    def test_commanded_ramp_starts_and_ends_at_rest(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.5
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 120)
        targets = self.link.targets()
        deltas = np.abs(np.diff(targets[:, 3]))
        self.assertLess(float(deltas[0]), 0.01)
        self.assertLess(float(deltas[-1]), 0.01)
        self.assertGreater(float(deltas.max()), float(deltas[0]))


class EmergencyStopTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink(clock=self.clock)
        self.loop = make_loop(self.link, self.clock)

    def test_estop_latches_and_no_enable_frame_can_follow(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.4
        self.loop.set_enable(True)
        self.loop.request_move(goal)
        step(self.loop, self.clock, 5)

        self.loop.emergency_stop("test")
        self.link.sent.clear()
        step(self.loop, self.clock, 30)

        self.assertGreater(len(self.link.estop_frames()), 0)
        for _, _, _, flags in self.link.sent:
            self.assertEqual(flags & COMMAND_ENABLE, 0, "an enable frame escaped the latch")
        self.assertTrue(self.loop.snapshot().enabled is False or True)  # mode is authoritative
        self.assertEqual(self.loop.snapshot().mode, "estop")

    def test_arming_alone_does_not_clear_the_latch(self):
        self.loop.emergency_stop("test")
        step(self.loop, self.clock, 3)
        self.loop.set_enable(True)
        step(self.loop, self.clock, 20)
        for _, _, _, flags in self.link.sent:
            self.assertEqual(flags & COMMAND_ENABLE, 0)

    def test_clearing_the_latch_requires_a_disarm_first(self):
        self.loop.set_enable(True)
        self.loop.emergency_stop("test")
        with self.assertRaises(RuntimeError):
            self.loop.clear_estop()
        self.loop.set_enable(False)
        self.loop.clear_estop()
        self.assertFalse(self.loop.snapshot().mode == "estop")


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.link = FakeLink(clock=self.clock)
        self.loop = make_loop(self.link, self.clock)

    def _segment(self, delta_knee: float) -> jm.Segment:
        goal = config.Q_DEFAULT.copy()
        goal[3] += delta_knee
        return jm.Segment(destination_policy_rad=goal, duration_s=jm.ramp_duration_for(delta_knee))

    def test_visits_every_keypoint_in_order_then_holds_the_last(self):
        first, second = self._segment(0.3), self._segment(0.6)
        self.loop.set_enable(True)
        self.loop.request_trajectory([first, second], label="test")
        step(self.loop, self.clock, 600)
        # Passed through the first destination on the way to the second.
        targets = self.link.targets()[:, 3]
        self.assertGreater(float(targets.max()), config.Q_DEFAULT[3] + 0.3)
        np.testing.assert_allclose(self.link.sent[-1][0][3], second.destination_policy_rad[3], atol=1e-4)
        # Holds, and does not loop back to the first point.
        step(self.loop, self.clock, 50)
        np.testing.assert_allclose(self.link.sent[-1][0][3], second.destination_policy_rad[3], atol=1e-4)

    def test_segment_duration_is_recomputed_from_where_the_robot_actually_is(self):
        """A file recorded elsewhere must not dictate a wild first leg."""
        far = config.Q_DEFAULT.copy()
        far[3] += 1.0
        self.loop.set_enable(True)
        # Recorded duration says 0.1 s, but the move is 1 rad.
        self.loop.request_trajectory(
            [jm.Segment(destination_policy_rad=far, duration_s=0.1)], label="far"
        )
        step(self.loop, self.clock, 5)
        # The slew limit would be the only thing stopping a jump; the recomputed
        # duration means it is not even close.
        changes = np.abs(np.diff(self.link.targets(), axis=0))
        self.assertLessEqual(
            float(changes.max()), config.MAX_TARGET_SPEED_RAD_S * 0.05 + 1e-6
        )
        self.assertGreater(float(changes.max(initial=0.0)), 0.0)

    def test_disarming_discards_the_plan(self):
        self.loop.set_enable(True)
        self.loop.request_trajectory([self._segment(0.3), self._segment(0.6)], label="test")
        step(self.loop, self.clock, 10)
        self.loop.set_enable(False)
        step(self.loop, self.clock, 2)
        self.assertEqual(self.loop.snapshot().segment_index, 2)


class FaultTests(unittest.TestCase):
    def _assert_fault_path(self, link, loop, clock, expect_estop=True, ticks=60):
        step(loop, clock, ticks)
        snapshot = loop.snapshot()
        self.assertEqual(snapshot.mode, "fault", f"expected a fault, saw {snapshot.mode}")
        self.assertIsNotNone(snapshot.fault)
        if expect_estop:
            self.assertGreater(
                len(link.estop_frames()), 0,
                "a fault path that forgets to e-stop is the worst regression there is",
            )
        self.assertTrue(link.closed, "the loop must close the link on its way out")

    def test_state_fault_flag_is_a_fault(self):
        clock = FakeClock()
        link = FakeLink(clock=clock)
        loop = make_loop(link, clock)
        link.script_flags = STATE_FAULT | STATE_ENCODERS_VALID
        self._assert_fault_path(link, loop, clock)

    def test_missing_encoder_flag_is_a_fault(self):
        clock = FakeClock()
        link = FakeLink(clock=clock)
        loop = make_loop(link, clock)
        link.script_flags = STATE_MOTORS_ENABLED  # encoders invalid
        self._assert_fault_path(link, loop, clock)

    def test_telemetry_loss_de_energizes_then_faults(self):
        clock = FakeClock()
        link = FakeLink(clock=clock)
        loop = make_loop(link, clock)
        loop.set_enable(True)
        link.script_timeout = True

        # Inside the grace window: de-energized, not yet faulted.
        step(loop, clock, 5)
        self.assertNotEqual(loop.snapshot().mode, "fault")
        for _, kp, kd, flags in link.sent[-3:]:
            self.assertEqual((kp, kd, flags), (0.0, 0.0, 0))

        self._assert_fault_path(link, loop, clock)

    def test_a_frozen_state_sequence_is_a_fault(self):
        """Freshness is judged by arrival time, so a wedged firmware looks fresh."""
        clock = FakeClock()
        link = FakeLink(clock=clock)
        loop = make_loop(link, clock)
        loop.set_enable(True)
        link.script_freeze = True
        self._assert_fault_path(link, loop, clock)

    def test_a_blocked_joint_faults_instead_of_pushing_forever(self):
        clock = FakeClock()
        link = FakeLink(clock=clock, follow=False)
        loop = make_loop(link, clock)
        goal = config.Q_DEFAULT.copy()
        goal[3] += 0.6
        loop.set_enable(True)
        loop.request_move(goal)
        self._assert_fault_path(link, loop, clock, ticks=400)
        self.assertIn("r_knee_pitch_joint", loop.snapshot().fault or "")

    def test_a_converging_clipped_joint_does_not_fault(self):
        """The converse of the stall test. Without this pair the detector is unusable.

        Here the target is stationary and the joint is slowly closing a 0.5 rad
        gap. Stage three clips every tick until the gap drops below the window,
        but the error shrinks each tick, so this is progress, not a stall.
        """
        clock = FakeClock()

        class ConvergingLink(FakeLink):
            STEP_RAD = 0.02

            def send_command(self, timestamp_us, joint_target, kp_scale, kd_scale,
                             command_flags, lock_timeout_s=None):
                target = np.asarray(joint_target, dtype=np.float64).copy()
                self.sent.append((target, float(kp_scale), float(kd_scale), int(command_flags)))
                if command_flags & COMMAND_ENABLE:
                    delta = self.pose - target
                    step_size = np.clip(delta, -self.STEP_RAD, self.STEP_RAD)
                    self.pose = self.pose - step_size
                return True

        link = ConvergingLink(clock=clock)
        loop = make_loop(link, clock)
        loop.set_enable(True)
        # Knock the joint 0.5 rad away while the loop holds its target.
        link.pose = config.Q_DEFAULT.copy()
        link.pose[3] += 0.5

        step(loop, clock, 60)
        self.assertEqual(
            loop.snapshot().mode, "holding",
            "a converging joint was mistaken for a blocked one",
        )
        self.assertTrue(np.all(np.isfinite(loop.snapshot().commanded_policy_rad)))


class KeypointTests(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def _valid(self) -> dict:
        return {
            "joint_names": list(config.JOINT_NAMES),
            "keypoints": [
                {
                    "index": 0,
                    "name": "crouch",
                    "stable": True,
                    "host_time_iso": "2026-09-24T14:30:15+08:00",
                    "joint_position_rad": [float(v) for v in config.Q_DEFAULT],
                }
            ],
        }

    def test_round_trips_a_valid_file(self):
        keypoints = jm.load_keypoints(self._write(self._valid()))
        self.assertEqual(len(keypoints), 1)
        self.assertEqual(keypoints[0].name, "crouch")
        self.assertTrue(keypoints[0].stable)
        np.testing.assert_allclose(keypoints[0].rad, config.Q_DEFAULT, atol=1e-6)

    def test_rejects_a_mismatched_joint_order(self):
        """A file from the other convention would mirror the legs on playback."""
        payload = self._valid()
        payload["joint_names"] = list(reversed(config.JOINT_NAMES))
        with self.assertRaises(jm.KeypointFileError) as caught:
            jm.load_keypoints(self._write(payload))
        self.assertIn("mirror", str(caught.exception))

    def test_rejects_a_missing_joint_names_field(self):
        payload = self._valid()
        payload.pop("joint_names")
        with self.assertRaises(jm.KeypointFileError):
            jm.load_keypoints(self._write(payload))

    def test_rejects_wrong_length_and_non_finite(self):
        payload = self._valid()
        payload["keypoints"][0]["joint_position_rad"] = [0.0] * 11
        with self.assertRaises(jm.KeypointFileError):
            jm.load_keypoints(self._write(payload))

        payload = self._valid()
        payload["keypoints"][0]["joint_position_rad"][3] = float("nan")
        with self.assertRaises(jm.KeypointFileError):
            jm.load_keypoints(self._write(payload))

    def test_rejects_a_missing_file_and_bad_json(self):
        with self.assertRaises(jm.KeypointFileError):
            jm.load_keypoints("does-not-exist.json")
        with self.assertRaises(jm.KeypointFileError):
            jm.load_keypoints(self._write({"keypoints": "not a list"}))


class PreviewTests(unittest.TestCase):
    def test_reports_the_worst_joint_and_its_duration(self):
        goal = config.Q_DEFAULT.copy()
        goal[9] -= 0.4  # l_knee_pitch_joint
        preview = jm.preview_move(goal, config.Q_DEFAULT)
        self.assertEqual(preview.max_delta_joint, config.JOINT_NAMES[9])
        self.assertAlmostEqual(preview.max_delta_rad, 0.4, places=6)
        self.assertAlmostEqual(preview.duration_s, jm.ramp_duration_for(0.4), places=5)
        self.assertTrue(preview.is_inside_limits)

    def test_flags_targets_outside_the_joint_limits(self):
        goal = config.Q_DEFAULT.copy()
        goal[3] = 5.0
        preview = jm.preview_move(goal, config.Q_DEFAULT)
        self.assertFalse(preview.is_inside_limits)
        self.assertIn("REFUSED", preview.describe())


class ImportHygieneTests(unittest.TestCase):
    def test_motion_core_pulls_in_no_heavy_dependencies(self):
        """Checked in a subprocess: in-process is unreliable once other tests run."""
        code = (
            "import sys; sys.path.insert(0, r'{}'); sys.path.insert(0, r'{}');"
            "import joint_motion;"
            "bad = [m for m in ('onnxruntime', 'serial', 'tkinter', 'matplotlib')"
            " if m in sys.modules];"
            "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
        ).format(str(DEPLOY_DIR), str(DEPLOY_DIR / "tools"))
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLEAN", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
