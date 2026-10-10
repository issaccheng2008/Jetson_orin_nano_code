"""Button startup without cameras, GPIO or motor commands."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new_vision" / "jetson"))
sys.path.insert(0, str(ROOT / "humanoid_jetson_deploy"))

import protocol
from serial_link import SerialLink
from start_gate import StartGate
import run_policy_vision


class ButtonStartTests(unittest.TestCase):
    def test_first_card_walk_time_is_configurable_and_validated(self):
        with patch('sys.argv', ['run_policy_vision.py']):
            self.assertEqual(run_policy_vision.parse_args().startup_first_walk_s, .5)
        with patch('sys.argv', ['run_policy_vision.py', '--no-camera-async', '--startup-first-walk-s', '.3']):
            self.assertEqual(run_policy_vision.parse_args().startup_first_walk_s, .3)
        for value in ('0', '-1', 'nan', 'inf'):
            with patch('sys.argv', ['run_policy_vision.py', '--no-camera-async', '--startup-first-walk-s', value]), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_policy_vision.parse_args()

    def test_card_readiness_does_not_release_button_gate(self):
        gate = StartGate(mode="button", shape_confirm=2)
        gate.observe_shape("square")
        gate.observe_shape("square")
        self.assertTrue(gate.shape_passed)
        self.assertFalse(gate.passed)
        self.assertFalse(gate.require_qr)
        self.assertTrue(gate.observe_button(True))
        self.assertTrue(gate.passed)
        self.assertFalse(gate.observe_button(True))

    def test_button_releases_without_a_card(self):
        gate = StartGate(mode="button")
        self.assertFalse(gate.observe_button(False))
        self.assertTrue(gate.observe_button(True))
        self.assertTrue(gate.passed)
        self.assertFalse(gate.shape_passed)

    def test_startup_packet_and_serial_write_are_separate_from_motor_command(self):
        link = SerialLink.__new__(SerialLink)
        link._sequence = 7
        link._write_lock = threading.Lock()
        link.serial = Mock()
        flags = protocol.STARTUP_ARM | protocol.STARTUP_CARD_READY
        link.send_startup_control(flags)
        frame = link.serial.write.call_args.args[0]
        self.assertEqual(len(frame), 11)
        decoder = protocol.FrameDecoder()
        messages = []
        for offset in range(0, len(frame), 3):
            messages.extend(decoder.feed(frame[offset:offset + 3]))
        self.assertEqual(messages, [protocol.StartupControlPacket(7, flags)])
        self.assertEqual(link._sequence, 8)
        self.assertEqual(decoder.crc_errors + decoder.format_errors, 0)
        with self.assertRaises(ValueError):
            protocol.pack_startup_control(protocol.StartupControlPacket(0, 4))

    def test_state_keeps_existing_size_and_button_bits(self):
        flags = protocol.STATE_START_BUTTON | protocol.STATE_STARTUP_ACTIVE
        state = protocol.StatePacket(1, 1000, np.zeros(12), np.zeros(12), np.zeros(3),
                                     np.zeros(3), np.array([1, 0, 0, 0]), flags, 0, 0)
        frame = protocol.pack_state(state)
        self.assertEqual(len(frame), 162)
        decoded = list(protocol.FrameDecoder().feed(frame))[0]
        self.assertEqual(decoded.status_flags, flags)

    def test_button_gate_requires_automatic_policy_launch(self):
        with patch("sys.argv", ["run_policy_vision.py", "--no-camera-async", "--start-gate", "button"]), \
                redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                run_policy_vision.parse_args()


class StartupSerialTests(unittest.TestCase):
    def test_reset_and_arm_ack_prevent_stale_press_and_close_before_handoff(self):
        import startup_button_link
        clock = [0.0]
        link = Mock(spec=SerialLink)
        link.reader_alive.return_value = True
        link.get_latest_state.return_value = SimpleNamespace(
            status_flags=protocol.STATE_STARTUP_ACTIVE | protocol.STATE_START_BUTTON)
        with patch.object(startup_button_link, "SerialLink", return_value=link), \
                patch.object(startup_button_link.time, "monotonic", lambda: clock[0]):
            button = startup_button_link.StartupButtonLink("/dev/ttyACM0")
            self.assertFalse(button.poll(False))
            link.send_startup_control.assert_called_with(0)
            clock[0] = 0.3
            self.assertFalse(button.poll(True))  # previous session press, still armed
            link.send_startup_control.assert_called_with(0)
            link.get_latest_state.return_value.status_flags = 0
            clock[0] = 0.4
            self.assertFalse(button.poll(True))
            link.send_startup_control.assert_called_with(protocol.STARTUP_ARM | protocol.STARTUP_CARD_READY)
            link.get_latest_state.return_value.status_flags = protocol.STATE_START_BUTTON
            clock[0] = 0.5
            self.assertFalse(button.poll(True))  # no arm ACK yet
            link.get_latest_state.return_value.status_flags |= protocol.STATE_STARTUP_ACTIVE
            self.assertTrue(button.poll(True))
            link.reader_alive.return_value = False
            button.close()
            link.send_startup_control.assert_called_with(0)
            link.close.assert_called_once()
            link.send_command.assert_not_called()
            link.send_action.assert_not_called()

    def test_missing_device_retries_without_starting_policy(self):
        import startup_button_link
        clock = [0.0]
        with patch.object(startup_button_link, "SerialLink", side_effect=OSError("not enumerated")) as factory, \
                patch.object(startup_button_link.time, "monotonic", lambda: clock[0]), \
                redirect_stdout(io.StringIO()):
            button = startup_button_link.StartupButtonLink("/dev/ttyACM0")
            self.assertFalse(button.poll(False))
            clock[0] = 0.1
            self.assertFalse(button.poll(False))
            self.assertEqual(factory.call_count, 1)
            clock[0] = 1.1
            self.assertFalse(button.poll(False))
            self.assertEqual(factory.call_count, 2)
            button.close()

    def test_live_reader_blocks_serial_handoff(self):
        import startup_button_link
        link = Mock(spec=SerialLink)
        link.reader_alive.return_value = True
        with patch.object(startup_button_link, "SerialLink", return_value=link):
            button = startup_button_link.StartupButtonLink("/dev/ttyACM0")
            self.assertFalse(button.poll(False))
            with self.assertRaisesRegex(RuntimeError, "reader"):
                button.close()


class ButtonVisionIntegrationTests(unittest.TestCase):
    def run_scenario(self, card, press_at=None, walk_s=None, sequence=None):
        clock, reads = [0.0], [0]
        camera = Mock()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera.isOpened.return_value = True
        camera.get.side_effect = [1280, 720]

        def read():
            reads[0] += 1
            clock[0] += 0.1
            if reads[0] > 30:
                run_policy_vision.signal.signal.call_args.args[1](None, None)
                return False, frame
            return True, frame

        camera.read.side_effect = read
        detector = Mock()
        detector.process.return_value = (0, 0, 0.9, None, dict(
            fused_err=0., fused_err_cm=0., angle_err_deg=0., lost_frames=0,
            curve_mode=False, bottom_lock_valid=True, curve_px=0., base_err_cm=0.))
        shape = Mock()
        shape.action_map = {"circle": 1}
        shape.update.return_value = (None, dict(presence=card, card_found=card,
            presence_cy_frac=0.9 if card else None, shape="circle" if card else None))
        order, ready_values = [], []

        def poll(ready):
            ready_values.append(ready)
            return press_at is not None and reads[0] >= press_at

        extra = [] if walk_s is None else ['--startup-first-walk-s', str(walk_s)]
        if sequence is not None:
            extra += ['--startup-sequence', sequence]
        timed_commands = []
        with (patch("sys.argv", ["run_policy_vision.py", "--no-camera-async", "--headless", "--start-gate", "button",
                                "--start-policy-on-gate", "--shape-every", "1", "--attitude-port", "0", *extra]),
              patch.object(run_policy_vision.signal, "signal"),
              patch.object(run_policy_vision, "ConnectorClient") as client_cls,
              patch("utils.open_camera", return_value=camera),
              patch("line_detector_v1_warp.LineDetector", return_value=detector),
              patch("shape_detector.ShapeDetector", return_value=shape),
              patch("qr_reader.QrReader") as qr_cls,
              patch("startup_button_link.StartupButtonLink") as button_cls,
              patch("policy_gate_launcher.PolicyGateLauncher") as launcher_cls,
              patch.object(run_policy_vision.time, "monotonic", lambda: clock[0]),
              redirect_stdout(io.StringIO())):
            button = button_cls.return_value
            button.poll.side_effect = poll
            button.close.side_effect = lambda: order.append("serial_closed")
            launcher = launcher_cls.return_value
            launcher.started = False
            launcher.process.poll.return_value = None

            def launch():
                order.append("gait_started")
                launcher.started = True

            launcher.start.side_effect = launch
            launcher.ready.side_effect = lambda: reads[0] >= (press_at or 100) + 2
            client_cls.return_value.publish.side_effect = lambda vx,wz,*a,**kw: timed_commands.append((clock[0],vx,wz,kw))
            self.assertEqual(run_policy_vision.main(), 0)
            launcher.timed_commands = timed_commands
        qr_cls.assert_not_called()
        return launcher, client_cls.return_value.publish.call_args_list, order, ready_values

    def test_first_card_action_follows_configured_walk_duration(self):
        for duration in (.3, .5, 1.):
            with self.subTest(duration=duration):
                launcher, *_ = self.run_scenario(card=True, press_at=6, walk_s=duration)
                first_walk = next(t for t,vx,wz,kw in launcher.timed_commands if vx > 0)
                first_action = next(t for t,vx,wz,kw in launcher.timed_commands if kw.get('event_id'))
                self.assertGreaterEqual(first_action-first_walk+1e-8, duration)
                self.assertLessEqual(first_action-first_walk, duration+.100001)

    def test_first_card_sequence_straight_then_turn_then_action(self):
        sequence = '[{"duration_s":0.5,"vx":0.2,"wz":0},{"duration_s":0.5,"vx":0.2,"wz":0.5}]'
        launcher, commands, *_ = self.run_scenario(card=True, press_at=6, sequence=sequence)
        timed = launcher.timed_commands
        straight_at = next(t for t,vx,wz,kw in timed if vx > 0)
        turn_at = next(t for t,vx,wz,kw in timed if wz > 0)
        action_at = next(t for t,vx,wz,kw in timed if kw.get('event_id'))
        self.assertAlmostEqual(turn_at-straight_at, .5, places=6)
        self.assertGreaterEqual(action_at-turn_at+1e-8, .5)
        self.assertLessEqual(action_at-turn_at, .600001)
        self.assertTrue(all((vx,wz)==(.2,.5) for t,vx,wz,kw in timed if turn_at <= t < action_at))
        self.assertEqual({c.kwargs['event_action'] for c in commands if c.kwargs.get('event_id')}, {1})

    def test_sequence_does_not_run_without_latched_first_card(self):
        sequence = '[{"duration_s":0.5,"vx":0.2,"wz":0.5}]'
        launcher, commands, *_ = self.run_scenario(card=False, press_at=6, sequence=sequence)
        self.assertFalse(any(wz for t,vx,wz,kw in launcher.timed_commands))
        self.assertFalse(any(c.kwargs.get('event_id') for c in commands))

    def test_recognized_card_lights_hint_but_does_not_start_without_button(self):
        launcher, commands, order, ready = self.run_scenario(card=True)
        launcher.start.assert_not_called()
        self.assertIn(True, ready)
        self.assertTrue(all(call.args[:2] == (0.0, 0.0) for call in commands))
        self.assertNotIn("gait_started", order)

    def test_button_without_card_starts_gait_after_serial_handoff(self):
        launcher, commands, order, ready = self.run_scenario(card=False, press_at=6)
        launcher.start.assert_called_once()
        self.assertEqual(order[:2], ["serial_closed", "gait_started"])
        self.assertFalse(any(ready))
        self.assertTrue(all(call.args[:2] == (0.0, 0.0) for call in commands[:8]))
        self.assertTrue(any(call.args[0] > 0 for call in commands[8:]))
        self.assertFalse(any(call.kwargs.get("event_id") for call in commands))

    def test_button_with_card_preserves_first_card_action_once(self):
        launcher, commands, order, ready = self.run_scenario(card=True, press_at=6)
        launcher.start.assert_called_once()
        self.assertEqual(order[:2], ["serial_closed", "gait_started"])
        self.assertIn(True, ready)
        events = {call.kwargs["event_id"]: call.kwargs["event_action"]
                  for call in commands if call.kwargs.get("event_id")}
        self.assertEqual(list(events.values()), [1])


if __name__ == "__main__":
    unittest.main()
