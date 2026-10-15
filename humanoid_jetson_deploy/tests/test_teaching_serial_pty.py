"""Linux integration: exercise the real serial reader against a fake STM32 PTY.

No /dev/ttyACM device is opened and no physical motor commands are sent.
"""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import (CommandPacket, FrameDecoder, StatePacket, pack_state,
                      COMMAND_ENABLE, COMMAND_ESTOP, COMMAND_TEACHING,
                      STATE_ENCODERS_VALID, STATE_TEACHING_READY)
from teaching import TeachingController, TeachingStore


@unittest.skipUnless(sys.platform == "linux", "requires Linux pseudo-terminal; no hardware")
class SerialTeachingIntegration(unittest.TestCase):
    def test_50hz_record_echo_hold_and_exclusive_port(self):
        import pty
        from serial_link import SerialLink
        import serial

        master, slave = pty.openpty()
        port = os.ttyname(slave)
        os.close(slave)
        link = SerialLink(port, queue_states=True)
        os.set_blocking(master, False)
        stopped = threading.Event()
        commands = []
        errors = []

        def firmware():
            decoder = FrameDecoder()
            deadline = time.monotonic()
            seq = 0
            try:
                while not stopped.is_set():
                    try:
                        data = os.read(master, 16384)
                    except BlockingIOError:
                        data = b""
                    commands.extend(p for p in decoder.feed(data) if isinstance(p, CommandPacket))
                    now = time.monotonic()
                    if now >= deadline:
                        seq += 1
                        q = np.full(12, .02*np.sin(seq*.02))
                        packet = StatePacket(seq & 0xffff, seq*20000, q, np.zeros(12),
                                             np.zeros(3), np.zeros(3), np.array([1,0,0,0]),
                                             STATE_ENCODERS_VALID | STATE_TEACHING_READY)
                        os.write(master, pack_state(packet))
                        deadline += .02
                    stopped.wait(.002)
            except Exception as exc:
                if not stopped.is_set():
                    errors.append(exc)

        worker = threading.Thread(target=firmware, daemon=True)
        worker.start()
        controller = None
        try:
            with self.assertRaises(serial.SerialException):
                other = serial.Serial(port, 921600)
                other.close()
            with tempfile.TemporaryDirectory() as root:
                controller = TeachingController(link, TeachingStore(root), enable_motors=True)
                controller.start()
                deadline = time.monotonic()+3
                while controller.snapshot()["position_rad"] is None and time.monotonic()<deadline:
                    time.sleep(.01)
                controller.execute("release")
                time.sleep(.16)
                controller.execute("start-recording")
                time.sleep(.60)
                controller.execute("stop-recording")
                self.assertEqual(controller.mode,"free",controller.error)
                times, poses=controller.store.recording(controller.record_name)
                self.assertGreaterEqual(len(poses),25)
                self.assertLessEqual(len(poses),35)
                np.testing.assert_allclose(np.diff(times), .02, atol=1e-8)
                controller.execute("checkpoint", dict(label="PTY position"))
                controller.execute("hold")
                fixed=controller.target.copy()
                time.sleep(.12)
                np.testing.assert_allclose(controller.target, fixed)
                self.assertTrue(any(np.all(p.kp==0) and np.all(p.kd==0) and
                                    p.command_flags & COMMAND_ENABLE for p in commands))
                self.assertTrue(any(np.all(p.kp>0) and np.all(p.kd>0) for p in commands))
                self.assertTrue(all(p.command_flags & COMMAND_TEACHING for p in commands))
                controller.execute("estop")
                time.sleep(.06)
                self.assertTrue(commands[-1].command_flags & COMMAND_ESTOP)
                self.assertFalse(commands[-1].command_flags & COMMAND_ENABLE)
                self.assertEqual(errors,[])
                controller.close()
                controller = None
        finally:
            if controller:
                controller.close()
            stopped.set()
            worker.join(timeout=1)
            link.close()
            os.close(master)


if __name__ == "__main__":
    unittest.main()
