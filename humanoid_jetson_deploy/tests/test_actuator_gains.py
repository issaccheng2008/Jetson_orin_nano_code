"""Exercise actual serial frames without a serial port or motor hardware."""
import threading
import unittest
from unittest.mock import patch

import numpy as np

import config
from protocol import COMMAND_ENABLE, CommandPacket, FrameDecoder, HEADER, pack_command
from serial_link import SerialLink


class RecordingSerial:
    def write(self, frame):
        self.frame = frame


class ActuatorGainTests(unittest.TestCase):
    def send(self, kp_scale=1.0, kd_scale=1.0):
        link = SerialLink.__new__(SerialLink)
        link.serial = RecordingSerial()
        link._sequence = 7
        link._write_lock = threading.Lock()
        target = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
        self.assertTrue(link.send_command(123, target, kp_scale, kd_scale, COMMAND_ENABLE))
        frame = link.serial.frame
        packet = list(FrameDecoder().feed(frame))[0]
        self.assertTrue(hasattr(packet, "kp"), "commands must carry absolute per-joint KP")
        return packet, frame, target

    def test_defaults_send_original_firmware_baseline(self):
        packet, frame, target = self.send()
        np.testing.assert_allclose(packet.kp, [35, 30, 20, 35, 30, 12] * 2)
        np.testing.assert_allclose(packet.kd, [1.5, 1.2, 1, 1.5, 1.6, 0.7] * 2)
        np.testing.assert_array_equal(packet.joint_target, target)
        self.assertEqual(HEADER.unpack_from(frame)[3], 152)
        self.assertEqual(len(frame), 162)
        self.assertEqual(packet.command_flags, COMMAND_ENABLE)
        self.assertEqual(packet.sequence, 7)

    def test_separate_scales_multiply_only_the_baseline(self):
        packet, _, _ = self.send(1.5, 2.0)
        np.testing.assert_allclose(packet.kp, np.array([35, 30, 20, 35, 30, 12] * 2) * 1.5)
        np.testing.assert_allclose(packet.kd, np.array([1.5, 1.2, 1, 1.5, 1.6, 0.7] * 2) * 2.0)

    def test_no_additional_common_multiplier(self):
        self.assertFalse(hasattr(config, "GAIN_SCALE"), "remove the extra common multiplier")

    def test_one_joint_can_be_tuned_independently(self):
        self.assertTrue(hasattr(config, "JOINT_KP"), "editable per-joint gains are required")
        kp = np.array(config.JOINT_KP, copy=True)
        kp[7] = 42
        with patch.object(config, "JOINT_KP", kp):
            packet, _, _ = self.send()
        self.assertEqual(packet.kp[7], 42)
        self.assertEqual(packet.kp[1], 30)

    def test_zero_scale_produces_zero_gains(self):
        packet, _, _ = self.send(0, 0)
        np.testing.assert_array_equal(packet.kp, np.zeros(12))
        np.testing.assert_array_equal(packet.kd, np.zeros(12))

    def test_invalid_multiplier_is_rejected_before_write(self):
        for scales in ((-1, 1), (1, float("nan")), (float("inf"), 1)):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                self.send(*scales)

    def test_invalid_absolute_gains_are_rejected(self):
        self.assertIn("kp", CommandPacket.__dataclass_fields__, "absolute gains are required")
        for field, invalid in (("kp", -1), ("kp", 501), ("kd", 5.1),
                               ("kd", float("nan")), ("kp", float("inf"))):
            kp = np.ones(12)
            kd = np.ones(12)
            (kp if field == "kp" else kd)[5] = invalid
            with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                pack_command(CommandPacket(1, 2, np.zeros(12), kp, kd, COMMAND_ENABLE))


if __name__ == "__main__":
    unittest.main()
