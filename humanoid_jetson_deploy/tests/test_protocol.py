from __future__ import annotations

import unittest

import numpy as np

from protocol import (
    COMMAND_ENABLE,
    CommandPacket,
    FrameDecoder,
    StatePacket,
    pack_command,
    pack_state,
)


class ProtocolTests(unittest.TestCase):
    def test_fragmented_state_round_trip(self):
        source = StatePacket(
            sequence=65535,
            timestamp_us=0xFFFFFFFE,
            joint_position=np.arange(12, dtype=np.float32) * 0.1,
            joint_velocity=-np.arange(12, dtype=np.float32),
            accel_m_s2=np.array([1.0, 2.0, 9.0], dtype=np.float32),
            gyro_rad_s=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            status_flags=12,
        )
        frame = pack_state(source)
        decoder = FrameDecoder()
        result = []
        for index in range(0, len(frame), 3):
            result.extend(decoder.feed(frame[index : index + 3]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].sequence, source.sequence)
        np.testing.assert_allclose(result[0].joint_position, source.joint_position)
        np.testing.assert_allclose(result[0].accel_m_s2, source.accel_m_s2)

    def test_command_round_trip_with_noise_prefix(self):
        source = CommandPacket(
            sequence=7,
            timestamp_us=99,
            joint_target=np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            kp_scale=0.5,
            kd_scale=0.25,
            command_flags=COMMAND_ENABLE,
        )
        decoded = list(FrameDecoder().feed(b"line noise" + pack_command(source)))
        self.assertEqual(len(decoded), 1)
        np.testing.assert_allclose(decoded[0].joint_target, source.joint_target)
        self.assertAlmostEqual(decoded[0].kp_scale, 0.5)

    def test_crc_error_is_rejected_and_next_frame_recovers(self):
        source = CommandPacket(1, 2, np.zeros(12), 1.0, 1.0, 0)
        damaged = bytearray(pack_command(source))
        damaged[20] ^= 0x40
        decoder = FrameDecoder()
        decoded = list(decoder.feed(bytes(damaged) + pack_command(source)))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoder.crc_errors, 1)

    def test_wire_sizes(self):
        state = StatePacket(0, 0, np.zeros(12), np.zeros(12), np.zeros(3), np.zeros(3), 0)
        command = CommandPacket(0, 0, np.zeros(12), 0.0, 0.0, 0)
        self.assertEqual(len(pack_state(state)), 138)
        self.assertEqual(len(pack_command(command)), 74)


if __name__ == "__main__":
    unittest.main()
