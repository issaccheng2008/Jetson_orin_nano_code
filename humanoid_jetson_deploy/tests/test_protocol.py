from __future__ import annotations

import unittest

import numpy as np

from protocol import (
    ActionRequestPacket,
    ActionStatusPacket,
    pack_action_request,
    pack_action_status,
    COMMAND_ENABLE,
    CommandPacket,
    FrameDecoder,
    StatePacket,
    pack_command,
    pack_state,
)


class ProtocolTests(unittest.TestCase):
    def test_action_request_and_status_round_trip_without_changing_existing_frames(self):
        request = ActionRequestPacket(sequence=7, event_id=123456, action_id=5)
        status = ActionStatusPacket(sequence=8, event_id=123456, action_id=5, status=2)
        decoded = list(FrameDecoder().feed(pack_action_request(request) + pack_action_status(status)))
        self.assertEqual(decoded, [request, status])
        self.assertEqual(len(pack_action_request(request)), 15)
        self.assertEqual(len(pack_action_status(status)), 16)

    def test_the_card_re_pose_events_are_sendable(self):
        """7/8 走的是同一个请求包。白名单里少了它们的话，send_action 会在写串口
        之前就抛 ValueError —— card_tilt 从来没上过线就是这个原因。"""
        from protocol import ACTION_CARD_RESTORE, ACTION_CARD_TILT
        for action_id in (ACTION_CARD_TILT, ACTION_CARD_RESTORE):
            with self.subTest(action_id=action_id):
                request = ActionRequestPacket(sequence=1, event_id=9, action_id=action_id)
                self.assertEqual(len(pack_action_request(request)), 15)

    def test_the_leg_card_ids_are_refused(self):
        """3/4 是正方形/菱形，走 Nano 自己的抬腿策略，从来不下发 STM32 ——
        发过去固件会把它当成别的东西。挡住是有意的，别顺手放宽白名单。"""
        for action_id in (3, 4):
            with self.subTest(action_id=action_id):
                request = ActionRequestPacket(sequence=1, event_id=9, action_id=action_id)
                with self.assertRaises(ValueError):
                    pack_action_request(request)

    def test_fragmented_state_round_trip(self):
        source = StatePacket(
            sequence=65535,
            timestamp_us=0xFFFFFFFE,
            joint_position=np.arange(12, dtype=np.float32) * 0.1,
            joint_velocity=-np.arange(12, dtype=np.float32),
            accel_m_s2=np.array([1.0, 2.0, 9.0], dtype=np.float32),
            gyro_rad_s=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
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
        np.testing.assert_allclose(result[0].orientation_wxyz, source.orientation_wxyz)
        self.assertIsNone(result[0].command_rx_count)
        self.assertIsNone(result[0].system_control_cycle)

    def test_state_counters_round_trip_and_legacy_state_still_decodes(self):
        old = StatePacket(1, 10, np.zeros(12), np.zeros(12), np.zeros(3),
                          np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 29)
        current = StatePacket(2, 20, np.zeros(12), np.zeros(12), np.zeros(3),
                              np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 29,
                              command_rx_count=0xFFFFFFFF, system_control_cycle=1234)
        self.assertEqual(len(pack_state(old)), 154)
        self.assertEqual(len(pack_state(current)), 162)
        stream = pack_state(old) + pack_state(current)
        decoder = FrameDecoder()
        result = []
        for offset in range(0, len(stream), 7):
            result.extend(decoder.feed(stream[offset:offset + 7]))
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0].command_rx_count)
        self.assertEqual(result[1].command_rx_count, 0xFFFFFFFF)
        self.assertEqual(result[1].system_control_cycle, 1234)
        self.assertEqual(decoder.format_errors, 0)

    def test_command_round_trip_with_noise_prefix(self):
        source = CommandPacket(
            sequence=7,
            timestamp_us=99,
            joint_target=np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            kp=np.linspace(10.0, 35.0, 12),
            kd=np.linspace(0.25, 1.5, 12),
            command_flags=COMMAND_ENABLE,
        )
        decoded = list(FrameDecoder().feed(b"line noise" + pack_command(source)))
        self.assertEqual(len(decoded), 1)
        np.testing.assert_allclose(decoded[0].joint_target, source.joint_target)
        np.testing.assert_allclose(decoded[0].kp, source.kp)
        np.testing.assert_allclose(decoded[0].kd, source.kd)

    def test_crc_error_is_rejected_and_next_frame_recovers(self):
        source = CommandPacket(1, 2, np.zeros(12), np.ones(12), np.ones(12), 0)
        damaged = bytearray(pack_command(source))
        damaged[20] ^= 0x40
        decoder = FrameDecoder()
        decoded = list(decoder.feed(bytes(damaged) + pack_command(source)))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoder.crc_errors, 1)

    def test_wire_sizes(self):
        state = StatePacket(
            0, 0, np.zeros(12), np.zeros(12), np.zeros(3), np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]), 0
        )
        command = CommandPacket(0, 0, np.zeros(12), np.zeros(12), np.zeros(12), 0)
        self.assertEqual(len(pack_state(state)), 154)
        self.assertEqual(len(pack_command(command)), 162)


if __name__ == "__main__":
    unittest.main()
