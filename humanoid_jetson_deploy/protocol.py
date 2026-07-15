"""Binary Jetson <-> STM32 protocol with framing, sequence IDs, and CRC-16."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np

from config import NUM_JOINTS


MAGIC = 0xA55A
MAGIC_BYTES = struct.pack("<H", MAGIC)
VERSION = 1
MSG_STATE = 1
MSG_COMMAND = 2
MAX_PAYLOAD = 512

COMMAND_ENABLE = 1 << 0
COMMAND_ESTOP = 1 << 1
COMMAND_CLEAR_FAULT = 1 << 2

STATE_MOTORS_ENABLED = 1 << 0
STATE_FAULT = 1 << 1
STATE_IMU_VALID = 1 << 2
STATE_ENCODERS_VALID = 1 << 3
STATE_COMMAND_FRESH = 1 << 4

HEADER = struct.Struct("<HBBHH")
CRC = struct.Struct("<H")
STATE_PAYLOAD = struct.Struct("<I" + "f" * NUM_JOINTS + "f" * NUM_JOINTS + "3f3fI")
COMMAND_PAYLOAD = struct.Struct("<I" + "f" * NUM_JOINTS + "ffI")


@dataclass(frozen=True)
class StatePacket:
    sequence: int
    timestamp_us: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    accel_m_s2: np.ndarray
    gyro_rad_s: np.ndarray
    status_flags: int


@dataclass(frozen=True)
class CommandPacket:
    sequence: int
    timestamp_us: int
    joint_target: np.ndarray
    kp_scale: float
    kd_scale: float
    command_flags: int


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: polynomial 0x1021, init 0xFFFF."""
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _pack_frame(message_type: int, sequence: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Payload exceeds protocol maximum")
    header = HEADER.pack(MAGIC, VERSION, message_type, len(payload), sequence & 0xFFFF)
    crc = crc16_ccitt(header[2:] + payload)
    return header + payload + CRC.pack(crc)


def pack_state(packet: StatePacket) -> bytes:
    q = np.asarray(packet.joint_position, dtype=np.float32).reshape(NUM_JOINTS)
    qd = np.asarray(packet.joint_velocity, dtype=np.float32).reshape(NUM_JOINTS)
    accel = np.asarray(packet.accel_m_s2, dtype=np.float32).reshape(3)
    gyro = np.asarray(packet.gyro_rad_s, dtype=np.float32).reshape(3)
    payload = STATE_PAYLOAD.pack(
        packet.timestamp_us & 0xFFFFFFFF,
        *q,
        *qd,
        *accel,
        *gyro,
        packet.status_flags & 0xFFFFFFFF,
    )
    return _pack_frame(MSG_STATE, packet.sequence, payload)


def pack_command(packet: CommandPacket) -> bytes:
    q = np.asarray(packet.joint_target, dtype=np.float32).reshape(NUM_JOINTS)
    payload = COMMAND_PAYLOAD.pack(
        packet.timestamp_us & 0xFFFFFFFF,
        *q,
        float(packet.kp_scale),
        float(packet.kd_scale),
        packet.command_flags & 0xFFFFFFFF,
    )
    return _pack_frame(MSG_COMMAND, packet.sequence, payload)


def decode_state(sequence: int, payload: bytes) -> StatePacket:
    values = STATE_PAYLOAD.unpack(payload)
    i = 1
    q = np.array(values[i : i + NUM_JOINTS], dtype=np.float32)
    i += NUM_JOINTS
    qd = np.array(values[i : i + NUM_JOINTS], dtype=np.float32)
    i += NUM_JOINTS
    accel = np.array(values[i : i + 3], dtype=np.float32)
    i += 3
    gyro = np.array(values[i : i + 3], dtype=np.float32)
    return StatePacket(sequence, values[0], q, qd, accel, gyro, values[-1])


def decode_command(sequence: int, payload: bytes) -> CommandPacket:
    values = COMMAND_PAYLOAD.unpack(payload)
    q = np.array(values[1 : 1 + NUM_JOINTS], dtype=np.float32)
    return CommandPacket(sequence, values[0], q, values[-3], values[-2], values[-1])


class FrameDecoder:
    """Incremental decoder that tolerates partial packets and line noise."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.valid_frames = 0
        self.crc_errors = 0
        self.format_errors = 0

    def feed(self, data: bytes) -> Iterable[StatePacket | CommandPacket]:
        self.buffer.extend(data)
        decoded: list[StatePacket | CommandPacket] = []

        while True:
            start = self.buffer.find(MAGIC_BYTES)
            if start < 0:
                if self.buffer[-1:] == MAGIC_BYTES[:1]:
                    self.buffer[:] = self.buffer[-1:]
                else:
                    self.buffer.clear()
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < HEADER.size:
                break

            magic, version, message_type, payload_len, sequence = HEADER.unpack_from(self.buffer)
            if magic != MAGIC or version != VERSION or payload_len > MAX_PAYLOAD:
                self.format_errors += 1
                del self.buffer[0]
                continue

            total_len = HEADER.size + payload_len + CRC.size
            if len(self.buffer) < total_len:
                break

            frame = bytes(self.buffer[:total_len])
            expected_crc = CRC.unpack_from(frame, HEADER.size + payload_len)[0]
            actual_crc = crc16_ccitt(frame[2 : HEADER.size + payload_len])
            if actual_crc != expected_crc:
                self.crc_errors += 1
                del self.buffer[0]
                continue

            payload = frame[HEADER.size : HEADER.size + payload_len]
            try:
                if message_type == MSG_STATE and payload_len == STATE_PAYLOAD.size:
                    decoded.append(decode_state(sequence, payload))
                elif message_type == MSG_COMMAND and payload_len == COMMAND_PAYLOAD.size:
                    decoded.append(decode_command(sequence, payload))
                else:
                    raise ValueError("Unknown message type or payload size")
            except (ValueError, struct.error):
                self.format_errors += 1
            else:
                self.valid_frames += 1
            del self.buffer[:total_len]

        return decoded
