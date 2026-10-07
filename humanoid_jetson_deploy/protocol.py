"""Binary Jetson <-> STM32 protocol with framing, sequence IDs, and CRC-16."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np

from config import NUM_JOINTS


MAGIC = 0xA55A
MAGIC_BYTES = struct.pack("<H", MAGIC)
VERSION = 2
MSG_STATE = 1
MSG_COMMAND = 2
MSG_ACTION_REQUEST = 3
MSG_ACTION_STATUS = 4
ACTION_ACCEPTED = 1
ACTION_DONE = 2
ACTION_BUSY = 3
ACTION_INVALID = 4
ACTION_FAILED = 5

# Card-stop body re-pose. 1-6 are the six printed shapes; these two are not shapes and
# never carry a card event. On START the STM32 re-poses the body and, for the rest of
# the stop, reports the attitude it held before that; on RESTORE it puts both back.
# The vision asks by publishing card_tilt, and main.py turns the two edges of that
# flag into these two requests.
ACTION_CARD_TILT = 7
ACTION_CARD_RESTORE = 8
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
STATE_PAYLOAD = struct.Struct("<I" + "f" * NUM_JOINTS + "f" * NUM_JOINTS + "3f3f4fI")
# The STM32 diagnostic branch appends two counters to the version-2 state
# payload. Accept both lengths so Nano can be deployed before STM32 is flashed.
STATE_PAYLOAD_WITH_COUNTERS = struct.Struct(STATE_PAYLOAD.format + "II")
COMMAND_PAYLOAD = struct.Struct("<I" + "f" * NUM_JOINTS + "ffI")
ACTION_REQUEST_PAYLOAD = struct.Struct("<IB")
ACTION_STATUS_PAYLOAD = struct.Struct("<IBB")


@dataclass(frozen=True)
class StatePacket:
    sequence: int
    timestamp_us: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    accel_m_s2: np.ndarray
    gyro_rad_s: np.ndarray
    orientation_wxyz: np.ndarray
    status_flags: int
    command_rx_count: int | None = None
    system_control_cycle: int | None = None


@dataclass(frozen=True)
class CommandPacket:
    sequence: int
    timestamp_us: int
    joint_target: np.ndarray
    kp_scale: float
    kd_scale: float
    command_flags: int


@dataclass(frozen=True)
class ActionRequestPacket:
    sequence: int
    event_id: int
    action_id: int


@dataclass(frozen=True)
class ActionStatusPacket:
    sequence: int
    event_id: int
    action_id: int
    status: int


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
    orientation = np.asarray(packet.orientation_wxyz, dtype=np.float32).reshape(4)
    has_counters = packet.command_rx_count is not None or packet.system_control_cycle is not None
    if has_counters and (packet.command_rx_count is None or packet.system_control_cycle is None):
        raise ValueError("Both STM32 counters must be present in an extended state")
    values = (
        packet.timestamp_us & 0xFFFFFFFF,
        *q,
        *qd,
        *accel,
        *gyro,
        *orientation,
        packet.status_flags & 0xFFFFFFFF,
    )
    if has_counters:
        payload = STATE_PAYLOAD_WITH_COUNTERS.pack(
            *values, packet.command_rx_count & 0xFFFFFFFF,
            packet.system_control_cycle & 0xFFFFFFFF)
    else:
        payload = STATE_PAYLOAD.pack(*values)
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


# action_id values that may go on the wire.
# 3/4 (square / diamond) are NOT here: those cards are walked by the Nano's own
# one-foot policy and never reach the STM32. 7/8 are the card re-pose
# (LEAN / RESTORE) - pose events rather than shape actions, but they travel on
# the same request, and leaving them out here is what made every card_tilt send
# raise "invalid upper-body action request" before it ever hit the wire.
SENDABLE_ACTION_IDS = (1, 2, 5, 6, ACTION_CARD_TILT, ACTION_CARD_RESTORE)


def pack_action_request(packet: ActionRequestPacket) -> bytes:
    if (not 0 < packet.event_id <= 0xFFFFFFFF
            or packet.action_id not in SENDABLE_ACTION_IDS):
        raise ValueError(f"invalid action request: event_id={packet.event_id} "
                         f"action_id={packet.action_id}")
    return _pack_frame(MSG_ACTION_REQUEST, packet.sequence,
                       ACTION_REQUEST_PAYLOAD.pack(packet.event_id, packet.action_id))


def pack_action_status(packet: ActionStatusPacket) -> bytes:
    return _pack_frame(MSG_ACTION_STATUS, packet.sequence,
                       ACTION_STATUS_PAYLOAD.pack(packet.event_id, packet.action_id, packet.status))


def decode_state(sequence: int, payload: bytes) -> StatePacket:
    if len(payload) == STATE_PAYLOAD.size:
        values = STATE_PAYLOAD.unpack(payload)
        has_counters = False
    elif len(payload) == STATE_PAYLOAD_WITH_COUNTERS.size:
        values = STATE_PAYLOAD_WITH_COUNTERS.unpack(payload)
        has_counters = True
    else:
        raise ValueError("Unknown state payload size")
    i = 1
    q = np.array(values[i : i + NUM_JOINTS], dtype=np.float32)
    i += NUM_JOINTS
    qd = np.array(values[i : i + NUM_JOINTS], dtype=np.float32)
    i += NUM_JOINTS
    accel = np.array(values[i : i + 3], dtype=np.float32)
    i += 3
    gyro = np.array(values[i : i + 3], dtype=np.float32)
    i += 3
    orientation = np.array(values[i : i + 4], dtype=np.float32)
    i += 4
    return StatePacket(sequence, values[0], q, qd, accel, gyro, orientation, values[i],
                       values[i + 1] if has_counters else None,
                       values[i + 2] if has_counters else None)


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

    def feed(self, data: bytes) -> Iterable[
        StatePacket | CommandPacket | ActionRequestPacket | ActionStatusPacket
    ]:
        self.buffer.extend(data)
        decoded: list[
            StatePacket | CommandPacket | ActionRequestPacket | ActionStatusPacket
        ] = []

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
                if message_type == MSG_STATE and payload_len in (
                        STATE_PAYLOAD.size, STATE_PAYLOAD_WITH_COUNTERS.size):
                    decoded.append(decode_state(sequence, payload))
                elif message_type == MSG_COMMAND and payload_len == COMMAND_PAYLOAD.size:
                    decoded.append(decode_command(sequence, payload))
                elif message_type == MSG_ACTION_REQUEST and payload_len == ACTION_REQUEST_PAYLOAD.size:
                    decoded.append(ActionRequestPacket(sequence, *ACTION_REQUEST_PAYLOAD.unpack(payload)))
                elif message_type == MSG_ACTION_STATUS and payload_len == ACTION_STATUS_PAYLOAD.size:
                    decoded.append(ActionStatusPacket(sequence, *ACTION_STATUS_PAYLOAD.unpack(payload)))
                else:
                    raise ValueError("Unknown message type or payload size")
            except (ValueError, struct.error):
                self.format_errors += 1
            else:
                self.valid_frames += 1
            del self.buffer[:total_len]

        return decoded
