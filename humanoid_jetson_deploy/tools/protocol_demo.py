#!/usr/bin/env python3
"""Show fragmented decoding without requiring a serial device."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import FrameDecoder, STATE_ENCODERS_VALID, STATE_IMU_VALID, StatePacket, pack_state


packet = StatePacket(
    sequence=42,
    timestamp_us=123456,
    joint_position=np.linspace(-0.2, 0.2, 12, dtype=np.float32),
    joint_velocity=np.zeros(12, dtype=np.float32),
    accel_m_s2=np.array([0.0, 0.0, 9.81], dtype=np.float32),
    gyro_rad_s=np.zeros(3, dtype=np.float32),
    status_flags=STATE_IMU_VALID | STATE_ENCODERS_VALID,
)
frame = pack_state(packet)
decoder = FrameDecoder()
messages = []
for index in range(0, len(frame), 7):
    messages.extend(decoder.feed(frame[index : index + 7]))
print(messages[0])
print(f"Frame bytes={len(frame)}, CRC errors={decoder.crc_errors}")
