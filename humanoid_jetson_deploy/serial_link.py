"""Threaded serial link that keeps the most recent valid STM32 state packet."""

from __future__ import annotations

import threading
import time

import serial

from protocol import CommandPacket, FrameDecoder, StatePacket, pack_command


class SerialLink:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.01, write_timeout=0.05)
        self.decoder = FrameDecoder()
        self._latest_state: StatePacket | None = None
        self._latest_state_host_time = 0.0
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._sequence = 0
        self._thread = threading.Thread(target=self._reader, name="stm32-serial-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self.serial.read(max(1, self.serial.in_waiting))
            except serial.SerialException:
                self._stop.set()
                return
            if not chunk:
                continue
            for message in self.decoder.feed(chunk):
                if isinstance(message, StatePacket):
                    with self._lock:
                        self._latest_state = message
                        self._latest_state_host_time = time.monotonic()

    def wait_for_state(self, timeout_s: float = 3.0) -> StatePacket:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                return self.get_latest_state(max_age_s=timeout_s)
            except TimeoutError:
                time.sleep(0.01)
        raise TimeoutError("No valid STM32 state packet received")

    def get_latest_state(self, max_age_s: float = 0.05) -> StatePacket:
        with self._lock:
            state = self._latest_state
            age = time.monotonic() - self._latest_state_host_time
        if state is None or age > max_age_s:
            raise TimeoutError(f"STM32 state is missing or stale ({age:.3f} s)")
        return state

    def send_command(
        self,
        timestamp_us: int,
        joint_target,
        kp_scale: float,
        kd_scale: float,
        command_flags: int,
    ) -> None:
        packet = CommandPacket(
            sequence=self._sequence,
            timestamp_us=timestamp_us,
            joint_target=joint_target,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            command_flags=command_flags,
        )
        frame = pack_command(packet)
        with self._write_lock:
            self.serial.write(frame)
        self._sequence = (self._sequence + 1) & 0xFFFF

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.2)
        self.serial.close()

    def __enter__(self) -> "SerialLink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
