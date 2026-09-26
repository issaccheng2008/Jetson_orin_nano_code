"""Threaded serial link that keeps the most recent valid STM32 state packet."""

from __future__ import annotations

import threading
import time


from protocol import (
    ActionRequestPacket, ActionStatusPacket, CommandPacket, FrameDecoder,
    StatePacket, pack_action_request, pack_command,
)


class SerialLink:
    def __init__(self, port: str, baudrate: int = 921600) -> None:
        import serial

        self._serial_module = serial
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.01, write_timeout=0.05)
        self.decoder = FrameDecoder()
        self._latest_state: StatePacket | None = None
        self._latest_state_host_time = 0.0
        self._action_status: dict[int, ActionStatusPacket] = {}
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
            except self._serial_module.SerialException:
                self._stop.set()
                return
            if not chunk:
                continue
            for message in self.decoder.feed(chunk):
                if isinstance(message, StatePacket):
                    with self._lock:
                        self._latest_state = message
                        self._latest_state_host_time = time.monotonic()
                elif isinstance(message, ActionStatusPacket):
                    with self._lock:
                        self._action_status[message.event_id] = message
                        if len(self._action_status) > 64:
                            self._action_status.pop(next(iter(self._action_status)))

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
        lock_timeout_s: float | None = None,
    ) -> bool:
        """Write one command frame. Returns False if nothing was written.

        ``lock_timeout_s`` bounds how long this waits for the write lock. The
        emergency path passes a timeout because an unbounded wait is the one way
        a stop request can be starved: if a control thread wedges while holding
        the lock, an e-stop that blocks behind it never gets sent. A racy or
        partial frame is survivable (the STM32 rejects it on CRC), a starvation
        deadlock is not. Callers that pass nothing keep the old blocking
        behaviour.
        """
        packet = CommandPacket(
            sequence=self._sequence,
            timestamp_us=timestamp_us,
            joint_target=joint_target,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            command_flags=command_flags,
        )
        frame = pack_command(packet)
        if lock_timeout_s is None:
            self._write_lock.acquire()
            acquired = True
        else:
            acquired = self._write_lock.acquire(timeout=lock_timeout_s)
        if not acquired:
            return False
        try:
            self.serial.write(frame)
        finally:
            self._write_lock.release()
        self._sequence = (self._sequence + 1) & 0xFFFF
        return True

    def send_action(self, event_id: int, action_id: int) -> None:
        """Send or retry an upper-body event; event_id identifies duplicates."""
        with self._write_lock:
            packet = ActionRequestPacket(self._sequence, event_id, action_id)
            self.serial.write(pack_action_request(packet))
            self._sequence = (self._sequence + 1) & 0xFFFF

    def get_action_status(self, event_id: int) -> int:
        with self._lock:
            packet = self._action_status.get(event_id)
        return packet.status if packet is not None else 0

    def reader_alive(self) -> bool:
        """True while the background reader thread is still running.

        Call this after ``close()`` to confirm the port is really released
        before another process opens the same device. ``close()`` only waits
        0.2 s for the reader, and a thread still blocked in ``read()`` on a
        closed descriptor can consume bytes from a later open that reuses the
        same file descriptor -- silent frame corruption with no error.
        """
        return self._thread.is_alive()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.2)
        self.serial.close()

    def __enter__(self) -> "SerialLink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
