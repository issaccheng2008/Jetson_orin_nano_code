"""Temporary serial ownership before C starts; no motor commands are sent."""

from pathlib import Path
import sys
import time

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "humanoid_jetson_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from protocol import STARTUP_ARM, STARTUP_CARD_READY, STATE_START_BUTTON, STATE_STARTUP_ACTIVE
from serial_link import SerialLink


class StartupButtonLink:
    def __init__(self, port: str, baud: int = 921600) -> None:
        self.port = port
        self.baud = baud
        self._link = None
        self._next_connect = 0.0
        self._next_write = 0.0
        self._reset_pending = True
        self._last_sent_flags = None
        self._state_flags = None

    def poll(self, card_ready: bool) -> bool:
        now = time.monotonic()
        if self._link is None:
            if now < self._next_connect:
                return False
            self._next_connect = now + 1.0
            try:
                self._link = SerialLink(self.port, self.baud)
                self._link.send_startup_control(0)
                self._last_sent_flags = 0
            except OSError as exc:
                self.close()
                print(f"[button-start] waiting for {self.port}: {exc}", flush=True)
                return False
            self._reset_pending = True
            self._next_write = now + 0.2
            return False
        if not self._link.reader_alive():
            self.close()
            return False
        try:
            state = self._link.get_latest_state(max_age_s=0.5)
        except TimeoutError:
            state = None
        self._state_flags = state.status_flags if state is not None else None
        if self._reset_pending and state is not None and not state.status_flags & STATE_STARTUP_ACTIVE:
            self._reset_pending = False
            self._next_write = 0.0
        flags = (0 if self._reset_pending else
                 STARTUP_ARM | (STARTUP_CARD_READY if card_ready else 0))
        try:
            if flags != self._last_sent_flags or now >= self._next_write:
                self._link.send_startup_control(flags)
                if flags != self._last_sent_flags:
                    print(f"[button-start] startup flags=0x{flags:02x} "
                          f"card_ready={bool(flags & STARTUP_CARD_READY)} "
                          f"state_flags={self._state_flags}", flush=True)
                self._last_sent_flags = flags
                self._next_write = now + 0.2
        except OSError as exc:
            self.close()
            print(f"[button-start] reconnecting {self.port}: {exc}", flush=True)
            return False
        return bool(not self._reset_pending and state is not None
                    and state.status_flags & STATE_STARTUP_ACTIVE
                    and state.status_flags & STATE_START_BUTTON)

    def diagnostics(self):
        return dict(startup_serial_reset_pending=self._reset_pending,
                    startup_serial_sent_flags=self._last_sent_flags,
                    startup_serial_state_flags=self._state_flags)

    def close(self) -> None:
        link, self._link = self._link, None
        if link is None:
            return
        try:
            link.send_startup_control(0)
        except OSError:
            pass
        finally:
            link.close()
        if link.reader_alive():
            raise RuntimeError("Startup serial reader is still alive; cannot hand port to gait")
