"""One-shot shape-card task scheduler, independent of camera and serial I/O."""

from __future__ import annotations

from dataclasses import dataclass
import math


UPPER_CARDS = (1, 2, 5, 6)
LEG_CARDS = (3, 4)


@dataclass(frozen=True)
class ShapeDecision:
    policy: str = "walking"
    lift_command: float = 0.0
    support_foot: str = "right"
    busy: bool = False
    send_upper: bool = False
    event_id: int = 0
    action_id: int = -1


class ShapeActionController:
    """Stop before each task, route arms/head to STM32 and legs to ONNX."""

    def __init__(self, lift_seconds=3.0, recovery_seconds=0.3,
                 upper_seconds=3.0, stop_timeout=4.0, action_timeout=6.0):
        for value in (lift_seconds, recovery_seconds, upper_seconds,
                      stop_timeout, action_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("action durations and timeouts must be positive and finite")
        self.lift_seconds = lift_seconds
        self.recovery_seconds = recovery_seconds
        self.upper_seconds = upper_seconds
        self.stop_timeout = stop_timeout
        self.action_timeout = action_timeout
        self.phase = "idle"
        self.phase_start = 0.0
        self.event_id = 0
        self.action_id = -1
        self.upper_done = False
        self.seen_event_ids: set[int] = set()

    def accept(self, event_id: int, action_id: int, now: float) -> bool:
        if event_id <= 0 or action_id not in UPPER_CARDS + LEG_CARDS:
            return False
        if event_id in self.seen_event_ids:
            return False
        self.seen_event_ids.add(event_id)
        if self.phase != "idle":
            return False
        self.event_id = event_id
        self.action_id = action_id
        self.upper_done = False
        self.phase = "wait_stop"
        self.phase_start = now
        return True

    def advance(self, now: float, stopped: bool, upper_status: int = 0,
                ready_to_lift: bool = True) -> ShapeDecision:
        phase = self.phase
        if phase == "idle":
            return ShapeDecision()
        if phase == "wait_stop":
            if stopped and (self.action_id in UPPER_CARDS or ready_to_lift):
                self.phase = "upper" if self.action_id in UPPER_CARDS else "lift"
                self.phase_start = now
            elif now - self.phase_start >= self.stop_timeout:
                raise TimeoutError("Robot did not stop before shape action")
        elif phase == "upper":
            if upper_status == 2:                       # ACTION_DONE
                self.upper_done = True
            elif upper_status in (4, 5):                # ACTION_INVALID / ACTION_FAILED
                raise RuntimeError(f"STM32 rejected shape action: status={upper_status}")
            # status 3 (ACTION_BUSY) is not a rejection: the STM32 runs one action at
            # a time and is still finishing the card untilt when the shape request
            # first goes out. Keep requesting until it takes it - the send repeats at
            # 10 Hz in main.py and the timeout below still bounds the wait.
            elif not self.upper_done and now - self.phase_start >= self.action_timeout:
                raise TimeoutError("STM32 shape action did not complete")
            if self.upper_done and now - self.phase_start >= self.upper_seconds:
                self.phase = "idle"
        elif phase == "lift" and now - self.phase_start >= self.lift_seconds:
            self.phase = "recover"
            self.phase_start = now
        elif phase == "recover" and now - self.phase_start >= self.recovery_seconds:
            self.phase = "idle"

        if self.phase == "idle":
            return ShapeDecision()
        if self.action_id in LEG_CARDS and self.phase == "lift":
            return ShapeDecision(
                policy="one-foot", lift_command=1.0,
                support_foot="right" if self.action_id == 3 else "left",
                busy=True, event_id=self.event_id, action_id=self.action_id,
            )
        return ShapeDecision(
            busy=True, send_upper=self.phase == "upper" and not self.upper_done,
            event_id=self.event_id, action_id=self.action_id,
        )
