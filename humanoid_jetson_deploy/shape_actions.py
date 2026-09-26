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

    def __init__(self, stand_seconds=0.5, lift_seconds=4.0,
                 lower_seconds=0.8, stop_timeout=4.0, action_timeout=6.0):
        for value in (stand_seconds, lift_seconds, lower_seconds, stop_timeout, action_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("action durations and timeouts must be positive and finite")
        self.stand_seconds = stand_seconds
        self.lift_seconds = lift_seconds
        self.lower_seconds = lower_seconds
        self.stop_timeout = stop_timeout
        self.action_timeout = action_timeout
        self.phase = "idle"
        self.phase_start = 0.0
        self.event_id = 0
        self.action_id = -1
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
        self.phase = "wait_stop"
        self.phase_start = now
        return True

    def advance(self, now: float, stopped: bool, upper_status: int = 0) -> ShapeDecision:
        phase = self.phase
        if phase == "idle":
            return ShapeDecision()
        if phase == "wait_stop":
            if stopped:
                self.phase = "upper" if self.action_id in UPPER_CARDS else "stand"
                self.phase_start = now
            elif now - self.phase_start >= self.stop_timeout:
                raise TimeoutError("Robot did not stop before shape action")
        elif phase == "upper":
            if upper_status == 2:
                self.phase = "idle"
            elif upper_status in (3, 4, 5):
                raise RuntimeError(f"STM32 rejected shape action: status={upper_status}")
            elif now - self.phase_start >= self.action_timeout:
                raise TimeoutError("STM32 shape action did not complete")
        elif phase == "stand" and now - self.phase_start >= self.stand_seconds:
            self.phase = "lift"
            self.phase_start = now
        elif phase == "lift" and now - self.phase_start >= self.lift_seconds:
            self.phase = "lower"
            self.phase_start = now
        elif phase == "lower" and now - self.phase_start >= self.lower_seconds:
            self.phase = "idle"

        if self.phase == "idle":
            return ShapeDecision()
        if self.action_id in LEG_CARDS and self.phase != "wait_stop":
            return ShapeDecision(
                policy="one-foot", lift_command=float(self.phase == "lift"),
                support_foot="right" if self.action_id == 3 else "left",
                busy=True, event_id=self.event_id, action_id=self.action_id,
            )
        return ShapeDecision(
            busy=True, send_upper=self.phase == "upper",
            event_id=self.event_id, action_id=self.action_id,
        )
