#!/usr/bin/env python3
"""Fixed-policy motion core: a 50 Hz loop that drives deliberate poses.

The robot runs in two alternating halves. The learned policy is run by
``main.py``. This module is the other half: the operator types target angles,
picks a key point, and the robot moves there. The two never run at once --
pyserial does not lock the device, so a second opener silently corrupts the
frame stream instead of failing.

Dependency-free at import: no tkinter, no pyserial, no onnxruntime, no
matplotlib. ``serial_link`` is imported lazily by :func:`open_link`, so this
module and its tests work on a machine with no hardware stack installed.

Nothing here energizes a motor unless a caller arms the loop explicitly, and
the ordering that matters is copied from ``main.py`` rather than reinvented:
joint limits, then slew limit, then the measured-position window, applied in
that order every tick.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

import config
from motor_test_common import CONTROL_DT, monotonic_us
from protocol import (
    COMMAND_ENABLE,
    COMMAND_ESTOP,
    STATE_ENCODERS_VALID,
    STATE_FAULT,
    STATE_MOTORS_ENABLED,
)

if TYPE_CHECKING:
    from serial_link import SerialLink


# --- timing and limits -------------------------------------------------------

#: Same freshness bound main.py uses.
MAX_STATE_AGE_S = 0.10
#: Telemetry may be missing this long before it is a fault. During the grace
#: window the loop de-energizes instead of holding blindly, because closed-loop
#: control without feedback is not control.
STALE_FAULT_S = 0.50
#: Consecutive ticks the loop may push a joint that is at the measured-position
#: window without it moving before declaring it blocked. This is the only signal
#: that a joint has stalled: the firmware clamps its feedback to within 0.5 rad
#: of the last target, so a blocked joint otherwise looks ordinary while the
#: motor pushes against it with a constant position error.
CLIP_FAULT_TICKS = 25
#: How far the measured pose must travel for the push to count as progress.
#: Roughly 0.6 degrees; well below anything a working joint does in a tick.
CLIP_PROGRESS_RAD = 0.01
#: A state packet whose sequence stops advancing is a wedged firmware, not a
#: fresh link: get_latest_state judges freshness by host arrival time.
FROZEN_SEQUENCE_FAULT_S = 0.50
#: Peak speed a deliberate pose change may reach, in rad/s. 0.5 puts a 1 rad
#: move at about 3 s, which is the pace the operator asked for. This must stay
#: well under config.MAX_TARGET_SPEED_RAD_S, or the slew limiter becomes the
#: binding constraint and the cosine silently degrades into a trapezoid -- at
#: which point ramp_duration_for no longer describes what the robot does.
RAMP_PEAK_RAD_S = 0.5
RAMP_MIN_S = 1.0
#: How often the latched e-stop re-asserts while it waits to be cleared.
ESTOP_RESEND_S = 0.1
#: Bounded wait for the write lock on the emergency path.
EMERGENCY_LOCK_TIMEOUT_S = 0.05

DEV_WINDOW_RAD = config.MAX_TARGET_DEVIATION_RAD
JOINT_MARGIN_RAD = config.JOINT_LIMIT_MARGIN_RAD


class FaultError(RuntimeError):
    """A condition that must stop motion and de-energize the motors."""


def validate_state(state, *, context: str = "state") -> None:
    """Reject faulty or non-finite encoder feedback before commanding motion."""
    if state.status_flags & STATE_FAULT:
        raise RuntimeError(f"STM32 reports a motor fault: flags=0x{state.status_flags:08X}")
    if not state.status_flags & STATE_ENCODERS_VALID:
        raise RuntimeError(
            f"STM32 encoder-valid flag is missing on {context}: "
            f"flags=0x{state.status_flags:08X}"
        )
    if not np.isfinite(state.joint_position).all():
        raise RuntimeError(f"STM32 joint position contains NaN or Inf on {context}")


# --- pure math ---------------------------------------------------------------


def cosine_blend(phase: float) -> float:
    """Blend 0 -> 1 with zero velocity at both ends.

    Matches ``MotorTestLoop.transition``: ``0.5 - 0.5*cos(pi*phase)``.
    """
    phase = min(1.0, max(0.0, float(phase)))
    return 0.5 - 0.5 * math.cos(math.pi * phase)


def ramp_duration_for(
    delta_rad: float,
    peak_rad_s: float = RAMP_PEAK_RAD_S,
    minimum_s: float = RAMP_MIN_S,
) -> float:
    """Duration for a cosine ramp covering ``delta_rad``.

    A cosine blend has peak rate ``pi*delta/(2*T)``, so ``T = pi*delta/(2*peak)``.

    One shared duration is used for all twelve joints. Per-joint durations would
    make the legs fight; on a legged robot the slowest joint should set the pace.

    ``peak_rad_s`` must stay well under ``config.MAX_TARGET_SPEED_RAD_S``. The
    minimum floor matters too: without it a 0.01 rad correction becomes a 50 ms
    jerk rather than a move.
    """
    delta = abs(float(delta_rad))
    if not math.isfinite(delta):
        raise ValueError("delta_rad must be finite")
    if delta <= 0.0:
        return float(minimum_s)
    return max(float(minimum_s), math.pi * delta / (2.0 * float(peak_rad_s)))


def slew_limit(
    target: np.ndarray,
    previous: np.ndarray,
    dt: float,
    max_speed_rad_s: float | None = None,
) -> np.ndarray:
    """Bound how far one frame may move, mirroring ``main.py``'s helper."""
    speed = config.MAX_TARGET_SPEED_RAD_S if max_speed_rad_s is None else max_speed_rad_s
    maximum_change = speed * dt
    return previous + np.clip(target - previous, -maximum_change, maximum_change)


def clamp_to_current_checked(
    target: np.ndarray,
    q_current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage three of the safety chain, reporting *which* joints it clamped.

    Wrapping ``config.clamp_policy_target_to_current`` is not enough on its own,
    because it can succeed every tick while silently pinning the command at the
    window edge. Callers need to see that happening.

    Caveat worth knowing: on this firmware the measured position that this
    window is computed against is itself clamped to within 0.5 rad of the last
    commanded target. When the tracking error is large the measurement stops
    being independent evidence of where the joint is, and this window's value
    as a safety bound is reduced. The stall detector exists for that case.

    Raises ``ValueError`` when the measured pose leaves no safe window at all.
    """
    target = np.asarray(target, dtype=np.float32)
    limited = config.clamp_policy_target_to_current(target, q_current)
    clipped = np.abs(limited - target) > np.float32(1.0e-6)
    return limited, clipped


def send_disable(
    link,
    q_motor: np.ndarray,
    estop: bool = False,
    repeats: int = 5,
    gap_s: float = 0.005,
    sleep=time.sleep,
    lock_timeout_s: float | None = EMERGENCY_LOCK_TIMEOUT_S,
) -> int:
    """Send repeated de-energizing frames; return how many went out.

    Unlike ``main.py``'s helper and ``MotorTestLoop.emergency_stop_and_close``,
    this keeps retrying after a dropped frame instead of aborting the burst on
    the first exception: one lost write must not cancel the stop.
    """
    flags = COMMAND_ESTOP if estop else 0
    sent = 0
    for _ in range(max(1, int(repeats))):
        try:
            link.send_command(
                monotonic_us(), q_motor, 0.0, 0.0, flags, lock_timeout_s=lock_timeout_s
            )
            sent += 1
        except Exception:  # noqa: BLE001 - a failed stop frame must not stop the retries
            pass
        sleep(gap_s)
    return sent


# --- preview -----------------------------------------------------------------


@dataclass(frozen=True)
class MovePreview:
    """What a move will do, computed before anything is sent."""

    delta_policy_rad: np.ndarray
    max_delta_rad: float
    max_delta_joint: str
    duration_s: float
    clipped_by_window: np.ndarray
    outside_limits: np.ndarray

    @property
    def is_inside_limits(self) -> bool:
        return not bool(np.any(self.outside_limits))

    @property
    def clipped_joint_names(self) -> list[str]:
        return [
            name for name, hit in zip(config.JOINT_NAMES, self.clipped_by_window) if hit
        ]

    def describe(self) -> str:
        text = (
            f"{self.max_delta_joint} moves {np.rad2deg(self.max_delta_rad):.1f} deg "
            f"over {self.duration_s:.1f} s"
        )
        if self.clipped_joint_names:
            text += f"; the safety window will ease: {', '.join(self.clipped_joint_names)}"
        if not self.is_inside_limits:
            text += "; REFUSED: a target is outside the joint limits"
        return text


def preview_move(goal_policy_rad, measured_policy_rad) -> MovePreview:
    """Describe a move to the goal before committing to it."""
    goal = np.asarray(goal_policy_rad, dtype=np.float64).reshape(config.NUM_JOINTS)
    measured = np.asarray(measured_policy_rad, dtype=np.float64).reshape(config.NUM_JOINTS)
    delta = goal - measured
    worst = int(np.argmax(np.abs(delta)))
    return MovePreview(
        delta_policy_rad=delta,
        max_delta_rad=float(abs(delta[worst])),
        max_delta_joint=config.JOINT_NAMES[worst],
        duration_s=ramp_duration_for(delta[worst]),
        clipped_by_window=np.abs(delta) > DEV_WINDOW_RAD,
        outside_limits=(goal < config.Q_LOWER + JOINT_MARGIN_RAD)
        | (goal > config.Q_UPPER - JOINT_MARGIN_RAD),
    )


# --- key points --------------------------------------------------------------


class KeypointFileError(ValueError):
    """A keypoints.json that cannot be played back safely."""


@dataclass(frozen=True)
class Keypoint:
    index: int
    name: str
    rad: np.ndarray
    stable: bool
    host_time_iso: str
    raw: dict


def load_keypoints(path) -> list[Keypoint]:
    """Read a ``keypoints.json`` containing named policy-angle poses.

    Rejects a file whose ``joint_names`` disagree with this checkout. That field
    exists for exactly this check: the repo has two competing joint orders, and
    a file recorded under the other one would mirror the robot's legs on
    playback.
    """
    path = Path(path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeypointFileError(f"{path} does not exist") from exc
    except OSError as exc:
        raise KeypointFileError(f"{path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise KeypointFileError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise KeypointFileError(f"{path} must contain a JSON object")

    recorded_names = list(payload.get("joint_names") or [])
    if recorded_names != list(config.JOINT_NAMES):
        raise KeypointFileError(
            f"{path} was recorded with a different joint order, so playing it "
            f"back would mirror the legs.\n  recorded: {recorded_names}\n"
            f"  expected: {list(config.JOINT_NAMES)}"
        )

    entries = payload.get("keypoints")
    if not isinstance(entries, list):
        raise KeypointFileError(f"{path} has no 'keypoints' list")

    keypoints: list[Keypoint] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise KeypointFileError(f"{path}: keypoints[{position}] is not an object")
        values = entry.get("joint_position_rad")
        if values is None:
            raise KeypointFileError(
                f"{path}: keypoints[{position}] has no 'joint_position_rad'"
            )
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (config.NUM_JOINTS,):
            raise KeypointFileError(
                f"{path}: keypoints[{position}] has {array.size} values, "
                f"expected {config.NUM_JOINTS}"
            )
        if not np.all(np.isfinite(array)):
            raise KeypointFileError(
                f"{path}: keypoints[{position}] contains non-finite values"
            )
        keypoints.append(
            Keypoint(
                index=position,
                name=str(entry.get("name") or f"keypoint_{position:03d}"),
                rad=array,
                stable=bool(entry.get("stable", False)),
                host_time_iso=str(entry.get("host_time_iso", "")),
                raw=entry,
            )
        )

    if not keypoints:
        raise KeypointFileError(f"{path} contains no key points")
    return keypoints


# --- intent and telemetry ----------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One leg of a trajectory.

    ``duration_s`` is the recorded hint only. The loop always re-derives the
    duration from the pose the robot is actually at, so a plan recorded
    elsewhere cannot demand a wild first leg.
    """

    destination_policy_rad: np.ndarray
    duration_s: float


@dataclass(frozen=True)
class Intent:
    """The operator's complete wish, replaced wholesale.

    This is *state*, never an event queue. A queue would let a stale move run
    after a newer e-stop or a newer goal; last-writer-wins is the correct
    semantic for every field here. The monotonic ``seq`` lets the control thread
    tell a fresh intent from a stale one without any extra handshake.
    """

    seq: int
    enable: bool
    trajectory: tuple[Segment, ...] = ()
    source_label: str = ""


@dataclass(frozen=True)
class Telemetry:
    """An immutable snapshot the GUI can render without locking anything."""

    tick: int
    mode: str  # waiting | disabled | holding | moving | estop | fault
    enabled: bool
    fault: str | None
    telemetry_ok: bool
    motors_reported_on: bool
    state_sequence: int
    state_flags: int
    measured_policy_rad: np.ndarray
    commanded_policy_rad: np.ndarray
    segment_index: int
    segment_count: int
    segment_phase: float
    clipped: bool
    source_label: str


# --- the loop ----------------------------------------------------------------


class ControlLoop:
    """The 50 Hz thread that owns the serial link while the GUI commands motion.

    Exactly one thing may hold the link at a time, in this process or any other.
    The thread is the sole writer: nothing else in the process may call
    ``send_command`` except the GUI's wedged-thread watchdog, and only because
    the single writer is by hypothesis already dead.
    """

    def __init__(
        self,
        link: "SerialLink",
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        log_path: Path | None = None,
        log_hz: float = 20.0,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._link = link
        # The control thread owns this file, so every row lands next to the
        # frames it describes and the Tk thread never blocks on a write. After
        # an incident "what did the GUI command, and when" has to be answerable.
        self._log_path = Path(log_path) if log_path is not None else None
        self._log_hz = float(log_hz)
        self._log_file = None
        self._log_writer = None
        self._log_rows = 0
        self._next_log = 0.0
        self._started_at = 0.0
        # The firmware currently comments out both gain scales and hardcodes
        # 1.0, so these are carried for protocol completeness only.
        self._kp_scale = float(kp_scale)
        self._kd_scale = float(kd_scale)
        self._clock = clock
        self._sleep = sleep

        self._intent_lock = threading.Lock()
        self._intent = Intent(seq=0, enable=False)
        self._applied_seq = 0
        self._telemetry_lock = threading.Lock()
        self._telemetry = Telemetry(
            tick=0,
            mode="waiting",
            enabled=False,
            fault=None,
            telemetry_ok=False,
            motors_reported_on=False,
            state_sequence=0,
            state_flags=0,
            measured_policy_rad=np.zeros(config.NUM_JOINTS, dtype=np.float64),
            commanded_policy_rad=np.zeros(config.NUM_JOINTS, dtype=np.float64),
            segment_index=0,
            segment_count=0,
            segment_phase=0.0,
            clipped=False,
            source_label="",
        )

        self._estop = threading.Event()
        self._estop_reason = ""
        self._fault = threading.Event()
        self._fault_reason: str | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._run, name="joint-motion", daemon=True)
        self._tick_count = 0
        self._last_tick_at = 0.0
        self._last_sequence: int | None = None
        self._frozen_ticks = 0
        self._stale_ticks = 0
        self._clip_ticks = 0
        self._clip_reference: np.ndarray | None = None
        self._prev_tick = self._clock()
        self._next_deadline = self._clock()

        self._last_commanded = np.zeros(config.NUM_JOINTS, dtype=np.float64)
        self._trajectory: tuple[Segment, ...] = ()
        self._segment_index = 0
        self._segment_start_pose = self._last_commanded.copy()
        self._segment_started_at = 0.0
        self._segment_duration = 0.0
        self._seed_valid = False

    # -- public API (any thread) ---------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def tick_age(self) -> float:
        """Seconds since the last completed tick. The GUI's watchdog input."""
        return self._clock() - self._last_tick_at

    def snapshot(self) -> Telemetry:
        with self._telemetry_lock:
            return self._telemetry

    def set_enable(self, enabled: bool) -> int:
        """Arm or disarm. Disarming de-energizes; it does not hold the pose."""
        with self._intent_lock:
            seq = self._intent.seq + 1
            self._intent = Intent(
                seq=seq,
                enable=bool(enabled),
                trajectory=self._intent.trajectory,
                source_label=self._intent.source_label,
            )
        self._wake.set()
        return seq

    def request_move(self, goal_policy_rad, label: str = "") -> int:
        """Travel to one pose and hold there.

        The duration is not a parameter: it is derived from how far the robot
        actually has to travel, so a move can never be asked for faster than
        RAMP_PEAK_RAD_S.
        """
        goal = np.asarray(goal_policy_rad, dtype=np.float64).reshape(config.NUM_JOINTS)
        return self._replace_trajectory(
            (Segment(destination_policy_rad=goal, duration_s=0.0),),
            label or "move",
        )

    def request_trajectory(self, segments, label: str = "") -> int:
        """Play a sequence of poses in order, then hold the last one."""
        prepared = tuple(
            Segment(
                destination_policy_rad=np.asarray(segment.destination_policy_rad,
                                                   dtype=np.float64).reshape(config.NUM_JOINTS),
                duration_s=float(segment.duration_s),
            )
            for segment in segments
        )
        if not prepared:
            raise ValueError("a trajectory needs at least one segment")
        if any(segment.duration_s <= 0.0 for segment in prepared):
            raise ValueError("every segment duration must be positive")
        return self._replace_trajectory(prepared, label or "sequence")

    def request_hold(self, label: str = "hold") -> int:
        """Stop advancing and hold wherever the loop currently is."""
        return self._replace_trajectory((), label)

    def _replace_trajectory(self, trajectory: tuple[Segment, ...], label: str) -> int:
        with self._intent_lock:
            seq = self._intent.seq + 1
            self._intent = Intent(
                seq=seq,
                enable=self._intent.enable,
                trajectory=trajectory,
                source_label=label,
            )
        self._wake.set()
        return seq

    def emergency_stop(self, reason: str = "operator") -> None:
        """Latch the stop.

        Deliberately does not touch the link: the control thread is the sole
        writer. Setting the event and waking the tick means the next frame is an
        e-stop within microseconds rather than after a 20 ms sleep.
        """
        self._estop_reason = str(reason)
        self._estop.set()
        self._wake.set()

    def clear_estop(self) -> None:
        """Release the latch. Requires that the loop is disarmed."""
        with self._intent_lock:
            if self._intent.enable:
                raise RuntimeError("disarm before clearing the emergency stop")
        self._estop_reason = ""
        self._estop.clear()

    def reset_fault(self) -> None:
        """Clear a latched fault. Requires that the loop is disarmed."""
        with self._intent_lock:
            if self._intent.enable:
                raise RuntimeError("disarm before clearing a fault")
        self._fault_reason = None
        self._fault.clear()
        self._estop_reason = ""
        self._estop.clear()

    def shutdown(self, timeout_s: float = 1.0) -> None:
        """Stop the thread. The loop itself closes the link on its way out."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout_s)

    # -- control thread ------------------------------------------------------

    def _read_intent(self) -> Intent:
        with self._intent_lock:
            return self._intent

    @staticmethod
    def _validate(state, context: str) -> None:
        """Promote the reader's RuntimeError into this loop's own fault type.

        Everything the tick raises must be a FaultError, so the contract is one
        type rather than a menu the caller has to enumerate.
        """
        try:
            validate_state(state, context=context)
        except RuntimeError as exc:
            raise FaultError(str(exc)) from exc

    def _run(self) -> None:
        try:
            if not self._seed_from_measured(timeout_s=3.0):
                raise FaultError("no STM32 state packet arrived before the loop started")
            self._open_log()
            self._last_tick_at = self._clock()
            while not self._stop.is_set():
                if self._estop.is_set():
                    # Latched. Keep telling the firmware to stay stopped, but
                    # stay alive so the GUI can see us and the operator can reset.
                    send_disable(self._link, self._last_commanded, estop=True, repeats=3)
                    self._wake.wait(ESTOP_RESEND_S)
                    self._wake.clear()
                    self._publish(mode="estop")
                    continue
                now = self._clock()
                self._tick(now)
                self._last_tick_at = self._clock()
                self._pace()
        except BaseException as exc:  # noqa: BLE001 - the loop must never die quietly
            self._enter_fault(exc)
        finally:
            self._shutdown_link()

    def _pace(self) -> None:
        self._next_deadline += CONTROL_DT
        remaining = self._next_deadline - self._clock()
        if remaining > 0.0:
            # Waiting on the event rather than sleeping means an e-stop cuts the
            # tick immediately instead of after up to 20 ms.
            self._wake.wait(remaining)
            self._wake.clear()
        else:
            # Behind schedule. Reset rather than catching up with a burst of
            # commands, exactly as main.py does.
            self._next_deadline = self._clock()

    def _seed_from_measured(self, timeout_s: float) -> bool:
        """Seed last_commanded from the measured pose.

        This is what makes handing over from ``main.py`` safe. Reusing a target
        remembered from before the handover -- or from before an e-stop -- would
        command a discontinuous jump on the first enabled frame.
        """
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            try:
                state = self._link.get_latest_state(max_age_s=1.0)
            except TimeoutError:
                self._sleep(0.01)
                continue
            self._validate(state, "loop start")
            self._last_commanded = np.asarray(
                config.motor_to_policy_position(state.joint_position), dtype=np.float64
            )
            self._segment_start_pose = self._last_commanded.copy()
            self._last_sequence = state.sequence
            self._seed_valid = True
            self._next_deadline = self._clock()
            self._publish(mode="disabled", telemetry_ok=True)
            return True
        return False

    def _tick(self, now: float) -> None:
        intent = self._read_intent()
        if intent.seq != self._applied_seq:
            self._apply_intent(intent, now)

        # -- telemetry -------------------------------------------------------
        try:
            state = self._link.get_latest_state(max_age_s=MAX_STATE_AGE_S)
        except TimeoutError:
            self._stale_ticks += 1
            if self._stale_ticks * CONTROL_DT > STALE_FAULT_S:
                raise FaultError("no STM32 telemetry")
            # Closed-loop control without feedback is not control. De-energize.
            self._write(self._last_commanded, enable=False)
            self._publish(mode="disabled", telemetry_ok=False)
            return
        self._stale_ticks = 0
        self._validate(state, "control loop")

        if state.sequence == self._last_sequence:
            # Freshness is judged by host arrival time, so a wedged firmware that
            # keeps transmitting would look fresh forever. Require progress.
            self._frozen_ticks += 1
            if self._frozen_ticks * CONTROL_DT > FROZEN_SEQUENCE_FAULT_S:
                raise FaultError("STM32 state sequence stopped advancing")
            self._publish(self._telemetry.mode, telemetry_ok=False)
            return
        self._last_sequence = state.sequence
        self._frozen_ticks = 0

        q_now = np.asarray(
            config.motor_to_policy_position(state.joint_position), dtype=np.float64
        )

        # -- desired target from the plan ------------------------------------
        desired = self._desired(now)

        # -- three-stage safety chain, in main.py's order --------------------
        dt = float(np.clip(now - self._prev_tick, 0.005, 0.05))
        self._prev_tick = now
        target = config.clamp_policy_target(desired)
        target = slew_limit(target, self._last_commanded, dt)
        # Sample stage three's input before it runs. A target resting exactly on
        # the window edge is still being pushed, but stage three would leave it
        # untouched and report no clipping -- which is precisely the steady state
        # a blocked joint settles into.
        fighting_mask = np.abs(target - q_now) >= DEV_WINDOW_RAD
        target, clipped_mask = clamp_to_current_checked(target, q_now)

        self._update_stall_detector(q_now, fighting_mask)
        if self._clip_ticks > CLIP_FAULT_TICKS:
            stalled = [
                name for name, hit in zip(config.JOINT_NAMES, fighting_mask) if hit
            ]
            raise FaultError(
                "cannot follow the commanded target; these joints are not moving "
                f"toward it: {', '.join(stalled) or 'none'}"
            )
        clipped = bool(np.any(clipped_mask))

        # -- send ------------------------------------------------------------
        if self._estop.is_set():
            send_disable(self._link, target, estop=True, repeats=3)
            self._publish(mode="estop")
            return

        enabled = bool(intent.enable) and not self._fault.is_set()
        if self._write(target, enable=enabled):
            self._last_commanded = np.asarray(target, dtype=np.float64)
        self._tick_count += 1
        self._log_row(now, q_now, target, enabled, intent.source_label)
        self._publish(
            mode="moving" if (enabled and self._segment_index < len(self._trajectory))
            else ("holding" if enabled else "disabled"),
            telemetry_ok=True,
            state=state,
            q_now=q_now,
            target=target,
            clipped=clipped,
            source_label=intent.source_label,
        )

    def _write(self, target: np.ndarray, enable: bool) -> bool:
        flags = COMMAND_ENABLE if enable else 0
        kp = self._kp_scale if enable else 0.0
        kd = self._kd_scale if enable else 0.0
        try:
            return bool(
                self._link.send_command(
                    monotonic_us(),
                    np.asarray(target, dtype=np.float32),
                    kp,
                    kd,
                    flags,
                    lock_timeout_s=EMERGENCY_LOCK_TIMEOUT_S,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a dead link is a fault, not a crash
            raise FaultError(f"could not write to the STM32: {exc}") from exc

    def _apply_intent(self, intent: Intent, now: float) -> None:
        """Materialise a new intent's cursor, re-based on the last commanded pose.

        Triggered by any change to the whole Intent, not just ``seq``, so that an
        arm/disarm/re-arm cycle cannot resume a ramp from the wrong origin.
        """
        self._applied_seq = intent.seq
        self._trajectory = tuple(intent.trajectory)
        self._segment_index = 0
        self._segment_start_pose = self._last_commanded.copy()
        self._segment_started_at = now
        self._segment_duration = self._segment_duration_for(0)
        # A disarmed loop drops its plan outright; the operator gets a clean
        # start rather than a silently resumed ramp.
        if not intent.enable:
            self._segment_index = len(self._trajectory)

    def _segment_duration_for(self, index: int) -> float:
        if index >= len(self._trajectory):
            return 0.0
        destination = self._trajectory[index].destination_policy_rad
        # Recompute from what is left to travel, not from the recorded value:
        # the robot may not be where the file assumed it was.
        delta = float(np.max(np.abs(destination - self._segment_start_pose)))
        return ramp_duration_for(delta)

    def _desired(self, now: float) -> np.ndarray:
        while self._segment_index < len(self._trajectory):
            segment = self._trajectory[self._segment_index]
            phase = (
                1.0
                if self._segment_duration <= 0.0
                else (now - self._segment_started_at) / self._segment_duration
            )
            if phase < 1.0:
                blend = cosine_blend(phase)
                return self._segment_start_pose + blend * (
                    segment.destination_policy_rad - self._segment_start_pose
                )
            # Arrived. The endpoint becomes the origin of the next leg.
            self._segment_start_pose = segment.destination_policy_rad.copy()
            self._segment_index += 1
            self._segment_started_at = now
            self._segment_duration = self._segment_duration_for(self._segment_index)
        return self._last_commanded.copy()

    def _update_stall_detector(self, q_now: np.ndarray, fighting_mask: np.ndarray) -> None:
        """Track how long the loop has been pushing a joint that will not move.

        The discriminator is physical movement, not the error: a joint closing
        the gap -- however slowly -- is not stalled, so the counter restarts
        whenever the measured pose has travelled far enough.
        """
        if not bool(np.any(fighting_mask)):
            self._clip_ticks = 0
            self._clip_reference = None
            return
        if self._clip_reference is None:
            self._clip_reference = np.asarray(q_now, dtype=np.float64).copy()
        elif float(np.max(np.abs(q_now - self._clip_reference))) > CLIP_PROGRESS_RAD:
            self._clip_ticks = 0
            self._clip_reference = np.asarray(q_now, dtype=np.float64).copy()
        self._clip_ticks += 1

    def _enter_fault(self, exc: BaseException) -> None:
        self._fault_reason = f"{type(exc).__name__}: {exc}"
        try:
            send_disable(self._link, self._last_commanded, estop=True, repeats=5)
        except Exception:  # noqa: BLE001
            pass
        self._fault.set()
        try:
            self._publish(mode="fault", enabled=False, fault=self._fault_reason)
        except Exception:  # noqa: BLE001
            pass
        # Latched until reset_fault(), which requires a disarm first.

    def _open_log(self) -> None:
        if self._log_path is None or self._log_hz <= 0.0:
            return
        import csv
        from datetime import datetime

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("w", encoding="utf-8", newline="", buffering=1)
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            ["host_time_iso", "elapsed_s", "tick", "mode", "enabled", "source_label",
             "state_sequence", "flags"]
            + [f"measured_{name}_rad" for name in config.JOINT_NAMES]
            + [f"commanded_{name}_rad" for name in config.JOINT_NAMES]
        )
        self._started_at = self._clock()
        self._next_log = self._started_at
        self._datetime = datetime

    def _log_row(self, now: float, q_now, target, enabled: bool, label: str) -> None:
        """Record one commanded frame. Runs on the control thread only."""
        if self._log_writer is None or now < self._next_log:
            return
        measured = np.asarray(q_now, dtype=np.float64).reshape(config.NUM_JOINTS)
        commanded = np.asarray(target, dtype=np.float64).reshape(config.NUM_JOINTS)
        self._log_writer.writerow(
            [
                self._datetime.now().astimezone().isoformat(timespec="milliseconds"),
                f"{now - self._started_at:.6f}",
                int(self._tick_count),
                self._telemetry.mode,
                int(bool(enabled)),
                label,
                self._telemetry.state_sequence,
                f"0x{self._telemetry.state_flags:08X}",
                *[f"{value:.8f}" for value in measured],
                *[f"{value:.8f}" for value in commanded],
            ]
        )
        self._log_rows += 1
        self._next_log = max(now, self._next_log) + 1.0 / self._log_hz

    def _log_rows_written(self) -> int:
        return self._log_rows

    def _shutdown_link(self) -> None:
        try:
            if self._log_file is not None and not self._log_file.closed:
                self._log_file.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            send_disable(self._link, self._last_commanded, estop=False, repeats=3)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._link.close()
        except Exception:  # noqa: BLE001
            pass

    def _publish(
        self,
        mode: str,
        *,
        telemetry_ok: bool | None = None,
        enabled: bool | None = None,
        fault: str | None = None,
        state=None,
        q_now=None,
        target=None,
        clipped: bool | None = None,
        source_label: str | None = None,
    ) -> None:
        """Publish an immutable snapshot.

        Arrays are copied: a Telemetry handed to the GUI must never alias a
        buffer this thread is still writing.
        """
        with self._intent_lock:
            intent_enable = self._intent.enable
        with self._telemetry_lock:
            current = self._telemetry
            flags = int(getattr(state, "status_flags", current.state_flags))
            self._telemetry = Telemetry(
                tick=self._tick_count,
                mode=mode,
                enabled=intent_enable if enabled is None else bool(enabled),
                fault=self._fault_reason if fault is None else fault,
                telemetry_ok=current.telemetry_ok if telemetry_ok is None else bool(telemetry_ok),
                motors_reported_on=bool(flags & STATE_MOTORS_ENABLED)
                if state is not None
                else current.motors_reported_on,
                state_sequence=int(getattr(state, "sequence", current.state_sequence)),
                state_flags=flags,
                measured_policy_rad=(
                    np.asarray(q_now, dtype=np.float64).copy()
                    if q_now is not None
                    else current.measured_policy_rad
                ),
                commanded_policy_rad=(
                    np.asarray(target, dtype=np.float64).copy()
                    if target is not None
                    else current.commanded_policy_rad
                ),
                segment_index=self._segment_index,
                segment_count=len(self._trajectory),
                segment_phase=self._current_phase(),
                clipped=current.clipped if clipped is None else bool(clipped),
                source_label=current.source_label if source_label is None else source_label,
            )

    def _current_phase(self) -> float:
        if self._segment_duration <= 0.0 or self._segment_index >= len(self._trajectory):
            return 0.0
        return min(
            1.0,
            max(0.0, (self._clock() - self._segment_started_at) / self._segment_duration),
        )


# --- link helpers ------------------------------------------------------------


def open_link(port: str, baud: int) -> "SerialLink":
    """Open the serial link, importing pyserial only when actually needed."""
    from serial_link import SerialLink

    return SerialLink(port, baud)


def open_link_with_retry(
    port: str,
    baud: int,
    attempts: int = 10,
    delay_s: float = 0.2,
    sleep=time.sleep,
) -> "SerialLink":
    """Reopen the port, tolerating the kernel's release lag.

    A busy port is worth retrying; a missing device is terminal and the operator
    needs to look at the cable, so that case raises immediately. The retry lives
    here rather than inside ``SerialLink.__init__`` because each failed attempt
    there would leak a reader thread.
    """
    last: Exception | None = None
    for _ in range(max(1, int(attempts))):
        try:
            return open_link(port, baud)
        except Exception as exc:  # noqa: BLE001
            last = exc
            text = str(exc).lower()
            if "no such file" in text or "cannot find" in text or "could not open" in text:
                raise
            sleep(delay_s)
    assert last is not None
    raise last
