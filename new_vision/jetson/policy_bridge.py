"""Convert new_vision steering to the historical policy UDP JSON protocol.

No serial port, ONNX runtime, or camera dependency. yaw_sign maps image steering
to policy yaw and defaults to 1: a positive steer (the track lies to the camera's
right) means a positive policy yaw. The documented frame convention implies -1,
but that steered the real robot the wrong way.
"""

from __future__ import annotations

import json
import math
import socket


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class SteeringController:
    """run_robot.py dual-mode PID and one-step preview, mapped to rad/s.

    Hold the last command for lost_hold_s on lost/invalid detection, then stop
    instead of running the serial controller's blind search. Reset PID on loss
    so reacquisition has no derivative kick.
    """

    DERIV_NOMINAL_DT = 0.05  # nominal vision frame period, seconds

    def __init__(self, vx=0.4, max_wz=0.5, steer_full_scale_cm=10.0,
                 yaw_sign=1, step_len_cm=8.0, preview_gain=0.0,
                 straight_gains=(0.83, 0.004, 0.095),
                 curve_gains=(0.83, 0.006, 0.16), integral_limit=60.0,
                 lost_hold_s=0.2, deriv_pole=0.78, bias_cm=0.0, bias_gate_px=12.0,
                 max_lateral_cm=0.0, max_wz_right=0.25, single_line_gain=1.0):
        values = (vx, max_wz, steer_full_scale_cm, yaw_sign, step_len_cm,
                  preview_gain, integral_limit, lost_hold_s, deriv_pole, bias_cm,
                  bias_gate_px, max_lateral_cm, max_wz_right, single_line_gain,
                  *straight_gains, *curve_gains)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("controller settings must be finite")
        if not 0 <= vx <= 1 or not 0 <= max_wz <= 0.5:
            raise ValueError("vx must be in [0, 1] and max_wz in [0, 0.5]")
        if steer_full_scale_cm <= 0 or yaw_sign not in (-1, 1):
            raise ValueError("steering full scale must be positive; yaw sign must be -1 or 1")
        if step_len_cm < 0 or preview_gain < 0 or integral_limit < 0:
            raise ValueError("preview and integral settings must be nonnegative")
        if lost_hold_s < 0:
            raise ValueError("lost hold window must be nonnegative")
        if not 0 <= deriv_pole < 1:
            raise ValueError("derivative filter pole must be in [0, 1)")
        if max_lateral_cm < 0:
            raise ValueError("lateral sanity bound must be nonnegative; 0 disables it")
        if not 0 < max_wz_right <= max_wz:
            raise ValueError("right yaw limit must be in (0, max_wz]")
        if single_line_gain <= 0:
            raise ValueError("single-line gain must be positive")
        self.deriv_pole = deriv_pole
        self.vx, self.max_wz = vx, max_wz
        self.full_scale, self.yaw_sign = steer_full_scale_cm, yaw_sign
        self.step_len, self.preview_gain = step_len_cm, preview_gain
        self.straight_gains, self.curve_gains = straight_gains, curve_gains
        self.integral_limit = integral_limit
        if bias_gate_px <= 0:
            raise ValueError("bias gate must be positive")
        # Additive trim on fused_err_cm, faded in by |curve_px|. The loop settles
        # where the P term balances the disturbance, so a standing lateral offset
        # is removed by shifting where that balance reads zero, not by offsetting
        # wz - a constant wz would only bend a straight into a very large circle.
        # Gating it keeps a curve-only offset from pushing the straights off centre.
        # Off by default. It is one track's measured standing offset, not a property of
        # the loop, and at 5 cm the robot also rode left of centre on the straights.
        self.bias_cm = bias_cm
        self.bias_gate_px = bias_gate_px
        # OFF by default. The near band reports the line's lateral offset in cm, and
        # the lane is 35 cm wide, so anything past half of that puts the robot off
        # the track - which cannot be true while it follows the line. Field runs on
        # 2026-09-26 showed such readings (err held -35.9 cm for twenty seconds while
        # walking), but the bound was never measured against normal running, and
        # turning a reading into an outright stop changes line following. Do not
        # default it on: measure base_err_cm over a full lap first, then set it.
        self.max_lateral_cm = max_lateral_cm
        # Asymmetric yaw limit: right turns capped lower than left. Right is the
        # negative wz the policy receives (wz sign is applied before this).
        self.max_wz_right = max_wz_right
        # Loop-gain multiplier that applies only while a single boundary is visible on
        # a curve. Left at 1.0 it leaves the loop exactly as it was.
        self.single_line_gain = single_line_gain
        # Last frame rejected on the lateral bound, for the caller to log. None
        # otherwise, so a caller can print on the transition instead of every frame.
        self.rejected_lateral = None
        self.last_err_eff = 0.0
        self.reset()
        self.lost_hold_s = lost_hold_s
        # Patience deliberately lives outside reset(): command() calls reset() on
        # every invalid frame, so holding it there would defeat the hold entirely.
        self.lost_s = 0.0
        self.hold = (0.0, 0.0)

    def reset(self, clear_hold=False):
        """Clear the loop state.

        clear_hold also drops the stored last-good command. Callers that are
        deliberately not driving - the card stop - want that: leaving it means the
        first invalid frame after the stop republishes the pre-stop vx and wz,
        which is the replayed command that was already tried and rejected.
        """
        self.integral = 0.0
        self.last_curve = False
        self.last_steer = 0.0
        self.err_window = []
        self.last_median = None
        self.derivative = 0.0
        if clear_hold:
            self.drop_held_command()

    def drop_held_command(self):
        """Forget the stored last-good command, keeping the loop state.

        The card stop needs exactly this much and no more. Leaving `hold` set means the
        first invalid frame after the stop republishes the pre-stop (vx, wz) as the
        lost-line fallback, which is the replay that was already tried and rejected.
        The integral and the derivative window are the walking state, and the stop is a
        pause rather than a fresh start, so they are deliberately left alone.
        """
        self.lost_s = 0.0
        self.hold = (0.0, 0.0)

    def filtered_derivative(self, err):
        """Median-of-3, then a first-order low-pass, over a nominal frame period.

        Differentiating the detector's already-EMA'd error mostly amplifies the
        high-frequency content it still carries: measured on real line video the
        raw term averaged 0.84 cm but swung with a 5.49 cm standard deviation,
        accounting for 5.93 of the 7.56 frame-to-frame steer jitter. The median
        rejects single-frame detection spikes and the nominal period keeps a
        jittering per-frame dt out of the division.
        """
        self.err_window.append(err)
        if len(self.err_window) > 3:
            del self.err_window[0]
        if len(self.err_window) < 3:
            return 0.0
        median = sorted(self.err_window)[1]
        if self.last_median is not None:
            raw = (median - self.last_median) / self.DERIV_NOMINAL_DT
            self.derivative = (self.deriv_pole * self.derivative
                               + (1.0 - self.deriv_pole) * raw)
        self.last_median = median
        return self.derivative

    def command(self, debug, confidence, dt):
        try:
            err = float(debug["fused_err_cm"])
            angle = float(debug["angle_err_deg"])
            lost = float(debug["lost_frames"])
            lateral = float(debug["base_err_cm"])
            valid = (all(math.isfinite(v) for v in (err, angle, lost, confidence, dt))
                     and confidence > 0 and lost == 0 and dt > 0)
            self.rejected_lateral = (
                lateral if valid and self.max_lateral_cm > 0
                and abs(lateral) > self.max_lateral_cm else None)
            valid = valid and self.rejected_lateral is None
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
            self.rejected_lateral = None
        if not valid:
            self.reset()
            # The detector's lost_frames has no hysteresis, so one missed frame
            # would otherwise step 0.4 m/s straight to zero.  Hold the last good
            # command briefly; the connector slews whatever we publish.
            self.lost_s += clamp(dt, 0.01, 0.2)
            if self.lost_s <= self.lost_hold_s:
                return self.hold
            self.hold = (0.0, 0.0)
            return self.hold
        dt = clamp(dt, 0.01, 0.2)
        # curve_mode cannot be this gate: it needs |curve_px| >= curve_switch_px
        # (18) while the real curve reads 9..14, so it opens on 6-13% of curve
        # frames and mis-fires on near-straight ones. |curve_px| fades instead.
        try:
            gate = abs(float(debug.get("curve_px", 0.0))) / self.bias_gate_px
        except (TypeError, ValueError):
            gate = 0.0
        err += self.bias_cm * (clamp(gate, 0.0, 1.0) if math.isfinite(gate) else 0.0)
        curve = bool(debug.get("curve_mode", False))
        # Scale after the bias, not before: on a curve the bias is most of the error
        # (bias 5 cm against a fused err near zero), so amplifying the raw reading
        # alone would be a no-op exactly where the correction is needed. Scaling here
        # takes P, I and D along together, which is what a loop gain should do.
        if curve and debug.get("single_line"):
            err *= self.single_line_gain
        self.last_err_eff = err
        # Clear only on the curve -> straight transition. The previous condition
        # (bottom lock valid AND previous frame was a curve) fired on every frame
        # of a long curve, so the integral never accumulated across it.
        if self.last_curve and not curve:
            self.integral = 0.0
        self.last_curve = curve
        kp, ki, kd = self.curve_gains if curve else self.straight_gains
        self.integral = clamp(self.integral + err * dt,
                              -self.integral_limit, self.integral_limit)
        steer = kp * err + ki * self.integral + kd * self.filtered_derivative(err)
        steer += self.preview_gain * self.step_len * math.sin(math.radians(angle))
        if not math.isfinite(steer):
            self.reset()
            self.hold = (0.0, 0.0)
            return self.hold
        self.last_steer = clamp(steer, -50.0, 50.0)
        wz = self.yaw_sign * clamp(self.last_steer / self.full_scale, -1.0, 1.0) * self.max_wz
        # Clamped here, not downstream, so the connector's slew limiter aims at the
        # capped value instead of ramping toward 0.5 and being cut later.
        if wz < -self.max_wz_right:
            wz = -self.max_wz_right
        self.lost_s = 0.0
        self.hold = (self.vx, wz)
        return self.hold


class ConnectorClient:
    """Use the legacy connector client's wire format and address."""

    def __init__(self, host="127.0.0.1", port=5006):
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, vx, wz, qr=-1, *, event_id=0, event_action=-1):
        if not math.isfinite(vx) or not math.isfinite(wz):
            raise ValueError("velocity must be finite")
        message = {"vx": float(vx), "vy": 0.0, "wz": float(wz), "qr": int(qr)}
        if event_id:
            if not 0 < int(event_id) <= 0xFFFFFFFF or int(event_action) not in (1, 2, 3, 4, 5, 6):
                raise ValueError("invalid shape event")
            message["event_id"] = int(event_id)
            message["event_action"] = int(event_action)
        self.socket.sendto(json.dumps(message, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8"), self.address)

    def close(self):
        try:
            self.publish(0.0, 0.0)
        finally:
            self.socket.close()
