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
                 lost_hold_s=0.2, deriv_pole=0.78, bias_cm=5.0, bias_gate_px=12.0,
                 max_lateral_cm=17.5):
        values = (vx, max_wz, steer_full_scale_cm, yaw_sign, step_len_cm,
                  preview_gain, integral_limit, lost_hold_s, deriv_pole, bias_cm,
                  bias_gate_px, max_lateral_cm, *straight_gains, *curve_gains)
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
        if max_lateral_cm <= 0:
            raise ValueError("lateral sanity bound must be positive")
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
        self.bias_cm = bias_cm
        self.bias_gate_px = bias_gate_px
        # The near band reports the line's lateral offset in cm. The lane is 35 cm
        # wide, so anything past half of that puts the robot off the track - which
        # cannot be true while it is following the line. Measured on the robot
        # 2026-09-26: standing still, with no card in the scene, err sat rock steady
        # at -35.9 cm for twenty seconds (near band +73 px, far +54 px, conf 0.78).
        # A wrong reading that stable is not noise, it is a lock onto something else
        # (the other line, a lane edge), and steering on it is how the robot ends up
        # turning hard the wrong way. Treat it as loss instead.
        self.max_lateral_cm = max_lateral_cm
        self.last_err_eff = 0.0
        self.reset()
        self.lost_hold_s = lost_hold_s
        # Patience deliberately lives outside reset(): command() calls reset() on
        # every invalid frame, so holding it there would defeat the hold entirely.
        self.lost_s = 0.0
        self.hold = (0.0, 0.0)

    def reset(self):
        self.integral = 0.0
        self.last_curve = False
        self.last_steer = 0.0
        self.err_window = []
        self.last_median = None
        self.derivative = 0.0

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
                     and confidence > 0 and lost == 0 and dt > 0
                     and abs(lateral) <= self.max_lateral_cm)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
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
        self.last_err_eff = err
        curve = bool(debug.get("curve_mode", False))
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
