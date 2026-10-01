"""Convert new_vision steering to the historical policy UDP JSON protocol.

No serial port, ONNX runtime, or camera dependency. Positive image steering is
rightward; positive policy yaw is counterclockwise (left), hence yaw_sign=-1.
"""

from __future__ import annotations

import json
import math
import socket


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class SteeringController:
    """run_robot.py dual-mode PID and one-step preview, mapped to rad/s.

    Stop on lost/invalid detection instead of using the serial controller's
    blind search. Reset PID on loss so reacquisition has no derivative kick.
    """

    def __init__(self, vx=0.4, max_wz=0.5, steer_full_scale_cm=50.0,
                 yaw_sign=-1, step_len_cm=8.0, preview_gain=1.0,
                 straight_gains=(0.83, 0.004, 0.095),
                 curve_gains=(0.78, 0.002, 0.16), integral_limit=60.0):
        values = (vx, max_wz, steer_full_scale_cm, yaw_sign, step_len_cm,
                  preview_gain, integral_limit, *straight_gains, *curve_gains)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("controller settings must be finite")
        if not 0 <= vx <= 1 or not 0 <= max_wz <= 0.5:
            raise ValueError("vx must be in [0, 1] and max_wz in [0, 0.5]")
        if steer_full_scale_cm <= 0 or yaw_sign not in (-1, 1):
            raise ValueError("steering full scale must be positive; yaw sign must be -1 or 1")
        if step_len_cm < 0 or preview_gain < 0 or integral_limit < 0:
            raise ValueError("preview and integral settings must be nonnegative")
        self.vx, self.max_wz = vx, max_wz
        self.full_scale, self.yaw_sign = steer_full_scale_cm, yaw_sign
        self.step_len, self.preview_gain = step_len_cm, preview_gain
        self.straight_gains, self.curve_gains = straight_gains, curve_gains
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_err = None
        self.last_curve = False
        self.last_steer = 0.0

    def command(self, debug, confidence, dt):
        try:
            err = float(debug["fused_err"])
            angle = float(debug["angle_err_deg"])
            lost = float(debug["lost_frames"])
            valid = (all(math.isfinite(v) for v in (err, angle, lost, confidence, dt))
                     and confidence > 0 and lost == 0 and dt > 0)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self.reset()
            return 0.0, 0.0
        dt = clamp(dt, 0.01, 0.2)
        curve = bool(debug.get("curve_mode", False))
        if debug.get("bottom_lock_valid", False) and self.last_curve:
            self.integral = 0.0
        self.last_curve = curve
        kp, ki, kd = self.curve_gains if curve else self.straight_gains
        self.integral = clamp(self.integral + err * dt,
                              -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.last_err is None else (err - self.last_err) / dt
        self.last_err = err
        steer = kp * err + ki * self.integral + kd * derivative
        steer += self.preview_gain * self.step_len * math.sin(math.radians(angle))
        if not math.isfinite(steer):
            self.reset()
            return 0.0, 0.0
        self.last_steer = clamp(steer, -50.0, 50.0)
        wz = self.yaw_sign * clamp(self.last_steer / self.full_scale, -1.0, 1.0) * self.max_wz
        return self.vx, wz


class ConnectorClient:
    """Same wire format/address as old_vision/connector_client.py."""

    def __init__(self, host="127.0.0.1", port=5005):
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, vx, wz, qr=-1):
        if not math.isfinite(vx) or not math.isfinite(wz):
            raise ValueError("velocity must be finite")
        message = {"vx": float(vx), "vy": 0.0, "wz": float(wz), "qr": int(qr)}
        self.socket.sendto(json.dumps(message, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8"), self.address)

    def close(self):
        try:
            self.publish(0.0, 0.0)
        finally:
            self.socket.close()
