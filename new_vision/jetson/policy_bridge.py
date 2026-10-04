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

    On lost/invalid detection the steering fades over lost_hold_s and the PID
    state is reset so reacquisition has no derivative kick; the walking speed
    itself never fades (see lost_command). A real stop is published elsewhere
    (start gate / card window / event hold).
    """

    DERIV_NOMINAL_DT = 0.05  # nominal vision frame period, seconds

    def __init__(self, vx=0.2, max_wz=0.5, steer_full_scale_cm=10.0,
                 yaw_sign=1, step_len_cm=8.0, preview_gain=0.0,
                 straight_gains=(0.83, 0.004, 0.095),
                 curve_gains=(0.83, 0.006, 0.16), integral_limit=60.0,
                 lost_hold_s=0.2, deriv_pole=0.78, bias_cm=3.0, bias_gate_px=12.0,
                 bias_dead_px=6.0, bias_straight_cm=1.0, max_lateral_cm=0.0,
                 max_wz_right=None, single_line_gain=1.0, center_dead_cm=4.0):
        # Unset means symmetric: right turns get the same limit as left. A caller that
        # wants them capped lower has to say so.
        if max_wz_right is None:
            max_wz_right = max_wz
        values = (vx, max_wz, steer_full_scale_cm, yaw_sign, step_len_cm,
                  preview_gain, integral_limit, lost_hold_s, deriv_pole, bias_cm,
                  bias_gate_px, bias_dead_px, bias_straight_cm, max_lateral_cm,
                  max_wz_right, single_line_gain, center_dead_cm,
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
        if center_dead_cm < 0:
            raise ValueError("centre dead band must be nonnegative; 0 disables it")
        self.deriv_pole = deriv_pole
        self.vx, self.max_wz = vx, max_wz
        self.full_scale, self.yaw_sign = steer_full_scale_cm, yaw_sign
        self.step_len, self.preview_gain = step_len_cm, preview_gain
        self.straight_gains, self.curve_gains = straight_gains, curve_gains
        self.integral_limit = integral_limit
        if bias_gate_px <= 0:
            raise ValueError("bias gate must be positive")
        if not 0 <= bias_dead_px < bias_gate_px:
            raise ValueError("bias dead band must be in [0, bias gate)")
        # Additive trim on fused_err_cm, faded in by |curve_px|. The loop settles
        # where the P term balances the disturbance, so a standing lateral offset
        # is removed by shifting where that balance reads zero, not by offsetting
        # wz - a constant wz would only bend a straight into a very large circle.
        # Gating it keeps a curve-only offset from pushing the straights off centre.
        # Off by default. It is one track's measured standing offset, not a property of
        # the loop, and at 5 cm the robot also rode left of centre on the straights.
        self.bias_cm = bias_cm
        self.bias_straight_cm = bias_straight_cm
        self.bias_gate_px = bias_gate_px
        self.bias_dead_px = bias_dead_px
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
        # The lane is 35 cm wide, so a reading of a centimetre or two is inside what
        # the near band resolves at all - but a small constant wz still integrates
        # into a drift over a lap. Inside this band the yaw authority is faded toward
        # zero at the centre rather than switched off, so the loop keeps correcting
        # and only how hard it corrects changes. 0 disables it.
        self.center_dead_cm = center_dead_cm
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

    def read_detection(self, debug, confidence, dt):
        """Validate the measurement for either continuous or discrete steering."""
        try:
            err = float(debug["fused_err_cm"])
            angle = float(debug["angle_err_deg"])
            lost = float(debug["lost_frames"])
            lateral = float(debug["base_err_cm"])
            valid = (all(math.isfinite(v) for v in (err, angle, lost, lateral, confidence, dt))
                     and confidence > 0 and lost == 0 and dt > 0)
            # P1 detectors explicitly distinguish a fresh measurement from a
            # diagnostic value retained for display. Legacy producers omit these
            # fields and keep their existing contract.
            valid = valid and bool(debug.get("measurement_valid", True))
            if debug.get("measurement_stale", False):
                valid = False
                # Expired geometry withdraws steering immediately. External
                # stops still use drop_held_command() to clear speed as well.
                self.hold = (self.hold[0], 0.0)
            self.rejected_lateral = (
                lateral if valid and self.max_lateral_cm > 0
                and abs(lateral) > self.max_lateral_cm else None)
            valid = valid and self.rejected_lateral is None
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
            self.rejected_lateral = None
        return (err, angle) if valid else None

    def lost_command(self, dt, *, clamp_elapsed=True):
        """Fade the *steering*; the speed keeps whatever the last good command had.

        丢线/无效帧只说明"转向看不可信"，不该把前进也停掉 —— 2026-10-04 实车：
        无效帧占四成，每次掉线都把车钉在 vx=0（C 侧的 0.5 s 最小保持再钉半秒），
        整趟走不起来。速度取上一条好命令的速度：走路时就是 --vx。
        真停（起跑门控、卡窗口、事件保持）走的是另一条路，而且已经
        `drop_held_command()` 把 hold 清成 0 —— 所以"无效帧不能凭空起步"
        这条契约不变：没有走路上下文时这里就是 (0, 0)。
        离散计时器用真实经过时间（clamp_elapsed=False）。
        """
        self.reset()
        if clamp_elapsed:
            elapsed = clamp(dt, 0.01, 0.2)
        elif math.isfinite(dt) and dt > 0.0:
            elapsed = dt
        else:
            # An invalid clock cannot justify replaying a stale motion command.
            self.lost_s = math.inf
            self.hold = (self.hold[0], 0.0)
            return self.hold
        self.lost_s += elapsed
        if self.lost_hold_s > 0.0 and self.lost_s <= self.lost_hold_s:
            fade = 1.0 - self.lost_s / self.lost_hold_s
            return (self.hold[0], self.hold[1] * fade)
        self.hold = (self.hold[0], 0.0)
        return self.hold

    def command(self, debug, confidence, dt):
        measurement = self.read_detection(debug, confidence, dt)
        if measurement is None:
            return self.lost_command(dt)
        err, angle = measurement
        dt = clamp(dt, 0.01, 0.2)
        # The gate reads the SMOOTHED curve_px, and it has a dead band. Neither is
        # decoration: on a straight the one-frame value jitters past any threshold a
        # real curve (9..14 px) also reaches, which is why curve_mode as a one-frame
        # test came out anti-correlated with curvature. The jitter has no mean and a
        # curve does, so the average is the signal - and under the dead band the trim
        # is exactly zero instead of the quarter it took at 3 px.
        try:
            source = debug.get("curve_px_smooth")
            if source is None:
                source = debug.get("curve_px", 0.0)
            gate = (abs(float(source)) - self.bias_dead_px) / (
                self.bias_gate_px - self.bias_dead_px)
        except (TypeError, ValueError):
            gate = 0.0
        # Two ends and the fade between them: the trim is bias_straight_cm on a straight
        # and bias_cm once the curve is full. A dead band on the smoothed reading is
        # what makes the straight end an exact number rather than wherever the ramp
        # happened to be sitting.
        blend = clamp(gate, 0.0, 1.0) if math.isfinite(gate) else 0.0
        err += self.bias_straight_cm + (self.bias_cm - self.bias_straight_cm) * blend
        curve = bool(debug.get("curve_mode", False))
        # Scale after the bias, not before: on a curve the bias is most of the error
        # (bias 5 cm against a fused err near zero), so amplifying the raw reading
        # alone would be a no-op exactly where the correction is needed. Scaling here
        # takes P, I and D along together, which is what a loop gain should do.
        # 2026-10-02：原来只认 single_line，条件太窄。实车在弯道末跑出去那几帧
        # **两条边都看到了**（single_line=False），只是偏出去超过对称容差所以
        # bottom_lock_valid=False —— 这个增益一次都没生效过。
        # 两个量在不同带上量（single_line 来自左右 ROI，bottom_lock_valid 来自
        # 底部那条 10 行的带），谁都不蕴含谁，所以是两个条件取或。
        # 默认 1.0 时这一改不改行为。
        # 2026-10-02：原来只认 single_line，条件太窄。实车在弯道末跑出去那几帧
        # **两条边都看到了**（single_line=False），只是偏出去超过对称容差所以
        # bottom_lock_valid=False —— 这个增益一次都没生效过。
        # 两个量在不同带上量（single_line 来自左右 ROI，bottom_lock_valid 来自
        # 底部那条 10 行的带），谁都不蕴含谁，所以是两个条件取或。
        # 默认 1.0 时这一改不改行为。
        if curve and (debug.get("single_line")
                      or not debug.get("bottom_lock_valid", True)):
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
        # capped value instead of ramping toward 0.5 and being cut later. Equal to
        # max_wz unless a caller asked for an asymmetric limit, in which case only the
        # negative side moves.
        if wz < -self.max_wz_right:
            wz = -self.max_wz_right
        # Applied to the output, after the caps: the band limits how hard the loop may
        # steer on a reading this small, it does not change where the loop settles.
        # Scaling after the PID (rather than dead-banding err on the way in) leaves the
        # integral integrating the real error, and inside the band that error is small
        # by definition, so there is nothing to wind up on.
        if self.center_dead_cm > 0.0:
            wz *= min(1.0, abs(err) / self.center_dead_cm)
        self.lost_s = 0.0
        self.hold = (self.vx, wz)
        return self.hold


class ConnectorClient:
    """Use the legacy connector client's wire format and address."""

    def __init__(self, host="127.0.0.1", port=5006):
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, vx, wz, qr=-1, *, event_id=0, event_action=-1, hold_upright=False,
                card_tilt=False, command_mode=None):
        if command_mode is not None and command_mode not in ("held", "continuous"):
            raise ValueError("invalid command mode")
        if not math.isfinite(vx) or not math.isfinite(wz):
            raise ValueError("velocity must be finite")
        message = {"vx": float(vx), "vy": 0.0, "wz": float(wz), "qr": int(qr)}
        if command_mode is not None:
            message["command_mode"] = command_mode
        if event_id:
            if not 0 < int(event_id) <= 0xFFFFFFFF or int(event_action) not in (1, 2, 3, 4, 5, 6):
                raise ValueError("invalid shape event")
            message["event_id"] = int(event_id)
            message["event_action"] = int(event_action)
        # Only sent when true, like event_id: an absent field means "don't", so a
        # connector older than this one keeps behaving exactly as before.
        if hold_upright:
            message["hold_upright"] = True
        if card_tilt:
            message["card_tilt"] = True
        self.socket.sendto(json.dumps(message, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8"), self.address)

    def close(self):
        try:
            self.publish(0.0, 0.0)
        finally:
            self.socket.close()
