"""Geometry-based steering with asymmetric command levels and a fixed 0.5 s hold.

Policy wz is a command, not a measured angular velocity. Prediction only brakes
an ongoing correction using observed visual trends; it never assumes wz*T is
the physical rotation. Explicit stops/loss may interrupt the normal hold.
"""
from __future__ import annotations

import math
from statistics import median

COMMAND_HOLD_S = 0.5  # Training contract: intentionally not a CLI parameter.


class HeadingSteeringController:
    def __init__(self, inner, lookahead_cm=50.0, right_tolerance_deg=12.0,
                 left_tolerance_deg=4.0, full_scale_deg=20.0, max_step=0.5,
                 allow_right=True, corridor_cm=8.0):
        values = (lookahead_cm, right_tolerance_deg, left_tolerance_deg,
                  full_scale_deg, max_step, corridor_cm)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("heading settings must be finite")
        if lookahead_cm <= 0 or full_scale_deg <= 0 or corridor_cm <= 0:
            raise ValueError("lookahead, full scale and corridor must be positive")
        if not 0 <= left_tolerance_deg < 90 or not 0 <= right_tolerance_deg < 90:
            raise ValueError("angular tolerances must be in [0, 90)")
        if not 0 < max_step <= inner.max_wz:
            raise ValueError("need 0 < max step <= model wz limit")
        self.inner = inner
        self.lookahead_cm = float(lookahead_cm)
        self.right_tolerance_deg = float(right_tolerance_deg)
        self.left_tolerance_deg = float(left_tolerance_deg)
        self.full_scale_deg = float(full_scale_deg)
        self.max_step = float(max_step)
        self.allow_right = bool(allow_right)
        self.corridor_cm = float(corridor_cm)
        # These are geometric left/right magnitudes; yaw_sign maps them onto
        # the robot wire convention. Caps remove levels, never create new ones.
        self.left_levels = tuple(v for v in (0.4, 0.5) if v <= self._cap(+1))
        self.right_levels = tuple(v for v in (0.1, 0.4, 0.5) if v <= self._cap(-1))
        if not self.left_levels or (self.allow_right and 0.4 not in self.right_levels):
            raise ValueError("wz caps must permit a 0.4 correction in each enabled direction")
        self._clock = 0.0
        self._started = None
        self._samples = []
        self._command = (0.0, 0.0)
        self._loss_s = 0.0
        self.diagnostics = {}

    @property
    def vx(self): return self.inner.vx
    @property
    def yaw_sign(self): return self.inner.yaw_sign
    @property
    def hold(self): return self._command
    @property
    def lost_s(self): return self._loss_s
    @property
    def last_steer(self): return self.inner.last_steer
    @property
    def last_err_eff(self): return self.inner.last_err_eff
    @property
    def rejected_lateral(self): return self.inner.rejected_lateral
    @property
    def turn_left(self):
        return (max(0.0, COMMAND_HOLD_S - (self._clock - self._started))
                if self._started is not None else 0.0)
    @property
    def gap_left(self): return 0.0

    def reset(self, clear_hold=False):
        self.inner.reset(clear_hold=True)
        self._started = None
        self._samples.clear()
        self._command = (0.0, 0.0)
        self._loss_s = 0.0
        self.diagnostics = {}

    def drop_held_command(self):
        self.reset(clear_hold=True)

    def _stop(self, reason):
        self._started = None
        self._samples.clear()
        self._command = self.inner.hold = (0.0, 0.0)
        self.inner.last_steer = 0.0
        self.diagnostics.update(steering_reason=reason, command_hold_remaining_s=0.0,
                                steering_applied_wz=0.0)
        return self._command

    def _geometry(self, debug):
        # New ground fit is preferred. Old P1 recordings remain replayable, with
        # an explicit source tag; an invalid new fit never falls back to stale data.
        if "heading_control_valid" in debug:
            if not debug["heading_control_valid"]:
                return None
            angle = float(debug["heading_control_deg"])
            source = "ground_x_z"
        else:
            if not debug.get("heading_valid", True):
                return None
            angle = float(debug["angle_err_deg"])
            source = "legacy_pixel_heading"
        near = float(debug.get("near_error_cm", debug["base_err_cm"]))
        z = float(debug.get("near_z_cm", 0.0))
        if not all(math.isfinite(v) for v in (near, z, angle)):
            return None
        if z < 0 or z >= self.lookahead_cm or abs(angle) >= 90:
            return None
        rad = math.radians(angle)
        # x(Z)=near-(Z-z_near)*tan(angle). atan2 in the scaled coordinates
        # avoids tan exploding close to a sideways line. Positive demand = left.
        target_x_scaled = near * math.cos(rad) - (self.lookahead_cm-z)*math.sin(rad)
        demand = -math.degrees(math.atan2(target_x_scaled, self.lookahead_cm*math.cos(rad)))
        return near, z, angle, demand, source

    def _cap(self, direction):
        return min(self.max_step, self.inner.max_wz_right) if direction*self.yaw_sign < 0 else self.max_step

    def _map_angle(self, demand):
        # Angular tolerance may be widened, but never widen the spatial corridor.
        corridor_angle = math.degrees(math.atan2(self.corridor_cm, self.lookahead_cm))
        positive_gate = min(self.right_tolerance_deg, corridor_angle)
        negative_gate = min(self.left_tolerance_deg, corridor_angle)
        if demand >= positive_gate and demand > 0:
            target = self._cap(+1) * min(1., max(0., demand-positive_gate)/self.full_scale_deg)
            return min(self.left_levels, key=lambda level: (abs(level-target), level))
        if self.allow_right and demand <= -negative_gate and demand < 0:
            target = self._cap(-1) * min(1., max(0., -demand-negative_gate)/self.full_scale_deg)
            level = min(self.right_levels, key=lambda level: (abs(level-target), level))
            if demand <= -corridor_angle:
                level = max(0.4, level)  # 0.1 is for small corrections, not corridor recovery.
            return -level
        return 0.

    def _decision(self, demand, near, heading):
        candidate = self._map_angle(demand)
        # Already pointing toward the line: do not keep turning just to erase
        # residual position error. Straight walking lets that error converge.
        if near < 0 and heading < 0 and demand > 0:
            return 0., "returning_from_right"
        if near > 0 and heading > 0 and demand < 0:
            return 0., "returning_from_left"
        # Near position protection is independent of the relaxed angular gate.
        if near <= -self.corridor_cm and heading >= 0:
            return max(candidate, self.left_levels[0]), "right_corridor"
        if self.allow_right and near >= self.corridor_cm and heading <= 0:
            return min(candidate, -0.4), "left_corridor"
        return candidate, "target_bearing"

    def _trend(self, column):
        if len(self._samples) < 3 or self._samples[-1][0]-self._samples[0][0] < .3:
            return None
        # Median of well-separated pair slopes reduces the impact of a single
        # shaken frame without pretending it measures body yaw or execution gain.
        slopes = [(b[column]-a[column])/(b[0]-a[0])
                  for i,a in enumerate(self._samples) for b in self._samples[i+1:]
                  if b[0]-a[0] >= .2]
        return median(slopes) if slopes else None

    def command(self, debug, confidence, dt):
        self.diagnostics = {}
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            return self._stop("invalid_clock")
        self._clock += dt
        reading = self.inner.read_detection(debug, confidence, dt)
        try:
            geometry = self._geometry(debug) if reading is not None else None
        except (KeyError, TypeError, ValueError, OverflowError):
            geometry = None
        if geometry is None:
            self._loss_s += dt
            self._samples.clear()
            # Brief missing frames keep the exact pair, never a per-frame fade.
            # Stale geometry / the existing loss deadline are immediate stops.
            if debug.get("measurement_stale", False) or self._loss_s >= self.inner.lost_hold_s:
                return self._stop("geometry_lost")
            self.diagnostics.update(steering_reason="brief_loss_hold",
                command_hold_remaining_s=self.turn_left, steering_applied_wz=self._command[1])
            return self._command
        self._loss_s = 0.0
        self.inner.lost_s = 0.0
        near, z, angle, demand, source = geometry
        self._samples.append((self._clock, demand, angle, near))
        self._samples = [s for s in self._samples if self._clock-s[0] <= COMMAND_HOLD_S+1e-9]
        filtered = median(s[1] for s in self._samples[-3:])
        filtered_near = median(s[3] for s in self._samples[-3:])
        filtered_heading = median(s[2] for s in self._samples[-3:])
        rate, angle_rate = self._trend(1), self._trend(2)
        predicted = filtered + COMMAND_HOLD_S*rate if rate is not None else filtered
        self.diagnostics.update(steering_heading_source=source, steering_near_cm=near,
            steering_near_z_cm=z, steering_heading_deg=angle,
            steering_demand_deg=demand, steering_filtered_demand_deg=filtered,
            steering_predicted_demand_deg=predicted, steering_demand_rate_deg_s=rate,
            steering_predicted_heading_deg=(angle+COMMAND_HOLD_S*angle_rate if angle_rate is not None else angle),
            steering_prediction_valid=rate is not None, steering_braked=False)
        self.inner.last_err_eff = filtered  # Units explicitly renamed in entry-point logging.
        if self._started is not None and self._clock-self._started < COMMAND_HOLD_S:
            self.diagnostics.update(steering_reason="minimum_hold",
                command_hold_remaining_s=self.turn_left, steering_applied_wz=self._command[1])
            return self._command
        candidate, decision = self._decision(filtered, filtered_near, filtered_heading)
        self.diagnostics["steering_decision"] = decision
        self.diagnostics["steering_corridor_cm"] = self.corridor_cm
        current = self._command[1] * self.yaw_sign
        # Forecast assumes the CURRENT action continues. Use it only to reduce
        # that same correction, never to predict a new action's unmeasured effect.
        if rate is not None and current*candidate > 0 and current*rate < 0:
            forecast = self._map_angle(predicted)
            reduced = min(abs(current), abs(candidate), abs(forecast)) if candidate*forecast > 0 else 0.0
            candidate = math.copysign(reduced, candidate) if reduced else 0.0
            self.diagnostics["steering_braked"] = True
        if (angle_rate is not None and current*candidate > 0
                and current*angle_rate < 0
                and current*(filtered_heading+COMMAND_HOLD_S*angle_rate) <= 0):
            candidate = 0.
            self.diagnostics["steering_braked"] = True
            self.diagnostics["steering_decision"] = "predicted_alignment"
        selected = (self.inner.vx, candidate*self.yaw_sign)
        changed = selected != self._command or self._started is None
        if changed:
            self._started = self._clock
            self._samples = [self._samples[-1]]  # No response estimate across action changes.
        self._command = self.inner.hold = selected
        self.inner.last_steer = selected[1]
        self.diagnostics.update(steering_reason="new_block" if changed else "continue_block",
            command_hold_remaining_s=self.turn_left, steering_applied_wz=selected[1])
        return selected
