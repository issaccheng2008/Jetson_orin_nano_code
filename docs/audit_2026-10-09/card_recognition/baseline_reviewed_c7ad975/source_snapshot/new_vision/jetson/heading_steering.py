"""Geometry-based steering; experiment 1 removes normal minimum command hold.

Policy wz is a command, not a measured angular velocity. Prediction only brakes
an ongoing correction using observed visual trends; it never assumes wz*T is
the physical rotation. Explicit stops and loss of steering may interrupt the
normal hold. Legacy loss clears yaw; configured recovery retains a recent turn
briefly, then searches left while keeping forward speed. External stops clear history.
When only one boundary is still in view, its direction supplies the heading —
the walk follows the single line instead of standing still.
"""
from __future__ import annotations

import math
from statistics import median

COMMAND_HOLD_S = 0.0  # Experiment 1: update on each accepted visual decision.
TREND_WINDOW_S = 0.5  # Observation history is independent of command timing.
PREDICTION_HORIZON_S = 0.5  # Preserve the baseline visual braking forecast.
LEFT_OFFSET_RELEASE_CM = 4.0  # Positive near offset: robot is left of the lane.
LEFT_OFFSET_CONFIRM_FRAMES = 2


class HeadingSteeringController:
    def __init__(self, inner, lookahead_cm=50.0, right_tolerance_deg=12.0,
                 left_tolerance_deg=4.0, full_scale_deg=20.0, max_step=0.5,
                 allow_right=True, corridor_cm=8.0,
                 left_levels=(0.37, 0.43, 0.5), right_levels=(0.3, 0.5),
                 straight_wz=0.0, min_hold_s=COMMAND_HOLD_S, filter_config=None,
                 recovery_config=None, segment_fallback=False,
                 position_gain=1.0, position_dead_cm=2.0,
                 position_lookahead_cm=50.0, position_max_deg=12.0,
                 position_recovery_cm=8.0, position_recovery_full_scale_cm=12.0,
                 position_confirm_frames=2):
        from steering_recovery import TurnHistory
        self._turn_history = (TurnHistory(recovery_config, straight_wz*inner.yaw_sign)
                              if recovery_config is not None else None)
        self.segment_fallback = segment_fallback
        position_values = (position_gain, position_dead_cm, position_lookahead_cm,
                           position_max_deg, position_recovery_cm,
                           position_recovery_full_scale_cm, position_confirm_frames)
        if (not all(math.isfinite(v) for v in position_values)
                or min(position_gain, position_dead_cm, position_max_deg, position_recovery_cm) < 0
                or position_lookahead_cm <= 0 or position_recovery_full_scale_cm <= 0
                or position_confirm_frames < 1 or int(position_confirm_frames) != position_confirm_frames):
            raise ValueError('invalid near position settings')
        self.position_gain = position_gain
        self.position_dead_cm = position_dead_cm
        self.position_lookahead_cm = position_lookahead_cm
        self.position_max_deg = position_max_deg
        self.position_recovery_cm = position_recovery_cm
        self.position_recovery_full_scale_cm = position_recovery_full_scale_cm
        self.position_confirm_frames = int(position_confirm_frames)
        self._position_side = self._position_frames = 0
        values = (lookahead_cm, right_tolerance_deg, left_tolerance_deg,
                  full_scale_deg, max_step, corridor_cm, straight_wz, min_hold_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("heading settings must be finite")
        if lookahead_cm <= 0 or full_scale_deg <= 0 or corridor_cm <= 0:
            raise ValueError("lookahead, full scale and corridor must be positive")
        if min_hold_s < 0:
            raise ValueError("minimum command hold must be nonnegative")
        if not 0 <= left_tolerance_deg < 90 or not 0 <= right_tolerance_deg < 90:
            raise ValueError("angular tolerances must be in [0, 90)")
        if not 0 < max_step <= inner.max_wz:
            raise ValueError("need 0 < max step <= model wz limit")
        left_levels = tuple(float(v) for v in left_levels)
        right_levels = tuple(float(v) for v in right_levels)
        for name, ladder in (("left", left_levels), ("right", right_levels)):
            if (not ladder or not all(math.isfinite(v) and v > 0 for v in ladder)
                    or any(a >= b for a, b in zip(ladder, ladder[1:]))):
                raise ValueError(f"heading {name} wz levels must be positive and increasing")
        if not 0 <= straight_wz < min(left_levels[0], right_levels[0]):
            raise ValueError("straight wz must be in [0, the smallest turn level)")
        self.inner = inner
        self.lookahead_cm = float(lookahead_cm)
        self.right_tolerance_deg = float(right_tolerance_deg)
        self.left_tolerance_deg = float(left_tolerance_deg)
        self.full_scale_deg = float(full_scale_deg)
        self.max_step = float(max_step)
        self.allow_right = bool(allow_right)
        self.corridor_cm = float(corridor_cm)
        self.straight_wz = float(straight_wz)
        self.min_hold_s = float(min_hold_s)
        # These are geometric left/right magnitudes; yaw_sign maps them onto
        # the robot wire convention. Caps remove levels, never create new ones.
        self.left_levels = tuple(v for v in left_levels if v <= self._cap(+1))
        self.right_levels = tuple(v for v in right_levels if v <= self._cap(-1))
        if not self.left_levels or (self.allow_right and not self.right_levels):
            raise ValueError("wz caps must leave at least one level in each enabled direction")
        self.filter_config = filter_config
        self._observation_filter = None
        if filter_config is not None:
            from steering_filter import GeometryFilter, validate_ladder
            if filter_config.algorithm != 'none':
                self._observation_filter = GeometryFilter(filter_config)
            validate_ladder(self.left_levels, self._cap(+1), self.full_scale_deg, filter_config.hysteresis_deg)
            validate_ladder(self.right_levels, self._cap(-1), self.full_scale_deg, filter_config.hysteresis_deg)
        self._clock = 0.0
        self._started = None
        self._samples = []
        self._command = (0.0, 0.0)
        self._loss_s = 0.0
        self._left_offset_frames = 0
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
        return (max(0.0, self.min_hold_s - (self._clock - self._started))
                if self._started is not None else 0.0)
    @property
    def gap_left(self): return 0.0

    def reset(self, clear_hold=False):
        self._position_side = self._position_frames = 0
        if self._turn_history is not None:
            self._turn_history.reset()
        self.inner.reset(clear_hold=True)
        self._started = None
        self._samples.clear()
        self._command = (0.0, 0.0)
        self._loss_s = 0.0
        self._left_offset_frames = 0
        self.diagnostics = {}
        if self._observation_filter is not None:
            self._observation_filter.reset()

    def drop_held_command(self):
        self.reset(clear_hold=True)

    def _stop(self, reason):
        self._position_side = self._position_frames = 0
        if self._turn_history is not None:
            self._turn_history.reset()
        if self._observation_filter is not None:
            self._observation_filter.reset()
        self._left_offset_frames = 0
        self._started = None
        self._samples.clear()
        self._command = self.inner.hold = (0.0, 0.0)
        self.inner.last_steer = 0.0
        self.diagnostics.update(steering_reason=reason, command_hold_remaining_s=0.0,
                                steering_applied_wz=0.0)
        return self._command

    def _lose_geometry(self):
        # Geometry loss cannot start walking or undo an external stop. Keep only
        # the speed actually applied before loss; never substitute configured vx.
        speed = self._command[0]
        self._stop("geometry_lost_yaw_zero")
        self._command = self.inner.hold = (speed, 0.0)
        return self._command

    def _geometry(self, debug):
        """Paired geometry first; the single-edge direction is the loss fallback."""
        geometry = self._paired_geometry(debug)
        if geometry is None:
            geometry = self._single_edge_geometry(debug)
        return geometry

    def _paired_geometry(self, debug):
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
        return self._bear_from(near, z, angle, source)

    def _single_edge_geometry(self, debug):
        """只剩一条边界时，沿着这条线的方向走（检测器 single_edge_* 字段）。

        P1 不让单边界供 heading_control / curve —— 那条契约不变；这里是丢线
        兜底：配对拟合不可用但检测器还看得见一条边（近期配对宽度支撑），就沿
        着它的方向继续走，而不是站住。位置/方向都用同一带的数据。
        """
        if not debug.get("single_edge_valid", False):
            return None
        try:
            angle = float(debug["single_edge_heading_deg"])
            near = float(debug.get("single_edge_near_cm", 0.0))
            z = float(debug.get("single_edge_z_cm", 0.0))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return self._bear_from(near, z, angle, "single_edge")

    def _bear_from(self, near, z, angle, source):
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

    def _target_distance_cm(self):
        """Target depth for spatial gates; subclasses may use measured support."""
        return self.lookahead_cm

    def _map_angle(self, demand):
        # Angular tolerance may be widened, but never widen the spatial corridor.
        corridor_angle = math.degrees(math.atan2(self.corridor_cm, self._target_distance_cm()))
        positive_gate = min(self.right_tolerance_deg, corridor_angle)
        negative_gate = min(self.left_tolerance_deg, corridor_angle)
        if demand >= positive_gate and demand > 0:
            target = self._cap(+1) * min(1., max(0., demand-positive_gate)/self.full_scale_deg)
            return min(self.left_levels, key=lambda level: (abs(level-target), level))
        if self.allow_right and demand <= -negative_gate and demand < 0:
            target = self._cap(-1) * min(1., max(0., -demand-negative_gate)/self.full_scale_deg)
            level = min(self.right_levels, key=lambda level: (abs(level-target), level))
            return -level
        return self.straight_wz

    def _filtered_level(self, demand):
        from steering_filter import select_level
        cfg = self.filter_config
        current = self._command[1]*self.yaw_sign
        direction = 1 if demand > 0 else -1
        corridor_angle = math.degrees(math.atan2(self.corridor_cm, self._target_distance_cm()))
        gate = min(self.right_tolerance_deg if direction > 0 else self.left_tolerance_deg, corridor_angle)
        turning = current != self.straight_wz and current*direction > 0
        threshold = max(gate, cfg.exit_deg if turning else cfg.enter_deg)
        if demand == 0 or abs(demand) <= threshold or (direction < 0 and not self.allow_right):
            return self.straight_wz
        levels = self.left_levels if direction > 0 else self.right_levels
        return direction*select_level(abs(demand), abs(current) if turning else 0.,
            levels=levels, cap=self._cap(direction), full_scale=self.full_scale_deg,
            gate=gate, width=cfg.hysteresis_deg)

    def _decision(self, demand, near, heading):
        candidate = (self._filtered_level(demand) if self.filter_config is not None
                     else self._map_angle(demand))
        # Already pointing toward the line: do not keep turning just to erase
        # residual position error. Straight walking lets that error converge.
        corridor_angle = math.degrees(math.atan2(self.corridor_cm, self._target_distance_cm()))
        projected_inside = abs(demand) <= corridor_angle
        if self.position_gain == 0 and near < 0 and heading < 0 and demand > 0 and projected_inside:
            return self.straight_wz, "returning_from_right"
        if self.position_gain == 0 and near > 0 and heading > 0 and demand < 0 and projected_inside:
            return self.straight_wz, "returning_from_left"
        # Near position protection is independent of the relaxed angular gate.
        if demand >= corridor_angle or (near <= -self.corridor_cm and heading >= 0):
            # The smallest left level follows a distant bend and is also the
            # floor for near right-boundary recovery.
            return max(candidate, self.left_levels[0]), "right_corridor"
        if self.allow_right and (demand <= -corridor_angle
                                 or (near >= self.corridor_cm and heading <= 0)):
            return min(candidate, -self.right_levels[0]), "left_corridor"
        return candidate, "target_bearing"

    def _position_correction(self, near):
        error = math.copysign(max(0., abs(near)-self.position_dead_cm), near)
        correction = -self.position_gain*math.degrees(math.atan2(error, self.position_lookahead_cm))
        return max(-self.position_max_deg, min(self.position_max_deg, correction))

    def _confirm_position(self, near):
        # Independent of trend/angle history, which may be reset or heavily filtered.
        side = (1 if near > 0 else -1) if (self.position_recovery_cm > 0
                and abs(near) >= self.position_recovery_cm) else 0
        self._position_frames = (min(self.position_confirm_frames, self._position_frames+1)
                                 if side and side == self._position_side else int(bool(side)))
        self._position_side = side
        return bool(side and self._position_frames >= self.position_confirm_frames)

    def _position_recovery_level(self, near):
        direction = -self._position_side
        if direction < 0 and not self.allow_right:
            return self.straight_wz
        levels = self.left_levels if direction > 0 else self.right_levels
        severity = max(0., abs(near)-self.position_recovery_cm)/self.position_recovery_full_scale_cm
        target = self._cap(direction)*min(1., severity)
        return direction*min(levels, key=lambda level: (abs(level-target), level))

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
            self._position_side = self._position_frames = 0
            self._left_offset_frames = 0
            self._loss_s += dt
            self._samples.clear()
            if self._turn_history is not None:
                pair, reason = self._turn_history.command(
                    self._clock, self._loss_s, self._command,
                    walking_vx=self.inner.vx, left_wz=min(.3, self._cap(+1))*self.yaw_sign)
                if pair == (0., 0.):
                    self._stop(reason)
                else:
                    self._command = self.inner.hold = pair
                    self.inner.last_steer = pair[1]
                self.diagnostics.update(steering_reason=reason, steering_applied_wz=pair[1],
                    steering_loss_age_s=self._loss_s, steering_history_applied=reason=='loss_history_turn')
                return pair
            # Brief missing frames keep the exact pair, never a per-frame fade.
            # Stale geometry / the loss deadline discard yaw, not forward speed.
            if debug.get("measurement_stale", False) or self._loss_s >= self.inner.lost_hold_s:
                return self._lose_geometry()
            self.diagnostics.update(steering_reason="brief_loss_hold",
                command_hold_remaining_s=self.turn_left, steering_applied_wz=self._command[1])
            return self._command
        self._loss_s = 0.0
        self.inner.lost_s = 0.0
        near, z, angle, demand, source = geometry
        position_recovery = self._confirm_position(near)
        filtered_geometry = (self._observation_filter.apply(geometry, self._clock)
                             if self._observation_filter is not None else None)
        # Confirm position independently of the trend samples, which are cleared
        # on command changes. A single shaken frame cannot veto a left command.
        offset_near = near  # Two-frame boundary safeguard must not lag behind position EMA.
        self._left_offset_frames = (min(LEFT_OFFSET_CONFIRM_FRAMES,
                                       self._left_offset_frames + 1)
                                    if offset_near >= LEFT_OFFSET_RELEASE_CM else 0)
        left_offset_confirmed = self._left_offset_frames >= LEFT_OFFSET_CONFIRM_FRAMES
        self._samples.append((self._clock, demand, angle, near))
        self._samples = [s for s in self._samples if self._clock-s[0] <= TREND_WINDOW_S+1e-9]
        filtered = median(s[1] for s in self._samples[-3:])
        filtered_near = median(s[3] for s in self._samples[-3:])
        filtered_heading = median(s[2] for s in self._samples[-3:])
        if filtered_geometry is not None:
            filtered_near, _, filtered_heading, filtered, _ = filtered_geometry
        if self.filter_config is not None and self.filter_config.algorithm == 'robust':
            # Forecast and spatial confirmation must not bypass robust smoothing.
            self._samples[-1] = (self._clock, filtered, filtered_heading, filtered_near)
        position_correction = self._position_correction(filtered_near)
        combined = filtered + position_correction
        if self.filter_config is not None:
            self.diagnostics.update(steering_filter_mode='active',
                steering_filter_algorithm=self.filter_config.algorithm,
                steering_filter_heading_deg=filtered_heading,
                steering_filter_demand_deg=filtered, steering_filter_near_cm=filtered_near,
                steering_hysteresis_deg=self.filter_config.hysteresis_deg)
        rate, angle_rate = self._trend(1), self._trend(2)
        predicted = filtered + PREDICTION_HORIZON_S*rate if rate is not None else filtered
        self.diagnostics.update(steering_heading_source=source, steering_near_cm=near,
            steering_near_z_cm=z, steering_heading_deg=angle,
            steering_filtered_heading_deg=filtered_heading,
            steering_demand_deg=demand, steering_filtered_demand_deg=filtered,
            steering_predicted_demand_deg=predicted, steering_demand_rate_deg_s=rate,
            steering_predicted_heading_deg=(angle+PREDICTION_HORIZON_S*angle_rate if angle_rate is not None else angle),
            steering_prediction_valid=rate is not None, steering_braked=False)
        self.diagnostics.update(steering_position_correction_deg=position_correction,
            steering_combined_demand_deg=combined, steering_position_recovery=position_recovery,
            steering_position_confirm_frames=self._position_frames,
            steering_position_recovery_cm=self.position_recovery_cm)
        self.diagnostics.update(steering_left_offset_confirmed=left_offset_confirmed,
                                steering_left_offset_release_cm=LEFT_OFFSET_RELEASE_CM)
        self.inner.last_err_eff = combined  # Units explicitly renamed in entry-point logging.
        if not position_recovery and self._started is not None and self._clock-self._started < self.min_hold_s:
            if self._turn_history is not None:
                self._turn_history.observe(self._clock, self._command, dt)
            self.diagnostics.update(steering_reason="minimum_hold",
                command_hold_remaining_s=self.turn_left, steering_applied_wz=self._command[1])
            return self._command
        candidate, decision = self._decision(combined, filtered_near, filtered_heading)
        # Subject to this branch's command timing, confirmed left position takes
        # priority over a distant leftward target. Only veto geometric left yaw;
        # forward speed and the original right-recovery decision stay intact.
        if self.position_gain == 0 and self.position_recovery_cm == 0 and left_offset_confirmed and candidate > 0:
            candidate, decision = self.straight_wz, "left_offset_release"
        self.diagnostics["steering_decision"] = decision
        self.diagnostics["steering_corridor_cm"] = self.corridor_cm
        current = self._command[1] * self.yaw_sign
        # Forecast assumes the CURRENT action continues. Use it only to reduce
        # that same correction, never to predict a new action's unmeasured effect.
        if rate is not None and current*candidate > 0 and current*rate < 0:
            forecast = self._map_angle(predicted + position_correction)
            reduced = min(abs(current), abs(candidate), abs(forecast)) if candidate*forecast > 0 else 0.0
            if decision in ("right_corridor", "left_corridor"):
                corridor_angle = math.degrees(math.atan2(self.corridor_cm, self._target_distance_cm()))
                near_boundary = (filtered_near <= -self.corridor_cm if candidate > 0
                                 else filtered_near >= self.corridor_cm)
                # A re-entry brake may fade to the straight level, but a
                # near-boundary recovery may not fade below the smallest turn
                # level of its side.
                if abs(predicted) <= corridor_angle:
                    reduced = 0.0
                elif near_boundary:
                    floor = self.left_levels[0] if candidate > 0 else self.right_levels[0]
                    if 0 < reduced < floor:
                        reduced = min(abs(candidate), max(floor, abs(forecast)))
            previous_candidate = candidate
            candidate = math.copysign(reduced, candidate) if reduced else self.straight_wz
            self.diagnostics["steering_braked"] = (abs(candidate) < abs(previous_candidate)
                                                   and abs(candidate) <= abs(current))
        # Confirmed near position owns the final choice. Far-target braking,
        # hysteresis and normal hold must not turn this recovery into straight walking.
        if position_recovery:
            recovery_level = self._position_recovery_level(near)
            # Keep a stronger inward target correction; only outward/straight
            # candidates are replaced by the independent position demand.
            if self._position_side > 0 and not self.allow_right:
                candidate = self.straight_wz
            else:
                candidate = (math.copysign(max(abs(candidate), abs(recovery_level)), recovery_level)
                             if candidate*recovery_level > 0 else recovery_level)
            decision = 'position_recovery_left' if self._position_side < 0 else 'position_recovery_right'
            self.diagnostics.update(steering_decision=decision, steering_braked=False)
        selected = (self.inner.vx, candidate*self.yaw_sign)
        changed = selected != self._command or self._started is None
        if changed:
            self._started = self._clock
            self._samples = [self._samples[-1]]  # No response estimate across action changes.
        self._command = self.inner.hold = selected
        if self._turn_history is not None:
            if (candidate == self.straight_wz and
                    (decision in ('left_offset_release','returning_from_left','returning_from_right',
                                  'position_recovery_right')
                     or self.diagnostics.get('steering_braked', False))):
                self._turn_history.reset()  # Never revive a turn vetoed by latest valid geometry.
            self._turn_history.observe(self._clock, selected, dt)
        self.inner.last_steer = selected[1]
        self.diagnostics.update(steering_reason="new_block" if changed else "continue_block",
            command_hold_remaining_s=self.turn_left, steering_applied_wz=selected[1])
        return selected
