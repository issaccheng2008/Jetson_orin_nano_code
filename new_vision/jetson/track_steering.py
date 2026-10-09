"""Opt-in steering toward an observed, imperfect lane path.

The old heading controller remains the fallback and the comparison shadow.
Commanded wz is never integrated as body yaw. Normal walking commands retain
the policy's fixed half-second hold.
"""
from __future__ import annotations

from copy import deepcopy
import math

from heading_steering import HeadingSteeringController


class TrackSteeringController(HeadingSteeringController):
    def __init__(self, inner, confirm_frames=3, **options):
        if not isinstance(confirm_frames, int) or confirm_frames < 1:
            raise ValueError("track confirm frames must be at least one")
        super().__init__(inner, **options)
        self.shadow = HeadingSteeringController(deepcopy(inner), **options)
        self.confirm_frames = confirm_frames
        self._target_depth = self.lookahead_cm
        self._phase = "uncertain"
        self._confirmed = 0
        self._previous = None
        self._track_active = False
        self._phase_candidate = None
        self._phase_count = 0
        self._latched_phase = "uncertain"

    def reset(self, clear_hold=False):
        super().reset(clear_hold=clear_hold)
        self.shadow.reset(clear_hold=clear_hold)
        self._target_depth = self.lookahead_cm
        self._phase = "uncertain"
        self._confirmed = 0
        self._previous = None
        self._track_active = False
        self._phase_candidate = None
        self._phase_count = 0
        self._latched_phase = "uncertain"

    def _target_distance_cm(self):
        return self._target_depth

    def _allow_geometry_without_legacy_reading(self, debug):
        return (debug.get("track_valid") is True
                and float(debug.get("track_confidence", 0.)) >= .4)

    def _geometry(self, debug):
        self._track_active = False
        self._phase = "uncertain"
        self._target_depth = self.lookahead_cm
        baseline = super()._geometry(debug)
        if not debug.get("track_valid", False):
            self._confirmed, self._previous = 0, None
            return baseline
        try:
            near = float(debug["track_near_cm"])
            z = float(debug["track_near_z_cm"])
            angle = float(debug["track_heading_deg"])
            bearing = float(debug["track_target_bearing_deg"])
            depth = float(debug["track_target_z_cm"])
            x = float(debug["track_target_x_cm"])
            confidence = float(debug["track_confidence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return baseline
        if (not all(math.isfinite(v) for v in (near, z, angle, bearing, depth, x, confidence))
                or confidence < .4 or depth <= z+8. or depth > self.lookahead_cm+1.
                or abs(angle) >= 80. or abs(bearing) >= 80.):
            self._confirmed, self._previous = 0, None
            return baseline
        previous = self._previous
        stable = (previous is not None and
                  abs(depth-previous[0]) <= 4. and abs(x-previous[1]) <= 6.
                  and abs(bearing-previous[2]) <= 10.)
        self._confirmed = min(self.confirm_frames, self._confirmed+1) if stable else 1
        self._previous = depth, x, bearing
        incoming_phase = str(debug.get("track_phase", "uncertain"))
        if incoming_phase == self._phase_candidate:
            self._phase_count = min(self.confirm_frames, self._phase_count+1)
        else:
            self._phase_candidate, self._phase_count = incoming_phase, 1
        if self._phase_count >= self.confirm_frames:
            self._latched_phase = incoming_phase
        if self._confirmed < self.confirm_frames:
            return baseline
        self._track_active = True
        self._target_depth = depth
        self._phase = self._latched_phase
        return near, z, angle, bearing, "observed_track"

    def _veto_left_for_offset(self, confirmed, candidate, near):
        if self._track_active and self._phase == "left_bend":
            # Four centimetres left of centre is not a reason to cancel the
            # bend's nominal turn while still inside the measured corridor.
            return confirmed and candidate > 0 and near >= self.corridor_cm
        return super()._veto_left_for_offset(confirmed, candidate, near)

    def _after_trend_brake(self, candidate, near):
        if (self._track_active and self._phase == "left_bend"
                and near < self.corridor_cm and candidate >= 0.):
            return max(candidate, self.left_levels[0])
        return candidate

    def command(self, debug, confidence, dt):
        if isinstance(dt, (int, float)) and dt > .25:
            self._confirmed, self._previous = 0, None
            self._phase_candidate, self._phase_count = None, 0
        actual = super().command(debug, confidence, dt)
        shadow = self.shadow.command(debug, confidence, dt)
        if (self.diagnostics.get("steering_reason") == "invalid_clock"
                or (not debug.get("track_valid") and self.diagnostics.get("steering_reason") in (
                    "brief_loss_hold", "geometry_lost_yaw_zero"))):
            self._confirmed, self._previous = 0, None
            self._track_active = False
        self.diagnostics.update(track_control_active=self._track_active,
                                track_confirm_frames=self._confirmed,
                                track_control_phase=self._phase,
                                track_shadow_wz=shadow[1],
                                track_shadow_reason=self.shadow.diagnostics.get("steering_reason"),
                                track_applied_wz=actual[1])
        return actual
