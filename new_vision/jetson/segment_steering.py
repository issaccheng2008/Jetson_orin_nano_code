"""Opt-in steering toward a measured lane target, with heading fallback.

The same heading state machine owns the applied command and this branch's
timing policy. The second controller only logs comparison commands; it never
publishes, owns a socket, or influences the applied controller's state.
"""
from __future__ import annotations

from copy import deepcopy
import math

from heading_steering import HeadingSteeringController
from lane_segments import SEGMENTS_CM

CONFIRM_FRAMES = 3
MAX_CONFIRMATION_GAP_S = 0.5
MAX_JOIN_CM = 6.0
MAX_DEPTH_GAP_CM = 4.0
MAX_HEADING_STEP_DEG = 35.0
MAX_TARGET_JUMP_CM = 6.0
MAX_BEARING_JUMP_DEG = 10.0


class SegmentSteeringController(HeadingSteeringController):
    def __init__(self, inner, **options):
        super().__init__(inner, **options)
        self.shadow = HeadingSteeringController(deepcopy(inner), **options)
        self._segment_depth = self.lookahead_cm
        self._confirmation = 0
        self._previous_target = None
        self._geometry_key = None
        self._geometry_depth = None
        self._segment_diagnostics = {}

    def reset(self, clear_hold=False):
        super().reset(clear_hold=clear_hold)
        self.shadow.reset(clear_hold=clear_hold)
        self._confirmation = 0
        self._previous_target = None
        self._geometry_key = self._geometry_depth = None
        self._segment_depth = self.lookahead_cm
        self._segment_diagnostics = {}

    def _target_distance_cm(self):
        return self._segment_depth

    @staticmethod
    def _segment(debug, index):
        prefix = f'fit_seg{index}_'
        if not debug.get(prefix + 'valid', False):
            return None
        try:
            values = {name: float(debug[prefix + name]) for name in
                      ('z_cm', 'x_cm', 'heading_deg', 'z_min_cm', 'z_max_cm',
                       'normal_width_cm', 'rmse_cm', 'points')}
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(v) for v in values.values()):
            return None
        lo, hi = SEGMENTS_CM[index]
        if not (lo <= values['z_min_cm'] <= values['z_cm'] <= values['z_max_cm'] < hi
                and values['z_max_cm'] - values['z_min_cm'] >= MAX_DEPTH_GAP_CM
                and abs(values['heading_deg']) < 80.
                and 26. <= values['normal_width_cm'] <= 44.
                and 0. <= values['rmse_cm'] <= 2.5 and values['points'] >= 5):
            return None
        return values

    @staticmethod
    def _x_at(segment, z):
        return segment['x_cm'] - (z-segment['z_cm'])*math.tan(math.radians(segment['heading_deg']))

    def _joined(self, first, second):
        gap = second['z_min_cm'] - first['z_max_cm']
        join = .5*(first['z_max_cm'] + second['z_min_cm'])
        return (0. <= gap <= MAX_DEPTH_GAP_CM
                and abs(first['heading_deg'] - second['heading_deg']) <= MAX_HEADING_STEP_DEG
                and abs(self._x_at(first, join) - self._x_at(second, join)) <= MAX_JOIN_CM)

    def _measured_target(self, debug, baseline):
        if baseline is None or baseline[4] != 'ground_x_z':
            return None, 'near_geometry_unavailable'
        if not debug.get('fit_seg_valid', False) or not debug.get('fit_seg_anchored', False):
            return None, 'unanchored_or_missing_segments'
        first, middle = self._segment(debug, 0), self._segment(debug, 1)
        if first is None or middle is None or not self._joined(first, middle):
            return None, 'near_middle_not_continuous'
        near, near_z = baseline[:2]
        if not (first['z_min_cm'] <= near_z <= first['z_max_cm']
                and abs(self._x_at(first, near_z) - near) <= MAX_JOIN_CM):
            return None, 'near_anchor_disagrees'
        segments = [(0, first), (1, middle)]
        far = self._segment(debug, 2)
        if far is not None and self._joined(middle, far):
            segments.append((2, far))
        target = None
        for index, segment in segments[1:]:
            if segment['z_min_cm'] > self.lookahead_cm:
                break
            z = min(self.lookahead_cm, segment['z_max_cm'])
            target = index, z, self._x_at(segment, z)
        if target is None or target[1] <= near_z:
            return None, 'no_observed_forward_target'
        index, z, x = target
        bearing = -math.degrees(math.atan2(x, z))
        return (near, near_z, first['heading_deg'], bearing, index, z, x), 'qualified'

    def _geometry(self, debug):
        baseline = super()._geometry(debug)
        target, reason = self._measured_target(debug, baseline)
        self._segment_depth = self.lookahead_cm
        self._segment_diagnostics.update(segment_control_active=False, segment_gate_reason=reason)
        selected = baseline
        if target is None:
            self._confirmation = 0
            self._previous_target = None
        else:
            near, near_z, angle, bearing, index, z, x = target
            previous = self._previous_target
            stable = (previous is not None and index == previous[0]
                      and abs(z-previous[1]) <= MAX_DEPTH_GAP_CM
                      and abs(x-previous[2]) <= MAX_TARGET_JUMP_CM
                      and abs(bearing-previous[3]) <= MAX_BEARING_JUMP_DEG)
            self._confirmation = min(CONFIRM_FRAMES, self._confirmation+1) if stable else 1
            self._previous_target = index, z, x, bearing
            self._segment_diagnostics.update(segment_target_index=index,
                segment_target_z_cm=z, segment_target_x_cm=x, segment_target_bearing_deg=bearing)
            if self._confirmation >= CONFIRM_FRAMES:
                self._segment_depth = z
                selected = near, near_z, angle, bearing, 'measured_segments'
                self._segment_diagnostics['segment_control_active'] = True
            else:
                self._segment_diagnostics['segment_gate_reason'] = 'confirming'
        # A trend in the old extrapolated target is not a trend in the measured
        # target. Clear only trend/median history, never the applied hold timer.
        key = ((selected[4], target[4] if selected[4] == 'measured_segments' else None)
               if selected is not None else None)
        if (key != self._geometry_key or self._geometry_depth is None
                or abs(self._segment_depth-self._geometry_depth) > MAX_DEPTH_GAP_CM):
            self._samples.clear()
        self._geometry_key, self._geometry_depth = key, self._segment_depth
        return selected

    def command(self, debug, confidence, dt):
        self._segment_diagnostics = dict(segment_control_active=False,
                                         segment_gate_reason='measurement_unavailable')
        # A fresh frame at 4 Hz can arrive just over 0.25 s after the previous
        # one. Freshness is checked independently by read_detection; only a
        # longer gap breaks continuity between otherwise qualified targets.
        if isinstance(dt, (int, float)) and dt > MAX_CONFIRMATION_GAP_S:
            self._confirmation = 0
            self._previous_target = None
        actual = super().command(debug, confidence, dt)
        shadow = self.shadow.command(debug, confidence, dt)
        if self.diagnostics.get('steering_reason') in (
                'brief_loss_hold', 'geometry_lost_yaw_zero', 'invalid_clock'):
            self._confirmation = 0
            self._previous_target = None
            self._segment_diagnostics['segment_control_active'] = False
        self._segment_diagnostics.update(segment_confirm_frames=self._confirmation,
            segment_shadow_vx=shadow[0], segment_shadow_wz=shadow[1],
            segment_shadow_reason=self.shadow.diagnostics.get('steering_reason'),
            segment_shadow_demand_deg=self.shadow.diagnostics.get('steering_demand_deg'),
            segment_applied_wz=actual[1])
        self.diagnostics.update(self._segment_diagnostics)
        return actual
