"""Known ground paths must change real steering, not just add diagnostics."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from lane_segments import describe_lane_segments
from line_detector_v1_warp import LineDetector
from policy_bridge import SteeringController
from segment_steering import SegmentSteeringController


def detection(path=lambda z: np.zeros_like(z), heading=0.0, **changes):
    z_lut = np.linspace(88., 20., 400)
    scale = np.linspace(.29, .22, 400)
    rows = np.arange(190, 399, 2)
    z = z_lut[rows]
    x = path(z)
    slope = np.gradient(x, z)
    segments = describe_lane_segments(rows, 160+x/scale[rows],
        35*np.sqrt(1+slope*slope)/scale[rows], z_lut, scale, 160, 35)
    near = float(path(np.asarray([25.]))[0])
    debug = dict(fused_err_cm=0., base_err_cm=near, near_error_cm=near,
        near_z_cm=25., angle_err_deg=heading, lost_frames=0,
        measurement_valid=True, measurement_stale=False, heading_control_valid=True,
        heading_control_deg=heading, fit_seg_anchored=True, **segments)
    debug.update(changes)
    return debug


def bend(z):
    return -.035*np.maximum(0., z-30.)**2


def controller(**options):
    inner = options.pop('inner_options', {})
    # Keep baseline geometry regressions under their original timing policy.
    options.setdefault('min_hold_s', .5)
    return SegmentSteeringController(SteeringController(**inner), **options)


def settle(c, debug, frames=12):
    return [c.command(debug, 1., .1) for _ in range(frames)]


class SegmentSteeringTests(unittest.TestCase):
    def test_real_image_scanner_feeds_bend_entry_and_straight_exit(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                d = LineDetector(z_calib=(1.13233, -2.4862), lane_width_cm=35)
                d.M = np.eye(3)
                d.startup_force_simple_bottom = False
                d.lane_fit_enable = d.lane_segments_enable = True
                c = controller()
                ys = np.arange(150, 400)
                z = np.asarray([d.z_cm_at(y) for y in ys])
                scale = np.asarray([d.cm_per_px_at(y) for y in ys])
                for curved in (True, False):
                    frame = np.full((400, 320, 3), 255, np.uint8)
                    x = direction*bend(z) if curved else np.zeros_like(z)
                    slope = direction*-.07*np.maximum(0., z-30.) if curved else np.zeros_like(z)
                    width = 35*np.sqrt(1+slope*slope)
                    for side in (-1, 1):
                        points = np.column_stack((160+(x+side*width/2)/scale, ys)).astype(np.int32)
                        cv2.polylines(frame, [points], False, (0, 0, 0), 5)
                    for _ in range(20):
                        _, _, confidence, _, debug = d.process(frame, dt=.1)
                        out = c.command(debug, confidence, .1)
                    self.assertTrue(c.diagnostics['segment_control_active'], c.diagnostics)
                    self.assertLessEqual(c.diagnostics['segment_target_z_cm'], 50.)
                    if curved:
                        self.assertGreater(out[1]*direction, 0.)
                        self.assertEqual(c.diagnostics['segment_shadow_wz'], 0.)
                    else:
                        self.assertEqual(out, (.2, 0.))

    def test_distant_left_bend_changes_command_when_near_heading_is_straight(self):
        c = controller()
        debug = detection(bend)
        commands = settle(c, debug)
        self.assertGreater(commands[-1][1], 0.)
        self.assertEqual(c.diagnostics['steering_heading_source'], 'measured_segments')
        self.assertEqual(c.diagnostics['segment_shadow_wz'], 0.)
        self.assertAlmostEqual(c.diagnostics['segment_target_z_cm'], 50.)
        self.assertAlmostEqual(c.diagnostics['segment_target_x_cm'], float(bend(50.)), delta=.6)

    def test_mirrored_bends_and_yaw_sign_reach_the_wire_direction(self):
        for sign in (-1, 1):
            for direction in (-1, 1):
                with self.subTest(sign=sign, direction=direction):
                    c = controller(inner_options=dict(yaw_sign=sign))
                    out = settle(c, detection(lambda z: direction*bend(z)))[-1]
                    self.assertGreater(out[1]*sign*direction, 0.)
                    self.assertLessEqual(abs(out[1]), .5)

    def test_exit_uses_observed_straight_path_instead_of_old_turning_tangent(self):
        c = controller()
        out = settle(c, detection(heading=30.))[-1]
        self.assertEqual(out, (.2, 0.))
        self.assertGreater(c.diagnostics['segment_shadow_wz'], 0.)

    def test_target_is_clamped_to_measured_middle_support_when_far_is_missing(self):
        debug = detection(bend)
        debug['fit_seg2_valid'] = False
        c = controller()
        settle(c, debug)
        self.assertEqual(c.diagnostics['steering_heading_source'], 'measured_segments')
        self.assertLessEqual(c.diagnostics['segment_target_z_cm'], debug['fit_seg1_z_max_cm'])
        self.assertLess(c.diagnostics['segment_target_z_cm'], 44.)

    def test_configured_target_inside_middle_segment_is_not_extended_to_far(self):
        c = controller(lookahead_cm=40.)
        settle(c, detection(bend))
        self.assertEqual(c.diagnostics['segment_target_z_cm'], 40.)
        self.assertEqual(c.diagnostics['segment_target_index'], 1)

    def test_unqualified_segment_paths_use_original_heading(self):
        changes = [dict(fit_seg_anchored=False), dict(fit_seg_valid=False),
                   dict(fit_seg1_valid=False), dict(fit_seg0_valid=False),
                   dict(fit_seg1_x_cm=30.), dict(fit_seg1_heading_deg=89.),
                   dict(fit_seg1_rmse_cm=9.), dict(fit_seg1_normal_width_cm=65.),
                   dict(fit_seg1_z_min_cm=float('nan')),
                   dict(fit_seg1_z_max_cm=99.)]
        for change in changes:
            with self.subTest(change=change):
                c = controller()
                settle(c, detection(bend, **change))
                self.assertEqual(c.diagnostics['steering_heading_source'], 'ground_x_z')
                self.assertEqual(c.hold, (.2, 0.))
                self.assertFalse(c.diagnostics['segment_control_active'])

    def test_three_consecutive_stable_frames_are_required(self):
        c = controller()
        good = detection(bend)
        for expected in (1, 2):
            c.command(good, 1., .1)
            self.assertFalse(c.diagnostics['segment_control_active'])
            self.assertEqual(c.diagnostics['segment_confirm_frames'], expected)
        c.command(detection(fit_seg_anchored=False), 1., .1)
        c.command(good, 1., .1)
        self.assertEqual(c.diagnostics['segment_confirm_frames'], 1)
        c.command(good, 1., .1)
        self.assertFalse(c.diagnostics['segment_control_active'])
        c.command(good, 1., .1)
        self.assertTrue(c.diagnostics['segment_control_active'])

    def test_large_target_jump_restarts_confirmation(self):
        c = controller()
        settle(c, detection(bend))
        c.command(detection(lambda z: -bend(z)), 1., .1)
        self.assertFalse(c.diagnostics['segment_control_active'])
        self.assertEqual(c.diagnostics['segment_confirm_frames'], 1)

    def test_source_change_does_not_cut_short_the_half_second_hold(self):
        c = controller()
        debug = detection(bend)
        first = next(out for out in settle(c, debug, 7) if out[1] > 0)
        until = c.turn_left
        self.assertGreater(until, 0.)
        no_segments = detection(fit_seg_valid=False)
        self.assertEqual(c.command(no_segments, 1., until/2), first)
        self.assertEqual(c.diagnostics['steering_reason'], 'minimum_hold')
        self.assertEqual(c.command(no_segments, 1., until/2+.000001), (.2, 0.))

    def test_bad_near_geometry_cannot_be_rescued_by_far_segments(self):
        c = controller()
        out = c.command(detection(bend, heading_control_valid=False), 1., .1)
        self.assertEqual(out, (0., 0.))
        self.assertFalse(c.diagnostics['segment_control_active'])

    def test_stale_or_lost_frame_discards_confirmation_and_preserves_applied_speed(self):
        for changes in (dict(measurement_stale=True), dict(measurement_valid=False, lost_frames=1)):
            with self.subTest(changes=changes):
                c = controller()
                settle(c, detection(bend))
                out = c.command(detection(bend, **changes), 0., .3)
                self.assertEqual(out, (.2, 0.))
                self.assertEqual(c.diagnostics['segment_confirm_frames'], 0)

    def test_external_stop_resets_both_control_and_comparison(self):
        c = controller()
        settle(c, detection(bend))
        c.drop_held_command()
        self.assertEqual(c.hold, (0., 0.))
        self.assertEqual(c.shadow.hold, (0., 0.))
        self.assertEqual(c.command(detection(measurement_stale=True), 0., .5), (0., 0.))
        c.command(detection(bend), 1., .1)
        self.assertFalse(c.diagnostics['segment_control_active'])

    def test_near_position_recovery_and_output_caps_are_retained(self):
        c = controller(inner_options=dict(max_wz_right=.3))
        debug = detection(lambda z: np.full_like(z, 12.))
        out = settle(c, debug)[-1]
        self.assertEqual(out, (.2, -.3))

    def test_segment_diagnostics_are_finite_scalar_telemetry(self):
        c = controller()
        settle(c, detection(bend))
        for key, value in c.diagnostics.items():
            with self.subTest(key=key):
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))


if __name__ == '__main__':
    unittest.main()
