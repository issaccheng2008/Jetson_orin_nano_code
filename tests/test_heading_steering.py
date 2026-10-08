"""Geometry/hold regressions using the real detection validator, no camera mock."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "new_vision" / "jetson"))
from heading_steering import COMMAND_HOLD_S, HeadingSteeringController
from policy_bridge import SteeringController


def detection(heading=30., near=0., z=0., **changes):
    # read_detection still checks legacy angle/error/lost/confidence. Ground
    # geometry is deliberately distinct, so a legacy fallback cannot hide.
    result = dict(fused_err_cm=0., angle_err_deg=0., lost_frames=0,
                  base_err_cm=near, near_error_cm=near, near_z_cm=z,
                  measurement_valid=True, measurement_stale=False,
                  heading_control_valid=True, heading_control_deg=heading,
                  heading_valid=True)
    result.update(changes)
    return result


def controller(**changes):
    inner_options = changes.pop("inner_options", {})
    # Preserve baseline geometry/strict-hold regressions explicitly; separate
    # experiment tests exercise the changed branch defaults.
    changes.setdefault("min_hold_s", .5)
    return HeadingSteeringController(SteeringController(**inner_options), **changes)


class HeadingSteeringTests(unittest.TestCase):
    def test_default_ladders_are_the_six_published_levels(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(inner_options=dict(yaw_sign=yaw_sign))
                self.assertEqual(c.left_levels, (.37, .43, .5))
                self.assertEqual(c.right_levels, (.3, .5))
                self.assertEqual(c.straight_wz, 0.)
                self.assertEqual(c.command(detection(20., z=25.), 1., .01),
                                 (.2, .37*yaw_sign))
                self.assertEqual(c.command(detection(-20., z=25.), 1., .501),
                                 (.2, -.3*yaw_sign))

    def test_stable_left_offset_releases_left_yaw_only_after_half_second(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(inner_options=dict(yaw_sign=yaw_sign))
                first = c.command(detection(30., z=25.), 1., .01)
                self.assertGreater(first[1]*yaw_sign, 0.)
                left = detection(60., near=6., z=25.)
                self.assertEqual(c.command(left, 1., .2), first)
                self.assertEqual(c.command(left, 1., .299), first)
                self.assertTrue(c.diagnostics["steering_left_offset_confirmed"])
                self.assertAlmostEqual(c.turn_left, .001)
                self.assertEqual(c.command(left, 1., .001001), (.2, 0.))
                self.assertEqual(c.diagnostics["steering_decision"], "left_offset_release")

    def test_stable_left_position_cannot_start_another_left_block(self):
        c = controller()
        self.assertEqual(c.command(detection(0., z=25.), 1., .01), (.2, 0.))
        left = detection(60., near=6., z=25.)
        self.assertEqual(c.command(left, 1., .3), (.2, 0.))
        self.assertEqual(c.command(left, 1., .201), (.2, 0.))
        self.assertEqual(c.diagnostics["steering_decision"], "left_offset_release")
        for _ in range(4):
            self.assertEqual(c.command(left, 1., .1), (.2, 0.))
        self.assertEqual(c.turn_left, 0.)
        # Returning to the centre rearms the original leftward correction.
        self.assertGreater(c.command(detection(60., near=0., z=25.), 1., .1)[1], 0.)

    def test_one_left_offset_spike_does_not_veto_a_left_correction(self):
        c = controller()
        c.command(detection(0., z=25.), 1., .01)
        output = c.command(detection(60., near=6., z=25.), 1., .501)
        self.assertGreater(output[1], 0.)
        self.assertFalse(c.diagnostics["steering_left_offset_confirmed"])

    def test_valid_single_edge_frames_can_confirm_left_offset(self):
        c = controller()
        c.command(detection(0., z=25.), 1., .01)
        left = self.single_edge(60., single_edge_near_cm=6.)
        c.command(left, 1., .3)
        self.assertEqual(c.command(left, 1., .201), (.2, 0.))
        self.assertTrue(c.diagnostics["steering_left_offset_confirmed"])
        self.assertEqual(c.diagnostics["steering_decision"], "left_offset_release")

    def test_left_offset_guard_preserves_right_recovery_and_right_position_recovery(self):
        for yaw_sign in (-1, 1):
            for near, heading, expected in ((9., -5., -.3), (-9., 5., .37)):
                with self.subTest(yaw_sign=yaw_sign, near=near):
                    c = controller(inner_options=dict(yaw_sign=yaw_sign))
                    c.command(detection(0., z=25.), 1., .01)
                    frame = detection(heading, near=near, z=25.)
                    c.command(frame, 1., .3)
                    output = c.command(frame, 1., .201)
                    self.assertEqual(output, (.2, expected*yaw_sign))

    def test_loss_and_external_stop_clear_left_position_confirmation(self):
        for loss in (True, False):
            with self.subTest(loss=loss):
                c = controller()
                left = detection(60., near=6., z=25.)
                c.command(detection(0., z=25.), 1., .01)
                c.command(left, 1., .3)
                c.command(left, 1., .201)
                self.assertTrue(c.diagnostics["steering_left_offset_confirmed"])
                if loss:
                    c.command(detection(measurement_valid=False), 0., .1)
                else:
                    c.drop_held_command()
                c.command(left, 1., .1)
                self.assertFalse(c.diagnostics["steering_left_offset_confirmed"])

    def test_real_near_planes_direction_and_offset_tradeoff(self):
        # Real detector sees near at 20-30cm, not the robot origin. The dead zone
        # is target bearing; it must not be misrepresented as a raw heading gate.
        for z in (20., 25., 30.):
            with self.subTest(z=z):
                c = controller()
                output = c.command(detection(35., z=z), 1., .05)
                expected = math.degrees(math.atan((50.-z)/50.*math.tan(math.radians(35.))))
                self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], expected)
                self.assertGreater(output[1], 0.)
        c = controller()
        self.assertEqual(c.command(detection(20., z=25.), 1., .05)[1], .37)
        self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], 10.314104815618196)

    def test_normal_speed_and_yaw_pair_hold_half_second(self):
        self.assertEqual(controller().min_hold_s, .5)
        c = controller(allow_right=True)
        initial = c.command(detection(30), 1., .01)
        self.assertAlmostEqual(initial[1], .5)
        c.inner.vx = .7
        self.assertEqual(c.command(detection(-40), 1., .1), initial)
        self.assertEqual(c.command(detection(-40), 1., .399), initial)
        self.assertAlmostEqual(c.turn_left, .001)
        selected = c.command(detection(-40), 1., .002)
        self.assertAlmostEqual(selected[0], .7)
        self.assertAlmostEqual(selected[1], -.5)
        self.assertEqual(c.diagnostics["steering_reason"], "new_block")

    def test_identical_heartbeat_does_not_extend_minimum_duration(self):
        c = controller()
        first = c.command(detection(20), 1., .01)
        for dt in (.2, .2, .100001):
            self.assertEqual(c.command(detection(20), 1., dt), first)
        self.assertEqual(c.turn_left, 0.)
        self.assertEqual(c.diagnostics["steering_reason"], "continue_block")
        # Median geometry may need two fresh frames, but a repeated heartbeat
        # must not force another half second before accepting their new demand.
        c.command(detection(40), 1., .01)
        self.assertAlmostEqual(c.command(detection(40), 1., .01)[1], .5)

    def test_only_requested_geometric_levels_are_used(self):
        values = []
        for heading, expected in ((0, 0), (10, .37), (20, .37), (27, .43),
                                  (35, .5), (-5, -.3), (-14, -.3), (-40, -.5)):
            with self.subTest(heading=heading):
                output = controller().command(detection(heading), 1., .01)[1]
                self.assertAlmostEqual(output, expected)
                values.append(round(output, 8))
        self.assertEqual(len(set(values)), 6)

    def test_all_outputs_stay_quantized_with_real_near_planes_and_history(self):
        geometric_levels = {0., .37, .43, .5, -.3, -.5}
        for yaw_sign in (-1, 1):
            c = controller(inner_options=dict(yaw_sign=yaw_sign))
            allowed = {yaw_sign * value for value in geometric_levels}
            step = 0
            for z in (20., 25., 30.):
                for near in (-16., -8., -4., 0., 4., 8., 16.):
                    for heading in (-70., -30., -10., -5., 0., 5., 10., 30., 70.):
                        with self.subTest(yaw_sign=yaw_sign, z=z, near=near, heading=heading):
                            dt = (.1, .05, .3, .01, .4, .5, .2)[step % 7]
                            output = c.command(detection(heading, near, z), 1., dt)
                            self.assertIn(output[1], allowed)
                            step += 1

    def test_wide_angular_tolerances_do_not_disable_corridor_recovery(self):
        for z in (20., 25., 30.):
            for near in (-8.1, 8.1):
                for tolerance in (12., 80.):
                    for outward_heading in (0., -math.copysign(3., near)):
                        with self.subTest(z=z, near=near, tolerance=tolerance, heading=outward_heading):
                            c = controller(right_tolerance_deg=tolerance, left_tolerance_deg=tolerance)
                            output = c.command(detection(outward_heading, near, z), 1., .01)[1]
                            self.assertEqual(output, .37 if near < 0 else -.3)
                            self.assertEqual(c.diagnostics["steering_decision"],
                                             "right_corridor" if near < 0 else "left_corridor")
        # The widened angle gate is still meaningful inside the spatial limit.
        for near in (-7.9, 7.9):
            c = controller(right_tolerance_deg=80., left_tolerance_deg=80.)
            self.assertEqual(c.command(detection(0., near, 25.), 1., .01)[1], 0.)

    def test_corridor_recovery_never_falls_below_the_first_level(self):
        c = controller()
        self.assertEqual(c.command(detection(20., 0., 25.), 1., .01)[1], .37)
        for near in (-12., -8.1, -8., 8., 8.1, 12.):
            for yaw_sign in (-1, 1):
                with self.subTest(near=near, yaw_sign=yaw_sign):
                    c = controller(inner_options=dict(yaw_sign=yaw_sign))
                    expected = (.37 if near < 0 else -.3) * yaw_sign
                    self.assertEqual(c.command(detection(0., near, 25.), 1., .01)[1],
                                     expected)

    def test_curve_level_follows_observations_without_a_fixed_straight_timer(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(inner_options=dict(yaw_sign=yaw_sign))
                bend = detection(20., z=25.)
                first = c.command(bend, 1., .01)
                self.assertEqual(first, (.2, .37*yaw_sign))
                for _ in range(10):
                    self.assertEqual(c.command(bend, 1., .1), first)
                self.assertEqual(c.turn_left, 0.)
                # Straight is selected from new geometry, not an elapsed timer.
                for _ in range(2):
                    output = c.command(detection(0., z=25.), 1., .1)
                self.assertEqual(output, (.2, 0.))

    def test_hold_finishes_before_strong_position_recovery(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(inner_options=dict(yaw_sign=yaw_sign))
                first = c.command(detection(20., z=25.), 1., .01)
                outside = detection(0., near=-30., z=25.)
                self.assertEqual(c.command(outside, 1., .2), first)
                self.assertEqual(c.command(outside, 1., .299), first)
                self.assertEqual(c.command(outside, 1., .001001), (.2, .5*yaw_sign))

    def test_returning_inward_coasts_after_hold_instead_of_overturning(self):
        for near in (-10., 10.):
            for yaw_sign in (-1, 1):
                with self.subTest(near=near, yaw_sign=yaw_sign):
                    c = controller(inner_options=dict(yaw_sign=yaw_sign))
                    first = c.command(detection(0., near, 25.), 1., .01)
                    inward_heading = math.copysign(5., near)
                    for dt in (.17, .17):
                        self.assertEqual(c.command(detection(inward_heading, near, 25.), 1., dt), first)
                    output = c.command(detection(inward_heading, near, 25.), 1., .170001)
                    self.assertEqual(output, (c.inner.vx, 0.))
                    self.assertEqual(c.diagnostics["steering_decision"],
                                     "returning_from_right" if near < 0 else "returning_from_left")

    def test_predicted_heading_alignment_alone_does_not_abandon_corridor_recovery(self):
        for near_sign in (-1, 1):
            for yaw_sign in (-1, 1):
                with self.subTest(near_sign=near_sign, yaw_sign=yaw_sign):
                    c = controller(inner_options=dict(yaw_sign=yaw_sign))
                    near = near_sign * 16.
                    first = c.command(detection(-near_sign * 20., near, 25.), 1., .01)
                    for heading, dt in ((16, .1), (12, .1), (8, .1), (4, .1)):
                        self.assertEqual(c.command(detection(-near_sign * heading, near, 25.), 1., dt), first)
                    output = c.command(detection(-near_sign * 2., near, 25.), 1., .100001)
                    # A predicted heading sign crossing is insufficient: the
                    # fused position/direction forecast is still outside ±8cm.
                    self.assertNotEqual(c._map_angle(c.diagnostics["steering_predicted_demand_deg"]), 0.)
                    corridor_angle = math.degrees(math.atan2(8., 50.))
                    self.assertGreater(abs(c.diagnostics["steering_predicted_demand_deg"]), corridor_angle)
                    self.assertLessEqual(-near_sign * c.diagnostics["steering_predicted_heading_deg"], 0.)
                    self.assertEqual(output[1], yaw_sign * (-.3 if near_sign > 0 else .37))
                    self.assertEqual(c.diagnostics["steering_decision"],
                                     "right_corridor" if near_sign < 0 else "left_corridor")

    def test_tiny_inward_heading_does_not_bypass_the_spatial_corridor(self):
        for near in (-12., 12.):
            for yaw_sign in (-1, 1):
                with self.subTest(near=near, yaw_sign=yaw_sign):
                    c = controller(inner_options=dict(yaw_sign=yaw_sign))
                    output = c.command(detection(math.copysign(.1, near), near, 25.), 1., .01)
                    self.assertGreater(abs(c.diagnostics["steering_demand_deg"]), math.degrees(math.atan2(8., 50.)))
                    self.assertEqual(output[1], yaw_sign * (-.3 if near > 0 else .37))
                    self.assertEqual(c.diagnostics["steering_decision"],
                                     "right_corridor" if near < 0 else "left_corridor")

    def test_boundary_brake_releases_to_zero_when_forecast_reenters_corridor(self):
        c = controller()
        first = c.command(detection(0., 12., 25.), 1., .01)
        self.assertEqual(first[1], -.3)
        for near, dt in ((12, .1), (11, .1), (10, .1), (9, .1)):
            self.assertEqual(c.command(detection(0., near, 25.), 1., dt), first)
        output = c.command(detection(0., 8.5, 25.), 1., .100001)
        # The re-entry brake must not be suppressed just because this forecast
        # quantizes to the smallest right level.
        self.assertEqual(c._map_angle(c.diagnostics["steering_predicted_demand_deg"]), -.3)
        self.assertEqual(output[1], 0.)
        self.assertTrue(c.diagnostics["steering_braked"])
        self.assertEqual(c.diagnostics["steering_decision"], "left_corridor")

    def test_stronger_boundary_correction_is_not_mislabeled_as_braking(self):
        c = controller()
        first = c.command(detection(10), 1., .01)
        self.assertEqual(first[1], .37)
        for near, dt in ((-16, .1), (-15.5, .1), (-15, .1), (-14.5, .1)):
            self.assertEqual(c.command(detection(0., near, 25.), 1., dt), first)
        output = c.command(detection(0., -14, 25.), 1., .100001)
        self.assertGreater(abs(c.diagnostics["steering_predicted_demand_deg"]), math.degrees(math.atan2(8., 50.)))
        self.assertEqual(output[1], .37)
        self.assertFalse(c.diagnostics["steering_braked"])
        self.assertEqual(c.diagnostics["steering_decision"], "right_corridor")

    def test_heading_and_position_cancel_at_projected_center(self):
        heading, near_z, lookahead = 30., 10., 50.
        center_crossing_near = (lookahead - near_z) * math.tan(math.radians(heading))
        c = controller(lookahead_cm=lookahead)
        self.assertAlmostEqual(c.command(detection(heading, center_crossing_near, near_z), 1., .01)[1], 0.)
        self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], 0., places=12)
        self.assertEqual(c.diagnostics["steering_heading_source"], "ground_x_z")
        # A parallel line to positive x demands a negative correction. A heading
        # alone at the optical center demands its own signed angle.
        c = controller(allow_right=True)
        self.assertLess(c.command(detection(0, near=10), 1., .01)[1], 0.)
        self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], -math.degrees(math.atan(.2)))
        c = controller()
        c.command(detection(30, z=10), 1., .01)
        self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], math.degrees(math.atan(.8 * math.tan(math.radians(30)))))

    def test_yaw_sign_and_asymmetric_tolerances_are_distinct(self):
        for yaw_sign in (-1, 1):
            for heading, expected in ((20, .37), (-20, -.3)):
                with self.subTest(yaw_sign=yaw_sign, heading=heading):
                    c = controller(allow_right=True, inner_options=dict(yaw_sign=yaw_sign))
                    self.assertAlmostEqual(c.command(detection(heading), 1., .01)[1], expected * yaw_sign)
        self.assertAlmostEqual(controller().command(detection(6), 1., .01)[1], 0.)
        self.assertEqual(controller().command(detection(-6), 1., .01)[1], -.3)
        self.assertEqual(controller(allow_right=False).command(detection(-60), 1., .01)[1], 0.)

    def test_asymmetric_cap_is_in_final_output_coordinate_for_both_signs(self):
        for yaw_sign in (-1, 1):
            for heading in (-70, 70):
                with self.subTest(yaw_sign=yaw_sign, heading=heading):
                    c = controller(allow_right=True, inner_options=dict(yaw_sign=yaw_sign, max_wz_right=.45))
                    output = c.command(detection(heading), 1., .01)[1]
                    capped = -.43 if yaw_sign < 0 else -.3
                    self.assertAlmostEqual(output, capped if heading * yaw_sign < 0 else .5)
        for options in (dict(max_step=.3), dict(max_step=.3, allow_right=False),
                        dict(inner_options=dict(max_wz_right=.2))):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "at least one level"):
                controller(**options)

    def test_approaching_trend_brake_never_increases_existing_amplitude(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(allow_right=True,
                               inner_options=dict(yaw_sign=yaw_sign))
                first = c.command(detection(14), 1., .01)[1]
                self.assertAlmostEqual(abs(first), .37)
                # A rise followed by a measured decline produces a larger
                # present/forecast candidate than the small action actually
                # being executed. Calling it braking may not raise that action.
                for heading, dt in ((42, .1), (41, .1), (40, .1), (39, .1), (38, .100001)):
                    result = c.command(detection(heading), 1., dt)[1]
                self.assertTrue(c.diagnostics["steering_prediction_valid"])
                self.assertTrue(c.diagnostics["steering_braked"])
                self.assertLessEqual(abs(result), abs(first) + 1e-12)
                self.assertGreaterEqual(result * first, 0.)

    def test_forecast_across_center_brakes_to_zero_without_reversing(self):
        for yaw_sign in (-1, 1):
            with self.subTest(yaw_sign=yaw_sign):
                c = controller(allow_right=True, inner_options=dict(yaw_sign=yaw_sign))
                first = c.command(detection(30), 1., .01)
                for heading, dt in ((28, .1), (24, .1), (20, .1), (16, .1)):
                    self.assertEqual(c.command(detection(heading), 1., dt), first)
                stopped = c.command(detection(12), 1., .100001)
                self.assertTrue(c.diagnostics["steering_braked"])
                self.assertEqual(stopped[1], 0.)
                self.assertEqual(stopped[0], c.inner.vx)

    def test_invalid_new_ground_fit_does_not_fall_back_to_legacy(self):
        cases = [dict(heading_control_valid=False), dict(heading_control_deg=float("nan")),
                 dict(near_z_cm=50.), dict(heading_control_deg=90.)]
        for changes in cases:
            with self.subTest(changes=changes):
                c = controller()
                invalid = detection(60, angle_err_deg=60., **changes)
                self.assertIsNotNone(c.inner.read_detection(invalid, 1., .01))
                self.assertEqual(c.command(invalid, 1., .01), (0., 0.))
                self.assertEqual(c.diagnostics["steering_reason"], "brief_loss_hold")
                self.assertNotIn("steering_heading_source", c.diagnostics)

    def test_legacy_only_geometry_is_explicitly_labeled(self):
        c = controller()
        old = detection(angle_err_deg=30)
        del old["heading_control_valid"]
        del old["heading_control_deg"]
        self.assertAlmostEqual(c.command(old, 1., .01)[1], .5)
        self.assertEqual(c.diagnostics["steering_heading_source"], "legacy_pixel_heading")

    def test_brief_loss_preserves_pair_then_deadline_discards_only_yaw(self):
        c = controller()
        first = c.command(detection(30), 1., .01)
        lost = detection(lost_frames=1, measurement_valid=False)
        self.assertEqual(c.command(lost, 0., .05), first)
        self.assertEqual(c.inner.hold, first)
        self.assertEqual(c.diagnostics["steering_reason"], "brief_loss_hold")
        self.assertEqual(c.command(lost, 0., .151), (first[0], 0.))
        self.assertEqual(c.diagnostics["steering_reason"], "geometry_lost_yaw_zero")
        self.assertEqual(c.command(lost, 0., 2.), (first[0], 0.))
        self.assertEqual(c.turn_left, 0.)

    def test_stale_is_immediate_and_reacquisition_starts_fresh(self):
        c = controller(allow_right=True, inner_options=dict(lost_hold_s=5.))
        first = c.command(detection(30), 1., .01)
        self.assertEqual(c.command(detection(measurement_stale=True), 1., .01), (first[0], 0.))
        self.assertEqual(c.inner.hold, (first[0], 0.))
        self.assertEqual(c.diagnostics["steering_reason"], "geometry_lost_yaw_zero")
        self.assertAlmostEqual(c.command(detection(-20), 1., .01)[1], -.3)
        self.assertAlmostEqual(c.turn_left, .5)

    def test_loss_cannot_start_walking_or_replay_speed_after_explicit_stop(self):
        c = controller()
        lost = detection(measurement_valid=False, measurement_stale=True)
        self.assertEqual(c.command(lost, 0., .5), (0., 0.))
        c.command(detection(30), 1., .01)
        c.drop_held_command()
        self.assertEqual(c.command(lost, 0., .5), (0., 0.))

    def test_geometry_loss_preserves_applied_speed_not_latest_configuration(self):
        c = controller(inner_options=dict(vx=.2))
        c.command(detection(30), 1., .01)
        c.inner.vx = .8
        self.assertEqual(c.command(detection(heading_control_valid=False), 1., .3), (.2, 0.))

    def single_edge(self, heading, **changes):
        values = dict(heading_control_valid=False, single_edge_valid=True,
                      single_edge_heading_deg=heading, single_edge_near_cm=0.,
                      single_edge_z_cm=25.)
        values.update(changes)
        return detection(30., near=0., z=25., **values)

    def test_single_edge_direction_supplies_heading_when_the_pair_fit_is_invalid(self):
        """丢线兜底：配对拟合无效、但检测器还看得见一条边时，沿着这条边的方向走。"""
        c = controller()
        output = c.command(self.single_edge(20.), 1., .01)
        self.assertEqual(c.diagnostics["steering_heading_source"], "single_edge")
        self.assertAlmostEqual(c.diagnostics["steering_demand_deg"], 10.314104815618196)
        self.assertEqual(output, (c.inner.vx, .37))
        self.assertEqual(c.diagnostics["steering_decision"], "right_corridor")

    def test_the_single_edge_sign_sets_the_turn_direction(self):
        for heading, expected in ((20., .37), (-20., -.3)):
            with self.subTest(heading=heading):
                c = controller()
                self.assertEqual(c.command(self.single_edge(heading), 1., .01)[1], expected)

    def test_paired_geometry_wins_over_the_single_edge_fallback(self):
        c = controller()
        output = c.command(self.single_edge(-40., heading_control_valid=True), 1., .01)
        self.assertEqual(c.diagnostics["steering_heading_source"], "ground_x_z")
        self.assertEqual(output[1], .37)

    def test_the_single_edge_fallback_shares_the_geometry_gates(self):
        cases = (dict(single_edge_heading_deg=90.), dict(single_edge_z_cm=50.),
                 dict(single_edge_heading_deg=float("nan")))
        for changes in cases:
            with self.subTest(changes=changes):
                c = controller()
                bad = self.single_edge(20., **changes)
                self.assertEqual(c.command(bad, 1., .01), (0., 0.))
                self.assertEqual(c.diagnostics["steering_reason"], "brief_loss_hold")
                self.assertNotIn("steering_heading_source", c.diagnostics)

    def test_invalid_clock_and_explicit_reset_cancel_the_contract(self):
        for dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=dt):
                c = controller()
                c.command(detection(30), 1., .01)
                self.assertEqual(c.command(detection(30), 1., dt), (0., 0.))
                self.assertEqual(c.diagnostics["steering_reason"], "invalid_clock")
        c = controller(allow_right=True)
        c.command(detection(30), 1., .01)
        c.drop_held_command()
        self.assertEqual(c.hold, (0., 0.))
        self.assertAlmostEqual(c.command(detection(-20), 1., .01)[1], -.3)

    def test_custom_ladders_and_the_straight_level_flow_through(self):
        self.assertEqual(controller(left_levels=(0.2, 0.45)).left_levels, (.2, .45))
        self.assertEqual(controller(right_levels=(0.15,)).right_levels, (.15,))
        self.assertEqual(controller(left_levels=(0.2, 0.45))
                         .command(detection(30), 1., .01)[1], .45)
        self.assertEqual(controller(right_levels=(0.15,))
                         .command(detection(-30), 1., .01)[1], -.15)
        self.assertEqual(controller(straight_wz=0.05)
                         .command(detection(0), 1., .01)[1], .05)

    def test_ladder_validation_rejects_bad_shapes(self):
        for changes in (dict(left_levels=()), dict(left_levels=(0.5, 0.37)),
                        dict(right_levels=(-0.3,)), dict(left_levels=(0.0, 0.4)),
                        dict(straight_wz=0.5)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                controller(**changes)


if __name__ == "__main__":
    unittest.main()
