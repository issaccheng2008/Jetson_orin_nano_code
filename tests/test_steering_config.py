"""Configuration changes must affect real steering and measured geometry."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_filter import FilterConfig
from line_detector_v1_warp import LineDetector
from lane_segments import describe_lane_segments
from segment_steering import SegmentSteeringController
from run_policy_vision import parse_args
from test_heading_steering import detection
import test_heading_far_band

TABLE = [[-90, -30, -.5], [-30, -10, -.3], [-10, 10, 0], [10, 30, .3], [30, 90, .5]]


class SteeringConfigTests(unittest.TestCase):
    def make(self, **options):
        return HeadingSteeringController(SteeringController(), angle_wz_table=TABLE,
            position_gain=0, position_recovery_cm=0, corridor_cm=100, **options)

    def test_table_boundaries_and_wire_sign(self):
        for sign in (-1, 1):
            c = HeadingSteeringController(SteeringController(yaw_sign=sign),
                angle_wz_table=TABLE, position_gain=0, position_recovery_cm=0,
                corridor_cm=100)
            for demand, expected in ((0, 0), (9.99, 0), (10, .3), (29.99, .3),
                                     (30, .5), (80, .5), (-10, 0), (-30, -.3), (-31, -.5)):
                with self.subTest(sign=sign, demand=demand):
                    self.assertAlmostEqual(c.command(detection(demand), 1, .1)[1], expected*sign)
                    c.reset()

    def test_table_uses_combined_filtered_demand(self):
        c = HeadingSteeringController(SteeringController(), angle_wz_table=TABLE,
            filter_config=FilterConfig(algorithm='none', hysteresis_deg=0),
            corridor_cm=100, position_recovery_cm=0)
        # Raw target bearing <10°, position correction puts final decision above 10°.
        out = c.command(detection(0, near=-6, z=25), 1, .1)
        self.assertLess(c.diagnostics['steering_filtered_demand_deg'], 10)
        self.assertGreater(c.diagnostics['steering_combined_demand_deg'], 10)
        self.assertEqual(out[1], .3)

    def test_hysteresis_follows_table_boundary(self):
        c = self.make(filter_config=FilterConfig(algorithm='none', hysteresis_deg=1))
        c._command = (.2, .3)
        self.assertEqual(c._filtered_level(30.5), .3)
        self.assertEqual(c._filtered_level(31.1), .5)
        c._command = (.2, .5)
        self.assertEqual(c._filtered_level(29.5), .5)
        self.assertEqual(c._filtered_level(28.9), .3)

    def test_protection_can_override_zero_table_entry(self):
        c = HeadingSteeringController(SteeringController(), angle_wz_table=TABLE,
            corridor_cm=5, position_gain=0, position_recovery_cm=0)
        candidate, reason = c._decision(7, 0, 7)
        self.assertEqual(candidate, .3)
        self.assertEqual(reason, 'right_corridor')

    def test_invalid_table_is_rejected(self):
        for table in ([[0, 10, 0], [12, 90, .3]], [[0, 30, 0], [20, 90, .3]],
                      [[0, 10, 0], [10, 90, .6]], [[0, 10, 0]],
                      [[0, 10, 0], [10, 90, float('nan')]]):
            with self.subTest(table=table), self.assertRaises(ValueError):
                HeadingSteeringController(SteeringController(), angle_wz_table=table)
        with self.assertRaises(ValueError):
            HeadingSteeringController(SteeringController(), angle_wz_table=TABLE, max_step=.3)

    def test_near_and_far_move_together_with_eight_rows(self):
        d = LineDetector()
        d.set_heading_distances(27, 34)
        near, far = test_heading_far_band.HeadingFarBandTests().scan(d)
        self.assertAlmostEqual(near['dist_cm'], 27, delta=.15)
        self.assertAlmostEqual(far['dist_cm'], 34, delta=.15)
        self.assertEqual(len(near['ys_list']), 8)
        self.assertEqual(len(far['ys_list']), 8)
        gray = np.full((400, 320), 255, np.uint8)
        gray[:, 87:94] = gray[:, 227:234] = 0
        lock = d._detect_bottom_center_lock(gray, np.repeat(gray[:, :, None], 3, axis=2), 80, True)
        self.assertTrue(lock['valid'])
        self.assertEqual(lock['center_ys'][0], near['ys_list'][0])
        self.assertAlmostEqual(d.err_scale_cm, .5*d.bird_w*d.cm_per_px_at(np.median(near['ys_list'])))
        d.set_camera_pitch_deg(53)
        near, far = test_heading_far_band.HeadingFarBandTests().scan(d)
        self.assertAlmostEqual(near['dist_cm'], 27, delta=.15)
        self.assertAlmostEqual(far['dist_cm'], 34, delta=.15)

    def test_bad_distance_change_is_atomic(self):
        d = LineDetector()
        d.set_heading_distances(25.07, 29)
        original = (d.band_low_y0, d.band_mid_y0, d.err_scale_cm)
        for near, far in ((29, 25), (28.8, 29), (10, 29), (25, 100), (float('nan'), 29)):
            with self.subTest(near=near, far=far), self.assertRaises(ValueError):
                d.set_heading_distances(near, far)
            self.assertEqual((d.band_low_y0, d.band_mid_y0, d.err_scale_cm), original)

    def segments(self, regions):
        z = np.linspace(88, 20, 400)
        scale = np.full(400, .25)
        rows = np.arange(110, 399, 2)
        return describe_lane_segments(rows, [160]*len(rows), [140]*len(rows),
            z, scale, 160, 35, regions_cm=regions)

    def test_two_and_four_regions_fit_and_control(self):
        for regions in (((20, 32), (32, 44)), ((20, 32), (32, 44), (44, 56), (56, 68))):
            c = SegmentSteeringController(SteeringController(), segment_regions_cm=regions,
                                          lookahead_cm=65)
            debug = detection(0, z=25, fit_seg_anchored=True, **self.segments(regions))
            self.assertEqual(debug['fit_seg_count'], len(regions))
            for _ in range(5):
                c.command(debug, 1, .1)
            self.assertTrue(c.diagnostics['segment_control_active'])
            self.assertEqual(c.diagnostics['segment_target_index'], len(regions)-1)
            self.assertLessEqual(c.diagnostics['segment_target_z_cm'], 65)

    def test_fourth_region_cannot_jump_across_missing_third(self):
        regions = ((20, 32), (32, 44), (44, 56), (56, 68))
        c = SegmentSteeringController(SteeringController(), segment_regions_cm=regions, lookahead_cm=65)
        debug = detection(0, z=25, fit_seg_anchored=True, **self.segments(regions))
        debug['fit_seg2_valid'] = False
        for _ in range(5):
            c.command(debug, 1, .1)
        self.assertEqual(c.diagnostics['segment_target_index'], 1)

    def test_cli_arrays_and_distance_validation(self):
        arguments = ['vision', '--heading-near-cm', '27', '--heading-far-cm', '34',
                     '--steering-angle-wz-table', json.dumps(TABLE),
                     '--segment-regions-cm', '[[20,32],[32,44]]']
        with patch.dict(os.environ, {}, clear=True), patch('sys.argv', arguments):
            args = parse_args()
        self.assertEqual(args.heading_near_cm, 27)
        self.assertEqual(len(args.segment_regions_cm), 2)
        self.assertEqual(args.steering_angle_wz_table[3][2], .3)
        for flag, value in (('--segment-regions-cm', '[[20,32],[35,44]]'),
                            ('--segment-regions-cm', '[[20,24],[24,44]]'),
                            ('--heading-near-cm', '30'),
                            ('--steering-angle-wz-table', 'bad-json')):
            with patch('sys.argv', ['vision', flag, value]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_signed_asymmetric_intervals_and_zero(self):
        table = [[-90,-3,-.2],[-3,0,-.1],[0,10,0],[10,90,.3]]
        c = HeadingSteeringController(SteeringController(), angle_wz_table=table,
            position_gain=0, position_recovery_cm=0, corridor_cm=100)
        for angle, expected in ((-4,-.2),(-3,-.1),(-.5,-.1),(0,0),(.5,0),(10,.3)):
            self.assertAlmostEqual(c.command(detection(angle),1,.1)[1],expected)
            c.reset()
        self.assertEqual(c.left_levels, (.3,))
        self.assertEqual(c.right_levels, (.1,.2))

    def test_signed_table_uses_each_wire_direction_cap(self):
        for sign in (-1,1):
            positive, negative = (.5,.2) if sign == 1 else (.2,.5)
            table = [[-90,-10,-negative],[-10,10,0],[10,90,positive]]
            c = HeadingSteeringController(SteeringController(yaw_sign=sign,max_wz_right=.2), angle_wz_table=table)
            self.assertEqual(c._map_angle(30)*sign, positive*sign)
            self.assertEqual(c._map_angle(-30)*sign, -negative*sign)
            bad = [list(row) for row in table]
            bad[0 if sign == 1 else 2][2] *= 2
            with self.assertRaises(ValueError):
                HeadingSteeringController(SteeringController(yaw_sign=sign,max_wz_right=.2),angle_wz_table=bad)

    def test_negative_side_hysteresis_does_not_use_positive_bins(self):
        table = [[-90,-20,-.2],[-20,-3,-.1],[-3,10,0],[10,90,.3]]
        c = HeadingSteeringController(SteeringController(),angle_wz_table=table,
            filter_config=FilterConfig(algorithm='none',hysteresis_deg=1))
        c._command=(.2,-.1)
        self.assertEqual(c._filtered_level(-20.5),-.1)
        self.assertEqual(c._filtered_level(-21.1),-.2)
        c._command=(.2,-.2)
        self.assertEqual(c._filtered_level(-19.5),-.2)
        self.assertEqual(c._filtered_level(-18.9),-.1)
        self.assertEqual(c._map_angle(20),.3)

    def test_invalid_signed_table_coverage_and_caps_are_rejected(self):
        for table in ([[-90,-3,-.6],[-3,10,0],[10,90,.3]],
                      [[-80,0,-.1],[0,90,.3]],
                      [[-90,10,0],[10,90,.3]],
                      [[-90,-10,-.1],[-10,10,float('nan')],[10,90,.3]],
                      [[-90,-10,-.1],[-10,10,0],[12,90,.3]]):
            with self.subTest(table=table),self.assertRaises(ValueError):
                HeadingSteeringController(SteeringController(),angle_wz_table=table)

    def test_zero_angle_rate_is_configurable_but_external_stop_stays_zero(self):
        for zero_rate in (-.1,.1):
            table=[[-90,-10,-.2],[-10,10,zero_rate],[10,90,.3]]
            for sign in (-1,1):
                c=HeadingSteeringController(SteeringController(yaw_sign=sign),angle_wz_table=table,
                    position_gain=0,position_recovery_cm=0,corridor_cm=100,
                    filter_config=FilterConfig(algorithm='none',hysteresis_deg=1))
                self.assertAlmostEqual(c.command(detection(0),1,.1)[1],zero_rate*sign)
                self.assertEqual(c.command(detection(0),1,0),(0,0))

    def test_zero_angle_bias_is_not_reused_as_loss_turn_history(self):
        from steering_recovery import RecoveryConfig
        c=HeadingSteeringController(SteeringController(),
            angle_wz_table=[[-90,-10,-.2],[-10,10,.1],[10,90,.3]],
            position_gain=0,position_recovery_cm=0,corridor_cm=100,recovery_config=RecoveryConfig())
        for _ in range(10):
            self.assertEqual(c.command(detection(0),1,.1)[1],.1)
        c.command(detection(measurement_valid=False),0,.1)
        self.assertEqual(c.diagnostics['steering_reason'],'loss_default_left')

    def test_heading_regions_sample_both_intervals_and_lock_with_fixed_budget(self):
        d=LineDetector()
        regions=((24,26),(28,30))
        d.set_heading_regions_cm(regions)
        near,far=test_heading_far_band.HeadingFarBandTests().scan(d)
        for band,(lo,hi) in zip((near,far),regions):
            self.assertEqual(len(band['ys_list']),8)
            z=[d.z_cm_at(y) for y in band['ys_list']]
            self.assertTrue(all(lo <= v < hi for v in z))
            self.assertGreater(max(z)-min(z),1.5)
        gray=np.full((400,320),255,np.uint8)
        gray[:,87:94]=gray[:,227:234]=0
        lock=d._detect_bottom_center_lock(gray,np.repeat(gray[:,:,None],3,axis=2),80,True)
        self.assertEqual(lock['center_ys'],near['ys_list'])
        self.assertAlmostEqual(d.err_scale_cm,.5*d.bird_w*d.cm_per_px_at(np.median(near['ys_list'])))
        d.set_camera_pitch_deg(53)
        for band,(lo,hi) in zip(test_heading_far_band.HeadingFarBandTests().scan(d),regions):
            self.assertTrue(all(lo <= d.z_cm_at(y) < hi for y in band['ys_list']))

    def test_heading_regions_invalid_change_keeps_old_geometry(self):
        d=LineDetector()
        d.set_heading_regions_cm(((24,26),(28,30)))
        old=(d.band_low_y0,d.band_mid_y0,d.err_scale_cm,d.heading_regions_cm)
        for regions in (((28,30),(24,26)),((24,29),(28,30)),((24,24),(28,30)),
                        ((24,24.1),(28,30)),((24,26),(100,110))):
            with self.subTest(regions=regions),self.assertRaises(ValueError):
                d.set_heading_regions_cm(regions)
            self.assertEqual((d.band_low_y0,d.band_mid_y0,d.err_scale_cm,d.heading_regions_cm),old)

    def test_heading_regions_cli_replaces_centres_and_keeps_modes_separate(self):
        with patch.dict(os.environ,{},clear=True),patch('sys.argv',['vision','--heading-regions-cm','[[24,26],[28,30]]']):
            args=parse_args()
        self.assertEqual(args.wz_mode,'heading')
        self.assertEqual(args.heading_regions_cm,((24,26),(28,30)))
        self.assertEqual(args.heading_near_cm,25)
        self.assertEqual(args.heading_far_cm,29)
        for regions in ('[[24,29],[28,30]]','[[24,26],[28,30],[32,40]]'):
            with patch('sys.argv',['vision','--heading-regions-cm',regions]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
