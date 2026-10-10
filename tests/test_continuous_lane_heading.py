"""Synthetic birdseye rows; no camera, IPM round trip, serial or motors."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision' / 'jetson'))
from continuous_lane_heading import HeadingScanConfig, trace_heading
from line_detector_v1_warp import LineDetector


class ContinuousHeadingTests(unittest.TestCase):
    def setUp(self):
        self.detector = LineDetector(z_calib=(1, 0))
        self.config = HeadingScanConfig()

    def make_mask(self, angle=0, offset_px=0, sx=.25, sz=.20,
                  missing_left=(), missing_right=(), missing_both=(),
                  wrong_rows=(), width_px=120):
        mask = np.zeros((400, 320), np.uint8)
        slope = math.tan(math.radians(angle))
        for y in range(400):
            if y in missing_both:
                continue
            z = (390-y)*sz
            center = 160 + offset_px + slope*(z-12)/sx
            if y in wrong_rows:
                center += 22
            half = (width_px + .03*(390-y))/2
            for side, missing in ((-1, missing_left), (1, missing_right)):
                if y in missing:
                    continue
                x = int(round(center + side*half))
                if 4 <= x < 316:
                    mask[y, x-3:x+4] = 255
        return mask

    def scan(self, mask, sx=.25, sz=.20, config=None, previous=(), age=float('inf')):
        ground = lambda x,y: ((x-160)*sx, (390-y)*sz)
        return trace_heading(
            mask,
            lambda y: self.detector._collect_track_runs_on_row(mask,y,0,319,128,False),
            ground, lambda y:sx, previous_rows=previous, previous_age_s=age,
            config=config or self.config)

    def test_full_range_straight_and_translation(self):
        for shift in (-35, 0, 35):
            result = self.scan(self.make_mask(offset_px=shift))
            self.assertTrue(result['valid'], result['reason'])
            self.assertEqual((result['rows'][0]['y'],result['rows'][-1]['y']), (390,220))
            self.assertGreaterEqual(len(result['rows']), 16)
            self.assertAlmostEqual(result['heading_right_deg'], 0, delta=.7)
            self.assertAlmostEqual(result['rows'][0]['center_x'],160+shift,delta=1)

    def test_physical_slopes_sign_and_non_square_pixels(self):
        for sx, sz in ((.25,.20),(.40,.10)):
            for angle in (-18,18):
                result = self.scan(self.make_mask(angle=angle,sx=sx,sz=sz),sx,sz)
                self.assertTrue(result['valid'],result['reason'])
                self.assertAlmostEqual(result['heading_right_deg'],angle,delta=1.2)
                # Public interfaces use the old positive-left convention.
                self.assertAlmostEqual(-result['heading_right_deg'],-angle,delta=1.2)

    def test_same_side_of_image_middle_and_single_edge_identity(self):
        # Both observed boundaries may be to the right of x=160.
        shifted = self.scan(self.make_mask(offset_px=95,width_px=100))
        self.assertTrue(shifted['valid'], shifted['reason'])
        self.assertGreater(shifted['rows'][0]['left_x'],160)
        sample_rows = [379,368,357,346]
        missing = {y+i for y in sample_rows for i in range(-2,3)}
        mask = self.make_mask(offset_px=95,width_px=100,missing_right=missing)
        result = self.scan(mask)
        self.assertTrue(result['valid'], result['reason'])
        self.assertTrue(any(r['source']=='left' for r in result['rows']))
        self.assertFalse(any(r['source']=='right' for r in result['rows']))
        self.assertAlmostEqual(result['heading_right_deg'],0,delta=1.0)
        left_shift = self.scan(self.make_mask(offset_px=-95,width_px=100,
                                              missing_left=missing))
        self.assertTrue(left_shift['valid'],left_shift['reason'])
        self.assertLess(left_shift['rows'][0]['right_x'],160)
        self.assertTrue(any(r['source']=='right' for r in left_shift['rows']))
        self.assertAlmostEqual(left_shift['heading_right_deg'],0,delta=1.0)

    def test_sparse_wrong_segments_and_outlier_centres(self):
        base = self.scan(self.make_mask(angle=10))
        mask = self.make_mask(angle=10,wrong_rows=set(range(323,327)))
        mask[280:284,5:12] = 255
        mask[302:306,180:186] = 255
        result = self.scan(mask)
        self.assertTrue(result['valid'],result['reason'])
        self.assertAlmostEqual(result['heading_right_deg'],base['heading_right_deg'],delta=2.0)

    def test_short_gap_predicts_but_long_gap_invalidates(self):
        short = {y+i for y in (357,346) for i in range(-2,3)}
        result = self.scan(self.make_mask(missing_both=short))
        self.assertTrue(result['valid'],result['reason'])
        self.assertEqual(sum(r['source']=='predicted' for r in result['rows']),2)
        self.assertGreater(result['observed_rows'],result['paired_rows']-1)
        long = set(range(360,400))
        result = self.scan(self.make_mask(missing_both=long))
        self.assertFalse(result['valid'])

    def test_empty_history_and_insufficient_span_do_not_claim_heading(self):
        good = self.scan(self.make_mask())
        empty = np.zeros((400,320),np.uint8)
        for age in (.1,.2,.3,.4):
            result = self.scan(empty,previous=good['rows'],age=age)
            self.assertFalse(result['valid'])
            self.assertEqual(result['observed_rows'],0)
        narrow = replace(self.config,fit_top_y=370)
        result = self.scan(self.make_mask(),config=narrow)
        self.assertFalse(result['valid'])
        self.assertLess(result['confidence'],.1)
        # A current, complete pair takes precedence over a displaced old track.
        old = self.scan(self.make_mask(offset_px=30))
        current = self.scan(self.make_mask(),previous=old['rows'],age=.1)
        self.assertTrue(current['valid'])
        self.assertAlmostEqual(current['rows'][0]['center_x'],160,delta=1)

    def test_out_of_frame_prediction_is_not_clamped_into_evidence(self):
        mask=np.zeros((400,320),np.uint8)
        for y,centre in ((390,160),(379,190),(368,220)):
            for x in (centre-60,centre+60):
                mask[y,x-3:x+4]=255
        result=self.scan(mask)
        row_by_y={r['y']:r for r in result['rows']}
        self.assertEqual(row_by_y[357]['source'],'predicted')
        self.assertEqual(row_by_y[346]['source'],'missing')
        self.assertIsNone(row_by_y[346]['center_x'])
        self.assertFalse(result['valid'])

    def test_process_output_and_loss_contract(self):
        d = self.detector
        d.M = np.eye(3)
        d.M_inv = np.eye(3)
        d.startup_force_simple_bottom = False
        d.bottom_lock_enable = False
        d.red_detect_enable = False
        d.photometric_mode = 'legacy'
        frame = np.full((400,320,3),255,np.uint8)
        for y in range(400):
            z=d.z_cm_at(y)
            x_cm=math.tan(math.radians(12))*(z-d.z_cm_at(330))
            centre=d.center_x+x_cm/d.cm_per_px_at(y)
            half=35/d.cm_per_px_at(y)/2
            for x in (int(round(centre-half)),int(round(centre+half))):
                if 4 <= x < 316:
                    frame[y,x-4:x+5]=0
        output = d.process(frame,dt=.1)
        self.assertEqual(len(output),5)
        dev,heading,confidence,vis,debug=output
        self.assertEqual(vis.shape,(400,320,3))
        self.assertTrue(debug['measurement_valid'],debug['heading_control_reject_reason'])
        self.assertTrue(debug['heading_control_valid'])
        self.assertAlmostEqual(heading,debug['angle_err_deg'])
        self.assertAlmostEqual(heading,debug['heading_control_deg'])
        self.assertAlmostEqual(heading,-debug['heading_right_deg'])
        self.assertAlmostEqual(heading,-12,delta=2)
        self.assertGreater(confidence,0)
        sampled=debug['heading_rows'][0]
        self.assertEqual(sampled['source'],'paired')
        np.testing.assert_array_equal(vis[sampled['y'],int(round(sampled['left_x']))],
                                      np.array([255,255,0],np.uint8))
        blank=np.full_like(frame,255)
        for _ in range(4):
            lost=d.process(blank,dt=.1)
            self.assertFalse(lost[-1]['measurement_valid'])
            self.assertFalse(lost[-1]['heading_control_valid'])
            self.assertEqual(lost[-1]['angle_err_deg'],0)
            self.assertEqual(lost[2],0)


if __name__ == '__main__':
    unittest.main()
