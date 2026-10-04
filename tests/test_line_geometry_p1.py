"""P1 lane geometry, finite preview and observation-time contracts."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_vision' / 'jetson'))
from line_detector_v1_warp import LineDetector, time_constant_ema


class GeometryP1Tests(unittest.TestCase):
    def detector(self, **kwargs):
        d = LineDetector(z_calib=(1.0, 0.0), lane_width_cm=35.0, **kwargs)
        d.startup_force_simple_bottom = False
        d.bottom_lock_enable = False
        d.robust_enable = False
        return d

    def band(self, d, name='low', offset=0.0, slope=0.0, conf=0.9, paired=True, widths=None):
        first = 350 if name == 'low' else 300
        ys = [first + i * 2 for i in range(8)]
        xs = [d.center_x + offset + slope * (y - first - 7) for y in ys]
        widths = widths or [140.0] * len(ys)
        return dict(band_name=name, ys_list=ys, centers_list=xs,
                    modes_list=[2 if paired else 1] * len(ys), widths_list=widths,
                    center_px=float(np.median(xs)),
                    center_cm=float(np.median([d._px_to_ground_cm(x, y)[0] for x, y in zip(xs, ys)])),
                    dist_cm=float(np.median([d.z_cm_at(y) for y in ys])),
                    lane_width_px=float(np.median(widths)), weight=0.65 if name=='low' else 0.35,
                    conf=conf, pair_ratio=1.0 if paired else 0.0,
                    single_ratio=0.0 if paired else 1.0,
                    left_seen=paired, right_seen=True, angle=math.degrees(math.atan(slope*d._asp)))

    def process(self, d, bands, dt=0.1, lock=None):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        with patch.object(d, '_detect_two_band_lanes', return_value=bands):
            if lock is None:
                return d.process(frame, dt=dt)[-1]
            with patch.object(d, '_detect_bottom_center_lock', return_value=lock):
                return d.process(frame, dt=dt)[-1]

    def lock(self, d, valid, offset=80.0):
        return dict(valid=valid, quality=0.9 if valid else 0.0,
                    pair_ratio=0.8, center_err_px=offset, center_px=d.center_x+offset,
                    centers_list=[d.center_x+offset]*10, center_ys=list(range(350,370,2)),
                    width_cv=0.0, fit_rmse_px=0.0)

    def test_large_offset_is_not_bad_bottom_geometry(self):
        d = self.detector()
        d.bottom_lock_enable = True
        gray=np.zeros((400,320),np.uint8)
        bgr=np.zeros((400,320,3),np.uint8)
        # A clear 140px paired lane shifted +50px is trustworthy despite old 24px gate.
        gray[:,137:144]=255
        gray[:,277:284]=255
        lock=d._detect_bottom_center_lock(gray,bgr,25,False)
        self.assertTrue(lock['valid'])
        self.assertGreater(lock['center_err_px'],24)
        self.assertGreater(lock['quality'],0.8)

    def test_missing_rows_inconsistent_width_and_fit_residual_are_rejected(self):
        d=self.detector(); d.bottom_lock_enable=True
        gray=np.zeros((400,320),np.uint8); bgr=np.zeros((400,320,3),np.uint8)
        gray[350:352,87:94]=255; gray[350:352,227:234]=255
        lock=d._detect_bottom_center_lock(gray,bgr,25,False)
        self.assertFalse(lock['valid']); self.assertEqual(lock['quality'],0)
        for widths, xs in [([70,210]*4,[160]*8),([140]*8,[120,200]*4)]:
            valid,q,cv,rmse=d._paired_geometry(list(range(350,366,2)),xs,widths,140)
            self.assertFalse(valid); self.assertEqual(q,0)

    def test_disabled_and_untrusted_lock_cannot_fuse_or_supply_heading(self):
        for enabled, valid in [(False,True),(True,False)]:
            d=self.detector(); d.bottom_lock_enable=enabled
            band=self.band(d,offset=10)
            dbg=self.process(d,[band],lock=self.lock(d,valid))
            self.assertEqual(dbg['bottom_lock_weight'],0)
            self.assertFalse(dbg['curve_mode'])
            self.assertAlmostEqual(dbg['base_err_px'],10)
            self.assertAlmostEqual(dbg['heading_deg'],0)
        d=self.detector(); d.bottom_lock_enable=True
        dbg=self.process(d,[self.band(d,offset=10)],lock=self.lock(d,True,offset=12))
        self.assertGreater(dbg['bottom_lock_weight'],0)

    def test_sparse_pairs_cannot_hide_behind_valid_row_denominator(self):
        d=self.detector()
        sparse=self.band(d)
        sparse.update(scan_rows=20, pair_support_ratio=.4)
        self.assertIsNone(d._qualified_band(sparse))
        dbg=self.process(d,[sparse])
        self.assertFalse(dbg['measurement_valid'])
        self.assertFalse(dbg['heading_valid'])

    def test_invalid_near_candidate_does_not_become_valid_through_band_path(self):
        d=self.detector()
        bad=self.band(d,widths=[70,210]*4)
        dbg=self.process(d,[bad],lock=self.lock(d,False))
        self.assertFalse(dbg['measurement_valid'])
        self.assertFalse(dbg['heading_valid'])
        self.assertFalse(dbg['preview_valid'])
        self.assertEqual(d._state['last_lane_center_x'],160)

    def test_preview_is_independent_of_zero_near_and_has_explicit_sign_and_bound(self):
        d=self.detector()
        low=self.band(d,offset=0,slope=0)
        mid=self.band(d,'mid',offset=-30,slope=0)
        dbg=self.process(d,[low,mid])
        self.assertTrue(dbg['measurement_valid'])
        self.assertTrue(dbg['heading_valid']); self.assertTrue(dbg['preview_valid'])
        self.assertAlmostEqual(dbg['near_error_cm'],0)
        self.assertLess(dbg['preview_error_cm'],0)
        self.assertGreater(dbg['heading_deg'],0)
        self.assertGreater(dbg['fused_err_cm'],0)
        self.assertAlmostEqual(dbg['lookahead_z_cm'],low['dist_cm']+.5*(mid['dist_cm']-low['dist_cm']))
        self.assertLessEqual(abs(dbg['preview_error_cm']),5)
        mirror=self.detector()
        flipped=self.process(mirror,[self.band(mirror,offset=0,slope=0),
                                     self.band(mirror,'mid',offset=30,slope=0)])
        self.assertAlmostEqual(flipped['fused_err_cm'],-dbg['fused_err_cm'])

    def test_preview_clamp_does_not_cancel_large_near(self):
        d=self.detector()
        dbg=self.process(d,[self.band(d,offset=60,slope=1.6),
                            self.band(d,'mid',offset=-20,slope=1.6)])
        self.assertTrue(dbg['preview_valid'])
        self.assertEqual(dbg['preview_error_cm'],-5)
        self.assertLess(dbg['fused_err_cm'],0)

    def test_single_edge_requires_recent_measured_width_and_has_no_heading(self):
        d=self.detector()
        single=self.band(d,paired=False,offset=8)
        self.assertFalse(self.process(d,[single])['measurement_valid'])
        self.process(d,[self.band(d)]); self.process(d,[self.band(d)])
        dbg=self.process(d,[single])
        self.assertTrue(dbg['measurement_valid'])
        self.assertLessEqual(dbg['measurement_quality'],.35)
        self.assertFalse(dbg['heading_valid']); self.assertFalse(dbg['preview_valid'])
        self.assertFalse(self.process(d,[single],dt=1.01)['measurement_valid'])

    def test_centimeter_filter_memory_is_not_reinterpreted_by_lateral_scale(self):
        d=self.detector()
        with patch.object(d,'_update_lateral_scale'):
            first=self.process(d,[self.band(d,offset=20)])
            previous_cm=d._state['smoothed_err_cm']
            previous_norm=d._state['smoothed_err']
            d.lateral_scale=1.2
            d._rebuild_err_scale()
            # Same physical measurement, different pixel-to-centimeter scale.
            second=self.process(d,[self.band(d,offset=20/1.2)])
        self.assertAlmostEqual(second['near_error_cm'],first['near_error_cm'])
        self.assertAlmostEqual(second['fused_err_cm'],first['fused_err_cm'])
        self.assertAlmostEqual(d._state['smoothed_err_cm'],previous_cm)
        self.assertNotAlmostEqual(d._state['smoothed_err'],previous_norm)
        self.assertIn('smoothed_err_cm',d.snapshot_tracking_state()['tracking'])

    def test_quality_gates_update_without_changing_time_constant(self):
        values=[]
        for quality in [.4,.9]:
            d=self.detector()
            self.process(d,[self.band(d,offset=0)])
            values.append(self.process(d,[self.band(d,offset=20,conf=quality)])['fused_err_cm'])
        self.assertAlmostEqual(*values)
        d=self.detector(); self.process(d,[self.band(d,offset=20)])
        before=d.snapshot_tracking_state()
        dbg=self.process(d,[self.band(d,offset=-80,conf=.19)])
        self.assertFalse(dbg['measurement_valid']); self.assertEqual(dbg['avg_conf'],0)
        self.assertEqual(d._state['smoothed_err'],before['tracking']['smoothed_err'])
        self.assertEqual(d._state['last_lane_center_x'],before['tracking']['last_lane_center_x'])

    def test_expired_observation_reinitializes_without_old_error(self):
        d=self.detector(); self.process(d,[self.band(d,offset=50)])
        expired=self.process(d,[],dt=.3)
        self.assertTrue(expired['measurement_stale']); self.assertFalse(expired['measurement_valid'])
        resumed=self.process(d,[self.band(d,offset=-30)])
        self.assertTrue(resumed['measurement_valid'])
        self.assertAlmostEqual(resumed['fused_err_cm'],-resumed['near_error_cm'])
        self.assertEqual(resumed['measurement_age_s'],0)

    def test_card_snapshot_restores_tracking_time_without_rewinding_clock(self):
        d=self.detector(); self.process(d,[self.band(d,offset=30)])
        snapshot=d.snapshot_tracking_state()
        for _ in range(4):
            self.process(d,[self.band(d,offset=-30)])
            d.restore_tracking_state(snapshot)
        self.assertAlmostEqual(d._observation_clock_s,.5)
        self.assertEqual(d._state['last_accepted_tracking_time'],snapshot['tracking']['last_accepted_tracking_time'])
        resumed=self.process(d,[self.band(d,offset=-10)])
        self.assertAlmostEqual(resumed['fused_err_cm'],-resumed['near_error_cm'])

    def test_red_gate_events_remain_live_with_invalid_lane_and_frozen_tracking(self):
        d=self.detector()
        snapshot=d.snapshot_tracking_state()
        with patch.object(d,'_detect_red_bar',return_value=(640,500,40)):
            for _ in range(d.red_bar_confirm_frames):
                dbg=self.process(d,[])
                d.restore_tracking_state(snapshot)
        self.assertFalse(dbg['measurement_valid'])
        self.assertTrue(dbg['red_bar_detected'])
        self.assertTrue(dbg['narrow_gate_detected'])
        self.assertEqual(dbg['narrow_gate_dir'],-1)
        self.assertTrue(dbg['narrow_red_visible'])
        self.assertEqual(dbg['ng_exit_z'],64)
        self.assertEqual(d._state['last_accepted_tracking_time'],None)
        self.assertTrue(d._derive_narrow_gate(True,15,0)['narrow_gate_detected'])
        self.assertEqual(d._derive_narrow_gate(True,15,0)['narrow_gate_dir'],1)
        self.assertEqual(d._derive_narrow_gate(False,0,35)['narrow_gate_dir'],1)

    def test_start_line_remains_live_when_lane_is_invalid(self):
        d=self.detector()
        with patch.object(d,'_detect_start_line',return_value=330):
            dbg=self.process(d,[])
        self.assertFalse(dbg['measurement_valid'])
        self.assertGreater(dbg['start_line_z'],0)

    def test_actual_bird_geometry_retains_straight_and_preview_direction(self):
        for slope in [0,.45,-.45]:
            d=self.detector(); d.M=np.eye(3)
            frame=np.full((400,320,3),255,np.uint8)
            ys=np.arange(180,400)
            centers=160+slope*(ys-357)
            for side in [-70,70]:
                pts=np.column_stack((centers+side,ys)).astype(np.int32)
                cv2.polylines(frame,[pts],False,(0,0,0),7)
            dbg=d.process(frame,dt=.1)[-1]
            self.assertTrue(dbg['measurement_valid'],dbg)
            self.assertTrue(dbg['heading_valid'])
            if slope:
                self.assertTrue(dbg['preview_valid'])
                self.assertGreater(dbg['fused_err_cm']*slope,0)
            else:
                self.assertLess(abs(dbg['fused_err_cm']),.1)

    def test_fixed_tau_is_invariant_to_six_fourteen_and_twenty_hz(self):
        values=[]
        for hz in [6,14,20]:
            x=0
            for _ in range(hz):
                x=time_constant_ema(x,1,1/hz,.18)
            values.append(x)
        np.testing.assert_allclose(values,1-math.exp(-1/.18),rtol=1e-12)
        self.assertEqual(time_constant_ema(.5,1,0,.18),.5)
        for dt in [float('nan'),-.1]:
            with self.assertRaises(ValueError): time_constant_ema(0,1,dt,.18)

    def test_config_uses_seconds_and_rejects_invalid_time_scale(self):
        d=self.detector(config={'vision':{'p1':{'filter_tau_s':.3,'preview_max_cm':2}}})
        self.assertEqual(d.filter_tau_s,.3); self.assertEqual(d.preview_max_cm,2)
        with self.assertRaises(ValueError): self.detector(config={'vision':{'p1':{'filter_tau_s':0}}})

if __name__=='__main__':
    unittest.main()
