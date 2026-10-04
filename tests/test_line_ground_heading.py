"""Ground-space heading is independent of the legacy fixed bird pixel aspect."""
from pathlib import Path
import importlib.util
import math
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
spec=importlib.util.spec_from_file_location('legacy_heading_reference',
       ROOT/'docs/audit_2026-10-04/vision_p1_after_release/line_detector_v1_warp.py')
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


class GroundHeadingTests(unittest.TestCase):
    def detector(self,cls=LineDetector):
        d=cls(z_calib=(1.13233,-2.4862),lane_width_cm=35)
        d.bottom_lock_enable=False
        d.lateral_scale=1
        return d

    def projected_ground_line(self,d,angle_deg,zlo,zhi,intercept_cm=0):
        # Project known true ground X(Z) through the current pinhole and then IPM;
        # the new detector must recover its direction using per-row ground units.
        true_z=np.linspace(zlo,zhi,12)
        true_x=intercept_cm-np.tan(np.radians(angle_deg))*true_z
        model_z=(true_z-d.z_b)/d.z_a
        cp,sp=math.cos(d.cam_pitch),math.sin(d.cam_pitch)
        depth=d.cam_height*sp+model_z*cp
        u=d.cx_px+d.fx_px*true_x/depth
        v=d.cy_px+d.fy_px*(d.cam_height*cp-model_z*sp)/depth
        original=np.stack([u,v,np.ones_like(u)])
        bird=d.M@original
        bx,by=bird[0]/bird[2],bird[1]/bird[2]
        self.assertTrue(np.all((by>=0)&(by<d.bird_h)))
        return dict(observation_paired=True,ys_list=by.tolist(),centers_list=bx.tolist())

    def test_same_ground_angle_projects_and_recovers_over_different_depths(self):
        for angle in [-25,-10,0,10,25]:
            estimates=[]
            for zlo,zhi in [(25,35),(40,55),(60,75)]:
                d=self.detector()
                # Keep the projected line near the image centre at each test depth.
                intercept=math.tan(math.radians(angle))*(zlo+zhi)/2
                r=self.projected_ground_line(d,angle,zlo,zhi,intercept)
                fit=d._fit_ground_control_heading([r],{})
                self.assertTrue(fit['heading_control_valid'],fit)
                self.assertAlmostEqual(fit['heading_control_deg'],angle,delta=.15)
                self.assertAlmostEqual(fit['heading_control_z_span_cm'],zhi-zlo,delta=.01)
                self.assertLess(fit['heading_control_rmse_cm'],.002)
                estimates.append(fit['heading_control_deg'])
            self.assertLess(max(estimates)-min(estimates),.05)

    def test_sign_and_no_fixed_aspect_dependency(self):
        d=self.detector()
        for angle in [-20,20]:
            r=self.projected_ground_line(d,angle,25,45,math.tan(math.radians(angle))*35)
            first=d._fit_ground_control_heading([r],{})
            old_angle=d._fit_trusted_heading([r],{})[0]
            d._asp*=2
            second=d._fit_ground_control_heading([r],{})
            self.assertGreater(first['heading_control_deg']*angle,0)
            self.assertEqual(first['heading_control_deg'],second['heading_control_deg'])
            self.assertNotEqual(old_angle,d._fit_trusted_heading([r],{})[0])

    def test_unpaired_and_disabled_or_invalid_lock_cannot_make_control_heading(self):
        d=self.detector()
        r=self.projected_ground_line(d,20,25,35,10)
        r['observation_paired']=False
        lock=dict(valid=True,center_ys=r['ys_list'],centers_list=r['centers_list'])
        self.assertFalse(d._fit_ground_control_heading([r],lock)['heading_control_valid'])
        d.bottom_lock_enable=True
        lock['valid']=False
        self.assertFalse(d._fit_ground_control_heading([r],lock)['heading_control_valid'])
        lock['valid']=True
        self.assertTrue(d._fit_ground_control_heading([],lock)['heading_control_valid'])

    def test_pixel_quality_gate_and_finite_ground_support_remain_required(self):
        d=self.detector()
        r=dict(observation_paired=True,ys_list=list(range(300,324,2)),
               centers_list=[100,220]*6)
        self.assertFalse(d._fit_ground_control_heading([r],{})['heading_control_valid'])
        good=self.projected_ground_line(d,0,25,35)
        with patch.object(d,'_px_to_ground_cm',return_value=(1,10)):
            fit=d._fit_ground_control_heading([good],{})
            self.assertFalse(fit['heading_control_valid'])
            self.assertEqual(fit['heading_control_z_span_cm'],0)
        with patch.object(d,'_px_to_ground_cm',return_value=(float('nan'),10)):
            self.assertFalse(d._fit_ground_control_heading([good],{})['heading_control_valid'])

    def test_shared_point_dedup_and_legacy_heading_are_unchanged(self):
        d=self.detector();old=self.detector(frozen.LineDetector)
        d.bottom_lock_enable=old.bottom_lock_enable=True
        rows=list(range(300,324,2))
        r=dict(observation_paired=True,ys_list=rows,centers_list=[150+.4*y for y in rows])
        ignored=dict(observation_paired=False,ys_list=rows,centers_list=[0]*len(rows))
        lock=dict(valid=True,center_ys=rows+[326],centers_list=[-999]*len(rows)+[280.4])
        ys,xs=d.trusted_heading_points([r,ignored],lock)
        self.assertEqual(ys,rows+[326])
        self.assertEqual(xs[:-1],r['centers_list'])
        for valid in [False,True]:
            lock['valid']=valid
            self.assertEqual(d._fit_trusted_heading([r,ignored],lock),
                             old._fit_trusted_heading([r,ignored],lock))

    def test_process_keeps_old_outputs_and_does_not_replay_control_heading_on_loss(self):
        d=self.detector();old=self.detector(frozen.LineDetector)
        for item in [d,old]:
            item.M=np.eye(3)
            item.startup_force_simple_bottom=False
        frame=np.full((400,320,3),255,np.uint8)
        ys=np.arange(160,400)
        center=160+.3*(ys-357)
        for side in [-70,70]:
            points=np.column_stack((center+side,ys)).astype(np.int32)
            cv2.polylines(frame,[points],False,(0,0,0),7)
        for _ in range(3):
            before=old.process(frame,dt=.1)
            after=d.process(frame,dt=.1)
            self.assertEqual(before[:3],after[:3])
            for key,value in before[-1].items():
                if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,after[-1][key])
                else:self.assertEqual(value,after[-1][key],key)
        self.assertTrue(after[-1]['heading_control_valid'])
        self.assertEqual(after[-1]['heading_control_source'],'ground_x_z')
        self.assertGreater(after[-1]['heading_control_z_span_cm'],0)
        lost=d.process(np.full(frame.shape,255,np.uint8),dt=.1)[-1]
        self.assertFalse(lost['heading_control_valid'])
        self.assertEqual(lost['heading_control_deg'],0)
        self.assertEqual(lost['heading_control_points'],0)

if __name__=='__main__':unittest.main()
