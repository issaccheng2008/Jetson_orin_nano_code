import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new_vision/jetson'))
from straight_road_guard import measure_near_road, StraightRoadGuard


class StraightRoadTests(unittest.TestCase):
    def mask(self, left, right=None):
        image = np.zeros((300,600), np.uint8)
        z = np.arange(300)*.15
        for curve in (left,right):
            if curve is not None:
                points = np.array([(round(300+curve(v)/.15), y) for y,v in enumerate(z)],np.int32)
                cv2.polylines(image,[points],False,255,10)
        return image

    def measure(self, image):
        return measure_near_road(image,np.ones_like(image,dtype=bool),
                                lambda x,y: ((x-300)*.15,y*.15))

    def test_parallel_straight_with_robot_yaw_and_lateral_offset(self):
        for k in (0., .3, -.5):
            half = 17.5*math.hypot(1,k)
            result = self.measure(self.mask(lambda z: 2+k*z-half,lambda z: 2+k*z+half))
            self.assertEqual(result['straight_evidence'],'STRAIGHT',result)
            self.assertAlmostEqual(result['straight_width_cm'],35.,delta=.3)

    def test_course_circles_are_curves_in_both_directions(self):
        for sign in (-1,1):
            image = self.mask(lambda z: sign*(77-math.sqrt(59.5**2-z*z)),
                              lambda z: sign*(77-math.sqrt(94.5**2-z*z)))
            result = self.measure(image)
            self.assertEqual(result['straight_evidence'],'CURVE',result)

    def test_single_straight_blank_card_horizontal_and_wrong_width_cannot_confirm(self):
        masks = [self.mask(lambda z:17.5),np.zeros((300,600),np.uint8),
                 self.mask(lambda z:-7,lambda z:7)]
        card = np.zeros((300,600),np.uint8)
        cv2.rectangle(card,(220,130),(380,190),255,8)
        masks.append(card)
        horizontal = np.zeros_like(card)
        cv2.line(horizontal,(0,150),(599,150),255,10)
        masks.append(horizontal)
        for mask in masks:
            self.assertEqual(self.measure(mask)['straight_evidence'],'UNKNOWN')

    def test_unobserved_ground_and_fragmented_or_noisy_edges_reject(self):
        image=self.mask(lambda z:-17.5,lambda z:17.5)
        result=measure_near_road(image,np.zeros_like(image,dtype=bool),
            lambda x,y:((x-300)*.15,y*.15))
        self.assertEqual(result['straight_evidence'],'UNKNOWN')
        image[160:205]=0
        self.assertEqual(self.measure(image)['straight_evidence'],'UNKNOWN')
        noisy=self.mask(lambda z:-17.5+3*math.sin(z*2),lambda z:17.5+3*math.sin(z*2))
        self.assertEqual(self.measure(noisy)['straight_evidence'],'UNKNOWN')

    def test_confirm_exit_unknown_hold_and_duplicate_frames(self):
        guard=StraightRoadGuard()
        for i in range(11):
            guard.update('STRAIGHT',i*.1,i)
        self.assertEqual(guard.state,'STRAIGHT')
        self.assertEqual(guard.limit(.5),.3)
        guard.update('CURVE',1.1,11)
        guard.update('STRAIGHT',1.2,12)  # transient does not flip state
        self.assertEqual(guard.state,'STRAIGHT')
        for i in range(13,20):
            guard.update('CURVE',i*.1,i)
        self.assertEqual(guard.state,'CURVE')
        self.assertEqual(guard.limit(.5),.5)
        guard=StraightRoadGuard()
        for i in range(11): guard.update('STRAIGHT',i*.1,i)
        guard.update('UNKNOWN',2.,20)
        self.assertEqual(guard.state,'STRAIGHT')
        guard.update('UNKNOWN',4.1,21)
        self.assertEqual(guard.state,'UNKNOWN')

    def test_duplicate_cannot_establish_straight_or_extend_support(self):
        guard=StraightRoadGuard()
        for i in range(15): guard.update('STRAIGHT',i*.1,1)
        self.assertEqual(guard.state,'UNKNOWN')
        guard=StraightRoadGuard()
        for i in range(11):guard.update('STRAIGHT',i*.1,i)
        guard.update('STRAIGHT',4.1,10)
        self.assertEqual(guard.state,'UNKNOWN')

    def test_first_frame_after_long_break_must_reconfirm(self):
        guard=StraightRoadGuard()
        for i in range(11):guard.update('STRAIGHT',i*.1,i)
        self.assertEqual(guard.state,'STRAIGHT')
        guard.update('STRAIGHT',5.,50)
        self.assertEqual(guard.state,'UNKNOWN')
        for i in range(51,61):guard.update('STRAIGHT',i*.1,i)
        self.assertEqual(guard.state,'STRAIGHT')

    def test_angle_table_straight_is_not_first_left_and_final_window_cannot_override(self):
        from steering_command_window import SteeringCommandWindow
        window=SteeringCommandWindow(.5,'lower')
        guard=StraightRoadGuard(straight_wz=.1,left_levels=(.1,.25,.5),right_levels=(.1,.2))
        guard.state='STRAIGHT'
        self.assertEqual(guard.allowed,(-.1,.1,.25))
        for i in range(15):
            vx,wz=window.update(.2,.5,i*.1,valid=False)
            self.assertLessEqual(guard.limit(wz),.25)
        self.assertEqual(guard.limit(-.5),-.1)
        self.assertEqual(guard.limit(0.,bypass=True),0.)
        self.assertEqual(guard.limit(.5,bypass=True),.5)
        flipped=StraightRoadGuard(yaw_sign=-1)
        flipped.state='STRAIGHT'
        self.assertEqual(flipped.limit(-.5),-.3)

    def test_detector_to_actual_entry_limits_lost_fallback_after_window(self):
        from line_detector_v1_warp import LineDetector
        import run_policy_vision
        detector=LineDetector()
        bird=np.full((detector.bird_h,detector.bird_w,3),210,np.uint8)
        for side in (-1,1):
            points=[]
            for y in range(detector.bird_h):
                x=round(detector.center_x+side*17.5/detector.cm_per_px_at(y))
                points.append((x,y))
            cv2.polylines(bird,[np.asarray(points,np.int32)],False,(10,10,10),10)
        raw=cv2.warpPerspective(bird,np.linalg.inv(detector.M),(1280,720),borderValue=(210,210,210))
        blank=np.full_like(raw,210)
        cap=Mock(); cap.isOpened.return_value=True
        cap.get.side_effect=lambda key:1280 if key==3 else 720
        clock=[0.]; index=[0]
        def read():
            index[0]+=1; clock[0]=index[0]*.1
            if index[0]>35:
                run_policy_vision.signal.signal.call_args.args[1](None,None)
                return False,None
            frame=(raw if index[0]<21 else blank).copy()
            frame[0,0,0]=index[0]
            return True,frame
        cap.read.side_effect=read
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch('sys.argv',['vision','--headless','--no-camera-async','--no-shape-detect',
                '--no-red-detect','--wz-mode','heading','--wz-step','0.5',
                '--heading-left-wz','0.3','0.4','0.5',
                '--line-preprocess','contrast','--photometric-mode','raw',
                '--straight-confirm-s','0.5','--steering-loss-max-s','0.2',
                '--steering-loss-fallback-wz','0.5','--line-log-dir',tmp]),
            patch.object(run_policy_vision.signal,'signal'),
            patch.object(run_policy_vision,'ConnectorClient') as client,
            patch.object(run_policy_vision,'LineTelemetry') as telemetry,
            patch('utils.open_camera',return_value=cap),
            patch('line_detector_v1_warp.LineDetector',return_value=detector),
            patch.object(run_policy_vision.time,'monotonic',lambda:clock[0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_policy_vision.main(),0)
        debug_rows=[c.args[0] for c in telemetry.return_value.write.call_args_list]
        self.assertTrue(any(d.get('straight_evidence')=='STRAIGHT' for d in debug_rows))
        limited=[i for i,d in enumerate(debug_rows) if d.get('straight_guard_before_wz')==.5
                 and d.get('straight_road_state')=='STRAIGHT']
        self.assertTrue(limited)
        for i in limited:
            self.assertEqual(client.return_value.publish.call_args_list[i].args[:2],(.2,.3))


if __name__=='__main__':unittest.main()
