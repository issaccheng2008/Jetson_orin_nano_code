"""Ground geometry and real bird mask regressions for sustained single boundaries."""
import math
import csv
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from continuous_lane_heading import trace_heading
from lane_boundary_path import BoundaryPathTracker
from line_detector_v1_warp import LineDetector
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from segment_steering import SegmentSteeringController
import run_policy_vision


class BoundaryPathTests(unittest.TestCase):
    def setUp(self):
        self.sx, self.sz = .25, .20
        self.to_ground = lambda x, y: ((x-160)*self.sx, (390-y)*self.sz)
        self.tracker = BoundaryPathTracker()
        self.time = 0.

    def mask(self, sides=('left', 'right'), shift=0., slope=0., width=30.,
             radius=None):
        image = np.zeros((400, 320), np.uint8)
        for y in range(220, 400):
            z = (390-y)*self.sz
            center = shift + slope*(z-22.)
            if radius is not None:
                center += (z-22.)**2/(2*radius)
            for side in sides:
                x_cm = center + (-width/2 if side == 'left' else width/2)
                x = round(160+x_cm/self.sx)
                if 4 <= x < 316:
                    image[y,x-3:x+4] = 255
        return image

    @staticmethod
    def runs(image, y):
        row = image[y] > 0
        changes = np.diff(np.pad(row.astype(np.int8),(1,1)))
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1))

    def scan(self, image, width=30., side=None, token=None, dt=.1):
        self.time += dt
        paired = trace_heading(image,lambda y:self.runs(image,y),
                               self.to_ground,lambda y:self.sx)
        return self.tracker.update(image,lambda y:self.runs(image,y),self.to_ground,
                                   now_s=self.time,width_cm=width,reference_z_cm=22.,
                                   paired_trace=paired,initial_side=side,frame_token=token)

    def test_current_pair_requirement_rejects_single_before_change(self):
        image=self.mask(sides=('right',))
        trace=trace_heading(image,lambda y:self.runs(image,y),self.to_ground,lambda y:self.sx)
        self.assertFalse(trace['valid'])
        self.assertEqual(trace['paired_rows'],0)

    def test_sustained_right_and_left_with_identity_crossing_middle(self):
        for side,shift in (('right',-21.),('left',21.)):
            with self.subTest(side=side):
                self.tracker=BoundaryPathTracker()
                self.time=0.
                self.scan(self.mask(shift=shift),token='pair')
                paired_width_time=self.tracker.width_measured_s
                result=None
                for frame in range(80):
                    result=self.scan(self.mask(sides=(side,),shift=shift),token=frame)
                    self.assertTrue(result['measurement_valid'],result['reject_reason'])
                    self.assertEqual(result['tracking_mode'],side.upper()+'_ONLY')
                self.assertAlmostEqual(result['near_error_cm'],shift,delta=2.)
                self.assertGreaterEqual(result['observed_'+side+'_points'],6)
                self.assertEqual(result['width_source'],'configured_equal_width')
                self.assertEqual(self.tracker.width_measured_s,paired_width_time)
                self.assertGreater(result['width_age_s'],7.)
                self.assertEqual(result['observed_'+side+'_age_s'],0.)

    def test_normal_offset_and_heading_match_paired_line(self):
        for slope in (-.35,0.,.35):
            with self.subTest(slope=slope):
                pair=self.scan(self.mask(slope=slope),token=('pair',slope))
                self.assertTrue(pair['measurement_valid'],pair['reject_reason'])
                for side in ('left','right'):
                    single=self.scan(self.mask(sides=(side,),slope=slope),token=(side,slope))
                    self.assertTrue(single['measurement_valid'],single['reject_reason'])
                    self.assertAlmostEqual(single['near_error_cm'],pair['near_error_cm'],delta=2.)
                    self.assertAlmostEqual(single['heading_control_deg'],pair['heading_control_deg'],delta=2.)

    def test_cold_ambiguous_explicit_side_and_duplicate_frame(self):
        image=self.mask(sides=('right',))
        cold=self.scan(image,token=1)
        self.assertFalse(cold['measurement_valid'])
        self.assertEqual(cold['reject_reason'],'side_ambiguous')
        self.assertEqual(cold['tracking_mode'],'UNKNOWN')
        known=self.scan(image,side='right',token=2)
        self.assertTrue(known['measurement_valid'],known['reject_reason'])
        repeated=self.scan(image,side='right',token=2)
        self.assertFalse(repeated['measurement_valid'])
        self.assertEqual(repeated['reject_reason'],'duplicate_frame')

    def test_blank_horizontal_blob_and_wrong_width_do_not_fake_path(self):
        self.scan(self.mask(),token='pair')
        blank=np.zeros((400,320),np.uint8)
        for image in (blank,):
            result=self.scan(image,token=id(image))
            self.assertFalse(result['measurement_valid'])
        horizontal=blank.copy()
        cv2.line(horizontal,(15,345),(300,345),255,7)
        result=self.scan(horizontal,token='horizontal')
        self.assertFalse(result['measurement_valid'])
        blob=blank.copy()
        cv2.rectangle(blob,(190,300),(210,315),255,-1)
        result=self.scan(blob,token='blob')
        self.assertFalse(result['measurement_valid'])
        result=self.scan(self.mask(sides=('right',)),width=20.,token='wrong_width')
        self.assertFalse(result['measurement_valid'])

    def test_reappearing_pair_and_unproven_other_side(self):
        for side in ('left','right'):
            with self.subTest(side=side):
                self.tracker=BoundaryPathTracker()
                self.time=0.
                self.scan(self.mask(),token=('both',side))
                single=self.scan(self.mask(sides=(side,)),token=('single',side))
                self.assertTrue(single['measurement_valid'])
                back=self.scan(self.mask(),token=('back',side))
                self.assertEqual(back['tracking_mode'],'BOTH')
                self.assertAlmostEqual(back['near_error_cm'],single['near_error_cm'],delta=2.)
        self.tracker=BoundaryPathTracker()
        self.time=0.
        self.scan(self.mask(),token='initial')
        for i in range(10):
            self.scan(self.mask(sides=('right',)),token=i)
        unknown=self.scan(self.mask(sides=('left',)),token='left_after_right')
        self.assertFalse(unknown['measurement_valid'])

    def test_conflicting_pair_needs_consistent_reacquisition(self):
        self.scan(self.mask(),token='initial')
        self.scan(self.mask(sides=('right',)),token='right')
        first=self.scan(self.mask(shift=12.),token='shifted_pair_1')
        self.assertFalse(first['measurement_valid'])
        self.assertEqual(first['reject_reason'],'pair_path_disagreement')
        second=self.scan(self.mask(shift=12.),token='shifted_pair_2')
        self.assertTrue(second['measurement_valid'],second['reject_reason'])
        self.assertEqual(second['tracking_mode'],'BOTH')

    def test_card_frame_far_from_robot_is_not_single_boundary(self):
        self.scan(self.mask(),token='pair')
        card=np.zeros((400,320),np.uint8)
        cv2.rectangle(card,(105,280),(215,330),255,6)
        result=self.scan(card,token='card')
        self.assertFalse(result['measurement_valid'])

    def test_virtual_target_can_be_outside_image_without_fake_boundary(self):
        mask=np.zeros((400,320),np.uint8)
        for y in range(220,400):
            mask[y,17:24]=255
        result=self.scan(mask,side='right',token='out_of_frame')
        self.assertTrue(result['measurement_valid'],result['reject_reason'])
        self.assertLess(result['near_error_cm'],-40.)
        self.assertEqual(result['observed_left_points'],0)
        self.assertEqual(result['left_points'],[])

    def test_normal_offset_on_curved_centerline(self):
        radius=90.
        mask=np.zeros((400,320),np.uint8)
        for center_z in np.arange(-2.,37.,.04):
            center_x=(center_z-22.)**2/(2*radius)
            k=(center_z-22.)/radius
            normal=np.array([1.,-k])/math.hypot(1.,k)
            for sign in (-1,1):
                bx=center_x+sign*15.*normal[0]
                bz=center_z+sign*15.*normal[1]
                px=round(160+bx/self.sx)
                py=round(390-bz/self.sz)
                if 3<=px<317 and 3<=py<397:
                    mask[py-2:py+3,px-2:px+3]=255
        paired=self.scan(mask,token='curve_pair')
        self.assertTrue(paired['measurement_valid'],paired['reject_reason'])
        for side in ('left','right'):
            # Remove the opposite component while retaining exact measured edge.
            only=mask.copy()
            if side=='left':
                only[:,160:]=0
            else:
                only[:,:160]=0
            single=self.scan(only,token='curve_'+side)
            self.assertTrue(single['measurement_valid'],single['reject_reason'])
            self.assertAlmostEqual(single['near_error_cm'],paired['near_error_cm'],delta=3.)
            # The old paired fit covers a longer asymmetric arc and reads +6.6°;
            # the exact centre tangent at the reference Z is 0°.
            self.assertAlmostEqual(single['heading_control_deg'],0.,delta=4.)
            self.assertAlmostEqual(single['heading_control_deg'],paired['heading_control_deg'],delta=8.)


class DetectorControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.detector=LineDetector(z_calib=(1,0))
        d=self.detector
        d.M=np.eye(3)
        d.M_inv=np.eye(3)
        d.ground_valid_mask[:]=True
        d.startup_force_simple_bottom=False
        d.bottom_lock_enable=False
        d.red_detect_enable=False
        d.preprocess_mode='contrast'
        self.controller=HeadingSteeringController(SteeringController(),
                                                  min_hold_s=.5,position_gain=0.,
                                                  position_recovery_cm=0.)

    def test_entry_rollback_and_cold_side_options(self):
        with patch('sys.argv',['run_policy_vision.py','--no-single-boundary-tracking',
                               '--initial-boundary-side','right']):
            args=run_policy_vision.parse_args()
        self.assertFalse(args.single_boundary_tracking)
        self.assertEqual(args.initial_boundary_side,'right')

    def frame(self, sides=('left','right'), offset=0., slope=0.):
        d=self.detector
        image=np.full((d.bird_h,d.bird_w,3),170,np.uint8)
        for y in range(205,d.bird_h):
            z=d.z_cm_at(y)
            x=offset+slope*(z-25.)
            for side in sides:
                edge=x+(-d.lane_width_true_cm/2 if side=='left' else
                        d.lane_width_true_cm/2)
                px=round(d.center_x+edge/d.cm_per_px_at(y))
                if 4<=px<d.bird_w-4:
                    image[y,px-4:px+5]=30
        return image

    def test_real_detector_single_boundary_reaches_existing_controller(self):
        d=self.detector
        paired=d.process(self.frame(),dt=.1,frame_token=0)[-1]
        self.assertTrue(paired['measurement_valid'],paired['heading_control_reject_reason'])
        paired_scale=d.lateral_scale
        for i in range(1,51):
            debug=d.process(self.frame(('right',),offset=-4.),dt=.1,frame_token=i)[-1]
            self.assertEqual(debug['tracking_mode'],'RIGHT_ONLY',debug['reject_reason'])
            self.assertTrue(debug['measurement_valid'])
            self.assertTrue(debug['heading_control_valid'])
            self.assertEqual(debug['lost_frames'],0)
            command=self.controller.command(debug,debug['measurement_quality'],.1)
            self.assertNotIn(self.controller.diagnostics['steering_reason'],
                             ('loss_history_turn','loss_default_left','geometry_lost_yaw_zero'))
            self.assertGreater(command[0],0.)
        self.assertAlmostEqual(d.lateral_scale,paired_scale,delta=1e-12)
        blank=np.full_like(self.frame(),170)
        lost=d.process(blank,dt=.1,frame_token=99)[-1]
        self.assertFalse(lost['measurement_valid'])

    def test_detector_left_only_remains_fresh_for_five_seconds(self):
        d=self.detector
        self.assertTrue(d.process(self.frame(),dt=.1,frame_token=0)[-1]
                        ['measurement_valid'])
        for i in range(1,51):
            debug=d.process(self.frame(('left',)),dt=.1,frame_token=i)[-1]
            self.assertEqual(debug['tracking_mode'],'LEFT_ONLY',debug['reject_reason'])
            self.assertTrue(debug['measurement_valid'])
            self.assertEqual(debug['observed_left_age_s'],0.)
            self.assertEqual(debug['lost_frames'],0)

    def test_explicit_rollback_restores_paired_only_acceptance(self):
        d=self.detector
        d.single_boundary_tracking=False
        self.assertTrue(d.process(self.frame(),dt=.1,frame_token=0)[-1]
                        ['measurement_valid'])
        single=d.process(self.frame(('right',)),dt=.1,frame_token=1)[-1]
        self.assertFalse(single['measurement_valid'])
        self.assertEqual(single['tracking_mode'],'LOST')

    def test_left_side_segment_fallback_and_duplicate_frame(self):
        d=self.detector
        d.process(self.frame(),dt=.1,frame_token=1)
        left=d.process(self.frame(('left',),offset=3.),dt=.1,frame_token=2)[-1]
        self.assertEqual(left['tracking_mode'],'LEFT_ONLY')
        c=SegmentSteeringController(SteeringController(),min_hold_s=.5,
                                    position_gain=0.,position_recovery_cm=0.)
        command=c.command(left,left['measurement_quality'],.1)
        self.assertGreater(command[0],0.)
        self.assertEqual(c.diagnostics['steering_heading_source'],'left_boundary')
        self.assertNotIn(c.diagnostics['steering_reason'],
                         ('loss_history_turn','loss_default_left','geometry_lost_yaw_zero'))
        repeated=d.process(self.frame(('left',),offset=3.),dt=.1,frame_token=2)[-1]
        self.assertFalse(repeated['measurement_valid'])
        self.assertEqual(repeated['reject_reason'],'duplicate_frame')

    def test_single_edge_position_controls_turn_sign_not_side_name(self):
        for side in ('left','right'):
            for offset,expected_sign in ((-7.,1),(7.,-1)):
                d=LineDetector(z_calib=(1,0))
                d.M=np.eye(3)
                d.ground_valid_mask[:]=True
                d.startup_force_simple_bottom=False
                d.bottom_lock_enable=False
                d.red_detect_enable=False
                d.preprocess_mode='contrast'
                self.detector=d
                d.process(self.frame(),dt=.1,frame_token=0)
                debug=d.process(self.frame((side,),offset=offset),dt=.1,frame_token=1)[-1]
                self.assertTrue(debug['measurement_valid'],debug['reject_reason'])
                c=HeadingSteeringController(SteeringController(),min_hold_s=.5,
                                            position_gain=0.,position_recovery_cm=0.)
                wz=c.command(debug,debug['measurement_quality'],.1)[1]
                # Within the spatial corridor straight is also a valid recovery.
                self.assertGreaterEqual(wz*expected_sign,0.,
                                        (side,offset,c.diagnostics))

    def test_real_entry_receives_single_path_and_preserves_external_stop(self):
        d=self.detector
        camera=Mock()
        camera.isOpened.return_value=True
        camera.get.side_effect=[d.bird_w,d.bird_h]
        counter=[0]
        sent=[]
        def read():
            counter[0]+=1
            if counter[0]>26:
                run_policy_vision.signal.signal.call_args.args[1](None,None)
                return False,None
            sides=('left','right') if counter[0]<=3 else ('right',)
            image=self.frame(sides,offset=-3.)
            image[0,0]=counter[0]  # new camera exposure, not a replayed buffer
            return True,image
        camera.read.side_effect=read
        clock=[0.]
        def tick():
            clock[0]+=.1
            return clock[0]
        with (tempfile.TemporaryDirectory() as tmp,
              patch('sys.argv',['run_policy_vision.py','--no-camera-async',
                   '--headless','--no-shape-detect','--no-record-video',
                   '--no-red-detect','--wz-mode','segments',
                   '--line-preprocess','contrast','--photometric-mode','raw',
                   '--attitude-port','0','--recording-root',tmp]),
              patch.object(run_policy_vision.signal,'signal'),
              patch.object(run_policy_vision,'ConnectorClient') as client,
              patch('utils.open_camera',return_value=camera),
              patch('line_detector_v1_warp.LineDetector',return_value=d),
              patch('camera_controls.apply_camera_controls',return_value={}),
              patch.object(run_policy_vision.time,'monotonic',side_effect=tick),
              contextlib.redirect_stdout(io.StringIO())):
            client.return_value.publish.side_effect=(
                lambda vx,wz,*args,**kw:sent.append((vx,wz)))
            self.assertEqual(run_policy_vision.main(),0)
            rows=[json.loads(line) for path in Path(tmp).rglob('line_frames.jsonl')
                  for line in path.read_text().splitlines()]
            single=[row for row in rows if row['measurement'].get('tracking_mode')=='RIGHT_ONLY']
            self.assertGreater(len(single),10)
            self.assertTrue(all(row['measurement']['measurement_valid'] for row in single))
            self.assertTrue(all(row['measurement']['steering_heading_source']=='right_boundary'
                                for row in single))
        self.assertGreater(counter[0],20)
        self.assertTrue(any(vx>0 for vx,_ in sent))
        self.assertEqual(sent[-1],(0.,0.))

    def test_recorded_pair_replay_ignores_old_binary_view_and_hud(self):
        """The replay input is only the recorded bird image, never its old mask."""
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from replay_sustained_boundary_video import replay
        with tempfile.TemporaryDirectory() as tmp:
            csv_contents=[]
            for right_fill in (0,255):
                path=Path(tmp)/f'pair_{right_fill}.avi'
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),
                                       10.,(960,468))
                self.assertTrue(writer.isOpened())
                for i in range(12):
                    image=np.zeros((468,960,3),np.uint8)
                    sides=('left','right') if i<3 else ('right',)
                    image[:358,:480]=cv2.resize(self.frame(sides), (480,358))
                    image[:358,480:]=right_fill
                    image[358:]=right_fill
                    writer.write(image)
                writer.release()
                output=Path(tmp)/f'out_{right_fill}'
                summary=replay(path,output,bird_height=358)
                self.assertEqual(summary['frames'],12)
                with (output/'frames.csv').open(newline='',encoding='utf-8') as handle:
                    csv_contents.append(list(csv.DictReader(handle)))
            self.assertEqual(csv_contents[0],csv_contents[1])
            self.assertTrue(any(row['valid']=='1' for row in csv_contents[0]))

    def test_entry_repeated_camera_buffer_cannot_refresh_single_edge(self):
        d=self.detector
        pair=self.frame()
        single=self.frame(('right',))
        camera=Mock()
        camera.isOpened.return_value=True
        camera.get.side_effect=[d.bird_w,d.bird_h]
        reads=[0]

        def read():
            reads[0]+=1
            if reads[0]>5:
                run_policy_vision.signal.signal.call_args.args[1](None,None)
                return False,None
            return True,(pair if reads[0]==1 else single)

        camera.read.side_effect=read
        clock=[0.]

        def tick():
            clock[0]+=.1
            return clock[0]

        with (tempfile.TemporaryDirectory() as tmp,
              patch('sys.argv',['run_policy_vision.py','--no-camera-async',
                   '--headless','--no-shape-detect','--no-record-video',
                   '--no-red-detect','--wz-mode','segments',
                   '--line-preprocess','contrast','--photometric-mode','raw',
                   '--attitude-port','0','--recording-root',tmp]),
              patch.object(run_policy_vision.signal,'signal'),
              patch.object(run_policy_vision,'ConnectorClient'),
              patch('utils.open_camera',return_value=camera),
              patch('line_detector_v1_warp.LineDetector',return_value=d),
              patch('camera_controls.apply_camera_controls',return_value={}),
              patch.object(run_policy_vision.time,'monotonic',side_effect=tick),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(run_policy_vision.main(),0)
            rows=[json.loads(line) for path in Path(tmp).rglob('line_frames.jsonl')
                  for line in path.read_text().splitlines()]
        measurements=[row['measurement'] for row in rows]
        self.assertTrue(any(m.get('tracking_mode')=='RIGHT_ONLY' and
                            m.get('measurement_valid') for m in measurements))
        self.assertTrue(any(m.get('reject_reason')=='duplicate_frame' and
                            not m.get('measurement_valid') for m in measurements))


if __name__=='__main__':
    unittest.main()
