"""Actual image fallback and endpoint completion regression checks."""
import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from lane_gap_completion import complete_lane_gaps
from visible_lane_direction import visible_lane_direction
from line_detector_v1_warp import LineDetector
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController

class RecoveryTests(unittest.TestCase):
    def test_gap_requires_two_aligned_real_fragments(self):
        mask=np.zeros((150,150),np.uint8)
        mask[10:60,70:77]=255; mask[75:140,70:77]=255
        repaired,added,info=complete_lane_gaps(mask,max_gap_px=25)
        self.assertEqual(info['lane_gap_bridges'],1)
        self.assertGreater(np.count_nonzero(added),0)
        self.assertEqual(cv2.connectedComponents(repaired)[0],2)
        valid=np.ones_like(mask);valid[60:75]=0
        self.assertEqual(complete_lane_gaps(mask,valid_mask=valid,max_gap_px=25)[2]['lane_gap_bridges'],0)
        self.assertFalse(complete_lane_gaps(np.zeros_like(mask))[0].any())

    def test_current_direction_has_correct_sign_and_no_blank_history(self):
        for shift in (-30,30):
            mask=np.zeros((160,160),np.uint8)
            cv2.line(mask,(80+shift,30),(80,150),255,7)
            result=visible_lane_direction(mask,lambda x,y:(x*.2,(160-y)*.2))
            self.assertLess(result['heading_deg']*shift,0)
        self.assertIsNone(visible_lane_direction(np.zeros_like(mask),lambda x,y:(x,y)))

    def test_unknown_side_reaches_controller_without_fake_position(self):
        d=LineDetector();d.set_lane_observation_mode('continuous')
        d.M=np.eye(3);d.M_inv=np.eye(3);d.ground_valid_mask[:]=True
        d.startup_force_simple_bottom=False;d.bottom_lock_enable=False
        d.red_detect_enable=False;d.preprocess_mode='contrast';d.photometric_mode='raw'
        frame=np.full((d.bird_h,d.bird_w,3),170,np.uint8)
        cv2.line(frame,(d.center_x+25,240),(d.center_x+60,d.bird_h-1),(25,25,25),9)
        debug=d.process(frame,dt=.1,frame_token=1)[-1]
        self.assertTrue(debug['measurement_valid'],debug['reject_reason'])
        self.assertEqual(debug['tracking_mode'],'VISIBLE_ONLY')
        self.assertFalse(debug['position_valid']);self.assertIsNone(debug['near_error_cm'])
        controller=HeadingSteeringController(SteeringController())
        controller.command(debug,debug['measurement_quality'],.1)
        self.assertNotIn(controller.diagnostics['steering_reason'],('loss_history_turn','loss_default_left'))
        repeated=d.process(frame,dt=.1,frame_token=1)[-1]
        self.assertFalse(repeated['measurement_valid'])
        blank=d.process(np.full_like(frame,170),dt=.1,frame_token=2)[-1]
        self.assertFalse(blank['measurement_valid'])
