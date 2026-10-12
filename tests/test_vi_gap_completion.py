"""VI baseline gap repair keeps synthetic support separate from real edges."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from lane_gap_completion import complete_lane_gaps
from line_detector_v1_warp import LineDetector


class ViGapTests(unittest.TestCase):
    def test_aligned_fragments_join_but_invalid_ground_does_not(self):
        mask=np.zeros((150,150),np.uint8)
        mask[10:60,70:77]=255;mask[75:140,70:77]=255
        repaired,added,info=complete_lane_gaps(mask,max_gap_px=25)
        self.assertEqual(info['lane_gap_bridges'],1)
        self.assertTrue(added.any())
        self.assertEqual(cv2.connectedComponents(repaired)[0],2)
        np.testing.assert_array_equal(repaired,mask|added)
        valid=np.ones_like(mask,dtype=bool);valid[60:75]=False
        self.assertEqual(complete_lane_gaps(mask,valid_mask=valid,max_gap_px=25)[2]['lane_gap_bridges'],0)

    def test_three_fragments_and_blank_are_not_extended_without_endpoints(self):
        mask=np.zeros((240,150),np.uint8)
        for a,b in ((10,70),(85,140),(155,225)):mask[a:b,70:77]=255
        repaired,added,info=complete_lane_gaps(mask,max_gap_px=25)
        self.assertEqual(info['lane_gap_bridges'],2)
        self.assertEqual(cv2.connectedComponents(repaired)[0],2)
        self.assertFalse(repaired[:9].any())
        self.assertFalse(repaired[227:].any())
        self.assertFalse(complete_lane_gaps(np.zeros_like(mask))[0].any())

    def test_side_by_side_boundaries_and_distant_fragments_do_not_join(self):
        mask=np.zeros((180,180),np.uint8)
        mask[10:170,45:52]=255;mask[10:170,100:107]=255
        self.assertEqual(complete_lane_gaps(mask,max_gap_px=80)[2]['lane_gap_bridges'],0)
        mask[:]=0;mask[10:50,70:77]=255;mask[110:170,70:77]=255
        self.assertFalse(complete_lane_gaps(mask,max_gap_px=25)[1].any())

    def test_actual_detector_separates_bridge_pixels_and_real_tracking_samples(self):
        d=LineDetector();d.M=np.eye(3);d.M_inv=np.eye(3)
        d.ground_valid_mask[:]=True;d.red_detect_enable=False
        d.startup_force_simple_bottom=False;d.bottom_lock_enable=False
        d.preprocess_mode='contrast';d.photometric_mode='raw';d.lane_gap_fill=True
        d.single_boundary_tracking=False
        frame=np.full((d.bird_h,d.bird_w,3),210,np.uint8)
        for side in (-1,1):
            points=np.array([(round(d.center_x+side*17.5/d.cm_per_px_at(y)),y)
                             for y in range(d.bird_h)],np.int32)
            cv2.polylines(frame,[points],False,(10,10,10),10)
        frame[340:365]=210
        d.lane_gap_fill=False
        before=d.process(frame,dt=.1,frame_token=1)[-1]
        self.assertFalse(before['measurement_valid'])
        d.reset_state();d.lane_gap_fill=True
        before_scale=d.lateral_scale
        debug=d.process(frame,dt=.1,frame_token=2)[-1]
        self.assertTrue(debug['measurement_valid'])
        self.assertGreater(debug['lane_gap_bridges'],0)
        self.assertTrue(debug['binary_gap_added'].any())
        self.assertFalse(np.any((debug['binary_observed']>0)&(debug['binary_gap_added']>0)))
        self.assertTrue(np.all(debug['binary'][debug['binary_gap_added']>0]>0))
        for row in debug['heading_rows']:
            if row['source']=='paired':
                self.assertGreater(debug['binary_observed'][row['y'],round(row['left_x'])],0)
                self.assertGreater(debug['binary_observed'][row['y'],round(row['right_x'])],0)
        for side in ('left','right'):
            for y,x in debug[side+'_points']:
                self.assertGreater(debug['binary_observed'][y,round(x)],0)
        self.assertEqual(d.lateral_scale,before_scale)


if __name__=='__main__':unittest.main()
