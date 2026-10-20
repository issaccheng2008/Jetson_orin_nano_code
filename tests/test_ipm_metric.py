"""Regression tests for metric IPM, independently projected ground geometry."""
import sys, unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from line_detector_v1_warp import LineDetector

def project(d, points):
    x,z=np.asarray(points).T
    z=(z-d.z_b)/d.z_a
    cp,sp=np.cos(d.cam_pitch),np.sin(d.cam_pitch)
    depth=d.cam_height*sp+z*cp
    return np.column_stack((d.fx_px*x/depth+d.cx_px,
        d.fy_px*(d.cam_height*cp-z*sp)/depth+d.cy_px))

def bird(d, points):
    return cv2.perspectiveTransform(project(d,points)[None].astype(np.float64),d.M)[0]

class MetricIPMTests(unittest.TestCase):
    def test_near_extension_preserves_rows_and_reaches_sensor_bottom(self):
        d = LineDetector()
        self.assertEqual((d.bird_w, d.bird_h), (623, 466))
        self.assertAlmostEqual(d.z_cm_at(399), d._to_true_z(20), places=8)
        self.assertLess(d.z_cm_at(d.bird_h - 1), 9.)
        pixel = d.M_inv @ np.array([d.center_x, d.bird_h - 1, 1.])
        self.assertLessEqual(pixel[1] / pixel[2], d.cam_h - 1)
        self.assertGreater(pixel[1] / pixel[2], d.cam_h - 3)
        shape = d.ground_valid_mask.shape
        d.set_heading_regions_cm([[20,22],[24,26]])
        for pitch in (40, 50):
            d.set_camera_pitch_deg(pitch)
            self.assertEqual(d.ground_valid_mask.shape, shape)
            self.assertAlmostEqual(d.z_cm_at(399), d._to_true_z(20), places=8)

    def test_square_ground_scale_near_and_far_and_pitch(self):
        for pitch in (35,45,55):
            d=LineDetector(cam_pitch_deg=pitch)
            for z in (25,45,75):
                q=bird(d,[[-5,z],[5,z],[5,z+10],[-5,z+10]])
                edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
                np.testing.assert_allclose(edges,10/d.cm_per_px,rtol=1e-9)
                self.assertAlmostEqual(float(np.dot(q[1]-q[0],q[2]-q[1])),0,places=6)
    def test_circle_curvature_constant_near_and_far(self):
        d=LineDetector()
        # Left bend radius 77.5cm, visible arc extending through near/far depths.
        t=np.linspace(.05,1.05,120)
        points=np.column_stack((-77.5+77.5*np.cos(t),77.5*np.sin(t)))
        q=bird(d,points)*d.cm_per_px
        for section in (q[:40],q[40:80],q[80:]):
            x,y=section.T
            a,b,c=np.linalg.lstsq(np.column_stack((2*x,2*y,np.ones(len(x)))),x*x+y*y,rcond=None)[0]
            self.assertAlmostEqual(np.sqrt(c+a*a+b*b),77.5,places=7)
    def test_lut_matches_metric_raster_and_pitch_rebuild(self):
        d=LineDetector()
        for pitch in (45,40,50):
            d.set_camera_pitch_deg(pitch)
            np.testing.assert_allclose(d._lut_cm_per_px,d.cm_per_px,rtol=1e-10)
            np.testing.assert_allclose(-np.diff(d._lut_z_cm),d.cm_per_px,rtol=1e-10)
            self.assertEqual(d.ground_valid_mask.shape,(d.bird_h,d.bird_w))
    def test_blank_frame_has_no_padding_lines(self):
        d=LineDetector()
        debug=d.process(np.full((720,1280,3),180,np.uint8),dt=.1)[-1]
        self.assertFalse(debug['measurement_valid'])
        self.assertEqual(np.count_nonzero(debug['binary']),0)
    def test_migrated_bands_keep_physical_distances(self):
        d=LineDetector()
        for old,new in ((350,d.band_low_y0),(399,d.band_low_y1),(300,d.band_mid_y0),(349,d.band_mid_y1)):
            self.assertLessEqual(abs(d.z_cm_at(new)-d._legacy_row_z[old]),d.cm_per_px)

if __name__=='__main__': unittest.main()
