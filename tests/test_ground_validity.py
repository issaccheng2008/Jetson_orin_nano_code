"""Unobserved IPM corners must not influence observed-ground extraction."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
from line_detector_v1_warp import LineDetector
from line_preprocess import extract_lane_candidates, sampled_otsu_threshold
from masked_ground import gaussian, morphology

class GroundValidityTests(unittest.TestCase):
    def setUp(self):
        self.detector = LineDetector()
        self.valid = self.detector.ground_valid_mask
        self.gray = np.full(self.valid.shape, 160, np.uint8)
        cv2.line(self.gray, (250,380), (205,50), 35, 8)
        cv2.line(self.gray, (400,380), (355,50), 35, 8)

    def test_default_metric_dimensions_and_scale(self):
        self.assertEqual((self.detector.bird_w,self.detector.bird_h),(623,400))
        self.assertFalse(self.valid[-1,0])
        self.assertFalse(self.valid[-1,-1])
        self.assertTrue(self.valid[-1,self.detector.center_x])
        np.testing.assert_allclose(self.detector._lut_cm_per_px,self.detector.cm_per_px)
        np.testing.assert_allclose(-np.diff(self.detector._lut_z_cm),self.detector.cm_per_px)

    def test_padding_colour_never_changes_threshold_statistics_or_candidates(self):
        rng=np.random.default_rng(7)
        for mode in ('legacy','contrast','canny'):
            baseline=None
            for fill in (0,255,None):
                with self.subTest(mode=mode,fill=fill):
                    frame=self.gray.copy()
                    frame[~self.valid]=fill if fill is not None else rng.integers(0,256,np.count_nonzero(~self.valid),dtype=np.uint8)
                    response,mask,threshold,diag=extract_lane_candidates(frame,mode,valid_mask=self.valid)
                    result=(response,mask,threshold,diag['preprocess_gray_mean'],diag['preprocess_gray_std'])
                    self.assertEqual(diag['preprocess_valid_pixels'],int(self.valid.sum()))
                    self.assertAlmostEqual(diag['preprocess_gray_mean'],self.gray[self.valid].mean())
                    self.assertAlmostEqual(diag['preprocess_gray_std'],self.gray[self.valid].std())
                    self.assertFalse(mask[~self.valid].any())
                    self.assertGreater(np.count_nonzero(mask),100)
                    if baseline is not None:
                        np.testing.assert_array_equal(result[0],baseline[0])
                        np.testing.assert_array_equal(result[1],baseline[1])
                        self.assertEqual(result[2:],baseline[2:])
                    baseline=result

    def test_blank_ground_produces_no_corner_edges(self):
        for mode in ('legacy','contrast','canny'):
            frame=np.where(self.valid,160,0).astype(np.uint8)
            self.assertEqual(np.count_nonzero(extract_lane_candidates(frame,mode,valid_mask=self.valid)[1]),0,mode)

    def test_full_mask_keeps_existing_preprocess(self):
        for mode in ('legacy','contrast','canny'):
            old=extract_lane_candidates(self.gray,mode)
            new=extract_lane_candidates(self.gray,mode,valid_mask=np.ones_like(self.valid))
            np.testing.assert_array_equal(old[0],new[0])
            np.testing.assert_array_equal(old[1],new[1])
            self.assertEqual(old[2],new[2])

    def test_local_mean_and_morphology_ignore_zero_padding(self):
        image=np.where(self.valid,137,0).astype(np.uint8)
        blurred=gaussian(image,self.valid,(31,31))
        np.testing.assert_allclose(blurred[self.valid],137,atol=1e-3)
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
        self.assertEqual(morphology(image,cv2.MORPH_BLACKHAT,k,self.valid).max(),0)

    def test_empty_and_invalid_masks(self):
        for mode in ('legacy','contrast','canny'):
            result=extract_lane_candidates(self.gray,mode,valid_mask=np.zeros_like(self.valid))
            self.assertFalse(result[1].any())
            self.assertEqual(result[3]['preprocess_valid_pixels'],0)
        with self.assertRaises(ValueError):
            extract_lane_candidates(self.gray,valid_mask=np.ones((10,10),bool))

    def test_output_images_keep_invalid_corners_black(self):
        d=self.detector
        result=d.process(np.full((720,1280,3),180,np.uint8),dt=.1)
        for key in ('bird','bird_color','binary','binary_raw'):
            self.assertFalse(result[-1][key][~self.valid].any(),key)
        self.assertFalse(result[3][~self.valid].any())
        self.assertFalse(result[-1]['measurement_valid'])

    def test_centroid_fallback_and_dark_runs_cannot_use_padding(self):
        d=self.detector
        image=np.where(self.valid,150,0).astype(np.uint8)
        row=390
        self.assertIsNone(d._centroid_pair_center(image,row,d.center_x,200,0,d.bird_w-1))
        self.assertEqual(d._collect_track_runs_on_row(image,row,0,d.bird_w-1,30,True),[])

    def test_otsu_ignores_padding(self):
        a=self.gray.copy();b=self.gray.copy();a[~self.valid]=0;b[~self.valid]=255
        self.assertEqual(sampled_otsu_threshold(a,self.valid),sampled_otsu_threshold(b,self.valid))

if __name__=='__main__':unittest.main()
