import sys, unittest, importlib.util
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'new_vision/jetson'))
import masked_ground as fast
from photometric_thresholds import max_channel, Photometry
spec=importlib.util.spec_from_file_location('mask_reference',Path(__file__).parent/'fixtures/masked_ground_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

class CPUOptimizationTests(unittest.TestCase):
    def test_exact_channel_and_lut(self):
        rng=np.random.default_rng(42)
        for shape in [(720,1280,3),(47,61,3),(47,61),(7,9,1),(7,9,4),(12,)]:
            im=rng.integers(0,256,shape,dtype=np.uint8)
            if len(shape)==3 and shape[2]==3:
                np.testing.assert_array_equal(max_channel(im),np.max(im,axis=2))
                np.testing.assert_array_equal(max_channel(im),np.maximum(np.max(im,axis=2),cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)))
            for std in (0.,.01,4.,30.,100.):
                p=Photometry(42.,std,154.,20.,'legacy')
                lut=np.clip(np.rint((np.arange(256,dtype=np.float32)-p.mean)*p.match_scale+p.reference_mean),0,255).astype(np.uint8)
                np.testing.assert_array_equal(p.match_image(im),lut[im])
    def test_masked_operations_exact_including_mutation(self):
        rng=np.random.default_rng(8)
        for shape in [(400,623),(466,623),(37,53),(3,5)]:
            im=rng.integers(0,256,shape,dtype=np.uint8);mask=rng.random(shape)>.15
            for i in range(3):
                if i==1:mask[:shape[0]//2]=False
                if i==2:mask[:]=True
                np.testing.assert_array_equal(fast.gaussian(im,mask,(31,31)),reference.gaussian(im,mask,(31,31)))
                np.testing.assert_array_equal(fast.clahe(im,mask),reference.clahe(im,mask))
                k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                for op in (cv2.MORPH_BLACKHAT,cv2.MORPH_CLOSE,cv2.MORPH_OPEN):
                    np.testing.assert_array_equal(fast.morphology(im,op,k,mask),reference.morphology(im,op,k,mask))
        self.assertLessEqual(fast._gaussian_weights.cache_info().currsize,4)
