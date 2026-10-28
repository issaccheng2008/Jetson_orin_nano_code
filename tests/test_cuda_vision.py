"""Device selection and CPU-reference pipeline tests without claiming GPU speed."""
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_vision/jetson'))
import cuda_vision
from line_preprocess import extract_lane_candidates
from photometric_thresholds import measure
from shape_detector import ShapeDetector
from line_detector_v1_warp import LineDetector
from card_presence import find_broken_border_presence


class FakeGpuMat:
    uploads = 0
    downloads = 0

    def __init__(self, data=None):
        self.data = data

    def upload(self, image):
        type(self).uploads += 1
        self.data = image.copy()

    def download(self):
        type(self).downloads += 1
        return self.data.copy()


class FakeCuda:
    def __init__(self, count=1, mismatch=False, gray_mismatch=False,
                 gray_morph_mismatch=False):
        self.count = count
        self.mismatch = mismatch
        self.gray_mismatch = gray_mismatch
        self.gray_morph_mismatch = gray_morph_mismatch
        self.fail_runtime = False
        self.filter_creations = 0

    def getCudaEnabledDeviceCount(self):
        return self.count

    def setDevice(self, device):
        if device != 0:
            raise AssertionError(device)

    def createMorphologyFilter(self, operation, dtype, kernel):
        self.filter_creations += 1
        owner = self
        class Filter:
            def apply(self, source):
                if owner.fail_runtime:
                    raise cv2.error('simulated device failure')
                result = cv2.morphologyEx(source.data, operation, kernel)
                if owner.mismatch:
                    result[0, 0] ^= 255
                if (owner.gray_morph_mismatch and operation != cv2.MORPH_BLACKHAT
                        and np.any(source.data > 1)
                        and np.any((source.data > 1) & (source.data < 255))):
                    result[0, 0] ^= 1
                return FakeGpuMat(result)
        return Filter()

    def cvtColor(self, source, code):
        result = cv2.cvtColor(source.data, code)
        if self.gray_mismatch:
            result[0, 0] ^= 1
        return FakeGpuMat(result)

    def resize(self, source, size, interpolation):
        return FakeGpuMat(cv2.resize(source.data, size, interpolation=interpolation))

    def warpPerspective(self, source, matrix, size, **kwargs):
        return FakeGpuMat(cv2.warpPerspective(source.data, matrix, size, **kwargs))


class CvFacade:
    def __init__(self, **options):
        self.cuda = FakeCuda(**options)
        self.cuda_GpuMat = FakeGpuMat

    def __getattr__(self, name):
        return getattr(cv2, name)


class CudaVisionTests(unittest.TestCase):
    def test_cli_device_is_explicit_and_cpu_remains_default(self):
        import run_policy_vision
        with patch.object(sys, 'argv', ['vision']):
            args = run_policy_vision.parse_args()
        self.assertEqual(args.vision_device, 'cpu')
        self.assertFalse(args.vision_cuda_fallback)
        self.assertEqual(args.line_preprocess, 'legacy')
        with patch.object(sys, 'argv', ['vision', '--vision-device', 'cuda', '--vision-cuda-fallback']):
            args = run_policy_vision.parse_args()
        self.assertEqual(args.vision_device, 'cuda')
        self.assertTrue(args.vision_cuda_fallback)

    def test_probe_failure_occurs_before_camera_or_connector(self):
        import run_policy_vision
        with patch.object(sys, 'argv', ['vision', '--vision-device', 'cuda']), \
                patch.object(cuda_vision, 'create_vision_backend', side_effect=cuda_vision.VisionCudaError('probe failed')), \
                patch('utils.open_camera') as camera, \
                patch.object(run_policy_vision, 'ConnectorClient') as connector:
            with self.assertRaisesRegex(cuda_vision.VisionCudaError, 'probe failed'):
                run_policy_vision.main()
        camera.assert_not_called()
        connector.assert_not_called()

    def test_cpu_default_does_not_probe_cuda(self):
        facade = CvFacade(count=0)
        with patch.object(cuda_vision, 'cv2', facade):
            backend = cuda_vision.create_vision_backend()
        self.assertEqual(backend.device, 'cpu')
        self.assertEqual(backend.diagnostics()['vision_cuda_calls_total'], 0)

    def test_cuda_without_device_fails_unless_fallback_was_explicit(self):
        with patch.object(cuda_vision, 'cv2', CvFacade(count=0)):
            with self.assertRaisesRegex(cuda_vision.VisionCudaError, 'no usable CUDA'):
                cuda_vision.create_vision_backend('cuda')
            with self.assertWarns(RuntimeWarning):
                backend = cuda_vision.create_vision_backend('cuda', allow_fallback=True)
        self.assertEqual(backend.device, 'cpu')
        self.assertEqual(backend.requested, 'cuda')
        self.assertTrue(backend.fallback_reason)

    def test_pixel_mismatch_refuses_required_morphology(self):
        with patch.object(cuda_vision, 'cv2', CvFacade(mismatch=True)):
            with self.assertRaisesRegex(cuda_vision.VisionCudaError, 'differs from CPU'):
                cuda_vision.create_vision_backend('cuda')

    def test_probe_checks_mask_chains_on_masks_not_arbitrary_gray(self):
        # The real line/card close-open stages receive threshold masks. A CUDA
        # difference for arbitrary grayscale values must not veto those paths.
        with patch.object(cuda_vision, 'cv2', CvFacade(gray_morph_mismatch=True)):
            backend = cuda_vision.create_vision_backend('cuda')
        self.assertIn('morphology', backend.enabled)

    def test_optional_color_mismatch_stays_cpu_and_is_reported(self):
        with patch.object(cuda_vision, 'cv2', CvFacade(gray_mismatch=True)):
            backend = cuda_vision.create_vision_backend('cuda')
            self.assertNotIn('gray', backend.enabled)
            self.assertIn('gray', backend.retained)
            image = np.full((8, 8, 3), 84, np.uint8)
            np.testing.assert_array_equal(backend.gray(image), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            self.assertIn('pixel mismatch', backend.diagnostics()['vision_cuda_retained_cpu'])

    def test_chain_has_one_upload_download_and_reuses_filters(self):
        facade = CvFacade()
        with patch.object(cuda_vision, 'cv2', facade):
            backend = cuda_vision.create_vision_backend('cuda')
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            operations = [(cv2.MORPH_CLOSE, kernel), (cv2.MORPH_OPEN, kernel)]
            image = np.zeros((80, 90), np.uint8)
            image[12:62, 20:28] = 255
            FakeGpuMat.uploads = FakeGpuMat.downloads = 0
            creations = facade.cuda.filter_creations
            actual = backend.morphology(image, operations)
            self.assertEqual((FakeGpuMat.uploads, FakeGpuMat.downloads), (1, 1))
            self.assertEqual(facade.cuda.filter_creations, creations)
            np.testing.assert_array_equal(actual, cuda_vision.CpuVisionBackend().morphology(image, operations))
            self.assertEqual(backend.diagnostics()['vision_cuda_calls_total'], 1)

    def test_runtime_failure_is_not_silently_replaced(self):
        facade = CvFacade()
        with patch.object(cuda_vision, 'cv2', facade):
            backend = cuda_vision.create_vision_backend('cuda')
            facade.cuda.fail_runtime = True
            with self.assertRaisesRegex(cuda_vision.VisionCudaError, 'simulated device failure'):
                backend.morphology(np.zeros((32, 32), np.uint8), [(cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))])

    def test_explicit_runtime_fallback_is_sticky_and_identified(self):
        facade = CvFacade()
        image = np.zeros((32, 32), np.uint8)
        operations = [(cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))]
        with patch.object(cuda_vision, 'cv2', facade):
            backend = cuda_vision.create_vision_backend('cuda', allow_fallback=True)
            facade.cuda.fail_runtime = True
            with self.assertWarns(RuntimeWarning):
                actual = backend.morphology(image, operations)
            self.assertEqual(backend.device, 'cpu')
            self.assertTrue(backend.fallback_reason)
            with warnings.catch_warnings(record=True) as seen:
                backend.morphology(image, operations)
            self.assertEqual(len(seen), 0)
            np.testing.assert_array_equal(actual, image)
            self.assertEqual(backend.diagnostics()['vision_cuda_enabled_ops'], '')

    def test_legacy_c_normalization_and_mask_are_preserved(self):
        gray = np.full((400, 320), 175, np.uint8)
        cv2.line(gray, (65, 399), (105, 0), 25, 11)
        cv2.line(gray, (255, 399), (215, 0), 25, 11)
        with patch.object(cuda_vision, 'cv2', CvFacade()):
            backend = cuda_vision.create_vision_backend('cuda')
            for gain in (1., .6, .4):
                image = (gray.astype(float)*gain).astype(np.uint8)
                photo = measure(image)
                reference = extract_lane_candidates(image, photometry=photo)
                actual = extract_lane_candidates(image, photometry=photo, backend=backend)
                np.testing.assert_array_equal(actual[0], reference[0])
                np.testing.assert_array_equal(actual[1], reference[1])
                self.assertEqual(actual[2:], reference[2:])
                self.assertEqual(actual[3]['preprocess_adaptive_c_effective'], photo.difference(-12))
            self.assertGreater(backend.diagnostics()['vision_cuda_calls_total'], 0)

    def test_detectors_and_broken_cue_retain_cpu_results(self):
        frame = np.full((540, 960, 3), 180, np.uint8)
        cv2.rectangle(frame, (380, 230), (530, 350), (40, 40, 40), 5)
        cv2.circle(frame, (455, 290), 25, (40, 40, 40), 5)
        with patch.object(cuda_vision, 'cv2', CvFacade()):
            backend = cuda_vision.create_vision_backend('cuda')
            reference = ShapeDetector()
            actual = ShapeDetector()
            actual.vision_backend = backend
            _, a = reference.update(frame)
            _, b = actual.update(frame)
            for key in ('binary', 'ink', 'gray'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('shape', 'card_found', 'presence', 'presence_cue', 'quad_total', 'quad_geom'):
                self.assertEqual(a[key], b[key])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old = find_broken_border_presence(gray, measure(gray), reference.cfg)
            new = find_broken_border_presence(gray, measure(gray), reference.cfg, backend=backend)
            self.assertEqual(old, new)
            line_old = LineDetector(960, 540)
            line_new = LineDetector(960, 540)
            line_new.vision_backend = backend
            old_result = line_old.process(frame, dt=.1)
            new_result = line_new.process(frame, dt=.1)
            self.assertEqual(old_result[:3], new_result[:3])
            for key in ('binary', 'bird', 'black_th', 'measurement_valid'):
                np.testing.assert_array_equal(old_result[4][key], new_result[4][key])


if __name__ == '__main__':
    unittest.main()
