"""Explicit OpenCV CUDA image backend, with CPU reference semantics.

Adaptive Gaussian thresholding, source photometry, DT/CC and sparse geometry
stay on CPU. Morphology chains use one upload/download and cached filters.
Startup checks real CUDA execution and exact CPU pixels for the kernels used
by this repository. Optional resampling/color operations remain CPU if their
probe differs; diagnostics identify the enabled operations. This is a single
vision-loop backend: cached CUDA filters are not shared across worker streams.
"""
from __future__ import annotations

import time
import warnings

import cv2
import numpy as np


class VisionCudaError(RuntimeError):
    """Requested CUDA execution is unavailable or failed."""


class CpuVisionBackend:
    device = "cpu"

    def __init__(self, requested="cpu", fallback_reason=""):
        self.requested = requested
        self.fallback_reason = fallback_reason

    def morphology(self, image, operations):
        out = image
        for operation, kernel in operations:
            out = cv2.morphologyEx(out, operation, kernel)
        return out

    def gray(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def resize(self, image, size):
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    def warp(self, image, matrix, size):
        return cv2.warpPerspective(image, matrix, size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def diagnostics(self):
        return dict(vision_device_requested=self.requested,
                    vision_device_actual=self.device,
                    vision_cuda_enabled_ops="", vision_cuda_calls_total=0,
                    vision_cuda_transfer_ms_total=0.,
                    vision_cuda_fallback_reason=self.fallback_reason)


class CudaVisionBackend(CpuVisionBackend):
    device = "cuda"

    def __init__(self, allow_fallback=False):
        super().__init__(requested="cuda")
        self.allow_fallback = bool(allow_fallback)
        self._filters = {}
        self._calls = 0
        self._transfer_ms = 0.
        self.enabled = {"morphology"}
        self.retained = {}
        self._probe()
        # Probe work is not per-frame execution evidence.
        self._calls = 0
        self._transfer_ms = 0.

    def _upload(self, image):
        started = time.perf_counter()
        gpu = cv2.cuda_GpuMat()
        gpu.upload(np.ascontiguousarray(image))
        self._transfer_ms += (time.perf_counter()-started)*1000.
        return gpu

    def _download(self, gpu):
        started = time.perf_counter()
        out = gpu.download()  # Synchronous: kernel work must complete here.
        self._transfer_ms += (time.perf_counter()-started)*1000.
        return out

    def _filter(self, operation, kernel):
        kernel = np.asarray(kernel, np.uint8)
        key = (int(operation), kernel.shape, kernel.tobytes())
        if key not in self._filters:
            self._filters[key] = cv2.cuda.createMorphologyFilter(
                int(operation), cv2.CV_8UC1, kernel)
        return self._filters[key]

    def _morphology_cuda(self, image, operations):
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("CUDA morphology requires a uint8 grayscale image")
        gpu = self._upload(image)
        for operation, kernel in operations:
            gpu = self._filter(operation, kernel).apply(gpu)
        self._calls += 1
        return self._download(gpu)

    def _gray_cuda(self, image):
        out = cv2.cuda.cvtColor(self._upload(image), cv2.COLOR_BGR2GRAY)
        self._calls += 1
        return self._download(out)

    def _resize_cuda(self, image, size):
        out = cv2.cuda.resize(self._upload(image), tuple(size),
                              interpolation=cv2.INTER_LINEAR)
        self._calls += 1
        return self._download(out)

    def _warp_cuda(self, image, matrix, size):
        out = cv2.cuda.warpPerspective(self._upload(image), matrix, tuple(size),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self._calls += 1
        return self._download(out)

    def _probe(self):
        if not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            raise VisionCudaError("OpenCV has no CUDA device API")
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count < 1:
                raise VisionCudaError("OpenCV reports no usable CUDA device")
            required = (hasattr(cv2, "cuda_GpuMat"),
                        hasattr(cv2.cuda, "createMorphologyFilter"))
            if not all(required):
                raise VisionCudaError("CUDA OpenCV lacks GpuMat/cudafilters bindings")
            cv2.cuda.setDevice(0)
            rng = np.random.default_rng(2941)
            gray = rng.integers(0, 256, (96, 128), dtype=np.uint8)
            gray[:4, :] = 0
            gray[:, -4:] = 255
            ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
            rect = lambda x, y: cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
            groups = [
                [(cv2.MORPH_BLACKHAT, ell(31))],
                [(cv2.MORPH_BLACKHAT, ell(9))],
                [(cv2.MORPH_CLOSE, ell(5)), (cv2.MORPH_OPEN, ell(5)),
                 (cv2.MORPH_CLOSE, ell(5)), (cv2.MORPH_OPEN, ell(3))],
                [(cv2.MORPH_CLOSE, ell(3)), (cv2.MORPH_CLOSE, rect(1, 5)),
                 (cv2.MORPH_CLOSE, rect(5, 1))],
                [(cv2.MORPH_BLACKHAT, rect(31, 31))],
                [(cv2.MORPH_BLACKHAT, rect(51, 51))],
            ]
            groups += [[(cv2.MORPH_CLOSE, rect(n, n))] for n in (5, 9, 15, 19, 25)]
            groups += [[(cv2.MORPH_DILATE, rect(n, n))] for n in (5, 7, 9, 11)]
            binary_images = ((gray > 127).astype(np.uint8)*255,
                             (gray > 127).astype(np.uint8))
            for operations in groups:
                # Black-hat consumes grayscale in this project. Close/open/
                # dilate consume threshold masks (0/255 or 0/1). Probing mask
                # chains on arbitrary grayscale can reject CUDA because of
                # values that the actual pipelines never send to those stages.
                probe_images = (gray,) if any(op == cv2.MORPH_BLACKHAT
                                              for op, _ in operations) else binary_images
                for image in probe_images:
                    reference = CpuVisionBackend.morphology(self, image, operations)
                    actual = self._morphology_cuda(image, operations)
                    if not np.array_equal(reference, actual):
                        changed = int(np.count_nonzero(reference != actual))
                        raise VisionCudaError(
                            f"CUDA morphology differs from CPU ({changed} pixels); "
                            "refusing to change card/lane evidence")
            bgr = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
            # Resampling/color may differ by rounding between builds. Retain
            # those operations on CPU unless the nontrivial probe is exact.
            matrix = np.array([[.81, .035, 2.3], [-.04, .89, 1.7],
                               [.0002, -.0003, 1.]], dtype=np.float64)
            optional = {
                "gray": ("cvtColor", lambda: self._gray_cuda(bgr),
                         lambda: CpuVisionBackend.gray(self, bgr)),
                "resize": ("resize", lambda: self._resize_cuda(bgr, (83, 71)),
                           lambda: CpuVisionBackend.resize(self, bgr, (83, 71))),
                "warp": ("warpPerspective", lambda: self._warp_cuda(bgr, matrix, (83, 71)),
                         lambda: CpuVisionBackend.warp(self, bgr, matrix, (83, 71))),
            }
            for name, (binding, operation, reference) in optional.items():
                if not hasattr(cv2.cuda, binding):
                    self.retained[name] = "CUDA binding unavailable; retained CPU"
                    continue
                try:
                    actual, expected = operation(), reference()
                    if np.array_equal(actual, expected):
                        self.enabled.add(name)
                    else:
                        self.retained[name] = (f"pixel mismatch "
                            f"({np.count_nonzero(actual != expected)}); retained CPU")
                except (cv2.error, AttributeError, TypeError) as exc:
                    self.retained[name] = f"CUDA probe failed; retained CPU: {exc}"
        except VisionCudaError:
            raise
        except (cv2.error, AttributeError, TypeError) as exc:
            raise VisionCudaError(f"CUDA startup probe failed: {exc}") from exc

    def _execute(self, name, cuda_call, cpu_call):
        if self.device != "cuda" or name not in self.enabled:
            return cpu_call()
        try:
            return cuda_call()
        except (cv2.error, AttributeError, TypeError) as exc:
            reason = f"CUDA {name} failed: {exc}"
            if not self.allow_fallback:
                raise VisionCudaError(reason) from exc
            self.device = "cpu"
            self.fallback_reason = reason
            warnings.warn(reason+"; explicit CPU fallback enabled", RuntimeWarning)
            return cpu_call()

    def morphology(self, image, operations):
        operations = list(operations)
        if not operations:
            return image
        return self._execute("morphology",
            lambda: self._morphology_cuda(image, operations),
            lambda: CpuVisionBackend.morphology(self, image, operations))

    def gray(self, image):
        return self._execute("gray", lambda: self._gray_cuda(image),
                             lambda: CpuVisionBackend.gray(self, image))

    def resize(self, image, size):
        return self._execute("resize", lambda: self._resize_cuda(image, size),
                             lambda: CpuVisionBackend.resize(self, image, size))

    def warp(self, image, matrix, size):
        return self._execute("warp", lambda: self._warp_cuda(image, matrix, size),
                             lambda: CpuVisionBackend.warp(self, image, matrix, size))

    def diagnostics(self):
        return dict(vision_device_requested=self.requested,
                    vision_device_actual=self.device,
                    vision_cuda_enabled_ops=",".join(sorted(self.enabled)) if self.device == "cuda" else "",
                    vision_cuda_retained_cpu="; ".join(f"{name}: {reason}" for name, reason in sorted(self.retained.items())),
                    vision_cuda_calls_total=self._calls,
                    # Download includes any device synchronization; not a pure
                    # PCIe-transfer or kernel-only measurement.
                    vision_cuda_transfer_ms_total=self._transfer_ms,
                    vision_cuda_fallback_reason=self.fallback_reason)


def create_vision_backend(device="cpu", allow_fallback=False):
    if device == "cpu":
        return CpuVisionBackend()
    if device != "cuda":
        raise ValueError("vision device must be cpu or cuda")
    try:
        return CudaVisionBackend(allow_fallback=allow_fallback)
    except VisionCudaError as exc:
        if not allow_fallback:
            raise
        warnings.warn(f"{exc}; explicit CPU fallback enabled", RuntimeWarning)
        return CpuVisionBackend(requested="cuda", fallback_reason=str(exc))
