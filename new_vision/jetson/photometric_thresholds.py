"""Mutually exclusive camera photometry: raw, threshold, guarded match, or LUT.

``legacy`` still means the old per-frame mean/std image matching. ``normalize``
is a deprecated alias for threshold-only adaptation; callers log the resolved mode.
``lut`` applies the calibrated fixed->auto camera LUT (load_lut_mapping) and
never stacks with the guarded match or with statistics-based threshold
compensation. Statistics use the original central ROI. A matched image is
never threshold-compensated again with its pre-match moments.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from photometric_lut import apply_mapping, load_mapping

GRAY_REFERENCE = (152.5821277006173, 20.665240198035168)
MAX_CHANNEL_REFERENCE = (154.5171163957077, 20.54726740198349)
STD_FLOOR = 4.0
MAX_MATCH_GAIN = 2.5


def resolve_mode(mode):
    aliases = {"legacy": "match", "normalize": "threshold"}
    result = aliases.get(mode, mode)
    if result not in ("raw", "threshold", "match", "lut"):
        raise ValueError("photometric mode must be raw, threshold, match, lut, legacy or normalize")
    return result


_LUT_MAPPING = None
_LUT_SOURCE = None


def load_lut_mapping(directory):
    """Load and register the fitted fixed->auto LUT pair used by mode ``lut``."""
    global _LUT_MAPPING, _LUT_SOURCE
    mapping = load_mapping(directory)
    lut_bgr = np.asarray(mapping["lut_bgr"])
    lut_gray = np.asarray(mapping["lut_gray"])
    if lut_bgr.shape != (3, 256) or lut_bgr.dtype != np.uint8:
        raise ValueError("lut_bgr must be uint8 with shape (3, 256)")
    if lut_gray.shape != (256,) or lut_gray.dtype != np.uint8:
        raise ValueError("lut_gray must be uint8 with shape (256,)")
    _LUT_MAPPING = mapping
    _LUT_SOURCE = str(directory)
    return mapping


def clear_lut_mapping():
    global _LUT_MAPPING, _LUT_SOURCE
    _LUT_MAPPING = None
    _LUT_SOURCE = None


@dataclass(frozen=True)
class Photometry:
    mean: float
    std: float
    reference_mean: float
    reference_std: float
    mode: str

    @property
    def effective_mode(self):
        return resolve_mode(self.mode)

    @property
    def scale(self):
        return (self.std / self.reference_std
                if self.effective_mode == "threshold" and self.std >= STD_FLOOR
                else 1.0)

    @property
    def match_allowed(self):
        return self.std >= STD_FLOOR and self.effective_mode == "match"

    @property
    def match_scale(self):
        if self.std < STD_FLOOR:
            return 0.0
        return min(MAX_MATCH_GAIN, self.reference_std / self.std)

    @property
    def match_offset(self):
        return self.reference_mean - self.mean * self.match_scale

    def match_image(self, image):
        """Guarded affine mapping. A flat/noisy input is passed through unchanged."""
        image = np.asarray(image)
        if not image.size or not np.issubdtype(image.dtype, np.number):
            raise ValueError("photometric matching needs a nonempty numeric image")
        if self.std < STD_FLOOR:
            return image.copy()
        if image.dtype == np.uint8:
            levels = np.arange(256, dtype=np.float32)
            lut = np.clip(np.rint(levels * self.match_scale + self.match_offset),
                          0, 255).astype(np.uint8)
            return (cv2.LUT(image, lut).reshape(image.shape) if image.ndim in (2, 3)
                    else lut[image])
        matched = image.astype(np.float32) * self.match_scale + self.match_offset
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            matched = np.clip(np.rint(matched), limits.min, limits.max)
        return matched.astype(image.dtype, copy=False)

    def apply(self, image):
        if self.effective_mode == "lut":
            if _LUT_MAPPING is None:
                raise RuntimeError("photometric mode 'lut' has no loaded mapping; "
                                   "run load_lut_mapping(directory) first")
            return apply_mapping(image, _LUT_MAPPING)
        return self.match_image(image) if self.match_allowed else image

    def intensity(self, reference_threshold):
        if self.effective_mode != "threshold" or self.std < STD_FLOOR:
            return float(reference_threshold)
        return self.mean + self.scale * (reference_threshold - self.reference_mean)

    def difference(self, reference_threshold):
        return float(reference_threshold) * self.scale

    def diagnostics(self):
        return dict(photometric_mode=self.mode,
                    photometric_effective_mode=self.effective_mode,
                    photometric_lut_source=_LUT_SOURCE,
                    photometric_lut_applied=self.effective_mode == "lut" and _LUT_MAPPING is not None,
                    photometric_mean=self.mean, photometric_std=self.std,
                    photometric_std_used=max(self.std, STD_FLOOR),
                    photometric_reference_mean=self.reference_mean,
                    photometric_reference_std=self.reference_std,
                    photometric_contrast_scale=self.scale,
                    photometric_match_scale=self.match_scale if self.match_allowed else 1.0,
                    photometric_match_offset=self.match_offset if self.match_allowed else 0.0,
                    photometric_match_gain_capped=(self.match_allowed and
                                                   self.match_scale >= MAX_MATCH_GAIN),
                    photometric_match_applied=self.match_allowed,
                    photometric_match_reject_reason=("low_contrast" if self.effective_mode == "match"
                        and not self.match_allowed else None),
                    photometric_threshold_reject_reason=("low_contrast" if
                        self.effective_mode == "threshold" and self.std < STD_FLOOR else None))


def measure(gray, mode="normalize", reference=GRAY_REFERENCE):
    """Measure the original central ROI, excluding caller padding/HUD."""
    resolve_mode(mode)
    if gray.ndim != 2 or not gray.size:
        raise ValueError("photometry needs a nonempty grayscale image")
    height, width = gray.shape
    roi = gray[int(.35*height):max(int(.35*height)+1, int(.75*height)),
               int(.25*width):max(int(.25*width)+1, int(.75*width))]
    return Photometry(float(roi.mean()), float(roi.std()), *reference, mode)


def max_channel(image):
    """Exact uint8 BGR channel maximum using compiled OpenCV kernels."""
    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        return cv2.max(cv2.max(b, g), r)
    return np.max(image, axis=2)
