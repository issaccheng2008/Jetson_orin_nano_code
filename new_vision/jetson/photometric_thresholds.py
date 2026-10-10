"""Match frame brightness moments to archived auto-exposure images.

The image mapping is I_ref = mean_ref + std_ref * (I - mean) / std. Callers
apply it once to the input image, before the existing vision pipeline. Optional
``normalize`` threshold helpers remain for compatibility; ``legacy`` keeps the
old fixed thresholds after image matching. Statistics use the same central
ground ROI as the archived video comparison.
"""
from dataclasses import dataclass
import numpy as np
import cv2

# Equal weight per valid sampled frame, four September 19 raw videos, 43 frames.
GRAY_REFERENCE = (152.5821277006173, 20.665240198035168)
MAX_CHANNEL_REFERENCE = (154.5171163957077, 20.54726740198349)
STD_FLOOR = 4.0  # Gray levels: do not amplify a nearly flat/noisy image without bound.


@dataclass(frozen=True)
class Photometry:
    mean: float
    std: float
    reference_mean: float
    reference_std: float
    mode: str

    @property
    def scale(self):
        return (max(self.std, STD_FLOOR) / self.reference_std
                if self.mode == "normalize" else 1.0)

    @property
    def match_scale(self):
        """Contrast gain for mapping the input image into the reference domain."""
        return self.reference_std / self.std if self.std > 1e-6 else 0.0

    def match_image(self, image):
        """Return a mean/std-matched copy, preserving shape and numeric dtype.

        For color images the same affine mapping is applied to every channel;
        the line detector measures statistics on max-channel grayscale, so its
        max-channel moments receive the same mapping before IPM and color checks.
        Integer images are rounded and clipped to their dtype's valid range.
        """
        image = np.asarray(image)
        if not image.size or not np.issubdtype(image.dtype, np.number):
            raise ValueError("photometric matching needs a nonempty numeric image")
        if image.dtype == np.uint8:
            levels = np.arange(256, dtype=np.float32)
            lut = np.clip(np.rint((levels - self.mean) * self.match_scale
                                  + self.reference_mean), 0, 255).astype(np.uint8)
            return (cv2.LUT(image, lut).reshape(image.shape) if image.ndim in (2, 3)
                    else lut[image])
        matched = ((image.astype(np.float32) - self.mean) * self.match_scale
                   + self.reference_mean)
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            matched = np.clip(np.rint(matched), limits.min, limits.max)
        return matched.astype(image.dtype, copy=False)

    def intensity(self, reference_threshold):
        if self.mode == "legacy":
            return float(reference_threshold)
        return self.mean + self.scale * (reference_threshold - self.reference_mean)

    def difference(self, reference_threshold):
        return float(reference_threshold) * self.scale

    def diagnostics(self):
        return dict(photometric_mode=self.mode, photometric_mean=self.mean,
                    photometric_std=self.std, photometric_std_used=max(self.std, STD_FLOOR),
                    photometric_reference_mean=self.reference_mean,
                    photometric_reference_std=self.reference_std,
                    photometric_contrast_scale=self.scale,
                    photometric_match_scale=self.match_scale)


def measure(gray, mode="normalize", reference=GRAY_REFERENCE):
    """Measure original grayscale; ROI excludes HUD and image padding in callers."""
    if mode not in ("normalize", "legacy"):
        raise ValueError("photometric mode must be normalize or legacy")
    if gray.ndim != 2 or not gray.size:
        raise ValueError("photometry needs a nonempty grayscale image")
    height, width = gray.shape
    roi = gray[int(.35*height):max(int(.35*height)+1, int(.75*height)),
               int(.25*width):max(int(.25*width)+1, int(.75*width))]
    return Photometry(float(roi.mean()), float(roi.std()), *reference, mode)


def max_channel(image):
    """Exact uint8 BGR channel maximum using compiled, vectorized OpenCV kernels."""
    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        return cv2.max(cv2.max(b, g), r)
    return np.max(image, axis=2)
