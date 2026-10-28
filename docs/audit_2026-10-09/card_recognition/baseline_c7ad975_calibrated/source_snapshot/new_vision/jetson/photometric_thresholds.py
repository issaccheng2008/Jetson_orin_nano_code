"""Reference z-score thresholds without clipping or rewriting the source image.

I_ref = mean_ref + std_ref * (I - mean) / std.
Thus an intensity threshold becomes mean + std/std_ref*(T_ref-mean_ref),
while differences, blackhat responses and adaptive C use std/std_ref only.
Statistics use the same central ground ROI as the archived video comparison.
"""
from dataclasses import dataclass
import numpy as np

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
                    photometric_contrast_scale=self.scale)


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
