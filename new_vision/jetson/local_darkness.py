"""Opt-in local darkness candidates, with independent card and lane scales.

No color transform and no shared final binary. The card mask is for finding a
frame; the rectified card interior is still read from the untouched gray image.
"""
import cv2
import numpy as np


def _relative_dark(gray, window, floor, noise_multiplier):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("local darkness requires uint8 grayscale")
    background = cv2.medianBlur(gray, window)
    darkness = cv2.subtract(background, gray)
    fine = gray.astype(np.int16) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.int16)
    # Robust absolute high-frequency residual. A low-contrast/noisy patch must
    # not be promoted to an observed dark stroke by global gain.
    noise = float(np.median(np.abs(fine)))
    threshold = max(floor, noise_multiplier * noise)
    ink = (darkness >= threshold).astype(np.uint8) * 255
    return ink, dict(local_dark_window=window, local_dark_noise=noise,
                     local_dark_threshold=threshold,
                     local_dark_fraction=float((ink > 0).mean()))


def card_candidates(gray):
    # 0.5 cm printed frame: about 3-7 px in the work image.
    return _relative_dark(gray, window=21, floor=8., noise_multiplier=3.)


def lane_candidates(gray):
    # Wider physical lane marks: keep this independent of card extraction.
    return _relative_dark(gray, window=41, floor=12., noise_multiplier=4.)
