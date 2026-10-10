"""Photometric lane candidates; geometry still decides which candidates are lanes.

The input is the unmodified 320x400 birdseye grayscale image. ``contrast``
uses limited local equalization and preserves thin/oblique fragments. ``legacy``
reproduces the former threshold/morphology/component filter for comparison.
No temporal history or motor commands belong in this stage.
"""
import cv2
import numpy as np


def sampled_otsu_threshold(gray):
    """Keep the original sampled histogram, including its constant-image fallback."""
    hist = np.bincount(gray[::max(1, gray.shape[0] // 30),
                           ::max(1, gray.shape[1] // 40)].ravel(), minlength=256)
    total = int(hist.sum())
    if not total:
        return 64
    sum_all = sum(i * int(hist[i]) for i in range(256))
    sum_b, weight_b, best_var, best_t = 0, 0, -1.0, 64
    for t in range(256):
        weight_b += int(hist[t])
        if not weight_b:
            continue
        weight_f = total - weight_b
        if not weight_f:
            break
        sum_b += t * int(hist[t])
        delta = sum_b / weight_b - (sum_all - sum_b) / weight_f
        variance = weight_b * weight_f * delta * delta
        if variance > best_var:
            best_var, best_t = variance, t
    return best_t


def extract_lane_candidates(gray, mode="contrast", th_offset=-12, th_min=25, th_max=80):
    """Return masked black-hat response, mask, threshold and scalar diagnostics.

    All geometric sampling continues in the original coordinates. Equalization
    only affects candidate extraction, never the source image, red detector,
    shape detector, or the raw-gray centroid fallback.
    """
    if mode == "canny":
        from canny_candidates import lane_canny
        mask, diagnostics = lane_canny(gray)
        # Downstream scanners require a thresholded response as well as a mask.
        # Use filled stroke evidence, never raw Canny double edges.
        return mask.copy(), mask, 127.0, diagnostics
    if mode not in ("legacy", "contrast"):
        raise ValueError("line preprocessing must be legacy, contrast or canny")
    source = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
              if mode == "contrast" else gray)
    response = cv2.morphologyEx(source, cv2.MORPH_BLACKHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    adaptive = cv2.adaptiveThreshold(response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, -12) > 0
    threshold = (float(np.median(response[adaptive])) if np.count_nonzero(adaptive) > 100
                 else sampled_otsu_threshold(response)) + th_offset
    threshold = float(np.clip(threshold, th_min, th_max))
    _, mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
    threshold_pixels = int(np.count_nonzero(mask))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    if mode == "legacy":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    if mode == "legacy":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    morph_pixels = int(np.count_nonzero(mask))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    retained = np.zeros(count, np.uint8)
    for i in range(1, count):
        width, height, area = stats[i, 2:5]
        # Direction-independent span: a horizontal bend is not short noise.
        accepted = (area >= 48 and max(width, height) >= 24 if mode == "contrast"
                    else area >= 300 and height >= 80)
        if accepted:
            retained[i] = 255
    mask = retained[labels]
    response[mask == 0] = 0
    diagnostics = {
        "preprocess_mode": mode,
        "preprocess_gray_mean": float(gray.mean()),
        "preprocess_gray_std": float(gray.std()),
        "preprocess_threshold_pixels": threshold_pixels,
        "preprocess_morph_pixels": morph_pixels,
        "preprocess_foreground_pixels": int(np.count_nonzero(mask)),
        "preprocess_components": int(np.count_nonzero(retained)),
    }
    return response, mask, threshold, diagnostics
