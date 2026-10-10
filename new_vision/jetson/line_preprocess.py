"""Photometric lane candidates; geometry still decides which candidates are lanes.

The input is a metric birdseye grayscale image with an optional validity mask. ``contrast``
uses limited local equalization and preserves thin/oblique fragments. ``legacy``
reproduces the former threshold/morphology/component filter for comparison.
No temporal history or motor commands belong in this stage.
"""
import cv2
import math
import numpy as np
from masked_ground import validity, gaussian as masked_gaussian, morphology as masked_morphology, clahe as masked_clahe


def sampled_otsu_threshold(gray, valid_mask=None):
    """Keep the original sampled histogram, including its constant-image fallback."""
    valid = validity(gray, valid_mask)
    sy, sx = max(1, gray.shape[0]//30), max(1, gray.shape[1]//40)
    values = gray[::sy, ::sx]
    values = values.ravel() if valid is None else values[valid[::sy, ::sx]]
    hist = np.bincount(values, minlength=256)
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


def extract_lane_candidates(gray, mode="legacy", th_offset=-12, th_min=25, th_max=80,
                            photometry=None, adaptive_c=-12, valid_mask=None):
    """Return masked black-hat response, mask, threshold and scalar diagnostics.

    All geometric sampling continues in the original coordinates. Equalization
    only affects candidate extraction, never the source image, red detector,
    shape detector, or the raw-gray centroid fallback.
    """
    valid = validity(gray, valid_mask)
    masked = valid is not None and not np.all(valid)
    values = gray.ravel() if valid is None else gray[valid]
    validity_debug = dict(preprocess_valid_pixels=int(values.size),
                          preprocess_ignored_pixels=int(gray.size-values.size),
                          preprocess_gray_mean=float(values.mean()) if values.size else 0.,
                          preprocess_gray_std=float(values.std()) if values.size else 0.)
    if not values.size:
        empty = np.zeros_like(gray)
        return empty, empty.copy(), float(th_min), dict(validity_debug,
            preprocess_mode=mode, preprocess_threshold_pixels=0,
            preprocess_morph_pixels=0, preprocess_foreground_pixels=0, preprocess_components=0)
    if mode == "canny":
        from canny_candidates import lane_canny
        mask, diagnostics = lane_canny(gray, valid_mask=valid)
        diagnostics.update(validity_debug)
        # Downstream scanners require a thresholded response as well as a mask.
        # Use filled stroke evidence, never raw Canny double edges.
        return mask.copy(), mask, 127.0, diagnostics
    if mode not in ("legacy", "contrast"):
        raise ValueError("line preprocessing must be legacy, contrast or canny")
    if masked:
        source = masked_clahe(gray, valid) if mode == "contrast" else gray
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        response = masked_morphology(source, cv2.MORPH_BLACKHAT, kernel, valid)
        validity_debug["preprocess_mask_backend"] = "cpu"
    else:
        source = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                  if mode == "contrast" else gray)
        response = cv2.morphologyEx(source, cv2.MORPH_BLACKHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # For I' = a*I+b, blackhat and (I-local_mean) both scale by a;
    # b cancels. Reference z-score normalization therefore maps C to C*a,
    # where a = std_current/std_reference (not the variance ratio).
    # CLAHE changes contrast nonlinearly, so this mapping is for legacy only.
    effective_c = (photometry.difference(adaptive_c)
                   if mode == "legacy" and photometry is not None else float(adaptive_c))
    if masked:
        local_mean = masked_gaussian(response, valid, (31, 31))
        adaptive = valid & (response > np.rint(local_mean) - math.ceil(effective_c))
    else:
        adaptive = cv2.adaptiveThreshold(response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, effective_c) > 0
    threshold = (float(np.median(response[adaptive])) if np.count_nonzero(adaptive) > 100
                 else sampled_otsu_threshold(response, valid)) + th_offset
    threshold = float(np.clip(threshold, th_min, th_max))
    _, mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
    if masked:
        mask[~valid] = 0
    threshold_pixels = int(np.count_nonzero(mask))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    operations = [(cv2.MORPH_CLOSE, k5)]
    if mode == "legacy":
        operations.append((cv2.MORPH_OPEN, k5))
    operations.append((cv2.MORPH_CLOSE, k5))
    if mode == "legacy":
        operations.append((cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))
    for operation, kernel in operations:
        mask = (masked_morphology(mask, operation, kernel, valid) if masked
                else cv2.morphologyEx(mask, operation, kernel))
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
        "preprocess_adaptive_c_reference": float(adaptive_c),
        "preprocess_adaptive_c_effective": float(effective_c),
        **validity_debug,
        "preprocess_threshold_pixels": threshold_pixels,
        "preprocess_morph_pixels": morph_pixels,
        "preprocess_foreground_pixels": int(np.count_nonzero(mask)),
        "preprocess_components": int(np.count_nonzero(retained)),
    }
    return response, mask, threshold, diagnostics
