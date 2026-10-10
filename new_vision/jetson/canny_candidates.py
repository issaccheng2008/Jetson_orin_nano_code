"""Canny-supported dark strokes, in each detector's original coordinates.

Canny marks both borders of black tape. These borders are evidence for one
filled stroke; they must not be passed to lane geometry as two lane boundaries.
The functions below combine local dark-ridge support with opposite gradient
polarities across a stroke. They do not infer lane identity or card shape.
"""
import cv2
import numpy as np
from masked_ground import validity, gaussian as masked_gaussian, morphology as masked_morphology


def _validate(gray, shape):
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8:
        raise ValueError("Canny candidates require a uint8 grayscale array")
    if gray.shape != shape:
        raise ValueError("Canny candidates require grayscale shape %s" % (shape,))


def _ray(mask, dx, dy, radius):
    """Whether there is evidence from this pixel along a short directed ray."""
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
    for t in range(radius + 1):
        kernel[radius + t * dy, radius + t * dx] = 1
    return cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT) > 0


def _extract(gray, mode, bh_size, radius, min_area, min_span, valid_mask=None):
    valid = validity(gray, valid_mask)
    masked = valid is not None and not np.all(valid)
    observed = np.ones_like(gray, bool) if valid is None else valid
    gradient_valid = (cv2.erode(observed.astype(np.uint8), np.ones((3,3),np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0) if masked else observed
    # A small blur suppresses sensor noise while retaining the card's thin ink.
    smooth = (np.clip(np.rint(masked_gaussian(gray, valid, (3,3), .7)),0,255).astype(np.uint8)
              if masked else cv2.GaussianBlur(gray, (3, 3), 0.7))
    residual = gray.astype(np.float32) - smooth.astype(np.float32)
    samples = residual[observed]
    noise = float(np.median(np.abs(samples-np.median(samples))) / 0.67448975) if samples.size else 0.
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # Sparse ink should not set a global image-intensity threshold. The floor
    # and noise term protect flat floors; the gradient percentile adapts to
    # exposure/texture. Sobel units match Canny aperture=3 and L2gradient=True.
    high = max(12.0, noise * 6.0, (float(np.percentile(magnitude[gradient_valid], 90)) if gradient_valid.any() else 0.) * 1.3)
    low = high * 0.4
    edges = cv2.Canny(smooth, low, high, apertureSize=3, L2gradient=True)
    edges[~gradient_valid] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
    response = (masked_morphology(smooth, cv2.MORPH_BLACKHAT, kernel, valid)
                if masked else cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel))
    ink_threshold = max(3.0, noise * 3.5,
                        (float(np.percentile(response[observed], 70)) if observed.any() else 0.) * 1.3)
    ink = np.where(response >= ink_threshold, 255, 0).astype(np.uint8)

    ink[~observed] = 0

    paired = np.zeros_like(gray, bool)
    # Four normal directions cover vertical/horizontal/oblique tape and bends.
    # Negative gradient on one side, positive on the other means a dark ridge.
    # The two supporting edges remain distinct; only observed dark pixels fill.
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        projection = (gx * dx + gy * dy) / np.hypot(dx, dy)
        negative = np.where((edges > 0) & (projection < -low * 0.35),
                            255, 0).astype(np.uint8)
        positive = np.where((edges > 0) & (projection > low * 0.35),
                            255, 0).astype(np.uint8)
        paired |= (_ray(negative, -dx, -dy, radius)
                   & _ray(positive, dx, dy, radius))

    supported = np.where((ink > 0) & paired, 255, 0).astype(np.uint8)
    # Only a 3px close: repair tiny raster gaps without completing missing card
    # sides or filling large closed contours. Holes inside rings remain holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = (masked_morphology(supported, cv2.MORPH_CLOSE, kernel, valid)
                 if masked else cv2.morphologyEx(supported, cv2.MORPH_CLOSE, kernel))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros(count, np.uint8)
    widths = []
    distances = (cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
                 if mode == "card" else None)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area or max(width, height) < min_span:
            continue
        if distances is not None:
            region = labels[y:y + height, x:x + width] == label
            stroke_width = float(2 * np.median(
                distances[y:y + height, x:x + width][region]))
            if stroke_width > 7.0:
                continue
            widths.append(stroke_width)
        keep[label] = 255
    mask = keep[labels]
    diagnostics = {
        "preprocess_mode": "canny",
        "canny_kind": mode,
        "canny_low": float(low),
        "canny_high": float(high),
        "canny_noise_sigma": noise,
        "canny_ink_threshold": float(ink_threshold),
        "canny_pair_radius": radius,
        "canny_blackhat_size": bh_size,
        "canny_raw_edge_pixels": int(np.count_nonzero(edges)),
        "preprocess_gray_mean": float(gray[observed].mean()) if observed.any() else 0.,
        "preprocess_gray_std": float(gray[observed].std()) if observed.any() else 0.,
        "preprocess_threshold_pixels": int(np.count_nonzero(ink)),
        "preprocess_morph_pixels": int(np.count_nonzero(candidate)),
        "preprocess_foreground_pixels": int(np.count_nonzero(mask)),
        "preprocess_components": int(np.count_nonzero(keep)),
        "card_stroke_widths": widths,
        "raw_edges": edges,
        "blackhat_response": response,
        "ink_mask": ink,
        "paired_mask": np.where(paired, 255, 0).astype(np.uint8),
    }
    return mask, diagnostics


def lane_canny(gray, valid_mask=None):
    """Filled lane strokes on the current metric raster (width is not fixed)."""
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("lane Canny requires a uint8 grayscale image")
    if min(gray.shape) < 33:
        raise ValueError("lane Canny image must fit its 33px neighbourhood")
    return _extract(gray, "lane", 33, 24, 48, 24, valid_mask=valid_mask)


def card_canny(gray960x540):
    """Return fine card ink and diagnostics for a (540, 960) image.

    The 11px local ridge scale suppresses broad tape; the existing 7px median
    stroke-width limit rejects remaining thick components. No ring is filled.
    Open/occluded outlines stay open for downstream closure/shape validation.
    """
    _validate(gray960x540, (540, 960))
    return _extract(gray960x540, "card", 11, 7, 18, 10)
