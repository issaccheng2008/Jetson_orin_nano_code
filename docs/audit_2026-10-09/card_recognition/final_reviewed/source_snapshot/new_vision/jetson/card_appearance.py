"""Extract a card symbol using local grayscale and contour nesting.

The frame encloses paper, which encloses the symbol. Keeping this topology
avoids cutting an interior outline with a fixed crop or mistaking paper
inside the outer frame for a square. Ground-plane card validation belongs to
the caller; this module does not detect arbitrary cards or issue actions.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CardAppearance:
    gray: np.ndarray
    binary: np.ndarray
    symbol: Optional[np.ndarray]
    reason: str
    contrast: float
    threshold: float
    frame_count: int = 0

    def diagnostics(self):
        return {"appearance_reason": self.reason,
                "appearance_contrast": self.contrast,
                "appearance_threshold": self.threshold,
                "appearance_frame_count": self.frame_count}


def rectify_card_gray(gray, quad, size=200, expand_px=4.0):
    """Rectify the entire frame with linear sampling, preserving gray edges."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = q.mean(axis=0)
    v = q-center
    radius = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(radius < 1e-3) or not np.isfinite(q).all():
        raise ValueError("card quadrilateral must have four finite distinct corners")
    q = center+v*((radius+expand_px)/radius)
    q = q[np.argsort(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    transform = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(gray, transform, (size, size),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def card_appearance(gray, quad, size=200, expand_px=4.0):
    """Return an enclosed ink exterior contour or a reason for abstaining.

    ``symbol`` includes the area enclosed by an outlined symbol and can go
    directly to the existing contour classifier. No class labels, filenames,
    frame history or action state are used to choose it.
    """
    rectified = rectify_card_gray(gray, quad, size, expand_px)
    p05, p95 = np.percentile(rectified, (5, 95))
    contrast = float(p95-p05)
    threshold, binary = cv2.threshold(rectified, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    result = CardAppearance(rectified, binary, None, "no_enclosing_frame",
                            contrast, float(threshold))
    if contrast < 3:
        result.reason = "insufficient_contrast"
        return result
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return result
    tree = hierarchy[0]
    area_ref = float(size*size)
    frames = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        child = int(tree[i][2])
        parent = int(tree[i][3])
        depth = 0
        while parent >= 0:
            depth += 1
            parent = int(tree[parent][3])
        if (depth % 2 or cv2.contourArea(contour) < .50*area_ref
                or min(w, h) < .60*size
                or child < 0):
            continue
        while child >= 0:
            if cv2.contourArea(contours[child]) >= .35*area_ref:
                frames.append((i, child))
            child = int(tree[child][0])
    result.frame_count = len(frames)
    candidates = []
    for frame, paper in frames:
        child = int(tree[paper][2])
        while child >= 0:
            contour = contours[child]
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if area >= .015*area_ref and moments['m00'] > 0:
                cx = moments['m10']/moments['m00']
                cy = moments['m01']/moments['m00']
                if (abs(cx-size/2) <= .22*size
                        and abs(cy-size/2) <= .22*size
                        and max(w, h) < .92*size):
                    candidates.append((area, contour))
            child = int(tree[child][0])
    if not candidates:
        # A broken outer frame may still leave a separately enclosed outlined
        # symbol. Its central closed hole is usable even if the ink exterior
        # is clipped at the edge. The outer paper hole is larger than this
        # symbol and has an enclosing boundary spanning the full card.
        for i, contour in enumerate(contours):
            if tree[i][3] != -1:
                continue
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if not (.03*area_ref <= area <= .65*area_ref and max(w, h) < .90*size):
                continue
            child = int(tree[i][2])
            while child >= 0:
                hole = contours[child]
                hole_area = cv2.contourArea(hole)
                hx, hy, hw, hh = cv2.boundingRect(hole)
                moments = cv2.moments(hole)
                if (hole_area >= .02*area_ref and hole_area >= .20*area
                        and hx > 0 and hy > 0 and hx+hw < size and hy+hh < size
                        and moments['m00'] > 0):
                    cx = moments['m10']/moments['m00']
                    cy = moments['m01']/moments['m00']
                    if abs(cx-size/2) <= .22*size and abs(cy-size/2) <= .22*size:
                        candidates.append((hole_area, hole))
                child = int(tree[child][0])
        if candidates:
            result.reason = "separate_outline_hole"
    if not candidates:
        # An outlined symbol may share ink with the frame. It can still have
        # a closed central hole, plus a separate surrounding paper hole.
        # A single blank-card paper hole and the off-center quadrants formed
        # by a cross provide no such central enclosed-outline evidence.
        for frame in set(frame for frame, paper in frames):
            child = int(tree[frame][2])
            holes = []
            while child >= 0:
                contour = contours[child]
                area = cv2.contourArea(contour)
                moments = cv2.moments(contour)
                if area >= .03*area_ref and moments['m00'] > 0:
                    cx = moments['m10']/moments['m00']
                    cy = moments['m01']/moments['m00']
                    holes.append((area, contour, cx, cy))
                child = int(tree[child][0])
            if len(holes) < 2:
                continue
            for area, contour, cx, cy in holes:
                if (.04*area_ref <= area <= .55*area_ref
                        and abs(cx-size/2) <= .15*size
                        and abs(cy-size/2) <= .15*size):
                    candidates.append((area, contour))
        if not candidates:
            result.reason = "no_separated_symbol"
            return result
        result.reason = "enclosed_outline_hole"
    candidates.sort(key=lambda candidate: -candidate[0])
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0]*.20:
        result.reason = "multiple_symbols"
        return result
    result.symbol = candidates[0][1]
    if result.reason not in ("enclosed_outline_hole", "separate_outline_hole"):
        result.reason = "enclosed_symbol"
    return result
