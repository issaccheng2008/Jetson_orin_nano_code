"""Conservative proposals for blurred card presence, never card classification.

Call this only when the existing card/ring paths fail. Morphology may propose a
closed border, but each side and the interior marking must have support in the
original contrast mask. The caller retains its proximity and history gates.
"""
import cv2
import numpy as np


def _quad(contour):
    hull = cv2.convexHull(contour)
    length = cv2.arcLength(hull, True)
    for fraction in (.025, .04, .06, .08):
        polygon = cv2.approxPolyDP(hull, fraction * length, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return polygon.reshape(4, 2).astype(np.float32)
    return None


def _edge_support(ink, quad, band, support=None):
    """Sample the observed mask, not the mask repaired by morphology."""
    if support is None:
        support = cv2.dilate(ink, np.ones((2 * band + 1,) * 2, np.uint8)) > 0
    fractions = []
    worst_gap = 0.0
    height, width = ink.shape
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        points = a + np.linspace(.04, .96, 40)[:, None] * (b - a)
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        hits = support[ys, xs]
        fractions.append(float(hits.mean()))
        run = maximum = 0
        for hit in hits:
            run = 0 if hit else run + 1
            maximum = max(maximum, run)
        worst_gap = max(worst_gap, maximum / len(hits))
    return min(fractions), float(np.mean(fractions)), worst_gap


def find_broken_border_presence(gray, photometry, cfg):
    """Return ``(box, reference_score, scalar_diagnostics)`` for a marked card.

    ``box`` uses the input/work-image coordinates. No temporal state, action or
    shape label is produced. The score has the existing cue's contrast units,
    so the caller must still apply ``cue_score_min`` and its history gate.
    Limited gap repair is capped at 35% of the candidate's short side. All four
    sides must remain observed and uncropped, a large bright paper region must survive, and
    dark ink must also occur away from the border. This rejects a lone cable,
    lane, solid blob, empty rectangular glare and wholly missing card sides.
    This weaker structural path additionally requires a score of at least 3.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("card presence needs a nonempty uint8 grayscale image")
    height, width = gray.shape
    best, score_best = None, 0.0
    accepted = []
    diagnostics = {"cue_broken_evaluated": True,
                   "cue_broken_candidates": 0, "cue_broken_accepted": False}
    for reason in ("edge", "core", "marking", "paper", "contrast", "score"):
        diagnostics["cue_broken_reject_" + reason] = 0
    scale = float(photometry.scale)
    blackhats = {}
    for kernel, close_size, threshold in ((31, 15, 12.), (31, 19, 16.),
                                           (51, 19, 12.), (51, 25, 8.)):
        if kernel not in blackhats:
            blackhats[kernel] = cv2.morphologyEx(
                gray, cv2.MORPH_BLACKHAT, np.ones((kernel, kernel), np.uint8))
        bh = blackhats[kernel]
        ink = (bh > max(3., photometry.difference(threshold))).astype(np.uint8)
        repaired = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                                   np.ones((close_size, close_size), np.uint8))
        contours, _ = cv2.findContours(repaired, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        supported_bands = {}
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            short, long = min(w, h), max(w, h)
            margin = cfg.get("cue_edge_margin", 4)
            if (short < cfg.get("cue_side_min", 30)
                    or long > cfg.get("cue_side_max", 340)
                    or long > cfg.get("cue_aspect_max", 2.6) * short
                    or close_size - 1 > .35 * short
                    or y + h < height * cfg.get("presence_top_frac", 1 / 3)
                    or x < margin or y < margin or x + w > width - margin
                    or y + h > height - margin):
                continue
            quad = _quad(contour)
            # A rotated/perspective square naturally occupies about half of its
            # axis-aligned box. Verify paper in the card plane below instead.
            if quad is None or cv2.contourArea(quad) < .35 * w * h:
                continue
            diagnostics["cue_broken_candidates"] += 1
            band = max(2, min(5, int(.035 * short)))
            if band not in supported_bands:
                supported_bands[band] = cv2.dilate(
                    ink, np.ones((2 * band + 1,) * 2, np.uint8)) > 0
            edge_min, edge_mean, edge_gap = _edge_support(
                ink, quad, band, supported_bands[band])
            if edge_min < .40 or edge_mean < .70 or edge_gap > .35:
                diagnostics["cue_broken_reject_edge"] += 1
                continue
            # Inspect a normalized card plane. Axis-aligned erosion amputates
            # interior ink on rotated cards, and bbox paper fractions penalize
            # their angle. Warp observed ink only; repaired pixels stay excluded.
            target = np.array([[0, 0], [95, 0], [95, 95], [0, 95]], np.float32)
            transform = cv2.getPerspectiveTransform(quad, target)
            values = cv2.warpPerspective(gray, transform, (96, 96))
            local_ink = cv2.warpPerspective(
                ink, transform, (96, 96), flags=cv2.INTER_NEAREST) > 0
            core = np.zeros((96, 96), bool)
            core[15:81, 15:81] = True
            marking = core & local_ink
            marking_fraction = float(marking.sum()) / float(core.sum())
            if not .025 <= marking_fraction <= .75:
                diagnostics["cue_broken_reject_marking"] += 1
                continue
            # Use unrepaired ink for contrast and bright area. A close operation
            # is only evidence of a bounded gap, never invented dark ink.
            paper = ~local_ink
            border = local_ink & (~core)
            paper_fraction = float(paper.mean())
            if paper_fraction < .30 or border.mean() < .025:
                diagnostics["cue_broken_reject_paper"] += 1
                continue
            paper_gray = float(values[paper].mean())
            border_gray = float(values[border].mean())
            contrast = (paper_gray - border_gray) / scale
            marking_contrast = (float(values[core & (~local_ink)].mean())
                                - float(values[marking].mean())) / scale
            if (paper_gray < photometry.intensity(cfg.get("cue_core_gray_min", 100.))
                    or contrast < cfg.get("cue_contrast_min", 4.)
                    or marking_contrast < 20.):
                diagnostics["cue_broken_reject_contrast"] += 1
                continue
            score = paper_fraction * contrast
            # Repaired contours carry less structural evidence than the closed
            # ring path. Keep an additional score floor for this path only.
            if score < max(3., cfg.get("cue_score_min", 1.2)):
                diagnostics["cue_broken_reject_score"] += 1
                continue
            details = dict(cue_broken_accepted=True,
                                   cue_broken_kernel=kernel,
                                   cue_broken_close=close_size,
                                   cue_broken_edge_min=edge_min,
                                   cue_broken_edge_mean=edge_mean,
                                   cue_broken_edge_gap=edge_gap,
                                   cue_broken_marking_fraction=marking_fraction,
                                   cue_broken_marking_contrast=marking_contrast,
                                   cue_broken_paper_fraction=paper_fraction,
                                   cue_broken_contrast=contrast,
                                   cue_broken_score_raw=score * scale,
                                   cue_broken_score_reference=float(score))
            accepted.append(((int(x), int(y), int(w), int(h)), float(score),
                             details, quad))
    # Different repair scales can connect the interior marking to only part of
    # the border. Keep the observed outer card when it contains such a smaller
    # proposal; an inner contour must not shift the stopping position.
    for box, score, details, quad in accepted:
        center = quad.mean(axis=0)
        area = cv2.contourArea(quad)
        nested = any(cv2.contourArea(other_quad) > 1.15 * area
                     and cv2.pointPolygonTest(other_quad.reshape(-1, 1, 2),
                                              (float(center[0]), float(center[1])),
                                              False) >= 0
                     for _, _, _, other_quad in accepted)
        if not nested and score > score_best:
            best, score_best = box, score
            diagnostics.update(details)
    return best, score_best, diagnostics
