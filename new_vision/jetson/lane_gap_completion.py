"""Join short tape gaps using observed endpoint positions and local tangents."""
import math

import cv2
import numpy as np


def _complete_pass(mask, *, valid_mask=None, max_gap_px=30., angle_deg=55.):
    """Return completed mask, added-only mask and diagnostic counts.

    A bridge requires two observed components. There is no endpoint extension,
    no virtual opposite boundary and no temporal image replay.
    """
    output = mask.copy()
    added = np.zeros_like(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    endpoints = []
    widths = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 12:
            continue
        yy, xx = np.nonzero(labels == index)
        points = np.column_stack((xx, yy)).astype(float)
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
        axis = vectors[0]
        projection = (points-center) @ axis
        if np.ptp(projection) < 12:
            continue
        thickness = float(np.clip(2*np.percentile(widths[yy, xx], 80), 5., 20.))
        for sign in (-1., 1.):
            edge = sign*projection
            tip = points[edge >= edge.max()-max(2., thickness/2)].mean(axis=0)
            local = points[edge >= edge.max()-max(18., 2*thickness)]
            _, _, local_vectors = np.linalg.svd(local-local.mean(axis=0), full_matrices=False)
            tangent = local_vectors[0]
            if np.dot(tangent, tip-center) < 0:
                tangent = -tangent
            endpoints.append((index, tip, tangent, thickness))
    candidates = []
    cosine = math.cos(math.radians(angle_deg))
    for i, first in enumerate(endpoints):
        for j in range(i+1, len(endpoints)):
            second = endpoints[j]
            if first[0] == second[0]:
                continue
            delta = second[1]-first[1]
            distance = float(np.linalg.norm(delta))
            if not 1. < distance <= max_gap_px:
                continue
            direction = delta/distance
            if (np.dot(first[2], direction) < cosine or
                    np.dot(second[2], -direction) < cosine or
                    max(first[3], second[3]) > 2.0*min(first[3], second[3])):
                continue
            candidates.append((distance, i, j))
    used = set()
    bridges = 0
    for distance, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        first, second = endpoints[i], endpoints[j]
        bridge = np.zeros_like(mask)
        cv2.line(bridge, tuple(np.rint(first[1]).astype(int)),
                 tuple(np.rint(second[1]).astype(int)), 255,
                 max(5, round(min(first[3], second[3]))), cv2.LINE_8)
        if valid_mask is not None and np.any((bridge > 0) & ~valid_mask.astype(bool)):
            continue
        output |= bridge
        used.update((i, j))
        bridges += 1
    added[(output > 0) & (mask == 0)] = 255
    return output, added, dict(lane_gap_bridges=bridges,
        lane_gap_added_pixels=int(np.count_nonzero(added)),
        lane_gap_max_px=float(max_gap_px))


def complete_lane_gaps(mask, *, valid_mask=None, max_gap_px=30., angle_deg=55.):
    """Three bounded joining rounds; added pixels never become real observations."""
    output = mask.copy()
    bridges = 0
    for _ in range(3):
        updated, added, info = _complete_pass(output, valid_mask=valid_mask,
            max_gap_px=max_gap_px, angle_deg=angle_deg)
        bridges += info['lane_gap_bridges']
        output = updated
        if not np.any(added):
            break
    added = np.where((output > 0) & (mask == 0), 255, 0).astype(np.uint8)
    return output, added, dict(lane_gap_bridges=bridges,
        lane_gap_added_pixels=int(np.count_nonzero(added)), lane_gap_max_px=float(max_gap_px))
