"""Current-image direction when a visible tape cannot support a centre position."""
import math

import cv2
import numpy as np


def visible_lane_direction(mask, to_ground):
    """Fit the nearest observed part; return direction without assigning a side."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 12 or max(width, height) < 6:
            continue
        # The current nearest portion is useful through a broken scan band.
        bottom = y+height-1
        yy, xx = np.nonzero((labels == index) &
                            (np.arange(mask.shape[0])[:, None] >= bottom-120))
        stride = max(1, len(xx)//180)
        pixels = list(zip(xx[::stride], yy[::stride]))
        ground = np.asarray([to_ground(float(px), float(py)) for px, py in pixels], np.float32)
        if len(ground) < 2 or not np.isfinite(ground).all():
            continue
        dz = float(np.ptp(ground[:, 1]))
        if dz < .5:
            continue
        vx, vz, cx, cz = (float(v) for v in cv2.fitLine(
            ground, cv2.DIST_L1, 0., .02, .02).reshape(-1))
        if vz < 0:
            vx, vz = -vx, -vz
        angle = float(np.clip(-math.degrees(math.atan2(vx, vz)), -85., 85.))
        candidates.append((bottom, dz, area, angle, len(ground)))
    if not candidates:
        return None
    bottom, span, area, angle, points = max(candidates, key=lambda p: (p[0], p[1], p[2]))
    return dict(heading_deg=angle, points=points, span_cm=span,
                component_count=len(candidates), nearest_row=bottom)
