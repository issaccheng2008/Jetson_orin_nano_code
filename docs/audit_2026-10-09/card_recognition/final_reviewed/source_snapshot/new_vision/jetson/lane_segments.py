"""Describe a measured lane with local ground-space segments.

Only paired boundary observations are accepted. Each segment is fitted on its
own; a distant false pair cannot bend the near segment or create an extrapolated
target. Width is checked perpendicular to the fitted lane tangent, since two
slanted parallel lines are farther apart on a horizontal image row.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


SEGMENTS_CM = ((20.0, 32.0), (32.0, 44.0), (44.0, 56.0))


def describe_lane_segments(ys, centres_px, widths_px, z_by_row,
                           cm_per_px_by_row, centre_column, lane_width_cm):
    result = {"fit_seg_valid": False, "fit_seg_pattern": "insufficient_support"}
    if not (len(ys) == len(centres_px) == len(widths_px)):
        return result
    by_row = defaultdict(list)
    for y, x, width in zip(ys, centres_px, widths_px):
        if (math.isfinite(float(y)) and math.isfinite(float(x))
                and math.isfinite(float(width)) and 0 <= y < len(z_by_row)
                and width > 0):
            by_row[round(float(y), 2)].append((float(x), float(width)))
    rows = np.asarray(sorted(by_row), dtype=float)
    if not len(rows):
        return result
    x_px = np.asarray([np.median([v[0] for v in by_row[y]]) for y in rows])
    width_px = np.asarray([np.median([v[1] for v in by_row[y]]) for y in rows])
    z = np.interp(rows, np.arange(len(z_by_row)), z_by_row)
    scale = np.interp(rows, np.arange(len(cm_per_px_by_row)), cm_per_px_by_row)
    x = (x_px - centre_column) * scale
    width_horizontal = width_px * scale
    accepted = []
    for index, (z_lo, z_hi) in enumerate(SEGMENTS_CM):
        chosen = (z >= z_lo) & (z < z_hi)
        zz, xx, ww = z[chosen], x[chosen], width_horizontal[chosen]
        prefix = f"fit_seg{index}_"
        result[prefix + "points"] = len(zz)
        if len(zz) < 5 or np.ptp(zz) < 4.0:
            continue
        design = np.column_stack((zz, np.ones(len(zz))))
        slope, intercept = np.linalg.lstsq(design, xx, rcond=None)[0]
        residuals = xx - (slope * zz + intercept)
        mad = float(np.median(np.abs(residuals)))
        inliers = np.abs(residuals) <= max(1.5, 3.0 * mad)
        if inliers.sum() < 5 or np.ptp(zz[inliers]) < 4.0:
            continue
        design = np.column_stack((zz[inliers], np.ones(inliers.sum())))
        slope, intercept = np.linalg.lstsq(design, xx[inliers], rcond=None)[0]
        rmse = float(np.sqrt(np.mean((xx[inliers] - slope * zz[inliers] - intercept) ** 2)))
        heading = -math.degrees(math.atan(float(slope)))
        normal_width = ww[inliers] * math.cos(math.radians(heading))
        width_median = float(np.median(normal_width))
        # The CAD specifies 35 cm. Permit camera-scale and line-centre error,
        # but reject a plausible-looking horizontal pair with the wrong normal width.
        if (rmse > 2.5 or not lane_width_cm - 9 <= width_median <= lane_width_cm + 9):
            continue
        mid_z = float(np.median(zz[inliers]))
        result.update({prefix + "valid": True, prefix + "z_cm": mid_z,
                       prefix + "z_min_cm": float(zz[inliers].min()),
                       prefix + "z_max_cm": float(zz[inliers].max()),
                       prefix + "x_cm": float(slope * mid_z + intercept),
                       prefix + "heading_deg": heading,
                       prefix + "normal_width_cm": width_median,
                       prefix + "rmse_cm": rmse})
        accepted.append((index, heading))
    result["fit_seg_count"] = len(accepted)
    # A far-only match across an unobserved middle cannot establish path
    # continuity, even if the near fit itself is sound.
    if len(accepted) < 2 or accepted[0][0] != 0 or accepted[1][0] != 1:
        return result
    # Distinguish a direction change along the path from simply entering a
    # straight lane at a yaw angle. No class is released without near support.
    delta = accepted[-1][1] - accepted[0][1]
    result["fit_seg_heading_change_deg"] = delta
    if len(accepted) == 3:
        steps = [accepted[i + 1][1] - accepted[i][1] for i in range(2)]
        if max(abs(v) for v in steps) < 4.0:
            pattern = "constant_heading"
        elif min(abs(v) for v in steps) >= 4.0 and steps[0] * steps[1] > 0:
            pattern = "left_bend" if delta > 0 else "right_bend"
        else:
            pattern = "transition_or_inconsistent"
    else:
        pattern = "heading_change_only"
    result.update(fit_seg_valid=True, fit_seg_pattern=pattern)
    return result
