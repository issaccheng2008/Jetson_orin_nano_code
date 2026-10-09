"""Optional ground-plane lane observation from independent boundary fragments.

The field layout is a weak prior. Neither parallel straights nor concentric
arcs are required. All returned target points are supported by image pixels.
This module never publishes commands and does not change the legacy detector.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _angle(points, at):
    lo, hi = max(0, at - 2), min(len(points) - 1, at + 2)
    dx, dz = points[hi] - points[lo]
    return math.degrees(math.atan2(-float(dx), float(dz)))


def _fragment_points(mask, z_by_row, scale_by_row, center_column):
    """Sample components along *their own* long axis, not along image rows."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    joined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    result = []
    for label in range(1, count):
        x0, y0, w, h, area = (int(v) for v in stats[label])
        if area < 35 or area > 18000:
            continue
        yy, xx = np.nonzero(labels[y0:y0+h, x0:x0+w] == label)
        yy, xx = yy + y0, xx + x0
        valid = (z_by_row[yy] >= 20.) & (z_by_row[yy] <= 58.)
        yy, xx = yy[valid], xx[valid]
        if len(yy) < 35:
            continue
        if len(yy) > 3000:
            stride = math.ceil(len(yy) / 3000)
            yy, xx = yy[::stride], xx[::stride]
        points = np.column_stack(((xx-center_column)*scale_by_row[yy], z_by_row[yy]))
        centre = np.median(points, axis=0)
        _, _, axes = np.linalg.svd(points-centre, full_matrices=False)
        along = (points-centre) @ axes[0]
        if np.ptp(along) < 6.:
            continue
        # Median bins remove line thickness and isolated bright pixels.
        bins = np.floor((along-along.min()) / 1.5).astype(int)
        reduced = np.asarray([np.median(points[bins == k], axis=0)
                              for k in np.unique(bins)])
        if len(reduced) < 4:
            continue
        if reduced[-1, 1] < reduced[0, 1]:
            reduced = reduced[::-1]
        result.append(reduced)
    return result


def _closest(fragment, point):
    return float(np.min(np.linalg.norm(fragment-point, axis=1)))


def _extend(seed, fragments, used):
    """Bridge short glare gaps without inventing distant centre points."""
    path = seed.copy()
    while True:
        best = None
        for index, piece in enumerate(fragments):
            if index in used:
                continue
            for reverse in (False, True):
                candidate = piece[::-1] if reverse else piece
                gap = np.linalg.norm(path[-1]-candidate[0])
                if gap > 6.:
                    continue
                first = _angle(path, len(path)-1)
                second = _angle(candidate, 0)
                if abs((first-second+180.) % 360.-180.) > 40.:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, index, candidate)
        if best is None:
            break
        _, index, candidate = best
        used.add(index)
        path = np.vstack((path, candidate))
    return path


def _centres(left, right, width):
    if left is not None and right is not None:
        candidates, widths = [], []
        for index, point in enumerate(left):
            tangent = math.radians(_angle(left, index))
            normal = np.array((math.cos(tangent), math.sin(tangent)))
            delta = right-point
            transverse = delta @ normal
            longitudinal = delta @ np.array((-math.sin(tangent), math.cos(tangent)))
            choices = np.where((transverse >= 22.) & (transverse <= 50.)
                               & (np.abs(longitudinal) <= 7.))[0]
            if len(choices):
                pick = choices[np.argmin(np.abs(longitudinal[choices]))]
                candidates.append((point+right[pick]) / 2.)
                widths.append(float(transverse[pick]))
        if len(candidates) >= 5:
            middle = float(np.median(widths))
            if float(np.median(np.abs(np.asarray(widths)-middle))) <= 6.:
                return np.asarray(candidates), middle, "paired"
    if width is None:
        return None, None, "unanchored_single"
    side, line = ("left", left) if left is not None else ("right", right)
    if line is None:
        return None, None, "no_boundary"
    sign = 1. if side == "left" else -1.
    centres = []
    for index, point in enumerate(line):
        angle = math.radians(_angle(line, index))
        centres.append(point + sign * width/2. * np.array((math.cos(angle), math.sin(angle))))
    return np.asarray(centres), float(width), "single"


def _path_observation(points, mode, width, lookahead):
    # Sort by forward distance only after component tracing and side association.
    # A nearly transverse line has too little forward support for a safe target.
    points = points[np.argsort(points[:, 1])]
    bins = np.floor(points[:, 1] / 1.5).astype(int)
    points = np.asarray([np.median(points[bins == k], axis=0) for k in np.unique(bins)])
    if len(points) < 5 or np.ptp(points[:, 1]) < 10.:
        return {"track_valid": False, "track_reason": "insufficient_forward_support"}
    near_index = int(np.argmin(np.abs(points[:, 1]-26.)))
    near_x, near_z = (float(v) for v in points[near_index])
    if abs(near_z-26.) > 8.:
        return {"track_valid": False, "track_reason": "near_not_observed"}
    choices = np.where((points[:, 1] <= lookahead) & (points[:, 1] >= near_z+8.))[0]
    if not len(choices):
        return {"track_valid": False, "track_reason": "target_not_observed"}
    target_index = int(choices[-1])
    target_x, target_z = (float(v) for v in points[target_index])
    headings = np.asarray([_angle(points, k) for k in range(len(points))])
    near_heading = float(np.median(headings[max(0, near_index-1):near_index+2]))
    # Curvature evidence is a change in local tangent, invariant to body yaw.
    # Three measured sections distinguish the two transition directions without
    # requiring an ideal parallel straight or a common centre for the arcs.
    cut1 = near_z + (target_z-near_z)/3.
    cut2 = near_z + 2.*(target_z-near_z)/3.
    sections = [headings[(points[:, 1] >= lo) & (points[:, 1] < hi)]
                for lo, hi in ((near_z, cut1), (cut1, cut2), (cut2, target_z+1.))]
    section_heading = [float(np.median(v)) if len(v) >= 2 else None for v in sections]
    steps = ((section_heading[1]-section_heading[0],
              section_heading[2]-section_heading[1])
             if all(v is not None for v in section_heading) else (0., 0.))
    heading_change = sum(steps)
    if steps[0] >= 2. and steps[1] >= 2. and heading_change >= 5.:
        phase = "left_bend"
    elif steps[0] <= 2. and steps[1] >= 4.:
        phase = "enter_left"
    elif steps[0] >= 4. and steps[1] <= 2.:
        phase = "exit_left"
    elif max(abs(v) for v in steps) < 2.:
        phase = "straight_like"
    else:
        phase = "uncertain"
    # A single boundary supports direction, but cannot settle its own identity.
    confidence = min(1., (target_z-near_z)/22.) * (0.85 if mode == "paired" else 0.45)
    return {"track_valid": True, "track_reason": mode,
            "track_mode": mode, "track_confidence": confidence,
            "track_near_cm": near_x, "track_near_z_cm": near_z,
            "track_heading_deg": near_heading,
            "track_target_x_cm": target_x, "track_target_z_cm": target_z,
            "track_target_bearing_deg": -math.degrees(math.atan2(target_x, target_z)),
            "track_heading_change_deg": heading_change,
            "track_step_near_deg": steps[0], "track_step_far_deg": steps[1],
            "track_phase": phase,
            "track_width_cm": width, "track_support_cm": target_z-near_z,
            "track_points": points}


class TrackGeometryTracker:
    def __init__(self, lookahead_cm=50.):
        self.lookahead_cm = float(lookahead_cm)
        self.left_near = self.right_near = None
        self.width_cm = None
        self.age_s = float("inf")

    def snapshot(self):
        return (None if self.left_near is None else self.left_near.copy(),
                None if self.right_near is None else self.right_near.copy(),
                self.width_cm, self.age_s)

    def restore(self, state):
        self.left_near, self.right_near, self.width_cm, self.age_s = state

    def observe(self, mask, z_by_row, scale_by_row, center_column,
                near_anchor=None, dt=.1):
        self.age_s += max(0., float(dt))
        fragments = _fragment_points(mask, z_by_row, scale_by_row, center_column)
        if near_anchor is not None:
            x, z, heading, width = near_anchor
            angle = math.radians(heading)
            normal = np.array((math.cos(angle), math.sin(angle)))
            centre = np.array((x, z))
            anchors = centre-width/2.*normal, centre+width/2.*normal
        elif self.age_s <= .6 and self.left_near is not None and self.right_near is not None:
            anchors = self.left_near, self.right_near
        else:
            # Bootstrap from two independent, mutually consistent boundaries.
            # One line alone cannot establish which side of the lane it is.
            pairs = []
            for i, first in enumerate(fragments[:12]):
                for j in range(i+1, min(len(fragments), 12)):
                    second = fragments[j]
                    left_i, right_i = ((i, j) if np.median(first[:, 0]) < np.median(second[:, 0])
                                       else (j, i))
                    centres, pair_width, mode = _centres(
                        fragments[left_i], fragments[right_i], None)
                    if mode != "paired" or not 25. <= pair_width <= 45.:
                        continue
                    near = centres[np.argmin(np.abs(centres[:, 1]-26.))]
                    if abs(near[1]-26.) > 6. or abs(near[0]) > 24.:
                        continue
                    score = abs(pair_width-35.) + .15*abs(near[0])
                    pairs.append((score, left_i, right_i))
            pairs.sort()
            if not pairs or (len(pairs) > 1 and pairs[1][0]-pairs[0][0] < 2.):
                return {"track_valid": False,
                        "track_reason": "ambiguous_bootstrap" if pairs else "no_paired_anchor",
                        "track_fragments": len(fragments)}
            _, left_i, right_i = pairs[0]
            selected, used = [left_i, right_i], {left_i, right_i}
        if near_anchor is not None or (self.age_s <= .6 and self.left_near is not None
                                      and self.right_near is not None):
            ranked = sorted(((_closest(piece, anchor), side, index)
                             for side, anchor in enumerate(anchors)
                             for index, piece in enumerate(fragments)))
            selected, used = [None, None], set()
            for distance, side, index in ranked:
                if distance > 13. or selected[side] is not None or index in used:
                    continue
                selected[side] = index
                used.add(index)
        if selected == [None, None]:
            return {"track_valid": False, "track_reason": "anchor_not_found",
                    "track_fragments": len(fragments)}
        left = _extend(fragments[selected[0]], fragments, used) if selected[0] is not None else None
        right = _extend(fragments[selected[1]], fragments, used) if selected[1] is not None else None
        centres, width, mode = _centres(left, right, self.width_cm)
        if centres is None:
            return {"track_valid": False, "track_reason": mode,
                    "track_fragments": len(fragments)}
        result = _path_observation(centres, mode, width, self.lookahead_cm)
        result["track_fragments"] = len(fragments)
        if not result["track_valid"]:
            return result
        if mode == "paired":
            self.width_cm = width if self.width_cm is None else .9*self.width_cm+.1*width
            self.age_s = 0.
        # Never let an inferred centre/width become its own paired anchor.
        if left is not None:
            self.left_near = left[np.argmin(np.abs(left[:, 1]-26.))].copy()
        if right is not None:
            self.right_near = right[np.argmin(np.abs(right[:, 1]-26.))].copy()
        return result
