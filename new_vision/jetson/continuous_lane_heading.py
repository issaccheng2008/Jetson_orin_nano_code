"""Near-field, row-by-row lane heading on an already preprocessed birdseye mask.

The caller owns image preprocessing and the pixel-to-ground calibration. A row
with no foreground is never converted into an observation by this module.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HeadingScanConfig:
    bottom_y: int = 390
    top_y: int = 220
    step_px: int = 11
    fit_top_y: int = 270
    min_line_width_px: int = 5
    min_lane_width_px: float = 24.0
    max_lane_width_px: float = 300.0
    initial_width_px: float = 140.0
    search_radius_px: float = 34.0
    max_gap_rows: int = 2
    min_observed_rows: int = 6
    min_paired_rows: int = 2
    min_span_cm: float = 8.0
    max_residual_cm: float = 2.5
    history_max_age_s: float = 0.25


def _median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _fit_weighted(z, x, weight):
    z_ref = float(np.average(z, weights=weight))
    x_ref = float(np.average(x, weights=weight))
    dz = z - z_ref
    den = float(np.dot(weight, dz * dz))
    if den <= 1e-9:
        return None
    k = float(np.dot(weight, dz * (x - x_ref)) / den)
    return x_ref, k, z_ref


def _robust_fit(z, x, weight, max_residual_cm):
    """Theil-Sen seed, then bounded-residual weighted least squares."""
    slopes = [(x[j] - x[i]) / (z[j] - z[i])
              for i in range(len(z)) for j in range(i + 1, len(z))
              if abs(z[j] - z[i]) >= 1.0]
    if not slopes:
        return None
    k0 = _median(slopes)
    b0 = _median(x - k0 * z)
    residual0 = np.abs(x - (b0 + k0 * z))
    scale = max(0.45, 1.4826 * _median(residual0))
    inlier = residual0 <= max(max_residual_cm, 3.0 * scale)
    if int(np.count_nonzero(inlier)) < 4:
        return None
    zi, xi, wi = z[inlier], x[inlier], weight[inlier].copy()
    fit = _fit_weighted(zi, xi, wi)
    if fit is None:
        return None
    for _ in range(3):
        x_ref, k, z_ref = fit
        residual = xi - (x_ref + k * (zi - z_ref))
        huber = np.minimum(1.0, max(0.7, 1.5 * scale) / np.maximum(np.abs(residual), 1e-9))
        fit = _fit_weighted(zi, xi, wi * huber)
        if fit is None:
            return None
    x_ref, k, z_ref = fit
    residual = xi - (x_ref + k * (zi - z_ref))
    rmse = float(np.sqrt(np.average(residual * residual, weights=wi)))
    return x_ref, k, z_ref, rmse, inlier


def trace_heading(mask, collect_runs, to_ground, cm_per_px_at, *,
                  previous_rows=(), previous_age_s=float('inf'),
                  config=HeadingScanConfig()):
    """Return one record per actual sample row and a ground-space heading fit.

    ``collect_runs(y)`` supplies inclusive (left, right) foreground runs from
    the detector's existing extraction. Previous-frame *observations* only
    guide candidate search; they never enter the fit or create measured rows.
    """
    height, width = mask.shape[:2]
    bottom = min(height - 1, config.bottom_y)
    top = max(0, config.top_y)
    if bottom < top or config.step_px <= 0:
        raise ValueError('invalid heading scan range')
    ys = list(range(bottom, top - 1, -config.step_px))
    if ys[-1] != top:
        ys.append(top)
    history = (tuple(r for r in previous_rows if r['source'] == 'paired')
               if previous_age_s <= config.history_max_age_s else ())
    rows = []
    observed = []
    paired = []
    gap = 0
    stopped = False

    def prior(y):
        if observed:
            last = observed[-1]
            if len(observed) >= 2:
                before = observed[-2]
                dy = last['y'] - before['y']
                trend_l = (last['left_x'] - before['left_x']) / dy if dy else 0.0
                trend_r = (last['right_x'] - before['right_x']) / dy if dy else 0.0
            else:
                trend_l = trend_r = 0.0
            dy = y - last['y']
            return (last['left_x'] + trend_l * dy,
                    last['right_x'] + trend_r * dy,
                    last['width_px'] * cm_per_px_at(last['y']) / cm_per_px_at(y))
        if history:
            near = min(history, key=lambda row: abs(row['y'] - y))
            return (near['left_x'], near['right_x'],
                    near['width_px'] * cm_per_px_at(near['y']) / cm_per_px_at(y))
        return None

    for y in ys:
        record = dict(y=y, left_x=None, right_x=None, center_x=None,
                      width_px=None, source='missing', confidence=0.0, valid=False)
        # A far-only pair is not a near-field tracking seed. Capture must start
        # in the first three configured sample rows.
        if not observed and len(rows) >= 3:
            stopped = True
        if stopped:
            rows.append(record)
            continue
        runs = collect_runs(y)
        centers = [0.5 * (a + b) for a, b in runs
                   if b - a + 1 >= config.min_line_width_px]
        prediction = prior(y)
        candidates = []
        for i, left in enumerate(centers):
            for right in centers[i + 1:]:
                lane_w = right - left
                if not config.min_lane_width_px <= lane_w <= config.max_lane_width_px:
                    continue
                expected_w = prediction[2] if prediction else config.initial_width_px
                if abs(lane_w - expected_w) > max(45.0, 0.45 * expected_w):
                    continue
                if prediction and observed:
                    radius = config.search_radius_px + gap * 10.0
                    if (abs(left - prediction[0]) > radius or
                            abs(right - prediction[1]) > radius):
                        continue
                # Centre proximity is only a weak tie-break on first capture;
                # it never defines which boundary is left or right.
                centre_hint = (prediction[0] + prediction[1]) * 0.5 if prediction else width * 0.5
                score = abs(lane_w - expected_w) + 0.35 * abs((left + right) * 0.5 - centre_hint)
                candidates.append((score, left, right, lane_w))
        if candidates:
            _, left, right, lane_w = min(candidates)
            record.update(left_x=left, right_x=right, center_x=(left + right) * 0.5,
                          width_px=lane_w, source='paired', confidence=1.0, valid=True)
            paired.append(record)
        elif prediction and centers and (paired or history) and gap <= config.max_gap_rows:
            left_p, right_p, lane_w = prediction
            if config.min_lane_width_px <= lane_w <= config.max_lane_width_px:
                options = sorted((abs(c - p), side, c)
                                 for c in centers for side, p in
                                 (('left', left_p), ('right', right_p)))
                if options:
                    distance, side, actual = options[0]
                    other_distance = min((d for d, other, _ in options if other != side),
                                         default=float('inf'))
                    if (distance <= config.search_radius_px + 8 * gap and
                            other_distance - distance >= 10):
                        left = actual if side == 'left' else actual - lane_w
                        right = actual if side == 'right' else actual + lane_w
                        centre = 0.5 * (left + right)
                        if 0 <= left < right < width and 0 <= centre < width:
                            record.update(left_x=left, right_x=right, center_x=centre,
                                          width_px=lane_w, source=side,
                                          confidence=0.55, valid=True)
        if record['valid']:
            observed.append(record)
            gap = 0
        else:
            gap += 1
            if prediction and observed and gap <= config.max_gap_rows:
                left, right, lane_w = prediction
                centre = 0.5 * (left + right)
                if 0 <= left < right < width and 0 <= centre < width:
                    record.update(left_x=left, right_x=right, center_x=centre,
                                  width_px=lane_w, source='predicted', confidence=0.15)
            if gap > config.max_gap_rows:
                stopped = bool(observed)
        rows.append(record)

    fit_rows = [r for r in rows if r['valid'] and r['y'] >= config.fit_top_y]
    fit_pairs = [r for r in fit_rows if r['source'] == 'paired']
    result = dict(rows=rows, valid=False, heading_right_deg=0.0, confidence=0.0,
                  residual_cm=0.0, observed_rows=len(fit_rows),
                  inferred_rows=sum(r['source'] in ('left', 'right', 'predicted') for r in rows),
                  paired_rows=len(fit_pairs), span_cm=0.0, fit=None,
                  reason='insufficient_observations')
    if len(fit_rows) < config.min_observed_rows or len(fit_pairs) < config.min_paired_rows:
        return result
    ground = np.asarray([to_ground(r['center_x'], r['y']) for r in fit_rows], dtype=float)
    if not np.all(np.isfinite(ground)):
        result['reason'] = 'nonfinite_ground_coordinates'
        return result
    x, z = ground[:, 0], ground[:, 1]
    weight = np.asarray([1.0 if r['source'] == 'paired' else 0.4 for r in fit_rows])
    result['span_cm'] = float(np.ptp(z))
    if result['span_cm'] < config.min_span_cm:
        result['reason'] = 'insufficient_span'
        return result
    fit = _robust_fit(z, x, weight, config.max_residual_cm)
    if fit is None:
        result['reason'] = 'fit_failed'
        return result
    x_ref, k, z_ref, rmse, inlier = fit
    support = [fit_rows[i] for i in np.flatnonzero(inlier)]
    support_pairs = sum(r['source'] == 'paired' for r in support)
    support_span = float(np.ptp(z[inlier]))
    result['residual_cm'] = rmse
    result['span_cm'] = support_span
    if (len(support) < config.min_observed_rows or
            support_pairs < config.min_paired_rows or
            support_span < config.min_span_cm or rmse > config.max_residual_cm):
        result['reason'] = 'insufficient_robust_support'
        return result
    confidence = min(1.0, len(support) / 10.0) * min(1.0, support_span / 15.0)
    confidence *= (0.55 + 0.45 * support_pairs / len(support))
    confidence *= math.exp(-rmse / config.max_residual_cm)
    result.update(valid=True, heading_right_deg=math.degrees(math.atan(k)),
                  confidence=confidence, fit=(x_ref, k, z_ref), reason='accepted')
    return result
