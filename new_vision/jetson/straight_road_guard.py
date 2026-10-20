"""Strict near-image road shape evidence and a final walking yaw restriction.

Road shape is curvature, independent of the robot's lateral position or yaw.
Only real pixels enter the fit; gap-completed pixels and control scan bands do not.
"""
import math

import cv2
import numpy as np


def measure_near_road(mask, valid_mask, to_ground, *, near_cm=10., far_cm=40.,
                      lane_width_cm=35., row_z_cm=None, row_x_scale_cm=None, center_x=None):
    result = dict(straight_evidence='UNKNOWN', straight_reject_reason='insufficient_edges',
                  straight_near_cm=near_cm, straight_far_cm=far_cm,
                  straight_edges=0, straight_max_turn_deg=None,
                  straight_max_rmse_cm=None, straight_width_cm=None,
                  straight_rmse_limit_cm=.6, straight_turn_limit_deg=4.,
                  straight_curve_min_deg=9., straight_support_min=.75)
    zs = (np.asarray(row_z_cm) if row_z_cm is not None else
          np.array([to_ground(mask.shape[1]/2, y)[1] for y in range(mask.shape[0])]))
    rows = np.flatnonzero((zs >= near_cm) & (zs <= far_cm))
    if len(rows) < 12 or far_cm-near_cm < 15:
        result['straight_reject_reason'] = 'near_region_unsupported'
        return result
    y0, y1 = int(rows.min()), int(rows.max())+1
    observed = ((mask[y0:y1] > 0) & valid_mask[y0:y1]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(observed, 8)
    fits = []
    for label in range(1, count):
        x0, top, width, height, area = stats[label]
        if height < .75*len(rows) or area < 30:
            continue
        yy, xx = np.nonzero(labels[top:top+height, x0:x0+width] == label)
        yy += top+y0
        xx += x0
        unique, starts, sizes = np.unique(yy, return_index=True, return_counts=True)
        if len(unique) < .75*len(rows):
            continue
        centers = np.add.reduceat(xx,starts)/sizes
        widths_px = np.maximum.reduceat(xx,starts)-np.minimum.reduceat(xx,starts)
        keep = (zs[unique] >= near_cm) & (zs[unique] <= far_cm)
        unique, centers, widths_px = unique[keep], centers[keep], widths_px[keep]
        if row_x_scale_cm is not None:
            scales = np.asarray(row_x_scale_cm)[unique]
            ground = np.column_stack(((centers-center_x)*scales, zs[unique]))
            thickness = widths_px*scales
        else:
            ground = np.asarray([to_ground(float(x),int(y)) for x,y in zip(centers,unique)])
            thickness = [abs(to_ground(float(x+w/2),int(y))[0]-p[0])*2
                         for x,y,w,p in zip(centers,unique,widths_px,ground)]
        if len(ground) < 12 or np.percentile(thickness, 90) > 5.:
            continue  # joined edges, card frames and transverse strokes
        if not np.isfinite(ground).all():
            continue
        x, z = ground[:, 0], ground[:, 1]
        if np.ptp(z) < .75*(far_cm-near_cm):
            continue
        ref = .5*(near_cm+far_cm)
        coeff = np.polyfit(z-ref, x, 2)
        error = x-np.polyval(coeff, z-ref)
        keep = np.abs(error) <= 1.0
        if keep.mean() < .9 or np.ptp(z[keep]) < .75*(far_cm-near_cm):
            continue
        coeff = np.polyfit(z[keep]-ref, x[keep], 2)
        k = float(coeff[1])
        rmse = float(np.sqrt(np.mean((x[keep]-np.polyval(coeff,z[keep]-ref))**2)) /
                     math.hypot(1.,k))
        if rmse > .6:
            continue
        ends = np.array([z[keep].min(), z[keep].max()])
        angles = np.degrees(np.arctan(2*coeff[0]*(ends-ref)+k))
        fits.append(dict(coeff=coeff, ref=ref, k=k, rmse=rmse,
                         turn=float(angles[1]-angles[0]),
                         lo=float(ends[0]), hi=float(ends[1])))
    result['straight_edges'] = len(fits)
    if not fits:
        return result
    result['straight_max_turn_deg'] = max(abs(f['turn']) for f in fits)
    result['straight_max_rmse_cm'] = max(f['rmse'] for f in fits)
    # Strong observed curvature can release a straight restriction even if
    # only one complete boundary remains visible. A straight needs BOTH.
    curved = (len(fits) <= 2 and all(abs(f['turn']) >= 9. for f in fits)
              and len({math.copysign(1., f['turn']) for f in fits}) == 1)
    if len(fits) == 1 and curved:
        result.update(straight_evidence='CURVE', straight_reject_reason='observed_curvature')
        return result
    if len(fits) != 2:
        result['straight_reject_reason'] = 'need_two_unambiguous_edges'
        return result
    left, right = sorted(fits, key=lambda f: f['coeff'][2])
    lo, hi = max(left['lo'],right['lo']), min(left['hi'],right['hi'])
    if hi-lo < .75*(far_cm-near_cm):
        result['straight_reject_reason'] = 'paired_span'
        return result
    sample = np.linspace(lo, hi, 7)
    k = .5*(left['k']+right['k'])
    widths = (np.polyval(right['coeff'],sample-right['ref'])-
              np.polyval(left['coeff'],sample-left['ref']))/math.hypot(1.,k)
    width = float(np.mean(widths))
    result['straight_width_cm'] = width
    if (width <= 0 or abs(width-lane_width_cm) > .25*lane_width_cm or
            float(np.std(widths)/width) > .08):
        result['straight_reject_reason'] = 'width_inconsistent'
    elif curved:
        result.update(straight_evidence='CURVE', straight_reject_reason='observed_curvature')
    elif abs(math.degrees(math.atan(left['k'])-math.atan(right['k']))) > 6.:
        result['straight_reject_reason'] = 'directions_disagree'
    elif max(abs(f['turn']) for f in fits) > 4.:
        result['straight_reject_reason'] = 'curvature_not_straight'
    else:
        result.update(straight_evidence='STRAIGHT', straight_reject_reason='accepted')
    return result


class StraightRoadGuard:
    def __init__(self, *, enabled=True, confirm_s=1., exit_s=.6, hold_s=3.,
                 straight_wz=0., left_levels=(.3,.4,.5), right_levels=(.3,.5), yaw_sign=1):
        self.enabled = enabled
        self.confirm_s, self.exit_s, self.hold_s = confirm_s, exit_s, hold_s
        self.straight_wz, self.yaw_sign = straight_wz, yaw_sign
        # A nonzero straight rate in an angle table is not also "left gear 1".
        left = sorted(v for v in left_levels if not math.isclose(v, straight_wz, abs_tol=1e-9))
        right = sorted(-v for v in right_levels if not math.isclose(-v, straight_wz, abs_tol=1e-9))
        self.allowed = tuple(sorted(set([straight_wz]+left[:1]+right[-1:])))
        self.state = 'UNKNOWN'
        self.pending = None
        self.pending_since = None
        self.last_supported = None
        self.last_fresh = None
        self.last_token = None

    def update(self, evidence, now, token, *, paused=False):
        # Expire BEFORE consuming a new frame: one image after a long gap
        # cannot revive an old straight classification without confirmation.
        if self.last_supported is not None and now-self.last_supported > self.hold_s:
            self.state = 'UNKNOWN'
            self.pending = self.pending_since = None
            self.last_supported = None
        fresh = token is not None and token != self.last_token and not paused
        if fresh:
            if self.last_fresh is not None and now-self.last_fresh > .35:
                self.pending = self.pending_since = None
            self.last_token, self.last_fresh = token, now
            if evidence in ('STRAIGHT','CURVE'):
                if self.pending != evidence:
                    self.pending, self.pending_since = evidence, now
                if evidence == self.state:
                    self.last_supported = now
                duration = self.confirm_s if evidence == 'STRAIGHT' else self.exit_s
                if now-self.pending_since >= duration-1e-9:
                    self.state = evidence
                    self.last_supported = now
            else:
                self.pending = self.pending_since = None
        return self.state

    def limit(self, wz, *, bypass=False):
        if not self.enabled or self.state != 'STRAIGHT' or bypass:
            return wz
        geometric = wz*self.yaw_sign
        value = min(self.allowed, key=lambda v: (abs(v-geometric), abs(v-self.straight_wz)))
        return value*self.yaw_sign

    def diagnostics(self, now, before, after):
        return dict(straight_guard_enabled=self.enabled, straight_road_state=self.state,
                    straight_guard_active=self.enabled and self.state=='STRAIGHT',
                    straight_guard_support_age_s=(None if self.last_supported is None else now-self.last_supported),
                    straight_guard_pending=self.pending,
                    straight_guard_pending_s=(0. if self.pending_since is None else now-self.pending_since),
                    straight_guard_before_wz=before, straight_guard_after_wz=after,
                    straight_guard_limited=not math.isclose(before,after,abs_tol=1e-9),
                    straight_guard_allowed_wz=','.join(f'{v*self.yaw_sign:g}' for v in self.allowed))
