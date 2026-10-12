"""Track real lane edges and derive a centre path from one measured boundary.

X is ground-right and Z is forward, both in centimetres. Only real paired
observations establish edge identity and update measured width. A configured
equal-width prior may support a continuously observed single edge indefinitely;
history alone can never publish a valid path.
"""

import math

import numpy as np


class BoundaryPathTracker:
    def __init__(self, *, min_points=6, min_span_cm=8., max_rmse_cm=2.5,
                 max_slope=1.2, frame_search_px=52., row_search_px=28.,
                 history_age_s=.6, width_tolerance=.18,
                 max_direction_step_deg=35.):
        self.min_points = min_points
        self.min_span_cm = min_span_cm
        self.max_rmse_cm = max_rmse_cm
        self.max_slope = max_slope
        self.frame_search_px = frame_search_px
        self.row_search_px = row_search_px
        self.history_age_s = history_age_s
        self.width_tolerance = width_tolerance
        self.max_direction_step_deg = max_direction_step_deg
        self.left_track = None
        self.right_track = None
        self.width_measured_cm = None
        self.width_measured_s = None
        self.width_prior_cm = None
        self.width_conflict_count = 0
        self.width_conflict_sample_cm = None
        self.width_conflict_sample_s = None
        self.width_conflict_until_s = None
        self.last_frame_token = None
        self.last_frame_s = None
        self.last_path = None
        self.last_mode = 'UNKNOWN'
        self.pending_pair = None

    @staticmethod
    def _empty(reason):
        return dict(tracking_mode='UNKNOWN' if reason == 'side_ambiguous' else 'LOST',
                    measurement_source=None, measurement_valid=False,
                    near_error_cm=0., near_z_cm=0., heading_control_deg=0.,
                    heading_control_valid=False, boundary_quality=0., path_quality=0.,
                    width_source='unavailable', width_cm=None, width_uncertainty_cm=None,
                    width_age_s=None, width_measured_cm=None,
                    width_conflict_count=0, width_conflict_active=False,
                    observed_left_age_s=None,
                    observed_right_age_s=None, identity_confidence=0.,
                    boundary_rmse_cm=None, boundary_span_cm=None,
                    observed_left_points=0, observed_right_points=0,
                    reject_reason=reason, target_points=[], left_points=[], right_points=[])

    @staticmethod
    def _fit(points, to_ground, min_points, min_span_cm, max_rmse_cm, max_slope):
        if len(points) < min_points:
            return None, 'insufficient_points'
        ground = np.asarray([to_ground(x,y) for y,x in points], dtype=float)
        if not np.isfinite(ground).all():
            return None, 'nonfinite_ground'
        x,z = ground[:,0],ground[:,1]
        span = float(np.ptp(z))
        if span < min_span_cm:
            return None, 'insufficient_span'
        # Robust seed, then fit only genuine points within a bounded residual.
        k0 = float(np.median([(x[j]-x[i])/(z[j]-z[i])
                              for i in range(len(z)) for j in range(i+1,len(z))
                              if abs(z[j]-z[i]) >= 2.]))
        b0 = float(np.median(x-k0*z))
        keep = np.abs(x-(b0+k0*z)) <= max_rmse_cm
        if int(keep.sum()) < min_points or float(np.ptp(z[keep])) < min_span_cm:
            return None, 'inconsistent_points'
        k,b = np.polyfit(z[keep],x[keep],1)
        rmse = float(np.sqrt(np.mean((x[keep]-(b+k*z[keep]))**2)))
        if not np.isfinite([k,b,rmse]).all() or abs(k) > max_slope or rmse > max_rmse_cm:
            return None, 'unreliable_fit'
        return dict(k=float(k),b=float(b),rmse=rmse,
                    z_min=float(np.min(z[keep])),z_max=float(np.max(z[keep])),
                    points=int(keep.sum()),ground=ground[keep]), 'accepted'

    def _fitted(self, points, to_ground):
        return self._fit(points,to_ground,self.min_points,self.min_span_cm,
                         self.max_rmse_cm,self.max_slope)

    def _record_paired_width(self, measured, prior, now_s):
        """A noisy single paired width must not veto an established track prior."""
        if not 20. <= measured <= 50.:
            self.width_conflict_count = 0
            return
        self.width_measured_cm = measured
        self.width_measured_s = now_s
        if abs(measured-prior) <= self.width_tolerance*prior:
            self.width_conflict_count = 0
            self.width_conflict_until_s = None
        elif (self.width_conflict_sample_s is not None and
              now_s-self.width_conflict_sample_s <= .4 and
              abs(measured-self.width_conflict_sample_cm) <= 1.5):
            self.width_conflict_count += 1
        else:
            self.width_conflict_count = 1
        self.width_conflict_sample_cm = measured
        self.width_conflict_sample_s = now_s
        if self.width_conflict_count >= 3:
            # A confirmed discrepancy remains actionable briefly. With no new
            # paired evidence, a continuously observed real edge can recover.
            self.width_conflict_until_s = now_s + .75

    def _width_conflict_active(self, now_s):
        return (self.width_conflict_until_s is not None and
                now_s <= self.width_conflict_until_s)

    def _local_fit(self, points, to_ground, side, width_cm, reference_z_cm):
        """Choose short supported tangents on a bend, never a distant extrapolation."""
        normal_sign=-1 if side=='right' else 1
        lengths=sorted(set((len(points),10,12,14)),reverse=True)
        choices=[]
        for length in lengths:
            if length<self.min_points or length>len(points):
                continue
            for start in range(len(points)-length+1):
                fit,reason=self._fitted(points[start:start+length],to_ground)
                if fit is None:
                    continue
                k=fit['k']
                dz=-(width_cm/2)*normal_sign*k/math.hypot(1.,k)
                target_min,target_max=fit['z_min']+dz,fit['z_max']+dz
                if not target_min-3.<=reference_z_cm<=target_max+3.:
                    continue
                target_mid=.5*(target_min+target_max)
                score=(fit['rmse']+.025*abs(target_mid-reference_z_cm)
                       -.015*(fit['z_max']-fit['z_min']))
                choices.append((score,fit))
        return min(choices,key=lambda item:item[0])[1] if choices else None

    @staticmethod
    def _predict(track, y):
        points=track['points']
        rows=np.asarray([p[0] for p in points],dtype=float)
        columns=np.asarray([p[1] for p in points],dtype=float)
        if len(points)>=3 and float(np.ptp(rows))>1.:
            slope=float(np.polyfit(rows,columns,1)[0])
        else:
            slope=0.
        i=int(np.argmin(np.abs(rows-y)))
        return float(columns[i]+slope*(y-rows[i]))

    def _rows(self, mask, collect_runs, *, bottom_y, top_y, step_px):
        bottom=min(mask.shape[0]-1,bottom_y)
        top=max(0,top_y)
        ys=list(range(bottom,top-1,-step_px))
        if ys and ys[-1]!=top:
            ys.append(top)
        for y in ys:
            centers=[.5*(a+b) for a,b in collect_runs(y)
                     if 5 <= b-a+1 <= 32]
            yield y,centers

    def _single_points(self, mask, collect_runs, side, now_s, initial_side,
                       *, bottom_y, top_y, step_px):
        track=self.left_track if side=='left' else self.right_track
        fresh=(track is not None and now_s-track['time_s']<=self.history_age_s)
        if not fresh and initial_side!=side:
            return [], 'side_ambiguous'
        points=[]
        for y, centers in self._rows(mask,collect_runs,bottom_y=bottom_y,
                                     top_y=top_y,step_px=step_px):
            if not points and y < min(mask.shape[0]-1,bottom_y)-2*step_px:
                break  # an isolated far object cannot seed near tracking
            if not centers:
                if points:
                    break
                continue
            if points:
                if len(points)>=2:
                    dy=points[-1][0]-points[-2][0]
                    trend=(points[-1][1]-points[-2][1])/dy if dy else 0.
                else:
                    trend=0.
                expected=points[-1][1]+trend*(y-points[-1][0])
                radius=self.row_search_px
            elif fresh:
                expected=self._predict(track,y)
                radius=self.frame_search_px
            else:
                if len(centers)!=1:
                    return [],'side_ambiguous'
                expected=centers[0]
                radius=1.
            distances=sorted((abs(c-expected),c) for c in centers)
            if (distances[0][0]>radius or
                    (len(distances)>1 and distances[1][0]-distances[0][0]<10.)):
                if points:
                    break
                return [],'association_ambiguous'
            points.append((y,distances[0][1]))
        return points,'accepted' if points else 'no_near_boundary'

    def update(self, mask, collect_runs, to_ground, *, now_s, width_cm,
               reference_z_cm, paired_trace=None, initial_side=None,
               frame_token=None, bottom_y=390, top_y=220, step_px=11,
               width_prior_applicable=True):
        result=self._empty('no_near_boundary')
        for side in ('left','right'):
            track=self.left_track if side=='left' else self.right_track
            result['observed_'+side+'_age_s']=(now_s-track['time_s']
                if track is not None and math.isfinite(now_s) else None)
        result['width_age_s']=(now_s-self.width_measured_s
            if self.width_measured_s is not None and math.isfinite(now_s) else None)
        result['width_measured_cm']=self.width_measured_cm
        result['width_conflict_count']=self.width_conflict_count
        result['width_conflict_active']=self._width_conflict_active(now_s)
        if (not math.isfinite(now_s) or not math.isfinite(width_cm)
                or not 20. <= width_cm <= 50. or not math.isfinite(reference_z_cm)):
            result['reject_reason']='invalid_width_prior'
            return result
        if self.width_prior_cm is not None and width_cm != self.width_prior_cm:
            result['reject_reason']='width_prior_changed'
            return result
        if self.last_frame_s is not None and now_s<=self.last_frame_s:
            result['reject_reason']='stale_frame'
            return result
        if frame_token is not None and frame_token==self.last_frame_token:
            result['reject_reason']='duplicate_frame'
            return result
        self.last_frame_s=now_s
        self.last_frame_token=frame_token
        self.width_prior_cm=width_cm
        pair_rows=([r for r in paired_trace['rows'] if r['source']=='paired']
                   if paired_trace is not None and paired_trace.get('valid') else [])
        if len(pair_rows)>=self.min_points:
            left=[(r['y'],r['left_x']) for r in pair_rows]
            right=[(r['y'],r['right_x']) for r in pair_rows]
            lf,lr=self._fitted(left,to_ground)
            rf,rr=self._fitted(right,to_ground)
            if lf and rf:
                k=float(paired_trace['fit'][1])
                widths=[(to_ground(r['right_x'],r['y'])[0]
                         -to_ground(r['left_x'],r['y'])[0])/math.hypot(1.,k)
                        for r in pair_rows]
                measured=float(np.median(widths))
                xref,k,zref=paired_trace['fit']
                x=float(xref+k*(reference_z_cm-zref))
                if (self.last_mode in ('LEFT_ONLY','RIGHT_ONLY') and
                        self.last_path is not None and
                        abs(x-(self.last_path[0]+self.last_path[1]*
                               (reference_z_cm-self.last_path[2])))>8.):
                    pending=self.pending_pair
                    confirmed=(pending is not None and now_s-pending[2]<.5 and
                               abs(x-pending[0])<4. and abs(k-pending[1])<.25)
                    self.pending_pair=None if confirmed else (x,k,now_s)
                    if not confirmed:
                        result['reject_reason']='pair_path_disagreement'
                        return result
                self.pending_pair=None
                self._record_paired_width(measured,width_cm,now_s)
                self.left_track=dict(points=left,fit=lf,time_s=now_s)
                self.right_track=dict(points=right,fit=rf,time_s=now_s)
                result.update(tracking_mode='BOTH',measurement_source='paired',
                    measurement_valid=True,near_error_cm=x,near_z_cm=reference_z_cm,
                    heading_control_deg=-math.degrees(math.atan(k)),
                    heading_control_valid=True,boundary_quality=.9,path_quality=.9,
                    width_source='paired_measured',width_cm=measured,
                    width_uncertainty_cm=max(.5,float(np.std(widths))),
                    width_age_s=0.,width_measured_cm=self.width_measured_cm,
                    width_conflict_count=self.width_conflict_count,
                    width_conflict_active=self._width_conflict_active(now_s),
                    observed_left_age_s=0.,
                    observed_right_age_s=0.,identity_confidence=1.,
                    boundary_rmse_cm=max(lf['rmse'],rf['rmse']),
                    boundary_span_cm=min(lf['z_max']-lf['z_min'],
                                         rf['z_max']-rf['z_min']),
                    observed_left_points=len(left),observed_right_points=len(right),
                    left_points=left,right_points=right,reject_reason='accepted')
                self.last_path=(x,k,reference_z_cm,now_s)
                self.last_mode='BOTH'
                return result
        if not width_prior_applicable:
            result['reject_reason']='width_prior_not_applicable'
            return result
        # Real side identity comes from previously paired, temporally associated
        # boundary points. Cold start requires an explicit side and one clear run.
        candidates=[]
        for side in ('left','right'):
            points,reason=self._single_points(mask,collect_runs,side,now_s,initial_side,
                bottom_y=bottom_y,top_y=top_y,step_px=step_px)
            if points:
                fit=self._local_fit(points,to_ground,side,width_cm,reference_z_cm)
                if fit:
                    candidates.append((side,points,fit))
        if len(candidates)!=1:
            result['reject_reason']=('side_ambiguous' if len(candidates)>1 or
                                     not (self.left_track or self.right_track or initial_side)
                                     else 'no_reliable_boundary')
            if result['reject_reason']=='side_ambiguous':
                result['tracking_mode']='UNKNOWN'
            return result
        side,points,fit=candidates[0]
        k,b=fit['k'],fit['b']
        old=self.left_track if side=='left' else self.right_track
        if old is not None and now_s-old['time_s']<=self.history_age_s:
            angle_step=abs(math.degrees(math.atan(k)-math.atan(old['fit']['k'])))
            if angle_step>self.max_direction_step_deg:
                result['reject_reason']='direction_jump'
                return result
        if self._width_conflict_active(now_s):
            # Preserve only the actually observed edge while the width is in
            # dispute; it cannot publish a virtual centre until the brief
            # conflicting evidence expires or a compatible pair returns.
            track=dict(points=points,fit=fit,time_s=now_s)
            if side=='left':
                self.left_track=track
            else:
                self.right_track=track
            result.update(reject_reason='width_prior_conflict',
                          width_source='paired_conflict',
                          width_cm=self.width_measured_cm,
                          width_conflict_count=self.width_conflict_count,
                          width_conflict_active=True,
                          observed_left_points=len(points) if side=='left' else 0,
                          observed_right_points=len(points) if side=='right' else 0,
                          left_points=points if side=='left' else [],
                          right_points=points if side=='right' else [])
            result['observed_'+side+'_age_s']=0.
            return result
        d=width_cm/2.
        normal_sign=-1 if side=='right' else 1
        # Offset BOTH X and Z, then re-intersect the offset line with the
        # common reference Z. Horizontal half-width translation is wrong on
        # a tilted boundary by d*(sqrt(1+k*k)-1).
        nx=normal_sign/math.hypot(1.,k)
        nz=-normal_sign*k/math.hypot(1.,k)
        target_b=b+d*(nx-k*nz)
        target_min=fit['z_min']+d*nz
        target_max=fit['z_max']+d*nz
        if not target_min-3.<=reference_z_cm<=target_max+3.:
            result['reject_reason']='reference_unsupported'
            return result
        x=target_b+k*reference_z_cm
        if (self.last_mode=='BOTH' and self.last_path is not None and
                now_s-self.last_path[3]<.5 and
                abs(x-(self.last_path[0]+self.last_path[1]*
                       (reference_z_cm-self.last_path[2])))>8.):
            result['reject_reason']='path_disagreement'
            return result
        quality=max(.21,min(.65,.75*fit['points']/10.*
                              math.exp(-fit['rmse']/self.max_rmse_cm)))
        result.update(tracking_mode=side.upper()+'_ONLY',
            measurement_source=side+'_boundary',measurement_valid=True,
            near_error_cm=float(x),near_z_cm=float(reference_z_cm),
            heading_control_deg=-math.degrees(math.atan(k)),
            heading_control_valid=True,boundary_quality=quality,
            path_quality=quality*.85,width_source='configured_equal_width',
            width_cm=width_cm,width_uncertainty_cm=max(1.5,fit['rmse']+1.),
            identity_confidence=.8 if old is not None else .55,
            boundary_rmse_cm=fit['rmse'],
            boundary_span_cm=fit['z_max']-fit['z_min'],
            observed_left_points=len(points) if side=='left' else 0,
            observed_right_points=len(points) if side=='right' else 0,
            left_points=points if side=='left' else [],
            right_points=points if side=='right' else [],
            target_points=[(z,target_b+k*z) for z in np.linspace(
                max(target_min,reference_z_cm-8.),min(target_max,reference_z_cm+15.),8)],
            reject_reason='accepted')
        result['observed_'+side+'_age_s']=0.
        track=dict(points=points,fit=fit,time_s=now_s)
        if side=='left':
            self.left_track=track
        else:
            self.right_track=track
        self.last_path=(float(x),k,reference_z_cm,now_s)
        self.last_mode=result['tracking_mode']
        return result
