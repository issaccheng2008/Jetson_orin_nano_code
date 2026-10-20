"""Classify final cached outputs, including intentional forecast-release exceptions."""
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"replay_final"
rows=list(csv.DictReader((OUT/"per_frame.csv").open(encoding="utf-8-sig")))
segments=list(csv.DictReader((OUT/"command_segments.csv").open(encoding="utf-8-sig")))
boundary_releases=[];boundary_weak_violations=[];raw_boundary_hold=[];stale_motion=[]
for row in rows:
    near=float(row["near_error_cm"]);heading=float(row["ground_heading_deg"])
    vx,wz=float(row["levels_vx"]),float(row["levels_wz"])
    loc=dict(video=row["video"],frame_index=int(row["frame_index"]))
    if row.get("levels_steering_decision") in ("left_corridor","right_corridor") and abs(wz)<.4-1e-12:
        if wz==0 and row.get("levels_steering_braked")=="True":
            boundary_releases.append(dict(loc,predicted_demand_deg=float(row["levels_steering_predicted_demand_deg"])))
        else:boundary_weak_violations.append(loc)
    if (row["measurement_valid"]=="True" and row["ground_heading_valid"]=="True"
        and ((near<=-8 and heading>=0) or (near>=8 and heading<=0)) and abs(wz)<.4-1e-12):
        raw_boundary_hold.append(dict(loc,near_cm=near,heading_deg=heading,wz=wz,
                                     reason=row.get("levels_steering_reason")))
    if row["measurement_stale"]=="True" and (vx!=0 or wz!=0):stale_motion.append(loc)
manifest=json.loads((HERE/"source_final/manifest.json").read_text())
result=dict(frames=len(rows),
    frozen_at_zh=datetime.fromtimestamp(manifest["frozen_at_unix_s"],timezone(timedelta(hours=8))).isoformat(),
    command_levels=dict(Counter(row["levels_wz"] for row in rows)),
    decisions=dict(Counter(row.get("levels_steering_decision") or "no_new_decision" for row in rows)),
    command_pair_different_frames=sum((float(row["levels_vx"]),float(row["levels_wz"]))!=(float(row["prior_heading_vx"]),float(row["prior_heading_wz"])) for row in rows),
    normal_completed_segments=sum(s["normal_hold_evaluable"]=="True" for s in segments),
    normal_short_changes=sum(s["normal_short_change"]=="True" for s in segments),
    geometry_stop_frames=sum(row.get("levels_steering_reason")=="geometry_lost" for row in rows),
    safety_interruptions=sum(s["ended_reason"]=="geometry_lost" for s in segments),
    prior_turning_frames=sum(abs(float(row["prior_heading_wz"]))>1e-12 for row in rows),
    levels_turning_frames=sum(abs(float(row["levels_wz"]))>1e-12 for row in rows),
    braked_frames=sum(row.get("levels_steering_braked")=="True" for row in rows),
    boundary_forecast_release_frames=boundary_releases,
    boundary_weak_violation_frames=boundary_weak_violations,
    raw_outside_not_inward_weak_command_frames=raw_boundary_hold,
    stale_motion=stale_motion,
    note="weak_corridor_decision_frames in raw replay summary is a probe of abs(wz)<.4, not an assertion of a bug: forecast re-entry releases to zero intentionally. Raw boundary cases during minimum_hold document latency, not a new-decision violation.")
(OUT/"validation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
