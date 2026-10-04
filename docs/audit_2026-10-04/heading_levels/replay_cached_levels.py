"""Replay frozen command-level controller against existing scalar detection cache.

No video decoding, image detection, CV dependency, UDP, model or motor execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=HERE.parent/"heading_hold/replay_final/per_frame.csv"

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar(value):
    if value=="":return None
    if value=="True":return True
    if value=="False":return False
    try:return int(value)
    except ValueError:
        try:return float(value)
        except ValueError:return value

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)

def finish(segment,now,reason,complete):
    result=dict(segment,end_time_s=now,duration_s=now-segment["start_time_s"],
                ended_reason=reason,completed=complete)
    safety=("geometry_lost","invalid_clock","initial_wait")
    result["normal_hold_evaluable"]=(complete and reason not in safety
        and result["start_reason"] not in safety)
    result["normal_short_change"]=(result["normal_hold_evaluable"]
                                   and result["duration_s"]<.5-1e-9)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    if args.freeze:
        args.snapshot.mkdir(parents=True,exist_ok=False)
        for name in ("heading_steering.py","policy_bridge.py"):
            shutil.copyfile(ROOT/"new_vision/jetson"/name,args.snapshot/name)
        manifest=dict(frozen_at_unix_s=time.time(),sha256={
            p.name:digest(p) for p in args.snapshot.iterdir() if p.is_file()})
        (args.snapshot/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    manifest=json.loads((args.snapshot/"manifest.json").read_text())
    assert all(digest(args.snapshot/name)==sha for name,sha in manifest["sha256"].items())
    args.output.mkdir(parents=True,exist_ok=False)
    controller_cls=load(args.snapshot/"heading_steering.py","cached_levels_controller").HeadingSteeringController
    inner_cls=load(args.snapshot/"policy_bridge.py","cached_levels_inner").SteeringController
    config=dict(lookahead_cm=50.,right_tolerance_deg=12.,left_tolerance_deg=4.,
        full_scale_deg=20.,max_step=.5,allow_right=True,corridor_cm=8.)
    inner_config=dict(vx=.5,max_wz=.5,max_wz_right=.5,yaw_sign=1,lost_hold_s=.2)
    cached=list(csv.DictReader(CACHE.open(encoding="utf-8-sig",newline="")))
    assert len(cached)==571
    groups={}
    for row in cached:groups.setdefault(row["video"],[]).append(row)
    assert len(groups)==4
    rows,segments,summary=[],[],{}
    allowed={-.5,-.4,-.1,0.,.4,.5}
    for video,source_rows in groups.items():
        controller=controller_cls(inner_cls(**inner_config),**config)
        active=None
        decision_counts=Counter();reason_counts=Counter();brake_increase=[];bad_levels=[]
        previous_pair=None;weak_corridor=[]
        for frame_index,source in enumerate(source_rows):
            assert int(source["frame_index"])==frame_index
            debug={key[4:]:scalar(value) for key,value in source.items() if key.startswith("det_")}
            dt=float(source["dt_s"]);stamp=float(source["time_s"])
            assert dt>0 and math.isfinite(dt)
            pair=controller.command(debug,float(source["confidence"]),dt)
            diag=controller.diagnostics
            reason=diag.get("steering_reason","unknown")
            decision=diag.get("steering_decision","no_new_decision")
            decision_counts[decision]+=1;reason_counts[reason]+=1
            if pair[1] not in allowed:bad_levels.append(frame_index)
            if diag.get("steering_braked",False) and previous_pair and abs(pair[1])>abs(previous_pair[1])+1e-12:
                brake_increase.append(frame_index)
            if decision in ("left_corridor","right_corridor") and abs(pair[1])<.4-1e-12:
                weak_corridor.append(frame_index)
            row=dict(video=video,frame_index=frame_index,time_s=stamp,dt_s=dt,
                prior_heading_vx=float(source["new_vx"]),prior_heading_wz=float(source["new_wz"]),
                levels_vx=pair[0],levels_wz=pair[1],confidence=float(source["confidence"]),
                measurement_valid=debug.get("measurement_valid"),measurement_stale=debug.get("measurement_stale"),
                ground_heading_valid=debug.get("heading_control_valid"),
                ground_heading_deg=debug.get("heading_control_deg"),near_error_cm=debug.get("near_error_cm"))
            row.update({"levels_"+key:value for key,value in diag.items()})
            rows.append(row)
            if active is None or (active["vx"],active["wz"])!=pair:
                if active is not None:segments.append(finish(active,stamp,reason,True))
                active=dict(video=video,start_frame=frame_index,start_time_s=stamp,
                    vx=pair[0],wz=pair[1],start_reason=("initial_wait" if frame_index==0 and pair==(0.,0.) else reason))
            previous_pair=pair
        segments.append(finish(active,float(source_rows[-1]["time_s"])+float(source_rows[-1]["dt_s"]),"cache_end",False))
        selected=[r for r in rows if r["video"]==video]
        video_segments=[s for s in segments if s["video"]==video]
        summary[video]=dict(frames=len(selected),command_level_frames=dict(Counter(str(r["levels_wz"]) for r in selected)),
            reason_counts=dict(reason_counts),decision_counts=dict(decision_counts),
            bad_command_levels=bad_levels,brake_amplitude_increases=brake_increase,
            weak_corridor_decision_frames=weak_corridor,
            normal_completed_segments=sum(s["normal_hold_evaluable"] for s in video_segments),
            normal_short_change_count=sum(s["normal_short_change"] for s in video_segments),
            normal_min_duration_s=min((s["duration_s"] for s in video_segments if s["normal_hold_evaluable"]),default=None),
            geometry_stop_frames=sum(r.get("levels_steering_reason")=="geometry_lost" for r in selected),
            safety_interruptions=sum(s["ended_reason"]=="geometry_lost" for s in video_segments),
            prior_full_stop_frames=sum(r["prior_heading_vx"]==0 and r["prior_heading_wz"]==0 for r in selected),
            levels_full_stop_frames=sum(r["levels_vx"]==0 and r["levels_wz"]==0 for r in selected),
            command_pair_different_frames=sum((r["levels_vx"],r["levels_wz"])!=(r["prior_heading_vx"],r["prior_heading_wz"]) for r in selected))
    write_csv(args.output/"per_frame.csv",rows)
    write_csv(args.output/"command_segments.csv",segments)
    metadata=dict(cache=str(CACHE.relative_to(ROOT)),cache_sha256=digest(CACHE),
        source_manifest=manifest,settings=config,inner_settings=inner_config,
        cached_detector_provenance=json.loads((CACHE.parent/"input_config_source.json").read_text(encoding="utf-8")),
        conditions="571 scalar detector cache rows, exact cached dt/video order, fresh controller state per video. No video redetection or new hardware data. Comparison is prior new_wz (continuous-amplitude heading, allow_right=False), not oldDiscrete.",
        limits="Open-loop cached data cannot validate trajectory, field safety, actuator response or causal effect of newly generated command. CSV scalar blank is restored to None; True/False are booleans; other numeric/text values are retained.")
    (args.output/"input_config_source.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
