"""Frozen, no-IO comparison of old discrete and new heading on four raw videos.

Use --freeze only after integration review has finished; existing evidence is
never overwritten. FPS defines dt. No inference, UDP, motor or IMU execution.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / "new_vision/jetson"
SOURCES = ("line_detector_v1_warp.py", "heading_steering.py", "discrete_steering.py",
           "policy_bridge.py", "utils.py", "run_policy_vision.py")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def csv_write(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)

def distribution(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return ({"n": len(values), "median": statistics.median(values),
             "p95": float(np.percentile(values,95)), "min": min(values), "max": max(values)}
            if values else {"n":0})

def close_segment(segments, segment, ended_s, ended_reason, completed):
    segment = dict(segment, end_time_s=ended_s,
                   duration_s=ended_s-segment["start_time_s"], ended_reason=ended_reason,
                   completed=completed)
    safety = ("geometry_lost", "invalid_clock", "loss", "stale")
    segment["normal_hold_evaluable"] = (completed and segment["controller"] == "new_heading"
        and segment["start_reason"] not in safety+("initial_wait",)
        and ended_reason not in safety)
    segment["normal_short_change"] = (segment["normal_hold_evaluable"]
                                     and segment["duration_s"] < .5-1e-9)
    segments.append(segment)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=HERE/"source_final")
    parser.add_argument("--output", type=Path, default=HERE/"replay_final")
    parser.add_argument("--freeze", action="store_true")
    args=parser.parse_args()
    if args.freeze:
        args.snapshot.mkdir(parents=True, exist_ok=False)
        for name in SOURCES:
            shutil.copyfile(RUNTIME/name,args.snapshot/name)
        shutil.copyfile(ROOT/"new_vision/config/cameras.json",args.snapshot/"cameras.json")
        shutil.copyfile(RUNTIME/"camera_config.py",args.snapshot/"camera_config.py")
        manifest={p.name:digest(p) for p in args.snapshot.iterdir() if p.is_file()}
        (args.snapshot/"manifest.json").write_text(json.dumps(
            {"frozen_at_unix_s":time.time(),"sha256":manifest},indent=2)+"\n",encoding="utf-8")
    manifest=json.loads((args.snapshot/"manifest.json").read_text())
    assert all(digest(args.snapshot/name)==sha for name,sha in manifest["sha256"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0,str(args.snapshot))
    detector_cls=load(args.snapshot/"line_detector_v1_warp.py","hold_frozen_detector").LineDetector
    inner_cls=load(args.snapshot/"policy_bridge.py","hold_frozen_policy").SteeringController
    old_cls=load(args.snapshot/"discrete_steering.py","hold_frozen_old").DiscreteSteeringController
    new_cls=load(args.snapshot/"heading_steering.py","hold_frozen_new").HeadingSteeringController
    camera=json.loads((args.snapshot/"cameras.json").read_text(encoding="utf-8"))["cameras"]["usb_main"]
    old_settings=dict(fire_cm=5.0,stop_cm=5.0,turn_s=1.0,gap_s=2.5,step=.5,allow_right=False)
    new_settings=dict(lookahead_cm=50.0,right_tolerance_deg=12.0,left_tolerance_deg=4.0,
                      full_scale_deg=20.0,max_step=.5,allow_right=False)
    inner_settings=dict(vx=.5,max_wz=.5,max_wz_right=.5,yaw_sign=1,lost_hold_s=.2)
    rows,segments,videos=[],[],[]
    for path in sorted((ROOT.parent.parent/"Robocup").glob("*_flip.mp4")):
        cap=cv2.VideoCapture(str(path))
        assert cap.isOpened(),path
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps>0
        videos.append(dict(video=path.name,sha256=digest(path),fps=fps,total_frames=count))
        detector=old=new=None
        active={}
        frame_index=0
        while True:
            ok,raw=cap.read()
            if not ok:break
            if detector is None:
                detector=detector_cls(raw.shape[1],raw.shape[0],
                    cam_height_cm=camera["mount_height_cm"],cam_pitch_deg=camera["pitch_deg"],
                    cam_vfov_deg=camera["vfov_deg"],lane_width_cm=camera["lane_width_cm"],
                    z_calib=(camera["distance_calib"]["a"],camera["distance_calib"]["b"]))
                old=old_cls(inner_cls(**inner_settings),**old_settings)
                new=new_cls(inner_cls(**inner_settings),**new_settings)
            start=time.perf_counter()
            _,heading,confidence,_,debug=detector.process(raw,dt=1/fps)
            detector_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter()
            old_pair=old.command(debug,confidence,1/fps)
            old_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter()
            new_pair=new.command(debug,confidence,1/fps)
            new_ms=1000*(time.perf_counter()-start)
            stamp=frame_index/fps
            row=dict(video=path.name,frame_index=frame_index,time_s=stamp,dt_s=1/fps,
                fps=fps,confidence=confidence,old_pixel_heading_deg=heading,
                new_ground_heading_deg=debug["heading_control_deg"],
                old_vx=old_pair[0],old_wz=old_pair[1],new_vx=new_pair[0],new_wz=new_pair[1],
                old_turn_remaining_s=old.turn_left,old_gap_remaining_s=old.gap_left,
                new_hold_remaining_s=new.turn_left,detector_ms=detector_ms,
                old_controller_ms=old_ms,new_controller_ms=new_ms)
            for key,value in debug.items():
                if value is None or isinstance(value,(str,int,float,bool)):
                    row["det_"+key]=value
            row.update({"new_"+key:value for key,value in new.diagnostics.items()})
            rows.append(row)
            for name,pair in (("old_discrete",old_pair),("new_heading",new_pair)):
                reason=(new.diagnostics.get("steering_reason","unknown") if name=="new_heading"
                    else "stale" if debug.get("measurement_stale",False)
                    else "loss" if old.lost_s>0 else "valid_frame")
                previous=active.get(name)
                if previous is None or (previous["vx"],previous["wz"])!=pair:
                    if previous is not None:
                        close_segment(segments,previous,stamp,reason,True)
                    active[name]=dict(video=path.name,controller=name,start_frame=frame_index,
                        start_time_s=stamp,vx=pair[0],wz=pair[1],
                        start_reason=("initial_wait" if frame_index==0 and pair==(0.,0.) else reason))
            frame_index+=1
        cap.release()
        assert frame_index==count
        for segment in active.values():
            close_segment(segments,segment,frame_index/fps,"video_end",False)
        print(f"{path.name}: {frame_index} frames at {fps:.3f} FPS",flush=True)
    assert len(rows)==571 and len(videos)==4
    csv_write(args.output/"per_frame.csv",rows)
    csv_write(args.output/"command_segments.csv",segments)
    summary={}
    for video in videos:
        selected=[r for r in rows if r["video"]==video["video"]]
        jointly_valid=[r for r in selected if r["det_heading_valid"] and r["det_heading_control_valid"]]
        new_segments=[s for s in segments if s["video"]==video["video"] and s["controller"]=="new_heading"]
        summary[video["video"]]=dict(frames=len(selected),
            measurement_valid=sum(bool(r["det_measurement_valid"]) for r in selected),
            ground_heading_valid=sum(bool(r["det_heading_control_valid"]) for r in selected),
            jointly_valid_headings=len(jointly_valid),
            ground_vs_pixel_abs_diff_deg=distribution(abs(r["new_ground_heading_deg"]-r["old_pixel_heading_deg"]) for r in jointly_valid),
            opposite_heading_sign=sum(r["new_ground_heading_deg"]*r["old_pixel_heading_deg"]<0 for r in jointly_valid),
            old_turning_frames=sum(abs(r["old_wz"])>1e-12 for r in selected),
            new_turning_frames=sum(abs(r["new_wz"])>1e-12 for r in selected),
            new_geometry_stop_frames=sum(r.get("new_steering_reason")=="geometry_lost" for r in selected),
            new_braked_frames=sum(bool(r.get("new_steering_braked",False)) for r in selected),
            normal_completed_segments=sum(s["normal_hold_evaluable"] for s in new_segments),
            normal_short_change_count=sum(s["normal_short_change"] for s in new_segments),
            new_normal_duration_s=distribution(s["duration_s"] for s in new_segments if s["normal_hold_evaluable"]),
            detector_ms=distribution(r["detector_ms"] for r in selected))
    config=dict(videos=videos,source_manifest=manifest,camera=camera,old_settings=old_settings,
        new_settings=new_settings,inner_settings=inner_settings,
        conditions="Sequential original robot frames, dt=1/video FPS, independent controller state, shared same detector output, no camera-mask modification, no card/posture simulation, no real motor/model execution.",
        limits="Open-loop replay cannot validate trajectory, actuator response, true geometry accuracy or physical command-effect prediction. Source recordings used differing model/config/closed-loop states; no claim their historical controller used these comparison settings.")
    (args.output/"input_config_source.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
