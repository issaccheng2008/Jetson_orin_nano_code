#!/usr/bin/env python3
"""Same-observation ablation; neither trajectory nor future camera frames are simulated."""
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_vision/jetson'))
from replay_steering_filter import replay, summarize
from steering_filter import FilterConfig
from steering_recovery import RecoveryConfig


def variation(rows):
    raw, filtered = [], []
    for a,b in zip(rows, rows[1:]):
        dt=b['process_monotonic_s']-a['process_monotonic_s']
        if (not a['controlled'] or not b['controlled'] or not 0<dt<=.5
                or a['geometry_source'] != b['geometry_source']): continue
        for name, output in (('raw_heading_deg',raw),('filtered_heading_deg',filtered)):
            x,y=a[name],b[name]
            if x is not None and y is not None: output.append(abs(y-x)/dt)
    def describe(values):
        s=sorted(values)
        return dict(samples=len(s),median_deg_s=median(s) if s else None,
                    p95_deg_s=s[min(len(s)-1,int(len(s)*.95))] if s else None)
    return dict(raw=describe(raw), filtered=describe(filtered))


def process(folder, tau=.45, slew=45.):
    files=list(folder.rglob('line_frames.jsonl'))
    if len(files)!=1: raise ValueError('Select one test with exactly one visual recording')
    path=files[0]
    all_rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    from analyze_recording import find_start
    start,_=find_start(all_rows)
    rows=[r for r in all_rows if r['host_time_ns']/1e9>=start]
    args=json.loads(path.with_name('run_manifest.json').read_text(encoding='utf-8'))['arguments']
    baseline=FilterConfig(algorithm=args.get('steering_filter_algorithm','one-euro'),
        min_cutoff_hz=args.get('steering_filter_min_hz',1.5),max_cutoff_hz=args.get('steering_filter_max_hz',4.),
        beta=args.get('steering_filter_beta',.03),position_tau_s=args.get('steering_filter_position_tau_s',.1),
        hysteresis_deg=args.get('steering_hysteresis_deg',1.),enter_deg=args.get('steering_enter_deg',2.),
        exit_deg=args.get('steering_exit_deg',1.)) if args.get('steering_filter_mode')=='active' else None
    robust=replace(baseline or FilterConfig(),algorithm='robust',robust_tau_s=tau,robust_slew_deg_s=slew)
    loss=RecoveryConfig()
    variants={name:replay(rows,args,cfg,recovery_config=rec,segment_fallback=fallback)
              for name,cfg,rec,fallback in (
                  ('baseline',baseline,None,False),('filter_only',robust,None,False),
                  ('loss_only',baseline,loss,False),('combined',robust,loss,True))}
    output=folder/'analysis/recovery_evaluation';output.mkdir(parents=True,exist_ok=True)
    report={name:dict(summarize(data),angle_variation=variation(data),
        history_frames=sum(r['reason']=='loss_history_turn' for r in data),
        stop_frames=sum(r['reason'] in ('loss_timeout_stop','loss_no_history_stop') for r in data),
        accepted_geometry_frames=sum(r['raw_heading_deg'] is not None for r in data)) for name,data in variants.items()}
    report['semantics']='Same original camera observations, visual candidate commands only. No hardware motion, IMU response, future images, model command hold or connector bias simulation. Missing old paired-near diagnostics cannot be reconstructed.'
    report['settings']=dict(tau_s=tau,slew_deg_s=slew,loss_max_s=loss.max_loss_s)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,data in variants.items():
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,constrained_layout=True)
    for name in ('baseline','combined'):
        data=variants[name];t=[(r['host_time_ns']/1e9-start) for r in data]
        axes[0].plot(t,[r['filtered_heading_deg'] for r in data],label=name+' filtered heading')
        axes[1].step(t,[r['new_wz'] for r in data],where='post',label=name+' WZ')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('Direction (deg)');axes[1].set_ylabel('Candidate WZ (rad/s)')
    axes[1].set_xlabel('Time after release (s)')
    axes[0].set_title('Offline comparison: identical original observations, not a new robot run')
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(folder.name,json.dumps({name:{k:v for k,v in value.items() if k in ('controlled_command_changes','history_frames','stop_frames','angle_variation','differs_from_recording_frames')} for name,value in report.items() if isinstance(value,dict) and name!='settings'}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('test_directory',type=Path)
    p.add_argument('--tau-s',type=float,default=.45);p.add_argument('--slew-deg-s',type=float,default=45.)
    args=p.parse_args()
    try:process(args.test_directory,args.tau_s,args.slew_deg_s)
    except (ValueError,OSError,KeyError,ImportError) as exc:p.error(str(exc))


if __name__=='__main__':main()
