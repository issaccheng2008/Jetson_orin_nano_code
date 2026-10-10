"""Replay four causal visual controllers on identical recorded observations.

No camera, serial port, socket, model or motor is opened. This compares visual
command decisions, not a simulated trajectory or model-input hold contract.
"""
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median
import time

from heading_steering import HeadingSteeringController
from segment_steering import SegmentSteeringController
from policy_bridge import SteeringController
from steering_filter import FilterConfig, add_arguments, config_from_args
from steering_recovery import RecoveryConfig

CONTROL_REASONS = {'new_block','continue_block','minimum_hold','brief_loss_hold','geometry_lost_yaw_zero',
                   'loss_history_turn','loss_timeout_stop','loss_no_history_stop',
                   'loss_default_left','loss_walking_disabled'}
_FROM_MANIFEST = object()


def replay(rows, arguments, filter_config, *, recovery_config=_FROM_MANIFEST,
           segment_fallback=_FROM_MANIFEST):
    get = arguments.get
    # Missing fields belong to older recordings, before these behaviors existed.
    # Explicit None/False still disable a stage for the recovery ablation tool.
    if recovery_config is _FROM_MANIFEST:
        recovery_config = (RecoveryConfig(get('steering_loss_max_s', .8),
                                          get('steering_loss_history_s', .8))
                           if get('steering_loss_mode', 'legacy') in ('history-turn', 'history-stop') else None)
    if segment_fallback is _FROM_MANIFEST:
        segment_fallback = get('steering_segment_fallback', False)
    inner = SteeringController(vx=get('vx',.2), max_wz=get('max_wz',.5),
        yaw_sign=get('yaw_sign',1), lost_hold_s=get('lost_hold_s',.2),
        max_lateral_cm=get('max_lateral_cm',0),
        max_wz_right=get('max_wz_right') or get('max_wz',.5))
    controller_type = SegmentSteeringController if get('wz_mode','heading') == 'segments' else HeadingSteeringController
    controller = controller_type(inner, lookahead_cm=get('heading_lookahead_cm',50.),
        angle_wz_table=get('steering_angle_wz_table'),
        **({'segment_regions_cm': get('segment_regions_cm', ((20,32),(32,44),(44,56)))}
           if controller_type is SegmentSteeringController else {}),
        right_tolerance_deg=get('heading_right_tolerance_deg',12.),
        left_tolerance_deg=get('heading_left_tolerance_deg',4.),
        full_scale_deg=get('heading_full_scale_deg',20.), max_step=get('wz_step',.5),
        corridor_cm=get('heading_corridor_cm',8.),
        left_levels=tuple(get('heading_left_wz',(.37,.43,.5))),
        right_levels=tuple(get('heading_right_wz',(.3,.5))),
        straight_wz=get('heading_straight_wz',0.), min_hold_s=0., filter_config=filter_config,
        recovery_config=recovery_config, segment_fallback=segment_fallback,
        position_gain=get('position_gain', 0.), position_dead_cm=get('position_dead_cm', 2.),
        position_lookahead_cm=get('position_lookahead_cm', 50.),
        position_max_deg=get('position_max_deg', 12.),
        position_recovery_cm=get('position_recovery_cm', 0.),
        position_recovery_full_scale_cm=get('position_recovery_full_scale_cm', 12.),
        position_confirm_frames=get('position_confirm_frames', 2))
    result, previous = [], None
    for row in rows:
        now = float(row['process_monotonic_s'])
        dt = now-previous if previous is not None else .1
        if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('replay requires increasing finite process_monotonic_s')
        previous = now
        debug = row['measurement']
        controlled = debug.get('steering_reason') in CONTROL_REASONS
        started = time.perf_counter_ns()
        if controlled:
            vx, wz = controller.command(debug, row['confidence'], dt)
            diag = controller.diagnostics
        else:
            controller.reset(clear_hold=True)
            vx, wz = row['vx'], row['wz']
            diag = {}
        elapsed_us = (time.perf_counter_ns()-started)/1000.
        result.append(dict(frame=row['frame'], host_time_ns=row['host_time_ns'],
            process_monotonic_s=now, controlled=controlled, recorded_vx=row['vx'], recorded_wz=row['wz'],
            new_vx=vx, new_wz=wz, raw_heading_deg=diag.get('steering_heading_deg'),
            filtered_heading_deg=diag.get('steering_filter_heading_deg',diag.get('steering_filtered_heading_deg')),
            raw_demand_deg=diag.get('steering_demand_deg'),
            filtered_demand_deg=diag.get('steering_filtered_demand_deg'),
            geometry_source=diag.get('steering_heading_source'),
            decision=diag.get('steering_decision'), reason=diag.get('steering_reason'),
            braked=diag.get('steering_braked'), compute_us=elapsed_us))
    return result


def summarize(rows):
    seconds, switches, reversals, prior = {}, 0, 0, None
    costs = []
    mismatches = 0
    for index, row in enumerate(rows):
        if not row['controlled']:
            prior = None
            continue
        costs.append(row['compute_us'])
        mismatches += int(abs(row['new_wz']-row['recorded_wz'])>1e-6
                          or abs(row['new_vx']-row['recorded_vx'])>1e-6)
        pair = row['new_vx'], row['new_wz']
        if prior is not None and pair != prior:
            switches += 1
            reversals += int(prior[1]*pair[1] < 0)
        prior = pair
        if index+1 < len(rows) and row['new_vx'] > 0:
            duration = rows[index+1]['process_monotonic_s']-row['process_monotonic_s']
            key = f"{row['new_wz']:g}"
            seconds[key] = seconds.get(key,0.)+duration
    ordered = sorted(costs)
    return dict(controlled_command_changes=switches, adjacent_nonzero_reversals=reversals,
        grade_seconds=seconds, differs_from_recording_frames=mismatches,
        desktop_compute_median_us=median(costs) if costs else 0.,
        desktop_compute_p95_us=ordered[int(.95*(len(ordered)-1))] if ordered else 0.)


def plot_variants(variants, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base, active = variants['legacy'], variants['combined']
    times = [(r['host_time_ns']-base[0]['host_time_ns'])/1e9 for r in base]
    fig, axes = plt.subplots(2,1,figsize=(16,7),sharex=True,layout='constrained')
    def values(rows, field):
        return [float('nan') if r[field] is None else r[field] for r in rows]
    axes[0].plot(times,values(active,'raw_demand_deg'),color='#94a3b8',label='Raw selected target demand')
    axes[0].plot(times,values(active,'filtered_demand_deg'),color='#15803d',label='Filtered target demand')
    axes[0].set_ylabel('Demand angle (degrees)')
    for name,color in (('legacy','#2563eb'),('hysteresis_only','#a855f7'),('filter_only','#f59e0b'),('combined','#15803d')):
        axes[1].step(times,[r['new_wz'] for r in variants[name]],where='post',label=name,color=color,alpha=.8,lw=1.2)
    axes[1].set_ylabel('Visual WZ command (rad/s)')
    axes[1].set_xlabel('Time since vision recording start (s)')
    for ax in axes:
        ax.legend(loc='upper left',ncol=2)
        ax.grid(alpha=.2)
    fig.suptitle('Recorded-observation replay; not a predicted robot trajectory or model-input hold')
    fig.savefig(output/'comparison.png',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('line_frames',type=Path)
    parser.add_argument('--manifest',type=Path,help='Defaults to run_manifest.json beside JSONL')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plot',action='store_true',help='Optional matplotlib plot; runtime filter needs no plotting library')
    add_arguments(parser)
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        rows = [json.loads(line) for line in args.line_frames.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError('empty line_frames recording')
        manifest = json.loads((args.manifest or args.line_frames.with_name('run_manifest.json')).read_text(encoding='utf-8'))
        arguments = manifest['arguments']
        if arguments.get('wz_mode') not in ('heading','segments'):
            raise ValueError('replay supports heading/segments recordings only')
        if arguments.get('steering_filter_mode','legacy') not in ('legacy', 'shadow'):
            raise ValueError('use a legacy or shadow recording for an exact legacy-output reference')
        variants = {name:replay(rows,arguments,cfg) for name,cfg in (
            ('legacy',None), ('hysteresis_only',replace(config,algorithm='none')),
            ('filter_only',replace(config,hysteresis_deg=0.,enter_deg=0.,exit_deg=0.)),
            ('combined',config))}
    except (ValueError,KeyError,OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary = {name:summarize(result) for name,result in variants.items()}
    summary['semantics'] = 'Same recorded observations, visual candidates only; no closed-loop trajectory, connector bias, watchdog or model hold simulation.'
    summary['recording_manifest'] = str(args.manifest or args.line_frames.with_name('run_manifest.json'))
    summary['filter_settings'] = config.__dict__
    for name,result in variants.items():
        with (args.output_dir/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.plot:
        plot_variants(variants,args.output_dir)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['legacy']['differs_from_recording_frames']:
        parser.error('legacy replay differs from recording; inspect source/timing before interpreting comparisons')


if __name__ == '__main__':
    main()
