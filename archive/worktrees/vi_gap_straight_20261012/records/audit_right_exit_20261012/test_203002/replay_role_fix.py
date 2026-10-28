"""Read-only controller replay on recorded geometry; no hardware or motion simulation."""
import csv
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'new_vision/jetson'))
from heading_steering import HeadingSteeringController
from policy_bridge import SteeringController
from steering_filter import configured_controller
from steering_command_window import SteeringCommandWindow
from command_video import line_lost

p = Path(__file__).resolve().parent
args = SimpleNamespace(**json.loads((p/'run_manifest.json').read_text())['arguments'])
old_namespace = {'__name__': 'old_heading_steering'}
source = subprocess.check_output(['git', 'show', 'a910cd1:new_vision/jetson/heading_steering.py'], cwd=ROOT, text=True)
exec(compile(source, '<baseline a910cd1>', 'exec'), old_namespace)
controllers = []
for cls in (old_namespace['HeadingSteeringController'], HeadingSteeringController):
    inner_options = {k: getattr(args,k) for k in inspect.signature(SteeringController).parameters if hasattr(args,k)}
    inner_options['max_wz_right'] = args.max_wz_right if args.max_wz_right is not None else args.max_wz
    options = dict(angle_wz_table=args.steering_angle_wz_table,
        lookahead_cm=args.heading_lookahead_cm, right_tolerance_deg=args.heading_right_tolerance_deg,
        left_tolerance_deg=args.heading_left_tolerance_deg, full_scale_deg=args.heading_full_scale_deg,
        max_step=args.wz_step, allow_right=True, corridor_cm=args.heading_corridor_cm,
        left_levels=args.heading_left_wz, right_levels=args.heading_right_wz, straight_wz=args.heading_straight_wz)
    for k in ('position_gain','position_dead_cm','position_lookahead_cm','position_max_deg',
              'position_recovery_cm','position_recovery_full_scale_cm','position_confirm_frames'):
        options[k] = getattr(args,k)
    controllers.append(configured_controller(cls, SteeringController(**inner_options), options, args))
windows = [SteeringCommandWindow(args.steering_command_window_s,args.steering_command_median) for _ in controllers]
rows = [json.loads(s) for s in (p/'line_frames.jsonl').read_text().splitlines()]
previous = rows[0]['process_monotonic_s']-.05
paused = True
out=[]
for r in rows:
    now=r['process_monotonic_s'];dt=now-previous;previous=now
    # External card/start commands do not call the walking controller.
    active = r['vx'] > 0 and r['measurement'].get('steering_reason') not in (None,'external_stop')
    if not active:
        paused=True
        for w in windows:w.reset()
        continue
    if paused:
        for c in controllers:c.drop_held_command()
        paused=False
    item={'frame':r['frame'],'time_s':now-rows[0]['process_monotonic_s'],
          'recorded_frame_wz':r['measurement']['steering_frame_wz'],'recorded_final_wz':r['wz']}
    for name,c,w in zip(('before','after'),controllers,windows):
        debug=dict(r['measurement'])
        vx,wz=c.command(debug,r['confidence'],dt)
        item[name+'_frame_wz']=wz
        item[name+'_decision']=c.diagnostics.get('steering_decision')
        item[name+'_final_wz']=w.update(vx,wz,now,valid=not line_lost(debug,r['confidence']))[1]
    out.append(item)
with (p/'role_fix_replay.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
summary=dict(frames=len(out),baseline_frame_matches=sum(x['before_frame_wz']==x['recorded_frame_wz'] for x in out),
    baseline_final_matches=sum(x['before_final_wz']==x['recorded_final_wz'] for x in out),
    changed_frame_commands=sum(x['before_frame_wz']!=x['after_frame_wz'] for x in out),
    changed_final_commands=sum(x['before_final_wz']!=x['after_final_wz'] for x in out),
    limitations='Same recorded geometry, controller/window replay. Does not predict driving trajectory; new commands would change future real observations.')
(p/'role_fix_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('frame 469',next(x for x in out if x['frame']==469))
