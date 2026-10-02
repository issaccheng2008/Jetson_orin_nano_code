"""Export and verify the 49/12 actor plus portable phase-clock deployment files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from fixed_stick_control import add_control_args, resolve_control, saved_control
    from deployment.policy_interface import JOINT_NAMES, DEFAULT_JOINT_POS, ACTION_SCALE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_control_args(parser)
    args = parser.parse_args()
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    control = resolve_control(args, raw)
    if control['command_mode'] != 'phase_clock':
        raise ValueError('This deployment bundle requires --command-mode phase_clock')
    if tuple(raw['actor_state_dict']['mlp.0.weight'].shape) != (512,49):
        raise ValueError('Expected the existing 49-dimensional actor')
    actor = MLPModel(TensorDict({'policy':torch.zeros(1,49)}, batch_size=[1]),
                     {'actor':['policy']}, 'actor', 12, hidden_dims=[512,256,128],
                     activation='elu', obs_normalization=False,
                     distribution_cfg={'class_name':'GaussianDistribution', 'init_std':1., 'std_type':'scalar'})
    actor.load_state_dict(raw['actor_state_dict'], strict=True)
    actor.eval()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = actor.as_onnx(False).eval()
    path = output/'policy.onnx'
    torch.onnx.export(model, torch.zeros(1,49), str(path), input_names=['obs'], output_names=['action'],
                      dynamic_axes={'obs':{0:'batch'}, 'action':{0:'batch'}},
                      opset_version=14, dynamo=False, external_data=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    evaluator = ReferenceEvaluator(graph)
    rng = np.random.default_rng(17)
    obs = rng.normal(0,.5,(64,49)).astype(np.float32)
    max_error = 0.
    with torch.inference_mode():
        for batch in (obs[:1], obs):
            expected = actor(TensorDict({'policy':torch.from_numpy(batch)}, batch_size=[len(batch)]), stochastic_output=False).numpy()
            actual = evaluator.run(['action'], {'obs':batch})[0]
            max_error = max(max_error, float(np.abs(expected-actual).max()))
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    validation = {'checker':'passed', 'numerical_check':'passed', 'test_inputs':64,
                  'tested_batch_sizes':[1,64], 'max_absolute_error':max_error,
                  'validation_backend':'onnx.reference.ReferenceEvaluator', 'opset':14}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    trained = saved_control(raw) or {'command_mode':'touchdown'}
    contract = {'checkpoint':str(args.checkpoint.resolve()), 'checkpoint_sha256':sha(args.checkpoint),
                'onnx_sha256':sha(path), 'iteration':raw.get('iter'),
                'training_stage':(raw.get('entropy_schedule_state') or {}).get('training_stage'),
                'trained_control':trained, 'deployment_control':control,
                'input':{'name':'obs','dtype':'float32','shape':['batch',49]},
                'output':{'name':'action','dtype':'float32','shape':['batch',12]},
                'joint_names':list(JOINT_NAMES), 'default_joint_positions_rad':DEFAULT_JOINT_POS.tolist(),
                'action_scale':ACTION_SCALE, 'action_clipping':None, 'obs_normalization':False,
                'observation_slices_zero_based':{'imu_acc_x0_1':[0,3],'imu_gyro':[3,6],
                    'projected_gravity':[6,9],'commands':[9,13],'q_minus_default':[13,25],
                    'joint_velocity':[25,37],'previous_unscaled_action':[37,49]},
                'initial_toe_bar_gap_m':.08, 'clock_inside_onnx':False,
                'sequence_finished_means':'command sequence ended; physical success is not measured on hardware',
                'hardware_validated':False, 'validation':validation}
    (output/'policy_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    (output/'phase_clock.json').write_text(json.dumps(control['phase_clock'], indent=2), encoding='utf-8')
    (output/'export_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    dest = output/'deployment'
    dest.mkdir(exist_ok=True)
    for name in ('__init__.py', 'phase_clock.py', 'policy_interface.py'):
        shutil.copy2(ROOT/'deployment'/name, dest/name)
    references = [
        'tasks/manager_based/humanoid_robot_policy_rsl_rl/humanoid_robot_policy_rsl_rl_env_cfg.py',
        'tasks/manager_based/humanoid_robot_policy_rsl_rl/humanoid_robot.py',
        'tasks/manager_based/humanoid_robot_policy_rsl_rl/fixed_stick_env_cfg.py',
        'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick_state.py',
        'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick.py',
        'hpc/fixed_stick_control.py', 'hpc/train_walk_stop_cross.py', 'hpc/play_walk_stop_cross.py',
        'hpc/fixed_stick_runner.py', 'hpc/export_fixed_stick_onnx.py',
    ]
    for relative in references:
        reference = output/'training_reference'/relative
        reference.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, reference)
    doc = ROOT/'docs/phase_clock_upper_computer_handoff_zh.md'
    if doc.is_file():
        shutil.copy2(doc, output/doc.name)
    print('[EXPORT]', json.dumps({'output':str(output), 'trained_control':trained, 'validation':validation}), flush=True)


if __name__ == '__main__':
    main()
