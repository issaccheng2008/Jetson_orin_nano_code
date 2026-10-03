"""Fine-tune p3: walk 10 cm then cross one fixed stick without stopping."""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from runtime_compat import check_runtime
    runtime = check_runtime()
    import gymnasium as gym
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab.utils.io import dump_yaml
    from fixed_stick_runner import FixedStickRunner as OnPolicyRunner
    from fixed_stick_stages import validate_stage_checkpoint
    from fixed_stick_control import add_control_args, resolve_control, validate_control_resume, configure_control
    from tasks.manager_based.humanoid_robot_policy_rsl_rl.fixed_stick_env_cfg import configure_fixed_stick_stage
    from fine_tune_checkpoint import load_fine_tune_state
    import tasks  # register only this checkout's task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Humanoid-Robot-RSLRL-v0')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'model_33999.pt')
    parser.add_argument('--usd', type=Path, default=ROOT/'v3.2/v3.2.usd')
    parser.add_argument('--stage', type=int, choices=(1, 2), default=1)
    parser.add_argument('--clean-success-threshold', type=float, default=.6)
    parser.add_argument('--clean-success-min-episodes', type=int, default=200)
    parser.add_argument('--resume', action='store_true', help='Restore a NEW fine-tune checkpoint including optimizer')
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--max-iterations', type=int, default=3000, help='Additional updates, not an absolute checkpoint index')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--run-name', default='finetune')
    add_control_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *remaining]
    if args.num_envs < 1 or args.max_iterations < 1:
        raise ValueError('num-envs and max-iterations must be positive')
    for path in (args.checkpoint, args.usd):
        if not path.is_file(): raise FileNotFoundError(path)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    validate_stage_checkpoint(checkpoint, args.stage, args.resume)
    control = resolve_control(args, checkpoint)
    validate_control_resume(checkpoint, control, args.resume)
    if not 0 < args.clean_success_threshold <= 1 or not 1 <= args.clean_success_min_episodes <= 1000:
        raise ValueError('Require 0 < clean-success-threshold <= 1 and 1 <= min-episodes <= 1000')
    if tuple(checkpoint['actor_state_dict']['mlp.0.weight'].shape) != (512,49):
        raise ValueError('Checkpoint must match the existing 49-dimensional policy')

    @hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
    def run(env_cfg, agent_cfg):
        configure_fixed_stick_stage(env_cfg, args.stage)
        configure_control(env_cfg, control)
        env_cfg.clean_success_threshold = args.clean_success_threshold
        env_cfg.clean_success_min_episodes = args.clean_success_min_episodes
        agent_cfg.algorithm.training_stage = env_cfg.training_stage
        agent_cfg.experiment_name = env_cfg.training_stage + ('_phase_clock' if control['command_mode'] == 'phase_clock' else '')
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.robot.spawn.usd_path = str(args.usd.resolve())
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.seed = agent_cfg.seed
        if args.device is not None:
            env_cfg.sim.device = args.device
            agent_cfg.device = args.device
        env_cfg.curriculum_start_step = (int(checkpoint['entropy_schedule_state']['phase_iteration'])
                                        * agent_cfg.num_steps_per_env if args.resume else 0)
        log_dir = ROOT/'logs/rsl_rl'/agent_cfg.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')+'_'+args.run_name)
        log_dir.mkdir(parents=True, exist_ok=False)
        env_cfg.log_dir = str(log_dir)
        agent_cfg.algorithm.learning_rate = args.learning_rate
        agent_cfg.max_iterations = args.max_iterations
        with launch_simulation(env_cfg, args):
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
            env = gym.make(args.task, cfg=env_cfg)
            try:
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
                load_fine_tune_state(runner, checkpoint, args.resume, args.learning_rate)
                obs = env.get_observations()
                if args.resume:
                    env.unwrapped._fixed_stick_state.clean_stats.restore(
                        (checkpoint.get('infos') or {}).get('fixed_stick_training', {}))
                if obs['policy'].shape[-1] != 49 or env.num_actions != 12:
                    raise RuntimeError('Observation/action interface changed')
                metadata = {'checkpoint': str(args.checkpoint.resolve()), 'usd': str(args.usd.resolve()),
                            'resume': args.resume, 'learning_rate': runner.alg.learning_rate,
                            'optimizer_lrs': [g['lr'] for g in runner.alg.optimizer.param_groups],
                            'stage': runner.alg.training_stage, 'collisionless_mode': env_cfg.collisionless_mode,
                            'clean_success_threshold': env_cfg.clean_success_threshold,
                            'clean_success_min_episodes': env_cfg.clean_success_min_episodes,
                            'start_iteration': runner.current_learning_iteration,
                            'additional_iterations': args.max_iterations,
                            'runtime': runtime, 'fixed_stick_control': control,
                            'initial_gap_m': env_cfg.fixed_initial_gap,
                            'walk_step_m': env_cfg.fixed_walk_step,
                            'crossing_step_m': env_cfg.fixed_crossing_step,
                            'distance_reference': 'initial frontmost sole to fixed bar near edge',
                            'live_distance_in_policy': False,
                            'initial_bar_xyz': env.unwrapped._fixed_stick_state.bar_pose[0, :3].tolist(),
                            'joint_names': list(env.unwrapped.scene['robot'].data.joint_names)}
                metadata['action_joint_names'] = list(env_cfg.actions.joint_pos.joint_names)
                metadata['action_scale'] = env_cfg.actions.joint_pos.scale
                metadata['initial_joint_positions_rad'] = dict(env_cfg.scene.robot.init_state.joint_pos)
                (log_dir/'fine_tune.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                dump_yaml(str(log_dir/'params/env.yaml'),env_cfg)
                dump_yaml(str(log_dir/'params/agent.yaml'),agent_cfg)
                print('[FINETUNE]', json.dumps(metadata), flush=True)
                print('[FINETUNE] logs:', log_dir, flush=True)
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            finally:
                env.close()
    run()


if __name__ == '__main__':
    main()
