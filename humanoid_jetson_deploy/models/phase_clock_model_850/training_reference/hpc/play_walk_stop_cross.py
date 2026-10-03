"""Record p3's first moving-crossing episode and its terminal outcome."""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import importlib.metadata
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def camera_view_for_root(root_position):
    x, y, z = (float(v) for v in root_position)
    return (x + 0.2, y + 1.4, z + 0.35), (x + 0.45, y, z - 0.15)


def move_kit_recording_camera(env, root_position) -> None:
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    eye, target = camera_view_for_root(root_position)
    set_kit_renderer_camera_view(eye, target, camera_prim_path=env.cfg.viewer.cam_prim_path)


def frame_has_content(frame) -> bool:
    if frame is None:
        return False
    pixels = np.asarray(frame)
    return pixels.ndim == 3 and pixels.shape[-1] >= 3 and bool(np.any(pixels[..., :3] > 2))


def annotate_command_frame(frame, stage, command):
    """Overlay current command values on a real Kit-rendered 3D frame."""
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    label = ('WALK', 'LEAD', 'FOLLOW', 'DONE')[stage]
    text = f'{label}  cross={int(command[3])}  step={command[2] * 100:.0f}cm'
    draw.rectangle((4, 4, 330, 26), fill=(20, 20, 20))
    draw.text((10, 9), text, fill=(130, 255, 140) if command[3] else (255, 255, 255))
    return np.asarray(canvas)


def main():
    from runtime_compat import check_runtime
    runtime = check_runtime()
    import gymnasium as gym
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    from fine_tune_checkpoint import load_fine_tune_state
    from fixed_stick_stages import checkpoint_stage
    from fixed_stick_control import add_control_args, resolve_control, configure_control
    from tasks.manager_based.humanoid_robot_policy_rsl_rl.fixed_stick_env_cfg import configure_fixed_stick_stage
    from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp.fixed_stick import _measure
    import tasks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model_33999.pt')
    parser.add_argument('--stage', type=int, choices=(1, 2), default=None,
                        help='Override stage; default follows checkpoint, old checkpoints use Stage 1')
    parser.add_argument('--usd', type=Path, default=ROOT / 'v3.2/v3.2.usd')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--video-backend', choices=('kit', 'software'), default='kit')
    parser.add_argument('--video-stride', type=int, default=4)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=None)
    add_control_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *remaining]
    if min(args.steps, args.video_stride) <= 0:
        raise ValueError('steps and video-stride must be positive')
    for path in (args.checkpoint, args.usd):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    stage_number = args.stage or checkpoint_stage(checkpoint) or 1
    control = resolve_control(args, checkpoint)

    if args.video_backend == 'kit' and not args.no_video:
        args.enable_cameras = True
        args.video = True

    @hydra_task_config('Humanoid-Robot-RSLRL-Play-v0', 'rsl_rl_cfg_entry_point')
    def run(env_cfg, agent_cfg):
        configure_fixed_stick_stage(env_cfg, stage_number)
        configure_control(env_cfg, control)
        agent_cfg.algorithm.training_stage = env_cfg.training_stage
        env_cfg.scene.num_envs = 1
        env_cfg.scene.robot.spawn.usd_path = str(args.usd.resolve())
        env_cfg.seed = agent_cfg.seed
        if args.device is not None:
            env_cfg.sim.device = agent_cfg.device = args.device

        if args.video_backend == 'kit':
            env_cfg.video_recorder.window_width = 640
            env_cfg.video_recorder.window_height = 360
            env_cfg.sim.render_interval = env_cfg.decimation * args.video_stride
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.eye = (0.2, 1.4, 0.6)
            env_cfg.viewer.lookat = (0.45, 0.0, 0.0)
            env_cfg.video_recorder.eye = (0.2, 1.4, 0.6)
            env_cfg.video_recorder.lookat = (0.45, 0.0, 0.0)

        recording_group = env_cfg.training_stage + ('_phase_clock' if control['command_mode'] == 'phase_clock' else '')
        output = args.output_dir or ROOT / 'recordings' / recording_group / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        output.mkdir(parents=True, exist_ok=True)
        with launch_simulation(env_cfg, args):
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
            env = gym.make('Humanoid-Robot-RSLRL-Play-v0', cfg=env_cfg,
                           render_mode="rgb_array" if args.video_backend == 'kit' else None)
            try:
                if args.video_backend == 'kit':
                    origin = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
                    initial_root = (origin[0], origin[1], origin[2] + 0.33)
                    eye, target = camera_view_for_root(initial_root)
                    capture = env.unwrapped.video_recorder._capture
                    if capture is not None:
                        capture.cfg.eye = eye
                        capture.cfg.lookat = target

                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                load_fine_tune_state(runner, checkpoint, resume=False)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                obs = env.get_observations()
                if obs['policy'].shape[-1] != 49 or env.num_actions != 12:
                    raise RuntimeError('Expected the old 49D/12D policy interface')
                state, soles, front, rear = _measure(env.unwrapped)
                robot = env.unwrapped.scene['robot']
                initial_gap = (state.bar_near - front.amax(dim=1))[0].item()
                summary = {'runtime': runtime, 'checkpoint': str(args.checkpoint.resolve()),
                           'initial_gap_m': initial_gap, 'initial_bar_xyz': state.bar_pose[0, :3].tolist(),
                           'policy_shape': list(obs['policy'].shape), 'action_dim': env.num_actions,
                           'command_switches': [], 'success': False, 'clean_success': False, 'done': False,
                           'training_stage': env_cfg.training_stage, 'collisionless_mode': env_cfg.collisionless_mode,
                           'foot_order': list(env_cfg.fixed_foot_names), 'fixed_stick_control': control,
                           'sequence_finished': False}
                print('[PLAY] initial gap:', initial_gap, 'bar:', summary['initial_bar_xyz'], flush=True)
                previous_stage = int(state.command_stage[0])
                with ExitStack() as stack:
                    file = stack.enter_context((output / 'trajectory.csv').open('w', newline='', encoding='utf-8'))
                    writer = csv.writer(file)
                    writer.writerow(['step', 'time_s', 'stage', 'forward_command', 'yaw_command', 'step_command',
                                     'cross_command', 'root_x', 'root_y', 'root_z',
                                     *robot.data.joint_names, 'applied_step_command', 'applied_cross_command',
                                     'left_touchdown', 'right_touchdown', 'geometry_stage', 'clock_elapsed_s', 'sequence_finished'])
                    video = None
                    if not args.no_video:
                        from obstacle_software_video import SoftwareVideoWriter
                        video_name = "obstacle-3d.mp4" if args.video_backend == 'kit' else "crossing.mp4"
                        if args.video_backend == 'kit':
                            video = stack.enter_context(SoftwareVideoWriter(
                                str(output / video_name),
                                fps=1. / (env.unwrapped.step_dt * args.video_stride),
                                width=env_cfg.video_recorder.window_width,
                                height=env_cfg.video_recorder.window_height,
                            ))
                        else:
                            from obstacle_software_video import render_side_frame
                            video = stack.enter_context(SoftwareVideoWriter(
                                str(output / video_name),
                                fps=1. / (env.unwrapped.step_dt * args.video_stride),
                            ))
                    for step in range(1, args.steps + 1):
                        applied_command = obs['policy'][0, 9:13].detach().cpu().tolist()
                        with torch.inference_mode():
                            obs, rewards, dones, infos = env.step(policy(obs))
                            policy.reset(dones)
                        done = bool(dones[0])
                        outcome = infos.get('fixed_stick_outcome', {}) if done else {}
                        stage = int(outcome['command_stage'][0]) if outcome else int(state.command_stage[0])
                        geometry_stage = int(outcome['stage'][0]) if outcome else int(state.stage[0])
                        clock_steps = int(outcome['control_steps'][0]) if outcome else int(state.control_steps[0])
                        sequence_finished = bool(outcome['sequence_finished'][0]) if outcome else bool(state.sequence_finished[0])
                        command = obs['policy'][0, 9:13].detach().cpu().tolist()
                        touchdown = (outcome['touchdown'][0] if outcome else state.touchdown[0]).detach().cpu().tolist()
                        if outcome:
                            command[0] = float(outcome['forward_command'][0])
                            command[1] = 0.
                            command[2] = float(outcome['step_command'][0])
                            command[3] = float(outcome['cross_command'][0])
                        changed_stage = stage != previous_stage
                        if changed_stage:
                            change = {'step': step, 'stage': stage, 'cross_command': command[3],
                                      'step_command_m': command[2], 'touchdown': touchdown,
                                      'clock_elapsed_s': clock_steps * env.unwrapped.step_dt, 'geometry_stage': geometry_stage}
                            summary['command_switches'].append(change)
                            print('[PLAY] command switch:', json.dumps(change), flush=True)
                            previous_stage = stage
                        root = (outcome['root_xyz'][0] if outcome else robot.data.root_pos_w[0]).detach().cpu().numpy()
                        joints = (outcome['joint_pos'][0] if outcome else robot.data.joint_pos[0]).detach().cpu().tolist()
                        writer.writerow([step, step * env.unwrapped.step_dt, stage, *command, *root.tolist(), *joints,
                                         *applied_command[2:4], *touchdown, geometry_stage,
                                         clock_steps * env.unwrapped.step_dt, sequence_finished])
                        if video and (step == 1 or step % args.video_stride == 0 or done
                                      or changed_stage):
                            if args.video_backend == 'kit':
                                move_kit_recording_camera(env.unwrapped, robot.data.root_pos_w[0].detach().cpu().tolist())
                                frame = env.unwrapped.render(recompute=True)
                                if frame_has_content(frame):
                                    video.append_data(annotate_command_frame(frame, stage, command))
                            elif not done:
                                video.append_data(render_side_frame(root, robot.data.body_names,
                                    robot.data.body_pos_w[0].detach().cpu().numpy(),
                                    state.bar_pose[0, :3].detach().cpu().numpy(), False))
                        summary['steps'] = step
                        summary['sequence_finished'] = sequence_finished
                        if done:
                            summary.update(done=True, success=bool(outcome['success'][0]) if outcome else False,
                                           hit=bool(outcome['hit'][0]) if outcome else False,
                                           clean_success=bool(outcome['clean_success'][0]) if outcome else False,
                                           walk_completed=bool(outcome['walk_completed'][0]) if outcome else False)
                            summary['termination_terms'] = [name for name, values in outcome.get('termination_terms', {}).items()
                                                            if bool(values[0])]
                            break
                (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                print('[PLAY]', json.dumps(summary), flush=True)
                print('[PLAY] output:', output, flush=True)
            finally:
                env.close()
    run()


if __name__ == '__main__':
    main()
