"""Isaac adapter for a single fixed stick and moving command transitions."""
import math

import torch
import warp as wp
from deployment.phase_clock import PhaseClockConfig
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.math import quat_apply
from isaaclab.utils.configclass import configclass

from .fixed_stick_state import FixedStickState, RecentCleanCrossings
from .wooden_bar import _as_env_ids, _sole_geometry_w, _sole_min_z_over_rectangle


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, 'torch') else wp.to_torch(value)


def fixed_soles_hit(soles, bar_pose):
    relative = soles - bar_pose[:, None, None, :3]
    # Evaluate height only where the sole intersects the bar footprint.
    # The heel can remain low behind the bar while the toe clears it.
    height_over_bar = _sole_min_z_over_rectangle(relative, .015, .40)
    return height_over_bar <= .015


def fixed_bar_force_hit(forces):
    return (forces.square().sum(dim=-1) > .01**2).flatten(1).any(dim=1)


def get_fixed_state(env):
    if not hasattr(env, '_fixed_stick_state'):
        state = FixedStickState(env.num_envs, env.device, env.step_dt,
                                initial_gap=env.cfg.fixed_initial_gap,
                                walk_step=env.cfg.fixed_walk_step,
                                crossing_step=env.cfg.fixed_crossing_step,
                                collisionless_mode=env.cfg.collisionless_mode,
                                command_mode=env.cfg.fixed_command_mode,
                                phase_clock=PhaseClockConfig(control_dt=env.step_dt,
                                    walk_end_s=env.cfg.fixed_walk_end_s, lead_end_s=env.cfg.fixed_lead_end_s,
                                    sequence_end_s=env.cfg.fixed_sequence_end_s,
                                    forward_velocity=env.cfg.fixed_forward_velocity,
                                    walk_step=env.cfg.fixed_walk_step, crossing_step=env.cfg.fixed_crossing_step))
        state.clean_stats = RecentCleanCrossings(env.cfg.clean_success_threshold,
                                                env.cfg.clean_success_min_episodes,
                                                env.cfg.clean_success_window)
        state.pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        state.initialized = torch.zeros_like(state.pending_reset)
        state.bar_pose = torch.zeros(env.num_envs, 7, device=env.device)
        state.bar_pose[:, 6] = 1.0  # IsaacLab 3.0 uses XYZW quaternions.
        state.previous_root_x = torch.zeros(env.num_envs, device=env.device)
        state.progress = torch.zeros_like(state.previous_root_x)
        state.sole_vertices = torch.tensor(env.cfg.fixed_sole_vertices, device=env.device, dtype=torch.float32)
        state.feet_cfg = SceneEntityCfg('robot', body_names=env.cfg.fixed_foot_names, preserve_order=True)
        state.sensor_cfg = SceneEntityCfg('contact_forces', body_names=env.cfg.fixed_foot_names, preserve_order=True)
        state.feet_cfg.resolve(env.scene)
        state.sensor_cfg.resolve(env.scene)
        env._fixed_stick_state = state
    return env._fixed_stick_state


def reset_fixed_stick(env, env_ids):
    state = get_fixed_state(env)
    ids = _as_env_ids(env, env_ids)
    state.pending_reset[ids] = True
    state.initialized[ids] = False
    # Geometry is read after sim.forward(), before the first policy action.
    # Reading feet here can reuse stale FK from the previous episode.


def _measure(env):
    state = get_fixed_state(env)
    soles = _sole_geometry_w(env, state.feet_cfg, env.cfg.fixed_sole_vertices)
    front = soles[..., 0].amax(dim=2)
    rear = soles[..., 0].amin(dim=2)
    if state.pending_reset.any():
        ids = state.pending_reset.nonzero(as_tuple=False).squeeze(1)
        state.reset(ids, front, int(env.common_step_counter))
        state.bar_pose[ids, 0] = state.bar_near[ids] + .015
        state.bar_pose[ids, 1] = _tensor(env.scene['robot'].data.root_pos_w)[ids, 1]
        state.bar_pose[ids, 2] = env.scene.env_origins[ids, 2] + .015
        bar = env.scene['wooden_bar']
        bar.write_root_pose_to_sim(state.bar_pose[ids], env_ids=ids)
        bar.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        state.previous_root_x[ids] = _tensor(env.scene['robot'].data.root_pos_w)[ids, 0]
        state.progress[ids] = 0.
        state.pending_reset[ids] = False
        state.initialized[ids] = True
    return state, soles, front, rear


def update_fixed_stick(env):
    state, soles, front, rear = _measure(env)
    step = int(env.common_step_counter)
    updated = state.last_step != step
    if not updated.any():
        return state
    robot = env.scene['robot']
    contact = _tensor(env.scene.sensors['contact_forces'].data.current_contact_time)[
        :, state.sensor_cfg.body_ids] > 0.
    # A disabled collider cannot supply useful contact forces. Stage 1 uses
    # exact sole/bar footprint intersection; Stage 2 adds all-link physical hits.
    hit = fixed_soles_hit(soles, state.bar_pose).any(dim=1)
    if not state.collisionless_mode:
        forces = env.scene.sensors['bar_contacts'].data.force_matrix_w_history
        if forces is None:
            raise RuntimeError('Fixed bar contact filter did not produce robot contact forces')
        hit |= fixed_bar_force_hit(_tensor(forces))
    root = _tensor(robot.data.root_pos_w)
    gravity = _tensor(robot.data.projected_gravity_b)
    lateral = soles[..., 1] - state.bar_pose[:, None, None, 1]
    failed = ((root[:, 2] - env.scene.env_origins[:, 2] < .20)
              | (gravity[:, 2] > -math.cos(math.radians(65)))
              | (lateral.abs().amax(dim=(1, 2)) > .40)
              | (env.episode_length_buf >= env.max_episode_length))
    state.progress[updated] = ((root[:, 0] - state.previous_root_x) / env.step_dt).clamp(-.4, .4)[updated]
    state.previous_root_x[updated] = root[updated, 0]
    height = _sole_min_z_over_rectangle(soles - state.bar_pose[:, None, None, :3], .015, .40)
    clearance = (height - .015).clamp(min=0.)
    velocity = _tensor(robot.data.body_lin_vel_w)[:, state.feet_cfg.body_ids, 0]
    state.update(step, front, rear, contact, hit, failed,
                 foot_clearance=clearance, foot_velocity=velocity)
    # RewardManager runs before CommandManager; synchronize the moving command.
    env.command_manager.get_term('base_velocity')._update_command()
    return state


class FixedStickVelocityCommand(UniformVelocityCommand):
    def _update_command(self):
        super()._update_command()
        self.is_standing_env[:] = False
        state = getattr(self._env, "_fixed_stick_state", None)
        # Same-step autoreset updates commands before reset FK/observations.
        # A new WALK must not inherit the previous DONE velocity of zero.
        self.vel_command_b[:, 0] = (torch.where(state.pending_reset, self.cfg.ranges.lin_vel_x[0],
                                              state.forward_command) if state is not None
                                    else self.cfg.ranges.lin_vel_x[0])
        self.vel_command_b[:, 1:] = 0.


@configclass
class FixedStickVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = FixedStickVelocityCommand


def fixed_step_command(env):
    return _measure(env)[0].step_distance.unsqueeze(1)


def fixed_crossing_command(env):
    return _measure(env)[0].crossing_command.float().unsqueeze(1)


def fixed_failure(env):
    return update_fixed_stick(env).failed


def fixed_completed(env):
    return update_fixed_stick(env).success


def fixed_step_reward(env, gaussian_std=.02):
    state = update_fixed_stick(env)
    score = torch.exp(-state.touchdown_error.square() / gaussian_std**2)
    return (score * state.touchdown_eligible).sum(dim=1)


def fixed_walk_completed_reward(env):
    return update_fixed_stick(env).walk_completed_event.float() / env.step_dt


def fixed_lead_completed_reward(env):
    return update_fixed_stick(env).lead_completed_event.float() / env.step_dt


def fixed_success_reward(env):
    # RewardManager integrates rates by dt; milestones are actual one-shot points.
    state = update_fixed_stick(env)
    # Completing a penetrating sequence is logged, but earns no clean-success bonus.
    return (state.success_event & ~state.hit).float() / env.step_dt


def fixed_progress_reward(env):
    state = update_fixed_stick(env)
    return state.progress * ~state.failed


def fixed_clearance_reward(env):
    state = update_fixed_stick(env)
    _, soles, front, rear = _measure(env)
    sensor = env.scene.sensors['contact_forces']
    airborne = _tensor(sensor.data.current_contact_time)[:, state.sensor_cfg.body_ids] <= 0.
    over_bar = (front >= state.bar_near[:, None]) & (rear <= state.bar_far[:, None])
    height = _sole_min_z_over_rectangle(soles - state.bar_pose[:, None, None, :3], .015, .40)
    clearance = (height - .015).clamp(0., .03) / .03
    expected = torch.arange(2, device=env.device)[None, :] == state.expected_foot[:, None]
    valid = airborne & over_bar & expected & state.crossing_command[:, None] & ~state.failed[:, None]
    return (clearance * valid).sum(dim=1)


def stick_entry_clearance_reward(env):
    return update_fixed_stick(env).entry_score


def stick_crossing_progress_reward(env):
    return update_fixed_stick(env).crossing_progress_score


def collisionless_hit_penalty(env):
    state = update_fixed_stick(env)
    # Penalize each penetrating sample, rather than every later frame of a dirty episode.
    return state.current_hit.float() * state.collisionless_mode


class FixedStickGeometryRecorder(RecorderTerm):
    """Latch sole penetration after each physics substep without a collider.

    Read the current PhysX link transforms directly: scene.update() has not yet
    invalidated the articulation data cache when this callback runs.
    No dataset is recorded; the recorder is the standard substep hook.
    """
    def record_post_physics_decimation_step(self):
        state = getattr(self._env, '_fixed_stick_state', None)
        if state is None or not state.collisionless_mode:
            return None, None
        robot = self._env.scene['robot']
        poses = _tensor(robot.root_view.get_link_transforms()).reshape(self._env.num_envs, -1, 7)
        feet = poses[:, state.feet_cfg.body_ids]
        vertices = state.sole_vertices[None].expand(self._env.num_envs, -1, -1, -1)
        quats = feet[..., 3:7].unsqueeze(2).expand(-1, -1, vertices.shape[2], -1)
        soles = quat_apply(quats.reshape(-1, 4), vertices.reshape(-1, 3)).reshape_as(vertices)
        soles = soles + feet[..., :3].unsqueeze(2)
        hit = fixed_soles_hit(soles, state.bar_pose).any(dim=1)
        state.record_physics_hit(hit & state.initialized & ~state.pending_reset)
        return None, None
