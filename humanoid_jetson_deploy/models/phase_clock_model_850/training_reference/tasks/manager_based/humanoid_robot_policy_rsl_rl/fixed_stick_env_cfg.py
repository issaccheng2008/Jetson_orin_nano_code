"""v3.2: one 10 cm walking step, then cross one fixed stick without stopping."""
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg, DatasetExportMode
from isaaclab.utils.configclass import configclass

from .humanoid_robot_policy_rsl_rl_env_cfg import (
    HumanoidRobotPolicyEnvCfg, FOOT_BODY_NAMES, FOOT_SOLE_VERTICES, LEG_JOINT_NAMES,
)
from .mdp import fixed_stick as mdp


@configclass
class FixedStickGeometryRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    geometry_hit = RecorderTermCfg(class_type=mdp.FixedStickGeometryRecorder)


@configclass
class FixedStickEnvCfg(HumanoidRobotPolicyEnvCfg):
    fixed_initial_gap: float = .08
    fixed_walk_step: float = .10
    fixed_crossing_step: float = .23
    fixed_command_mode: str = 'phase_clock'
    fixed_walk_end_s: float = .14
    fixed_lead_end_s: float = .44
    fixed_sequence_end_s: float = .76
    fixed_forward_velocity: float = .20
    fixed_foot_names: tuple = tuple(FOOT_BODY_NAMES)
    fixed_sole_vertices: tuple = FOOT_SOLE_VERTICES
    training_stage: str = 'fixed_stick_stage1_v32'
    collisionless_mode: bool = True
    clean_success_threshold: float = .60
    clean_success_min_episodes: int = 200
    clean_success_window: int = 1000

    def __post_init__(self):
        super().__post_init__()
        self.stop_before_crossing = False
        self.hurdle_gap_curriculum = False
        self.episode_length_s = 8.0
        self.is_finite_horizon = True
        self.scene.lazy_sensor_update = False
        self.scene.terrain.terrain_type = 'plane'
        self.scene.terrain.terrain_generator = None
        for j in range(10):
            setattr(self.scene, f'course_stick_{j}', None)
        self.scene.collisionless_wooden_bar = None
        bar = self.scene.wooden_bar.spawn
        bar.rigid_props.kinematic_enabled = True
        bar.rigid_props.disable_gravity = True
        bar.collision_props.collision_enabled = not self.collisionless_mode
        bar.activate_contact_sensors = not self.collisionless_mode
        bar.visual_material.diffuse_color = (.60, .32, .10)
        self.scene.bar_contacts = ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/WoodenBar',
            filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/Robot/{name}' for name in
                                   ['base_link', *(joint.replace('_joint', '_link') for joint in LEG_JOINT_NAMES)]],
            update_period=self.sim.dt,
            history_length=self.decimation,
        )
        if self.collisionless_mode:
            self.scene.bar_contacts = None
            self.recorders = FixedStickGeometryRecorderCfg()
        self.events.reset_base.params['pose_range'] = {}
        self.events.reset_base.params['velocity_range'] = {}
        self.events.reset_crossing_state = None
        self.events.update_crossing_state = None
        self.events.align_wooden_bars = None
        self.events.configure_collisionless_bar_collisions = None
        self.events.reset_fixed_stick = EventTermCfg(func=mdp.reset_fixed_stick, mode='reset')
        self.commands.base_velocity = mdp.FixedStickVelocityCommandCfg(
            asset_name='robot', resampling_time_range=(1.e9, 1.e9),
            rel_standing_envs=0., rel_heading_envs=0., heading_command=False,
            debug_vis=False,
            ranges=mdp.FixedStickVelocityCommandCfg.Ranges(
                lin_vel_x=(.20, .20), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)),
        )
        self.observations.policy.step_distance = ObservationTermCfg(func=mdp.fixed_step_command)
        self.observations.policy.crossing_command = ObservationTermCfg(func=mdp.fixed_crossing_command)
        for name in ('hurdle_forward_progress', 'stick_over_clearance', 'stick_landing_center',
                     'stick_cleared', 'stick_failed', 'all_sticks_completed', 'stick_collision',
                     'stop_stability', 'stop_completion', 'hurdle_target_approach',
                     'hurdle_wrong_landing', 'physical_bar_crossing_completion_reward',
                     'collisionless_bar_contact_penalty', 'stepping_wooden_bar_step_reward'):
            setattr(self.rewards, name, None)
        # The inherited config may gain additional legacy band terms later.
        for name in ('following_wooden_bar_step_reward', 'feet_height_entering_band_reward'):
            if hasattr(self.rewards, name):
                setattr(self.rewards, name, None)
        self.rewards.step_distance_tracking_reward = RewardTermCfg(func=mdp.fixed_step_reward, weight=50.)
        self.rewards.fixed_walk_completed = RewardTermCfg(func=mdp.fixed_walk_completed_reward, weight=25.)
        self.rewards.fixed_lead_completed = RewardTermCfg(func=mdp.fixed_lead_completed_reward, weight=30.)
        self.rewards.fixed_success = RewardTermCfg(func=mdp.fixed_success_reward, weight=100.)
        self.rewards.fixed_progress = RewardTermCfg(func=mdp.fixed_progress_reward, weight=6.)
        self.rewards.fixed_clearance = None
        self.rewards.stick_entry_clearance_reward = RewardTermCfg(func=mdp.stick_entry_clearance_reward, weight=75.)
        self.rewards.stick_crossing_progress_reward = RewardTermCfg(func=mdp.stick_crossing_progress_reward, weight=15.)
        self.rewards.collisionless_hit_penalty = RewardTermCfg(func=mdp.collisionless_hit_penalty, weight=-100.)
        self.rewards.termination_penalty.params['term_keys'] = [
            'bad_orientation', 'low_base_height', 'fixed_stick_failed']
        for name in ('hurdle_out_of_bounds', 'hurdle_skipped_gap', 'hurdle_course_completed',
                     'wooden_bar_moved', 'stop_failed', 'time_out'):
            setattr(self.terminations, name, None)
        self.terminations.fixed_stick_failed = TerminationTermCfg(func=mdp.fixed_failure)
        self.terminations.fixed_stick_completed = TerminationTermCfg(func=mdp.fixed_completed)
        self.curriculum.step_distance_gaussian = None
        self.curriculum.phase_5_ang_vel_z = None
        self.curriculum.wooden_bar_reward_weights = None


@configclass
class FixedStickEnvCfg_PLAY(FixedStickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = 'asset_root'
        self.viewer.env_index = 0
        self.viewer.asset_name = 'robot'
        self.viewer.eye = (1.2, 1.2, .7)
        self.viewer.lookat = (.2, 0., 0.)


def configure_fixed_stick_stage(cfg, stage):
    """Apply before launch; physics shapes and sensors are created once per run."""
    if stage not in (1, 2):
        raise ValueError('Fixed-stick stage must be 1 or 2')
    cfg.collisionless_mode = stage == 1
    cfg.training_stage = f'fixed_stick_stage{stage}_v32'
    bar = cfg.scene.wooden_bar.spawn
    bar.collision_props.collision_enabled = stage == 2
    bar.activate_contact_sensors = stage == 2
    cfg.scene.bar_contacts = None if stage == 1 else ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/WoodenBar',
        filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/Robot/{name}' for name in
                               ['base_link', *(joint.replace('_joint', '_link') for joint in LEG_JOINT_NAMES)]],
        update_period=cfg.sim.dt, history_length=cfg.decimation,
    )
    cfg.recorders = (FixedStickGeometryRecorderCfg() if stage == 1 else
                     RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE))
    cfg.rewards.collisionless_hit_penalty.weight = -100. if stage == 1 else 0.
    return cfg
