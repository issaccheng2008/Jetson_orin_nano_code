# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based rough-terrain locomotion environment config for the custom humanoid robot.

Suggested file name:
    humanoid_robot_policy_env_cfg.py

This file is designed for the project:
    Humanoid_Robot_Policy

Self-collision note:
    Fall detection is based on root orientation and root height. Contact sensors
    are used only for foot stepping rewards, not whole-body fall detection.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
try:
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
except ImportError:
    from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp

from .humanoid_robot import HUMANOID_ROBOT_CFG
from .training_phase import WOODEN_BAR_TRAINING_PHASE, STOP_BEFORE_CROSSING

SMALL_RANDOM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    # Size of each generated terrain patch.
    size=(8.0, 8.0),

    # Flat border around the complete terrain grid.
    border_width=10.0,

    # Creates 10 Ã— 20 = 200 terrain patches.
    num_rows=10,
    num_cols=20,

    # Resolution of the generated terrain.
    horizontal_scale=0.05,  # one mesh point every 5 cm
    vertical_scale=0.001,   # height resolution of 1 mm

    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,

    sub_terrains={
        "small_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,

            # Ground elevation varies from -5 mm to +5 mm.
            noise_range=(-0.005, 0.005),

            # Heights are generated in 1 mm increments.
            noise_step=0.001,

            # Random samples are generated every 10 cm and interpolated.
            # This produces smoother deviations instead of sharp noise.
            downsampled_scale=0.10,

            # Flat padding around each individual patch.
            border_width=0.25,
        ),
    },
)



##
# Robot-specific names
##

LEG_JOINT_NAMES = [
    # Right leg
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",

    # Left leg
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]



BASE_BODY_NAME = "base_link"

FOOT_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

COLLISIONLESS_BAR_FILTER_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
    "l_ankle_pitch_link",
    "r_ankle_pitch_link",
]

TARGET_BASE_HEIGHT = 0.32
MIN_BASE_HEIGHT = 0.20
MAX_BASE_TILT = math.radians(65.0)

WOODEN_BAR_LENGTH = 0.80  # 80cm across lateral Y (0.40m half-width)
WOODEN_BAR_WIDTH = 0.03   # 3cm along walking X
WOODEN_BAR_HEIGHT = 0.03  # 3cm vertical Z
PHYSICAL_BAR_HALF_WIDTH = 0.5 * WOODEN_BAR_WIDTH
PHYSICAL_BAR_HALF_LENGTH = 0.5 * WOODEN_BAR_LENGTH  # 0.40m
NUM_STICKS = 10
FIRST_STICK_X = 0.50
STICK_CLEAR_GAP = 0.25    # Initial clear gap; accuracy curriculum reduces it to 20cm
STICK_SPACING = STICK_CLEAR_GAP + WOODEN_BAR_WIDTH  # 0.28m initial pitch (center-to-center)
TEN_STICKS_USD_PATH = str(Path(__file__).resolve().parents[3] / "v3.2" / "ten_sticks.usd")

VIRTUAL_BAND_WIDTH = 0.04
VIRTUAL_BAND_HALF_WIDTH = 0.5 * VIRTUAL_BAND_WIDTH
VIRTUAL_BAND_NEAR_EDGE_OFFSET = 0.005
PHYSICAL_BAR_CENTER_DISTANCE = 0.03
PHYSICAL_BAR_POSITION_ERROR_RANGE = (-0.006, 0.006)
PHYSICAL_BAR_DROP_CLEARANCE = 0.010
HIDDEN_BAR_DEPTH = 2.0
FOOT_HEIGHT_SATURATION = 0.03
STEPPING_FOOT_DISTANCE_TO_BAND_EDGE = 0.22
PHYSICAL_WOODEN_BAR_NAME = "wooden_bar"
COLLISIONLESS_WOODEN_BAR_NAME = "collisionless_wooden_bar"

DEFAULT_STEP_DISTANCE = 0.15
CROSSING_STEP_DISTANCE = STICK_SPACING
PHASE_2_POST_CROSSING_STEP_DISTANCE = STICK_SPACING
PHASE_3_POST_CROSSING_STEP_DISTANCE = STICK_SPACING
PHASE_4_POST_CROSSING_STEP_DISTANCE = 0.0  # Course computes the exact paired-foot target

# Phase 5 mixes Phase 4 obstacle episodes with command-diversity episodes.
PHASE_5_BAR_EPISODE_PROBABILITY = 1.0 if NUM_STICKS > 1 else 0.80
PHASE_5_NO_BAR_STOP_PROBABILITY = 0.10
PHASE_5_STOP_TIME_RANGE_S = (0.0, 5.0)
PHASE_5_INITIAL_ANG_VEL_Z_RANGE = (-1.5, 1.5)
PHASE_5_FINAL_ANG_VEL_Z_RANGE = (-1.5, 1.5)
PHASE_5_ANG_VEL_Z_CURRICULUM_ITERATIONS = 4000

NORMAL_STEP_DEFAULT_PROBABILITY = 0.30
RANDOM_STEP_DISTANCE_RANGE = (0.02, 0.12)
CROSSING_TOUCHDOWN_INDEX_RANGE = (3, 10)
STEP_DISTANCE_GAUSSIAN_INITIAL_STD = 0.015
STEP_DISTANCE_GAUSSIAN_FINAL_STD = 0.002
STEP_DISTANCE_GAUSSIAN_START_STD = (
    STEP_DISTANCE_GAUSSIAN_INITIAL_STD
    if WOODEN_BAR_TRAINING_PHASE == 1
    else STEP_DISTANCE_GAUSSIAN_FINAL_STD
)
STEP_DISTANCE_TRACKING_REWARD_WEIGHT = 50.0
PHYSICAL_BAR_CROSSING_COMPLETION_REWARD_WEIGHT = 100.0
EXPECTED_POLICY_OBS_DIM = 49
CROSSING_STATE_UPDATE_INTERVAL_S = 0.02

step_reward_std_curriculum_end_step=1500

PPO_STEPS_PER_ITERATION = 24
# EL05 nominal torque, used only by the copied walking reward term.
EL05_RATED_TORQUE = 4

# Convex perimeters of the physical lowest sole surfaces, measured from the
# v3.2 USD collision meshes in each ankle-roll frame (front/rear trimmed).
# Order matches FOOT_BODY_NAMES: left, right. See tests/test_v32_sole_geometry.py.
FOOT_SOLE_VERTICES = (
    (
        (-0.103087232, -0.027696537, -0.043790001),
        (-0.102338016, -0.031525739, -0.043790001),
        (-0.100180455, -0.034776740, -0.043790001),
        (-0.096943028, -0.036954619, -0.043790001),
        (-0.093118601, -0.037727799, -0.043790001),
        (0.034880780, -0.038128614, -0.043790001),
        (0.038709980, -0.037379399, -0.043790001),
        (0.041960981, -0.035221837, -0.043790001),
        (0.044138860, -0.031984411, -0.043790001),
        (0.044912044, -0.028159978, -0.043790001),
        (0.045087397, 0.027839748, -0.043790001),
        (0.044338182, 0.031668946, -0.043790001),
        (0.042180620, 0.034919951, -0.043790001),
        (0.038943194, 0.037097827, -0.043790001),
        (0.035118762, 0.037871011, -0.043790001),
        (-0.092880614, 0.038271826, -0.043790001),
        (-0.096709818, 0.037522607, -0.043790001),
        (-0.099960819, 0.035365049, -0.043790001),
        (-0.102138698, 0.032127623, -0.043790001),
        (-0.102911875, 0.028303189, -0.043790001),
    ),
    (
        (-0.103087112, 0.027658129, -0.043790001),
        (-0.102911755, -0.028341597, -0.043790001),
        (-0.102138571, -0.032166030, -0.043790001),
        (-0.099960692, -0.035403457, -0.043790001),
        (-0.096709691, -0.037561014, -0.043790001),
        (-0.092880495, -0.038310234, -0.043790001),
        (0.035118882, -0.037909418, -0.043790001),
        (0.038943317, -0.037136234, -0.043790001),
        (0.042180739, -0.034958359, -0.043790001),
        (0.044338301, -0.031707354, -0.043790001),
        (0.045087520, -0.027878156, -0.043790001),
        (0.044912163, 0.028121570, -0.043790001),
        (0.044138979, 0.031946003, -0.043790001),
        (0.041961100, 0.035183430, -0.043790001),
        (0.038710099, 0.037340991, -0.043790001),
        (0.034880899, 0.038090207, -0.043790001),
        (-0.093118481, 0.037689392, -0.043790001),
        (-0.096942909, 0.036916208, -0.043790001),
        (-0.100180335, 0.034738332, -0.043790001),
        (-0.102337897, 0.031487327, -0.043790001),
    ),
)

ANKLE_JOINT_NAMES = [
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

YAW_ROLL_JOINT_NAMES = [
    ".*_leg_yaw_joint",
]


def _ordered_feet_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot", body_names=FOOT_BODY_NAMES, preserve_order=True
    )


def _ordered_feet_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True
    )


def _crossing_state_update_params() -> dict:
    """Return one authoritative set of configurable crossing parameters."""
    return {
        "feet_cfg": _ordered_feet_cfg(),
        "sensor_cfg": _ordered_feet_sensor_cfg(),
        "sole_vertices": FOOT_SOLE_VERTICES,
        "training_phase": WOODEN_BAR_TRAINING_PHASE,
        "collisionless_bar_name": COLLISIONLESS_WOODEN_BAR_NAME,
        "physical_bar_name": PHYSICAL_WOODEN_BAR_NAME,
        "bar_height": WOODEN_BAR_HEIGHT,
        "physical_bar_half_width": PHYSICAL_BAR_HALF_WIDTH,
        "physical_bar_half_length": PHYSICAL_BAR_HALF_LENGTH,
        "virtual_band_half_width": VIRTUAL_BAND_HALF_WIDTH,
        "virtual_band_near_edge_offset": VIRTUAL_BAND_NEAR_EDGE_OFFSET,
        "physical_bar_center_distance": PHYSICAL_BAR_CENTER_DISTANCE,
        "physical_bar_position_error_range": (
            PHYSICAL_BAR_POSITION_ERROR_RANGE
        ),
        "physical_bar_drop_clearance": PHYSICAL_BAR_DROP_CLEARANCE,
        "default_step_distance": DEFAULT_STEP_DISTANCE,
        "crossing_step_distance": CROSSING_STEP_DISTANCE,
        "phase_2_post_crossing_step_distance": (
            PHASE_2_POST_CROSSING_STEP_DISTANCE
        ),
        "phase_3_post_crossing_step_distance": (
            PHASE_3_POST_CROSSING_STEP_DISTANCE
        ),
        "phase_4_post_crossing_step_distance": (
            PHASE_4_POST_CROSSING_STEP_DISTANCE
        ),
        "normal_step_default_probability": (
            NORMAL_STEP_DEFAULT_PROBABILITY
        ),
        "random_step_distance_range": RANDOM_STEP_DISTANCE_RANGE,
    }


def _make_wooden_bar_cfg(prim_name: str, height: float) -> RigidObjectCfg:
    """Create one fixed-height bar variant for the obstacle curriculum."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(WOODEN_BAR_WIDTH, WOODEN_BAR_LENGTH, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.02, 0.02),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )


##
# Scene definition
##

@configclass
class HumanoidRobotPolicySceneCfg(InteractiveSceneCfg):
    """Scene configuration for rough-terrain walking with forward and turning commands."""

    # Phase 3's pre-startup USD collision filtering requires independently
    # authored physics schemas rather than replicated physics prims.
    replicate_physics: bool = WOODEN_BAR_TRAINING_PHASE != 3

    # Randomly rough ground used to improve locomotion robustness.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",

        # Generate small, smooth height variations instead of using an infinite plane.
        terrain_type="generator",
        terrain_generator=SMALL_RANDOM_ROUGH_TERRAIN_CFG,

        collision_group=-1,

        # Ground friction remains fixed.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),

        debug_vis=False,
    )

    # Robot.
    robot: ArticulationCfg = HUMANOID_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Simulated IMU attached to base_link.
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        debug_vis=False,

        # Replace these with the real IMU mounting pose relative to base_link.
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),  # this Isaac Lab version uses x, y, z, w
        ),
    )

    # Contact sensor used for foot stepping rewards only.
    # Do not use this for fall detection when self-collision is enabled,
    # because self-collision also produces contact forces.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # Fixed kinematic sticks can be repositioned on reset without stretching geometry.
    for _stick_index in range(NUM_STICKS):
        locals()[f"course_stick_{_stick_index}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/WoodenBars/stick_{_stick_index}",
            spawn=sim_utils.CuboidCfg(
                size=(WOODEN_BAR_WIDTH, WOODEN_BAR_LENGTH, WOODEN_BAR_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.88, .42, .12)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(FIRST_STICK_X + _stick_index * STICK_SPACING, 0., .5 * WOODEN_BAR_HEIGHT)
            ),
        )
    del _stick_index

    # Legacy/compatibility bars (hidden underground by default)
    collisionless_wooden_bar = _make_wooden_bar_cfg(
        "CollisionlessWoodenBar",
        WOODEN_BAR_HEIGHT,
    )

    # The Phase 4 physical bar is also used by Phase 5 bar episodes.
    wooden_bar = _make_wooden_bar_cfg(
        "WoodenBar",
        WOODEN_BAR_HEIGHT,
    )

    # Light.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


##
# MDP: Commands
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.ObstacleAwareVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(12.0, 12.0),

        # Lead steps use 0.20 m/s; paired following steps slow to 0.10 m/s.
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,

        # Playback/debug geometry uses the same sole and band dimensions as the
        # command, reward, and crossing calculations.
        feet_cfg=_ordered_feet_cfg(),
        sensor_cfg=_ordered_feet_sensor_cfg(),
        sole_vertices=FOOT_SOLE_VERTICES,
        virtual_band_length=WOODEN_BAR_LENGTH,
        virtual_band_height=FOOT_HEIGHT_SATURATION,

        debug_vis=False,
        phase_5_enabled=False,
        ranges=mdp.ObstacleAwareVelocityCommandCfg.Ranges(
            lin_vel_x=(0.20, 0.20),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# MDP: Actions
##

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


##
# MDP: Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the policy network."""

        # Three-axis acceleration measured in the simulated IMU frame.
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},

            # Initial estimate; later tune this using recordings from the real IMU.
            noise=Unoise(n_min=-0.3, n_max=0.3),

            # Convert typical acceleration magnitudes to roughly network-sized values.
            # For example, 9.81 m/s^2 becomes approximately 0.981.
            scale=0.1,
        )

        # Recommended: obtain all IMU-related observations from the same
        # simulated sensor frame.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Commanded walking velocity.
        velocity_commands = ObsTerm(
            func=mdp.forward_yaw_velocity_commands,
            params={"command_name": "base_velocity"},
        )

        # Signed longitudinal distance between the landing and support sole
        # fronts at the next valid touchdown.
        step_distance = ObsTerm(
            func=mdp.step_distance_command,
            params={"default_step_distance": DEFAULT_STEP_DISTANCE},
        )

        # 0 = normal walking, 1 = execute the crossing action.
        crossing_command = ObsTerm(func=mdp.crossing_command)

        # Joint state.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # Previous action.
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """Post initialization."""
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP: Events
##

@configclass
class EventCfg:
    """Configuration for events.

    Events handle startup/reset randomization.
    This first version is conservative for debugging.
    """

    configure_collisionless_bar_collisions = (
        EventTerm(
            func=mdp.configure_collisionless_bar_collisions,
            mode="prestartup",
            params={
                "training_phase": WOODEN_BAR_TRAINING_PHASE,
                "robot_name": "robot",
                "collisionless_bar_name": COLLISIONLESS_WOODEN_BAR_NAME,
                "rigid_body_names": COLLISIONLESS_BAR_FILTER_BODY_NAMES,
            },
        )
        if WOODEN_BAR_TRAINING_PHASE == 3
        else None
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.08, 0.12),
                "y": (-0.02, 0.02),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Keep default standing pose at first.
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_crossing_state = EventTerm(
        func=mdp.reset_crossing_state,
        mode="reset",
        params={
            "collisionless_bar_name": COLLISIONLESS_WOODEN_BAR_NAME,
            "physical_bar_name": PHYSICAL_WOODEN_BAR_NAME,
            "hidden_depth": HIDDEN_BAR_DEPTH,
            "training_phase": WOODEN_BAR_TRAINING_PHASE,
            "default_step_distance": DEFAULT_STEP_DISTANCE,
            "trigger_touchdown_range": CROSSING_TOUCHDOWN_INDEX_RANGE,
            "phase_5_bar_episode_probability": (
                PHASE_5_BAR_EPISODE_PROBABILITY
            ),
            "phase_5_no_bar_stop_probability": (
                PHASE_5_NO_BAR_STOP_PROBABILITY
            ),
            "phase_5_stop_time_range_s": PHASE_5_STOP_TIME_RANGE_S,
        },
    )

    # This is a safety caller. Reward/termination terms use the same cached
    # update earlier in the control step, so no event can be counted twice.
    update_crossing_state = EventTerm(
        func=mdp.update_crossing_state,
        mode="interval",
        interval_range_s=(
            CROSSING_STATE_UPDATE_INTERVAL_S,
            CROSSING_STATE_UPDATE_INTERVAL_S,
        ),
        is_global_time=False,
        params=_crossing_state_update_params(),
    )

    # Randomize the physics material of the two feet.
    randomize_foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),

            # One random material is assigned to each environment.
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.1),

            # Keep feet non-bouncy.
            "restitution_range": (0.0, 0.1),

            # Discretize the random range into material buckets.
            "num_buckets": 64,

            # Prevent physically inconsistent combinations such as
            # dynamic friction being greater than static friction.
            "make_consistent": True,
        },
    )

    # Randomize actuator response independently across simulated robots.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),

            # Multipliers applied to the nominal values in humanoid_robot.py.
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.80, 1.20),

            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            ),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    align_wooden_bars = EventTerm(
        func=mdp.align_wooden_bars_to_env_origins,
        mode="startup",
    )



@configclass
class RewardsCfg:
    """Reward terms for the bipedal humanoid walking MDP.

    This is G1-like, but modified to fight two early local optima:
    1. standing still
    2. shuffling/sliding feet without real stepping
    """

    # -------------------------------------------------------------------------
    # Main task rewards
    # -------------------------------------------------------------------------

    # 10-stick hurdle course rewards (4-Layer System)
    # Layer 1: Continuous forward progress throughout the hurdle course
    hurdle_forward_progress = RewTerm(
        func=mdp.hurdle_forward_progress_reward,
        weight=6.0,
    )

    # Layer 2: Dynamic foot clearance shaping (reward feet safely above stick)
    stick_over_clearance = RewTerm(
        func=mdp.stick_over_clearance_reward,
        weight=15.0,
    )

    # Layer 3: Precision landing in 20cm gap center on touchdown
    stick_landing_center = RewTerm(
        func=mdp.stick_landing_center_reward,
        weight=25.0,
    )

    # Layer 4: Milestone per stick cleared & completion of all 10 sticks
    # Rule 1: Clean pass without touching the stick -> +30.0
    stick_cleared = RewTerm(
        func=mdp.stick_cleared_reward,
        weight=30.0,
    )

    # Rule 3: Collided with stick and failed/fell -> -15.0 penalty
    stick_failed = RewTerm(
        func=mdp.stick_failed_penalty,
        weight=-15.0,
    )

    # Grand Prize: All 10 sticks cleared with zero touches across entire course -> +100.0
    all_sticks_completed = RewTerm(
        func=mdp.all_sticks_completed_reward,
        weight=100.0,
    )

    # First contact with each stick has a cost, even when the robot later clears it.
    stick_collision = RewTerm(
        func=mdp.stick_collision_penalty,
        weight=-5.0,
    )

    stop_stability = None
    stop_completion = None

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_quadratic_relative,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "moving_command_threshold": 0.05,
            "standing_std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Track the sampled yaw-rate commands so the policy learns turning together
    # with forward locomotion.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0 if WOODEN_BAR_TRAINING_PHASE == 1 else 4.0,
        params={
            "command_name": "base_velocity",
            "std": 0.3,
        },
    )

    # -------------------------------------------------------------------------
    # Anti-shuffling / stepping terms
    # -------------------------------------------------------------------------

    # G1-like foot timing reward using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            # 0.4 is G1-like. For your smaller/custom robot, start slightly lower.
            "threshold": 0.5,
        },
    )

    # G1-like foot slide penalty using only l_ankle_roll_link and r_ankle_roll_link.
    # This assumes the USD collision filters prevent foot self-collision from
    # corrupting foot contact timing. If that still happens later, replace this
    # with a custom ground-filtered reward.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
            ),
        },
    )

    ground_contact_flatness = RewTerm(
        func=mdp.ground_contact_flatness,
        weight=1,
        params={
            "flat_tolerance": math.radians(5.0),
            "penalty_start_angle": math.radians(10.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )

    # Match the walking task's effective iteration-zero weight and clearance
    # range without copying its separate reward-weight curriculum.
    swing_foot_clearance = RewTerm(
        func=mdp.swing_foot_clearance_reward,
        weight=1.5,
        params={
            "min_clearance": 0.01,
            "max_clearance": 0.04,
            "sole_vertices": FOOT_SOLE_VERTICES,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )

    # Score valid target-foot touchdowns during paired course crossing as well.
    step_distance_tracking_reward = RewTerm(
        func=mdp.step_distance_tracking_reward,
        weight=STEP_DISTANCE_TRACKING_REWARD_WEIGHT,
        params={
            "gaussian_std": .03 if NUM_STICKS > 1 else STEP_DISTANCE_GAUSSIAN_START_STD,
            **_crossing_state_update_params(),
        },
    )

    hurdle_target_approach = RewTerm(func=mdp.hurdle_target_approach_reward, weight=2.0)
    hurdle_wrong_landing = RewTerm(func=mdp.hurdle_wrong_landing_penalty, weight=-5.0)

    physical_bar_crossing_completion_reward = (
        None
        if NUM_STICKS > 1
        else (
            RewTerm(
                func=mdp.physical_bar_crossing_completion_reward,
                weight=PHYSICAL_BAR_CROSSING_COMPLETION_REWARD_WEIGHT,
                params=_crossing_state_update_params(),
            )
            if WOODEN_BAR_TRAINING_PHASE in (3, 4, 5)
            else None
        )
    )

    collisionless_bar_contact_penalty = (
        RewTerm(
            func=mdp.collisionless_bar_contact_penalty,
            weight=-100.0,
        )
        if WOODEN_BAR_TRAINING_PHASE == 3
        else None
    )

    # -------------------------------------------------------------------------
    # Contact / termination terms
    # -------------------------------------------------------------------------

    # Strong penalty for early termination, G1-style.
    termination_penalty = RewTerm(
        func=mdp.is_any_terminated_term,
        weight=-200.0,
        params={
            "term_keys": ["bad_orientation", "low_base_height", "hurdle_out_of_bounds", "hurdle_skipped_gap"]
        },
    )

    wooden_bar_moved_penalty = None

    # Disabled: fall is detected by root orientation and root height, not contact forces.
    illegal_non_foot_contact = None

    # -------------------------------------------------------------------------
    # Stability terms
    # -------------------------------------------------------------------------

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-3,
    )

    hurdle_body_heading = RewTerm(
        func=mdp.hurdle_body_heading_reward,
        weight=3.0,
        params={
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalize sudden sideways base acceleration.
    base_acc_y_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "y",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[BASE_BODY_NAME],
            ),
        },
    )

    # Penalize bouncing and sudden vertical base acceleration.
    base_acc_z_l2 = RewTerm(
        func=mdp.base_acceleration_l2,
        weight=-0.005,
        params={
            "axis": "z",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[BASE_BODY_NAME],
            ),
        },
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2,
    )

    # -------------------------------------------------------------------------
    # Joint / action penalties
    # -------------------------------------------------------------------------

    # Make these weaker at first.
    # If they are too strong, the easiest solution is "do not move".
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    dof_torque_over_nominal = RewTerm(
        func=mdp.joint_torque_over_nominal,
        weight=-0.1,
        params={
            "nominal_torque": EL05_RATED_TORQUE,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            ),
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
            )
        },
    )

    # Slightly weaker than G1 at first. Increase later when walking works.
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.002,
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ANKLE_JOINT_NAMES,
            )
        },
    )

    # Do not penalize leg pitch/knee pitch too much, because those are needed
    # for stepping. Only softly discourage sideways/yaw flailing.
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=YAW_ROLL_JOINT_NAMES,
            )
        },
    )

    #both feet airborn penalty
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODY_NAMES,
            ),
        },
    )

    stepping_wooden_bar_step_reward = RewTerm(
        func=mdp.stepping_wooden_bar_step_reward,
        weight=1.5 if WOODEN_BAR_TRAINING_PHASE == 2 else 0.0,
        params={
            "height_saturation": FOOT_HEIGHT_SATURATION,
            "forward_velocity_saturation": 0.15,
            "progress_unit": 0.22,
            "band_half_width": VIRTUAL_BAND_HALF_WIDTH,
            "sole_vertices": FOOT_SOLE_VERTICES,
            "feet_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )

    following_wooden_bar_step_reward = RewTerm(
        func=mdp.following_wooden_bar_step_reward,
        weight=3.0 if WOODEN_BAR_TRAINING_PHASE == 2 else 0.0,
        params={
            "height_saturation": FOOT_HEIGHT_SATURATION,
            "forward_velocity_saturation": 0.2,
            "progress_unit": 0.22,
            "stepping_foot_distance_to_band_edge": (
                STEPPING_FOOT_DISTANCE_TO_BAND_EDGE
            ),
            "band_half_width": VIRTUAL_BAND_HALF_WIDTH,
            "sole_vertices": FOOT_SOLE_VERTICES,
            "feet_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )

    feet_height_entering_band_reward = RewTerm(
        func=mdp.feet_height_entering_band_reward,
        weight=100 if WOODEN_BAR_TRAINING_PHASE == 2 else 0.0,
        params={
            "height_saturation": FOOT_HEIGHT_SATURATION,
            "band_half_width": VIRTUAL_BAND_HALF_WIDTH,
            "sole_vertices": FOOT_SOLE_VERTICES,
            "feet_cfg": _ordered_feet_cfg(),
            "sensor_cfg": _ordered_feet_sensor_cfg(),
        },
    )


##
# MDP: Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stop_failed = None

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": MAX_BASE_TILT,
        },
    )

    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": MIN_BASE_HEIGHT,
        },
    )

    hurdle_out_of_bounds = DoneTerm(
        func=mdp.hurdle_out_of_bounds,
        params={
            "feet_cfg": _ordered_feet_cfg(),
            "sole_vertices": FOOT_SOLE_VERTICES,
            "half_length": PHYSICAL_BAR_HALF_LENGTH,
            "course_end_x": FIRST_STICK_X + NUM_STICKS * STICK_SPACING - PHYSICAL_BAR_HALF_WIDTH,
        },
    )

    # Keep after every failure term: update the current crossing events before rewards.
    hurdle_skipped_gap = DoneTerm(
        func=mdp.hurdle_skipped_gap,
        params={"feet_cfg": _ordered_feet_cfg(), "sensor_cfg": _ordered_feet_sensor_cfg(),
                "sole_vertices": FOOT_SOLE_VERTICES},
    )

    hurdle_course_completed = DoneTerm(func=mdp.hurdle_course_completed)

    wooden_bar_moved = None


##
# MDP: Curriculum
##

@configclass
class CurriculumCfg:
    """Configure the selected independently resumed wooden-bar phase."""

    policy_observation_shape = CurrTerm(
        func=mdp.policy_observation_shape_check,
        params={
            "group_name": "policy",
            "expected_dim": EXPECTED_POLICY_OBS_DIM,
        },
    )

    step_distance_gaussian = None if NUM_STICKS > 1 else CurrTerm(
        func=mdp.step_distance_gaussian_curriculum,
        params={
            "reward_term_name": "step_distance_tracking_reward",
            "initial_std": STEP_DISTANCE_GAUSSIAN_START_STD,
            "final_std": STEP_DISTANCE_GAUSSIAN_FINAL_STD,
            "start_step": 0,
            "end_step": step_reward_std_curriculum_end_step*PPO_STEPS_PER_ITERATION,
        },
    )

    phase_5_ang_vel_z = (
        CurrTerm(
            func=mdp.phase_5_ang_vel_z_curriculum,
            params={
                "command_name": "base_velocity",
                "initial_range": PHASE_5_INITIAL_ANG_VEL_Z_RANGE,
                "final_range": PHASE_5_FINAL_ANG_VEL_Z_RANGE,
                "start_step": 0,
                "end_step": (
                    PHASE_5_ANG_VEL_Z_CURRICULUM_ITERATIONS
                    * PPO_STEPS_PER_ITERATION
                ),
            },
        )
        if WOODEN_BAR_TRAINING_PHASE == 5
        else None
    )

    wooden_bar_reward_weights = (
        CurrTerm(
            func=mdp.wooden_bar_reward_weight_curriculum,
            params={
                "pre_start_reward_weights": {
                    "stepping_wooden_bar_step_reward": 15,
                    "following_wooden_bar_step_reward": 20,
                    "feet_height_entering_band_reward": 400.0,
                },
                "reward_weight_ranges": {
                    "stepping_wooden_bar_step_reward": (15.0, 15.0),
                    "following_wooden_bar_step_reward": (24.0, 15.0),
                    "feet_height_entering_band_reward": (400.0, 400.0),
                },
                "start_step": (
                    0
                    * PPO_STEPS_PER_ITERATION
                ),
                "end_step": (
                    6000
                    * PPO_STEPS_PER_ITERATION
                ),
            },
        )
        if WOODEN_BAR_TRAINING_PHASE == 2
        else None
    )

##
# Environment configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for rough-terrain forward-velocity and yaw-rate tracking."""

    # Resume the global curriculum clock alongside model_650.
    # One PPO iteration collects 24 environment/control steps (see
    # HumanoidRobotRoughPPORunnerCfg.num_steps_per_env), therefore:
    #
    #     model_650 -> 650 * 24 = 15,600 curriculum steps
    #
    # Set this to 0 when training a new policy from scratch.  For another
    # model_N checkpoint, set it to N * num_steps_per_env.
    stop_before_crossing: bool = STOP_BEFORE_CROSSING
    stop_hold_s: float = 0.5
    stop_timeout_s: float = 3.0
    hurdle_initial_gap: float = .25
    hurdle_gap_levels: tuple[float, ...] = (.25, .225, .20)
    hurdle_gap_curriculum: bool = True
    hurdle_curriculum_min_episodes: int = 200
    hurdle_curriculum_landing_accuracy: float = .90
    hurdle_curriculum_completion_rate: float = .80
    hurdle_follow_speed: float = .20

    curriculum_start_step: int = 0


    # Scene settings.
    scene: HumanoidRobotPolicySceneCfg = HumanoidRobotPolicySceneCfg(
        num_envs=512,
        env_spacing=2.5,
    )

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings.
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # General settings.
        self.decimation = 4
        self.episode_length_s = 30.0

        # Simulation settings.
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Sensor update periods.
        self.scene.contact_forces.update_period = self.sim.dt

        # Viewer.
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)

        # Update the simulated IMU at every physics step: 0.005 s = 200 Hz.
        self.scene.imu.update_period = self.sim.dt


##
# Play / visualization configuration
##

@configclass
class HumanoidRobotPolicyEnvCfg_PLAY(HumanoidRobotPolicyEnvCfg):
    """Keyboard-controlled configuration for policy playback."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0

        # Keyboard controls the base_velocity command.
        # self.commands.base_velocity.class_type = (
        #     mdp.KeyboardVelocityCommand
        # )

        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.20)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.observations.policy.enable_corruption = False

        # Camera follows the robot in environment 0.
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"

        # Camera position and target relative to the robot root.
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)

