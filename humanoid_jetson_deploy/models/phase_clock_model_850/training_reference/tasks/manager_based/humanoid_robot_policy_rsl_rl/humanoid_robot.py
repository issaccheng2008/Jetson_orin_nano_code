# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Articulation configuration using this checkout's v3.2 USD asset."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Resolve from the project so Windows and HPC copies use the same asset.
HUMANOID_USD_PATH = str(Path(__file__).resolve().parents[3] / "v3.2" / "v3.2.usd")


HUMANOID_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HUMANOID_USD_PATH,
        # Keep contact sensors enabled for foot stepping rewards. Fall detection
        # is handled from root orientation and root height in the environment.
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collision is enabled because the USD collision filters and
            # solve-contact settings have been manually cleaned up for the
            # nearby leg links that can contact during locomotion.
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Estimated from the uploaded URDF joint chain. After importing the USD,
        # tune this so both feet are just touching the ground at reset.
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            # Right leg
            "r_leg_pitch_joint": 0.15,
            "r_leg_roll_joint": 0.0,
            "r_leg_yaw_joint": 0.0,
            "r_knee_pitch_joint": 0.30,
            "r_ankle_pitch_joint": -0.15,
            "r_ankle_roll_joint": 0.0,

            # Left leg: mirrored joint signs because the pitch axes are reversed.
            "l_leg_pitch_joint": -0.15,
            "l_leg_roll_joint": 0.0,
            "l_leg_yaw_joint": 0.0,
            "l_knee_pitch_joint": -0.30,
            "l_ankle_pitch_joint": 0.15,
            "l_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_pitch_joint",
                ".*_leg_roll_joint",
                ".*_leg_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            # Your URDF currently lists effort=5 and velocity=10 for every joint.
            # These are conservative placeholders; replace them with calibrated
            # motor limits when you have them.
            effort_limit_sim=5.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_leg_pitch_joint": 35.0,
                ".*_leg_roll_joint": 30.0,
                ".*_leg_yaw_joint": 20.0,
                ".*_knee_pitch_joint": 35.0,
            },
            damping={
                ".*_leg_pitch_joint": 1.5,
                ".*_leg_roll_joint": 1.2,
                ".*_leg_yaw_joint": 1.0,
                ".*_knee_pitch_joint": 1.5,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=5.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_ankle_pitch_joint": 15.0,
                ".*_ankle_roll_joint": 12.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.8,
                ".*_ankle_roll_joint": 0.7,
            },
            armature=0.005,
        ),
    },
)
