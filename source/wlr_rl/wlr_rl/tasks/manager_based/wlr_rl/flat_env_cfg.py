# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import RoughEnvCfg
from .velocity_env_cfg import FLAT_TERRAIN_CFG


@configclass
class FlatEnvCfg(RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to a generated local flat mesh instead of Isaac Sim's remote grid USD.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        # # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_lin_vel_x_exp.weight = 30.0
        self.rewards.track_lin_vel_x_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 15.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 1.0
        self.rewards.base_height_exp.weight = 30.0
        self.rewards.base_height_exp.params["target_height"] = 0.29
        self.rewards.base_height_exp.params["std"] = 0.1
        self.rewards.flat_orientation_roll_exp.weight = 5.0
        self.rewards.flat_orientation_roll_exp.params["std"] = 0.1
        self.rewards.flat_orientation_pitch_exp.weight = 5.0
        self.rewards.flat_orientation_pitch_exp.params["std"] = 0.1

        self.rewards.lin_vel_z_l2.weight = -10.0
        self.rewards.ang_vel_x_l2.weight = -1  # roll
        self.rewards.ang_vel_y_l2.weight = -2  # pitch

        self.rewards.dof_torques_l2.weight = -5.0e-4
        self.rewards.leg_acc_l2.weight = -2.0e-5
        self.rewards.wheel_acc_l2.weight = -1.0e-7
        self.rewards.leg_action_rate_l2.weight = -1.0
        self.rewards.wheel_action_rate_l2.weight = -0.5
  
        self.rewards.virtual_leg_angle_diff_l2.weight = -50.0
        self.rewards.virtual_leg_length_diff_l2.weight = -10.0
        self.rewards.virtual_leg_angle_deviation_l2.weight = -5.0

        self.rewards.undesired_contacts.weight = -30.0
        self.rewards.desired_contacts.weight = 0

        self.rewards.termination_penalty.weight = -500.0
        self.rewards.constant_bonus.weight = 25.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
class FlatEnvCfg_PLAY(FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
