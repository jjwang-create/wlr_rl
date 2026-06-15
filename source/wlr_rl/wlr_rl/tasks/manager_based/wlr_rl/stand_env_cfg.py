# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import wlr_rl.tasks.manager_based.wlr_rl.mdp as mdp

from .flat_env_cfg import FlatEnvCfg

STAND_BASE_HEIGHT = 0.392
STAND_FRONT_POINT = (0.257, 0.0, -0.135)
STAND_REAR_POINT = (-0.257, 0.0, -0.135)
STAND_MIN_BODY_POINT_HEIGHT = 0.08
FORWARD_BACK_BODY_LENGTH = 0.50
FORWARD_BACK_FORWARD_DURATION = 2.5
FORWARD_BACK_HOLD_DURATION = 1.5
FORWARD_BACK_RETURN_DURATION = 2.5
FORWARD_BACK_TRACK_SPEED = 0.08
FORWARD_BACK_TRACK_SEGMENT_DURATION = 2.5
FORWARD_BACK_TRACK_SETTLE_DURATION = 1.5


@configclass
class StandEnvCfg(FlatEnvCfg):
    """Strict short-horizon standing task for the wheel-legged robot."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 6.0
        self.scene.robot.init_state.pos = (0.0, 0.0, STAND_BASE_HEIGHT)
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # Start exactly from the nominal standing pose for the first standing curriculum stage.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params = {
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        }

        # Zero command, no heading controller. The policy objective is pure balance.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.heading_control_stiffness = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Keep the action space near the nominal pose during the short standing bootstrap.
        self.actions.hip_pos.scale = 0.15
        self.actions.knee_pos.scale = 0.15
        self.actions.wheel_vel.scale = 40.0

        # Reward only what is needed for stable standing.
        self.rewards.termination_penalty.weight = -2000.0
        self.rewards.track_lin_vel_x_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.base_height_exp.weight = 40.0
        self.rewards.base_height_exp.params["target_height"] = STAND_BASE_HEIGHT
        self.rewards.base_height_exp.params["std"] = 0.12
        self.rewards.flat_orientation_roll_exp.weight = 20.0
        self.rewards.flat_orientation_roll_exp.params["std"] = 0.12
        self.rewards.flat_orientation_pitch_exp.weight = 100.0
        self.rewards.flat_orientation_pitch_exp.params["std"] = 0.12
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-80.0)
        self.rewards.front_clearance = RewTerm(
            func=mdp.body_point_clearance_exp,
            weight=20.0,
            params={"body_point": STAND_FRONT_POINT, "minimum_height": STAND_MIN_BODY_POINT_HEIGHT, "std": 0.04},
        )
        self.rewards.rear_clearance = RewTerm(
            func=mdp.body_point_clearance_exp,
            weight=20.0,
            params={"body_point": STAND_REAR_POINT, "minimum_height": STAND_MIN_BODY_POINT_HEIGHT, "std": 0.04},
        )
        self.rewards.lin_vel_xy_l2 = RewTerm(func=mdp.lin_vel_xy_l2, weight=-5.0)
        self.rewards.lin_vel_z_l2.weight = -5.0
        self.rewards.ang_vel_x_l2.weight = -1.0
        self.rewards.ang_vel_y_l2.weight = -4.0
        self.rewards.ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.5)
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.leg_acc_l2.weight = -1.0e-5
        self.rewards.wheel_acc_l2.weight = -1.0e-7
        self.rewards.leg_action_rate_l2.weight = -0.5
        self.rewards.wheel_action_rate_l2.weight = -0.1
        self.rewards.virtual_leg_angle_diff_l2.weight = -20.0
        self.rewards.virtual_leg_length_diff_l2.weight = -5.0
        self.rewards.virtual_leg_angle_deviation_l2.weight = -15.0
        self.rewards.virtual_leg_angle_deviation_l2.params["target_angle"] = 0.0
        self.rewards.undesired_contacts.weight = -200.0
        self.rewards.undesired_contacts.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "hip.*", "knee.*"]),
            "threshold": 1.0,
        }
        self.rewards.desired_contacts.weight = 0.0
        self.rewards.constant_bonus.weight = 5.0

        self.terminations.turn_over.params["limit_angle"] = math.radians(20.0)
        self.terminations.base_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.26, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.roll_pitch = DoneTerm(
            func=mdp.roll_pitch_exceeds,
            params={"max_roll": math.radians(12.0), "max_pitch": math.radians(12.0)},
        )
        self.terminations.front_low = DoneTerm(
            func=mdp.body_point_height_below_minimum,
            params={"body_point": STAND_FRONT_POINT, "minimum_height": STAND_MIN_BODY_POINT_HEIGHT},
        )
        self.terminations.rear_low = DoneTerm(
            func=mdp.body_point_height_below_minimum,
            params={"body_point": STAND_REAR_POINT, "minimum_height": STAND_MIN_BODY_POINT_HEIGHT},
        )


@configclass
class StandEnvCfg_PLAY(StandEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandStrictEnvCfg(StandEnvCfg):
    """Standing task stage that rejects support from any non-wheel contact."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.non_wheel_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "hip.*", "knee.*"]),
                "threshold": 1.0,
            },
        )


@configclass
class StandStrictEnvCfg_PLAY(StandStrictEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackStrictEnvCfg(StandStrictEnvCfg):
    """Move forward one body length, stabilize, move back, and stabilize again."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = (
            FORWARD_BACK_FORWARD_DURATION + 2.0 * FORWARD_BACK_HOLD_DURATION + FORWARD_BACK_RETURN_DURATION
        )
        self.actions.wheel_vel.scale = 60.0

        command_params = {
            "body_length": FORWARD_BACK_BODY_LENGTH,
            "forward_duration": FORWARD_BACK_FORWARD_DURATION,
            "hold_duration": FORWARD_BACK_HOLD_DURATION,
            "return_duration": FORWARD_BACK_RETURN_DURATION,
        }
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_command,
            params=command_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_command,
            params=command_params,
        )

        self.rewards.forward_back_position = RewTerm(
            func=mdp.forward_back_position_exp,
            weight=160.0,
            params={**command_params, "std": 0.12},
        )
        self.rewards.forward_back_velocity = RewTerm(
            func=mdp.forward_back_velocity_exp,
            weight=60.0,
            params={**command_params, "std": 0.20},
        )
        self.rewards.final_stability = RewTerm(
            func=mdp.final_stability_exp,
            weight=60.0,
            params={**command_params, "std": 0.12},
        )
        self.rewards.lateral_position_l2 = RewTerm(func=mdp.lateral_position_l2, weight=-30.0)
        self.rewards.wheel_velocity_l2 = RewTerm(
            func=mdp.wheel_velocity_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.lin_vel_xy_l2.weight = -1.0
        self.rewards.wheel_acc_l2.weight = -5.0e-8
        self.rewards.wheel_action_rate_l2.weight = -0.03

        self.terminations.path_error = DoneTerm(
            func=mdp.forward_back_tracking_error_exceeds,
            params={**command_params, "max_x_error": 0.45, "max_y_error": 0.25},
        )


@configclass
class StandForwardBackStrictEnvCfg_PLAY(StandForwardBackStrictEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackBootstrapEnvCfg(StandForwardBackStrictEnvCfg):
    """Easier first stage for learning active forward/back wheel motion."""

    def __post_init__(self) -> None:
        super().__post_init__()

        body_length = 0.25
        forward_duration = 3.0
        hold_duration = 1.5
        return_duration = 3.0
        self.episode_length_s = forward_duration + 2.0 * hold_duration + return_duration
        self.actions.wheel_vel.scale = 80.0

        command_params = {
            "body_length": body_length,
            "forward_duration": forward_duration,
            "hold_duration": hold_duration,
            "return_duration": return_duration,
        }
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_command,
            params=command_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_command,
            params=command_params,
        )

        self.rewards.forward_back_position.params = {**command_params, "std": 0.18}
        self.rewards.forward_back_velocity.params = {**command_params, "std": 0.18}
        self.rewards.final_stability.params = {**command_params, "std": 0.16}
        self.rewards.forward_back_position.weight = 220.0
        self.rewards.forward_back_velocity.weight = 80.0
        self.rewards.final_stability.weight = 80.0
        self.rewards.wheel_velocity_l2.weight = -0.002
        self.rewards.wheel_action_rate_l2.weight = -0.01
        self.rewards.lin_vel_xy_l2.weight = -0.5
        self.terminations.path_error = None


@configclass
class StandForwardBackBootstrapEnvCfg_PLAY(StandForwardBackBootstrapEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackDirectedEnvCfg(StandStrictEnvCfg):
    """Forward/back sequence with velocity-like commands and strong straight-line constraints."""

    def __post_init__(self) -> None:
        super().__post_init__()

        body_length = 0.25
        forward_duration = 4.0
        hold_duration = 1.5
        return_duration = 4.0
        self.episode_length_s = forward_duration + 2.0 * hold_duration + return_duration
        self.actions.hip_pos.scale = 0.12
        self.actions.knee_pos.scale = 0.12
        self.actions.wheel_vel.scale = 50.0

        command_params = {
            "body_length": body_length,
            "forward_duration": forward_duration,
            "hold_duration": hold_duration,
            "return_duration": return_duration,
        }
        command_obs_params = {**command_params, "position_gain": 1.2, "max_command_velocity": 0.30}
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_velocity_command,
            params=command_obs_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_velocity_command,
            params=command_obs_params,
        )

        self.rewards.forward_back_position = RewTerm(
            func=mdp.forward_back_position_exp,
            weight=180.0,
            params={**command_params, "std": 0.14},
        )
        self.rewards.forward_back_position_l2 = RewTerm(
            func=mdp.forward_back_position_l2,
            weight=-320.0,
            params=command_params,
        )
        self.rewards.forward_back_velocity = RewTerm(
            func=mdp.forward_back_world_velocity_exp,
            weight=120.0,
            params={**command_params, "std": 0.12},
        )
        self.rewards.final_stability = RewTerm(
            func=mdp.final_stability_exp,
            weight=120.0,
            params={**command_params, "std": 0.12},
        )
        self.rewards.lateral_position_l2 = RewTerm(func=mdp.lateral_position_l2, weight=-260.0)
        self.rewards.lateral_velocity_l2 = RewTerm(func=mdp.lateral_velocity_l2, weight=-60.0)
        self.rewards.yaw_drift_l2 = RewTerm(func=mdp.yaw_drift_l2, weight=-80.0)
        self.rewards.wheel_velocity_l2 = RewTerm(
            func=mdp.wheel_velocity_l2,
            weight=-0.004,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.wheel_velocity_difference_l2 = RewTerm(
            func=mdp.wheel_velocity_difference_l2,
            weight=-0.004,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.lin_vel_xy_l2.weight = -0.1
        self.rewards.ang_vel_z_l2.weight = -1.0
        self.rewards.wheel_acc_l2.weight = -5.0e-8
        self.rewards.wheel_action_rate_l2.weight = -0.02

        self.terminations.path_error = DoneTerm(
            func=mdp.forward_back_tracking_error_exceeds,
            params={**command_params, "max_x_error": 0.35, "max_y_error": 0.16},
        )
        self.terminations.yaw_error = DoneTerm(
            func=mdp.yaw_exceeds,
            params={"max_yaw": math.radians(25.0)},
        )


@configclass
class StandForwardBackDirectedEnvCfg_PLAY(StandForwardBackDirectedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackDirectedSoftEnvCfg(StandForwardBackDirectedEnvCfg):
    """Soft bootstrap for the directed sequence before path/yaw hard termination."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.path_error = None
        self.terminations.yaw_error = None

        self.rewards.forward_back_position.weight = 220.0
        self.rewards.forward_back_position_l2.weight = -380.0
        self.rewards.forward_back_velocity.weight = 140.0
        self.rewards.final_stability.weight = 100.0
        self.rewards.lateral_position_l2.weight = -360.0
        self.rewards.lateral_velocity_l2.weight = -80.0
        self.rewards.yaw_drift_l2.weight = -140.0
        self.rewards.wheel_velocity_l2.weight = -0.003
        self.rewards.wheel_velocity_difference_l2.weight = -0.006


@configclass
class StandForwardBackDirectedSoftEnvCfg_PLAY(StandForwardBackDirectedSoftEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackVelocityPulseEnvCfg(StandStrictEnvCfg):
    """Gentle velocity-pulse curriculum for forward/hold/back/hold motion."""

    def __post_init__(self) -> None:
        super().__post_init__()

        body_length = 0.18
        forward_duration = 4.0
        hold_duration = 1.5
        return_duration = 4.0
        self.episode_length_s = forward_duration + 2.0 * hold_duration + return_duration
        self.actions.hip_pos.scale = 0.08
        self.actions.knee_pos.scale = 0.08
        self.actions.wheel_vel.scale = 35.0

        command_params = {
            "body_length": body_length,
            "forward_duration": forward_duration,
            "hold_duration": hold_duration,
            "return_duration": return_duration,
        }
        command_obs_params = {**command_params, "position_gain": 0.25, "max_command_velocity": 0.12}
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_velocity_command,
            params=command_obs_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.forward_back_sequence_velocity_command,
            params=command_obs_params,
        )

        self.rewards.forward_back_position = RewTerm(
            func=mdp.forward_back_position_exp,
            weight=35.0,
            params={**command_params, "std": 0.25},
        )
        self.rewards.forward_back_position_l2 = RewTerm(
            func=mdp.forward_back_position_l2,
            weight=-25.0,
            params=command_params,
        )
        self.rewards.forward_back_velocity = RewTerm(
            func=mdp.forward_back_world_velocity_exp,
            weight=85.0,
            params={**command_params, "std": 0.12},
        )
        self.rewards.final_stability = RewTerm(
            func=mdp.final_stability_exp,
            weight=45.0,
            params={**command_params, "std": 0.16},
        )
        self.rewards.lateral_position_l2 = RewTerm(func=mdp.lateral_position_l2, weight=-90.0)
        self.rewards.lateral_velocity_l2 = RewTerm(func=mdp.lateral_velocity_l2, weight=-25.0)
        self.rewards.yaw_drift_l2 = RewTerm(func=mdp.yaw_drift_l2, weight=-70.0)
        self.rewards.wheel_velocity_l2 = RewTerm(
            func=mdp.wheel_velocity_l2,
            weight=-0.002,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.wheel_velocity_difference_l2 = RewTerm(
            func=mdp.wheel_velocity_difference_l2,
            weight=-0.008,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.lin_vel_xy_l2.weight = -0.05
        self.rewards.ang_vel_z_l2.weight = -1.0
        self.rewards.wheel_acc_l2.weight = -5.0e-8
        self.rewards.wheel_action_rate_l2.weight = -0.02


@configclass
class StandForwardBackVelocityPulseEnvCfg_PLAY(StandForwardBackVelocityPulseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackCommandNeutralPulseEnvCfg(StandForwardBackVelocityPulseEnvCfg):
    """Very conservative pulse stage for command-neutral standing warm starts."""

    def __post_init__(self) -> None:
        super().__post_init__()

        body_length = 0.08
        forward_duration = 4.0
        hold_duration = 1.5
        return_duration = 4.0
        self.episode_length_s = forward_duration + 2.0 * hold_duration + return_duration
        # Keep the standing checkpoint's action semantics intact; constrain exploration with rewards instead.
        self.actions.hip_pos.scale = 0.15
        self.actions.knee_pos.scale = 0.15
        self.actions.wheel_vel.scale = 40.0

        command_params = {
            "body_length": body_length,
            "forward_duration": forward_duration,
            "hold_duration": hold_duration,
            "return_duration": return_duration,
        }
        command_obs_params = {**command_params, "position_gain": 0.12, "max_command_velocity": 0.055}
        self.observations.policy.velocity_commands.params = command_obs_params
        self.observations.critic.velocity_commands.params = command_obs_params

        self.rewards.forward_back_position.weight = 8.0
        self.rewards.forward_back_position.params = {**command_params, "std": 0.30}
        self.rewards.forward_back_position_l2.weight = -4.0
        self.rewards.forward_back_position_l2.params = command_params
        self.rewards.forward_back_velocity.weight = 28.0
        self.rewards.forward_back_velocity.params = {**command_params, "std": 0.10}
        self.rewards.final_stability.weight = 25.0
        self.rewards.final_stability.params = {**command_params, "std": 0.18}
        self.rewards.lateral_position_l2.weight = -45.0
        self.rewards.lateral_velocity_l2.weight = -18.0
        self.rewards.yaw_drift_l2.weight = -45.0
        self.rewards.wheel_velocity_l2.weight = -0.001
        self.rewards.wheel_velocity_difference_l2.weight = -0.01
        self.rewards.lin_vel_xy_l2.weight = -0.02
        self.rewards.ang_vel_z_l2.weight = -0.5
        self.rewards.leg_action_l2 = RewTerm(
            func=mdp.action_l2_joint,
            weight=-2.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*", "knee.*"])},
        )
        self.rewards.wheel_action_l2 = RewTerm(
            func=mdp.action_l2_joint,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )


@configclass
class StandForwardBackCommandNeutralPulseEnvCfg_PLAY(StandForwardBackCommandNeutralPulseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class StandForwardBackVelocityTrackEnvCfg(StandStrictEnvCfg):
    """Forward/back velocity following with two direction changes and a final zero-speed settle."""

    def __post_init__(self) -> None:
        super().__post_init__()

        speed = FORWARD_BACK_TRACK_SPEED
        segment_duration = FORWARD_BACK_TRACK_SEGMENT_DURATION
        settle_duration = FORWARD_BACK_TRACK_SETTLE_DURATION
        self.episode_length_s = 3.0 * segment_duration + settle_duration

        # Preserve standing-checkpoint action semantics when resuming from command-neutral standing.
        self.actions.hip_pos.scale = 0.15
        self.actions.knee_pos.scale = 0.15
        self.actions.wheel_vel.scale = 40.0

        command_params = {
            "speed": speed,
            "segment_duration": segment_duration,
            "settle_duration": settle_duration,
        }
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.forward_back_velocity_schedule_command,
            params=command_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=mdp.forward_back_velocity_schedule_command,
            params=command_params,
        )

        self.rewards.forward_back_velocity = RewTerm(
            func=mdp.forward_back_velocity_schedule_exp,
            weight=65.0,
            params={**command_params, "std": 0.08},
        )
        self.rewards.forward_back_velocity_l2 = RewTerm(
            func=mdp.forward_back_velocity_schedule_l2,
            weight=-18.0,
            params=command_params,
        )
        self.rewards.lateral_position_l2 = RewTerm(func=mdp.lateral_position_l2, weight=-45.0)
        self.rewards.lateral_velocity_l2 = RewTerm(func=mdp.lateral_velocity_l2, weight=-20.0)
        self.rewards.yaw_drift_l2 = RewTerm(func=mdp.yaw_drift_l2, weight=-55.0)
        self.rewards.wheel_velocity_difference_l2 = RewTerm(
            func=mdp.wheel_velocity_difference_l2,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )
        self.rewards.lin_vel_xy_l2.weight = -0.02
        self.rewards.ang_vel_z_l2.weight = -0.5
        self.rewards.wheel_acc_l2.weight = -5.0e-8
        self.rewards.wheel_action_rate_l2.weight = -0.02
        self.rewards.leg_action_l2 = RewTerm(
            func=mdp.action_l2_joint,
            weight=-2.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*", "knee.*"])},
        )
        self.rewards.wheel_action_l2 = RewTerm(
            func=mdp.action_l2_joint,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
        )


@configclass
class StandForwardBackVelocityTrackEnvCfg_PLAY(StandForwardBackVelocityTrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
