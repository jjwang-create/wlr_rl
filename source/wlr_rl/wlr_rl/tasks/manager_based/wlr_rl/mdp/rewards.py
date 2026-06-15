# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .observations import virtual_leg_angle, virtual_leg_length
from .observations import _env_origin_xy, _forward_back_sequence, _forward_back_velocity_schedule

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def track_lin_vel_x_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Only tracks forward/backward velocity (x-axis), ignoring lateral velocity (y-axis).
    Returns exp(-error²/std²) which is normalized to (0, 1].
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # Only track x-axis (forward/backward) velocity
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward asset height tracking using exponential kernel.

    Returns exp(-error²/std²) which is normalized to (0, 1].
    
    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO：增加command作为目标高度的选项
    adjusted_target_height = target_height
    # Compute the exponential reward
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / std**2)


def body_point_clearance_exp(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    body_point: tuple[float, float, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping a root-body local point above a minimum world height."""
    asset = env.scene[asset_cfg.name]
    point_b = torch.tensor(body_point, dtype=asset.data.root_pos_w.dtype, device=env.device).repeat(env.num_envs, 1)
    point_w = asset.data.root_pos_w + quat_apply(asset.data.root_quat_w, point_b)
    shortfall = torch.clamp(minimum_height - point_w[:, 2], min=0.0)
    return torch.exp(-torch.square(shortfall) / std**2)

def lin_vel_z_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel.
    
    Returns vz² for use with negative weight.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def lin_vel_xy_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize horizontal base linear velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def ang_vel_z_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize yaw-axis base angular velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def ang_vel_x_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize x-axis (roll) base angular velocity using L2 squared kernel.
    
    Returns ωx² for use with negative weight.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def ang_vel_y_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize y-axis (pitch) base angular velocity using L2 squared kernel.
    
    Returns ωy² for use with negative weight.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 1])


def flat_orientation_roll_exp(
    env: "ManagerBasedRLEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat orientation along roll axis using exponential kernel.
    
    Uses the x-component of projected gravity vector in body frame.
    When robot is flat, projected_gravity_b[:, 0] = 0.
    Returns exp(-gx²/std²) which is normalized to (0, 1].
    """
    asset = env.scene[asset_cfg.name]
    roll_error = torch.square(asset.data.projected_gravity_b[:, 1])
    return torch.exp(-roll_error / std**2)


def flat_orientation_pitch_exp(
    env: "ManagerBasedRLEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat orientation along pitch axis using exponential kernel.
    
    Uses the y-component of projected gravity vector in body frame.
    When robot is flat, projected_gravity_b[:, 1] = 0.
    Returns exp(-gy²/std²) which is normalized to (0, 1].
    """
    asset = env.scene[asset_cfg.name]
    pitch_error = torch.square(asset.data.projected_gravity_b[:, 0])
    return torch.exp(-pitch_error / std**2)



def action_rate_l2_joint(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the rate of change of actions for specific joints using L2 squared kernel.
    
    This function allows penalizing action rate for a subset of joints specified
    by asset_cfg.joint_ids, unlike the default action_rate_l2 which penalizes all actions.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying which joints to include via joint_names.
        
    Returns:
        Sum of squared action differences for the specified joints.
    """
    # Get the joint indices for the specified joints
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    # If joint_ids is specified, only use those indices
    if asset_cfg.joint_ids is not None:
        action = action[:, asset_cfg.joint_ids]
        prev_action = prev_action[:, asset_cfg.joint_ids]
    
    return torch.sum(torch.square(action - prev_action), dim=1)


def action_l2_joint(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize action magnitude for selected action dimensions."""
    action = env.action_manager.action
    if asset_cfg.joint_ids is not None:
        action = action[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(action), dim=1)


def virtual_leg_angle_diff_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the difference in angles between the left and right virtual legs.
    
    Uses the virtual leg model to compute leg angles from hip and knee joint positions.
    Returns the squared difference between left and right leg angles.
    
    Returns:
        Tensor [num_envs]: Squared angle difference for each environment.
    """
    # Get virtual leg angles: [num_envs, 2] = [left_angle, right_angle]
    angles = virtual_leg_angle(env)
    angle_diff = angles[:, 0] - angles[:, 1]
    return torch.square(angle_diff)

def virtual_leg_length_diff_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the difference in lengths between the left and right virtual legs.
    
    Uses the virtual leg model to compute leg lengths from hip and knee joint positions.
    Returns the squared difference between left and right leg lengths.
    
    Returns:
        Tensor [num_envs]: Squared length difference for each environment.
    """
    # Get virtual leg lengths: [num_envs, 2] = [left_length, right_length]
    lengths = virtual_leg_length(env)
    length_diff = lengths[:, 0] - lengths[:, 1]
    return torch.square(length_diff)


def virtual_leg_angle_deviation_l2(env: "ManagerBasedRLEnv", target_angle: float = 0.0) -> torch.Tensor:
    """Penalize deviation of virtual leg angles from target angle.
    
    The virtual leg angle is 0 when pointing straight down (vertical),
    positive when swinging forward. This reward encourages the robot
    to keep both virtual legs at the specified target angle.
    
    Args:
        env: The environment instance.
        target_angle: Target angle in radians. Default 0.0 (vertical).
                      Positive = forward lean, Negative = backward lean.
    
    Returns:
        Tensor [num_envs]: Sum of squared angle deviations for both legs.
    """
    # Get virtual leg angles: [num_envs, 2] = [left_angle, right_angle]
    angles = virtual_leg_angle(env)
    # Sum of squared deviations from target for both legs
    return torch.sum(torch.square(angles - target_angle), dim=1)


def constant_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant reward of 1.0 for each environment step."""
    return torch.ones(env.num_envs, device=env.device)


def wheel_velocity_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["wheel.*"])
) -> torch.Tensor:
    """Penalize wheel angular velocities using L2 squared kernel.
    
    Returns sum of squared wheel angular velocities for use with negative weight.
    This encourages the robot to minimize wheel spinning when not needed.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying wheel joints. Default matches "wheel.*".
        
    Returns:
        Tensor [num_envs]: Sum of squared angular velocities for wheel joints.
    """
    asset = env.scene[asset_cfg.name]
    # Get wheel joint velocities using joint_ids from asset_cfg
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(wheel_vel), dim=1)


def wheel_velocity_difference_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["wheel.*"])
) -> torch.Tensor:
    """Penalize left/right wheel speed mismatch to discourage yaw drift."""
    asset = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if wheel_vel.shape[1] < 2:
        return torch.zeros(env.num_envs, dtype=asset.data.joint_vel.dtype, device=env.device)
    return torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])


def forward_back_position_exp(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the scheduled forward/back root x position."""
    asset = env.scene[asset_cfg.name]
    target_x, _, _ = _forward_back_sequence(env, body_length, forward_duration, hold_duration, return_duration)
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    error = target_x - root_xy[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def forward_back_position_l2(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize scheduled x position tracking error without exponential saturation."""
    asset = env.scene[asset_cfg.name]
    target_x, _, _ = _forward_back_sequence(env, body_length, forward_duration, hold_duration, return_duration)
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    error = target_x - root_xy[:, 0]
    return torch.square(error)


def forward_back_velocity_exp(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the scheduled x velocity."""
    asset = env.scene[asset_cfg.name]
    _, target_vel_x, _ = _forward_back_sequence(env, body_length, forward_duration, hold_duration, return_duration)
    error = target_vel_x - asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def forward_back_world_velocity_exp(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking scheduled x velocity in the reset/world frame."""
    asset = env.scene[asset_cfg.name]
    _, target_vel_x, _ = _forward_back_sequence(env, body_length, forward_duration, hold_duration, return_duration)
    error = target_vel_x - asset.data.root_lin_vel_w[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def forward_back_velocity_schedule_exp(
    env: "ManagerBasedRLEnv",
    speed: float,
    segment_duration: float,
    settle_duration: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the pure scheduled forward/back world x velocity."""
    asset = env.scene[asset_cfg.name]
    target_vel_x, _ = _forward_back_velocity_schedule(env, speed, segment_duration, settle_duration)
    error = target_vel_x - asset.data.root_lin_vel_w[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def forward_back_velocity_schedule_l2(
    env: "ManagerBasedRLEnv",
    speed: float,
    segment_duration: float,
    settle_duration: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pure scheduled forward/back world x velocity tracking error."""
    asset = env.scene[asset_cfg.name]
    target_vel_x, _ = _forward_back_velocity_schedule(env, speed, segment_duration, settle_duration)
    error = target_vel_x - asset.data.root_lin_vel_w[:, 0]
    return torch.square(error)


def lateral_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame lateral velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 1])


def lateral_position_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift from the reset origin."""
    asset = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    return torch.square(root_xy[:, 1])


def yaw_drift_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw rotation away from the reset heading."""
    asset = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    heading_w = quat_apply(asset.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    yaw = torch.atan2(heading_w[:, 1], heading_w[:, 0])
    upright = torch.clamp(-gravity_b[:, 2], min=0.0, max=1.0)
    return torch.square(yaw) * upright


def final_stability_exp(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward being back near the origin and still after the return phase."""
    asset = env.scene[asset_cfg.name]
    elapsed = env.episode_length_buf.to(dtype=torch.float32, device=env.device) * env.step_dt
    final_start = forward_duration + hold_duration + return_duration
    final_mask = elapsed >= final_start
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    pos_error = torch.linalg.norm(root_xy[:, :2], dim=1)
    speed_error = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    score = torch.exp(-torch.square(pos_error) / std**2) * torch.exp(-torch.square(speed_error) / std**2)
    return torch.where(final_mask, score, torch.ones_like(score))



