# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Wheel-Legged Infantry Robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import (
    _get_joint_positions,
    _get_joint_velocities,
    _joint_to_virtual_leg,
)

# Gravitational acceleration magnitude
GRAVITY_MAGNITUDE = 9.81


def _env_origin_xy(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env.scene, "env_origins"):
        return env.scene.env_origins[:, :2]
    return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)


def _smoothstep(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.clamp(u, 0.0, 1.0)
    value = u * u * (3.0 - 2.0 * u)
    derivative = 6.0 * u * (1.0 - u)
    return value, derivative


def _forward_back_sequence(
    env: "ManagerBasedRLEnv",
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    elapsed = env.episode_length_buf.to(dtype=torch.float32, device=env.device) * env.step_dt
    forward_end = forward_duration
    hold_end = forward_end + hold_duration
    return_end = hold_end + return_duration

    target_x = torch.full((env.num_envs,), body_length, dtype=torch.float32, device=env.device)
    target_vel_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    forward_mask = elapsed < forward_end
    forward_u = elapsed / max(forward_duration, 1.0e-6)
    forward_s, forward_ds = _smoothstep(forward_u)
    target_x = torch.where(forward_mask, body_length * forward_s, target_x)
    target_vel_x = torch.where(forward_mask, body_length * forward_ds / max(forward_duration, 1.0e-6), target_vel_x)

    return_mask = torch.logical_and(elapsed >= hold_end, elapsed < return_end)
    return_u = (elapsed - hold_end) / max(return_duration, 1.0e-6)
    return_s, return_ds = _smoothstep(return_u)
    target_x = torch.where(return_mask, body_length * (1.0 - return_s), target_x)
    target_vel_x = torch.where(return_mask, -body_length * return_ds / max(return_duration, 1.0e-6), target_vel_x)

    final_mask = elapsed >= return_end
    target_x = torch.where(final_mask, torch.zeros_like(target_x), target_x)
    target_vel_x = torch.where(final_mask, torch.zeros_like(target_vel_x), target_vel_x)

    phase = torch.clamp(elapsed / max(return_end + hold_duration, 1.0e-6), 0.0, 1.0)
    return target_x, target_vel_x, phase


def _forward_back_velocity_schedule(
    env: "ManagerBasedRLEnv",
    speed: float,
    segment_duration: float,
    settle_duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a +speed, -speed, +speed, 0 schedule with two direction changes."""
    elapsed = env.episode_length_buf.to(dtype=torch.float32, device=env.device) * env.step_dt
    first_end = segment_duration
    second_end = 2.0 * segment_duration
    third_end = 3.0 * segment_duration
    total_duration = third_end + settle_duration

    target_vel_x = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    target_vel_x = torch.where(elapsed < first_end, torch.full_like(target_vel_x, speed), target_vel_x)
    target_vel_x = torch.where(
        torch.logical_and(elapsed >= first_end, elapsed < second_end),
        torch.full_like(target_vel_x, -speed),
        target_vel_x,
    )
    target_vel_x = torch.where(
        torch.logical_and(elapsed >= second_end, elapsed < third_end),
        torch.full_like(target_vel_x, speed),
        target_vel_x,
    )
    phase = torch.clamp(elapsed / max(total_duration, 1.0e-6), 0.0, 1.0)
    return target_vel_x, phase


def forward_back_sequence_command(
    env: "ManagerBasedRLEnv",
    body_length: float = 0.5,
    forward_duration: float = 2.5,
    hold_duration: float = 1.5,
    return_duration: float = 2.5,
) -> torch.Tensor:
    """Return a 3D command for a deterministic forward-hold-back-hold standing sequence.

    The command keeps the same dimensionality as the original velocity command observation so standing
    checkpoints can be resumed without changing the actor architecture.
    """
    asset: Articulation = env.scene["robot"]
    target_x, target_vel_x, phase = _forward_back_sequence(
        env, body_length, forward_duration, hold_duration, return_duration
    )
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    x_error = target_x - root_xy[:, 0]
    return torch.stack([x_error, target_vel_x, phase], dim=1)


def forward_back_sequence_velocity_command(
    env: "ManagerBasedRLEnv",
    body_length: float = 0.5,
    forward_duration: float = 4.0,
    hold_duration: float = 1.5,
    return_duration: float = 4.0,
    position_gain: float = 1.2,
    max_command_velocity: float = 0.35,
) -> torch.Tensor:
    """Return a velocity-like command for the forward/back sequence.

    Slot 0 keeps the original locomotion meaning: desired forward velocity.
    Slot 1 stays zero, and slot 2 carries phase progress.  This is easier to
    warm-start from standing/velocity policies than putting position error in
    the lateral velocity slot.
    """
    asset: Articulation = env.scene["robot"]
    target_x, target_vel_x, phase = _forward_back_sequence(
        env, body_length, forward_duration, hold_duration, return_duration
    )
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    x_error = target_x - root_xy[:, 0]
    command_vel_x = torch.clamp(
        target_vel_x + position_gain * x_error,
        min=-max_command_velocity,
        max=max_command_velocity,
    )
    command_vel_y = torch.zeros_like(command_vel_x)
    return torch.stack([command_vel_x, command_vel_y, phase], dim=1)


def forward_back_velocity_schedule_command(
    env: "ManagerBasedRLEnv",
    speed: float = 0.08,
    segment_duration: float = 2.5,
    settle_duration: float = 1.5,
) -> torch.Tensor:
    """Return a pure forward/back velocity command with unchanged 3D command shape."""
    target_vel_x, phase = _forward_back_velocity_schedule(env, speed, segment_duration, settle_duration)
    return torch.stack([target_vel_x, torch.zeros_like(target_vel_x), phase], dim=1)


# ============== IMU Observation Functions ==============

def imu_measured_gravity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Simulated IMU gravity direction measurement.
    
    Real IMU accelerometers measure specific force (a_measured = a_body - g).
    When normalized, this gives an estimate of gravity direction that is 
    corrupted by body acceleration.
    
    This function simulates this by:
        imu_gravity = projected_gravity - body_lin_acc_b / |g|
    
    Returns:
        Tensor [num_envs, 3]: Simulated gravity direction in body frame (not normalized)
    """
    asset: Articulation = env.scene["robot"]
    
    # Get true projected gravity (pure geometry, from quaternion)
    projected_gravity = asset.data.projected_gravity_b  # [num_envs, 3]
    
    # Get root body linear acceleration in world frame
    # body_lin_acc_w shape: [num_envs, num_bodies, 3], index 0 is root
    root_lin_acc_w = asset.data.body_lin_acc_w[:, 0, :]  # [num_envs, 3]
    
    # Transform acceleration to body frame
    root_quat_w = asset.data.root_link_quat_w  # [num_envs, 4]
    root_lin_acc_b = quat_apply_inverse(root_quat_w, root_lin_acc_w)  # [num_envs, 3]
    
    # Simulate IMU measurement: gravity direction is corrupted by body acceleration
    # a_measured = a_body - g, so measured_gravity_direction ≈ -a_measured/|g| = g/|g| - a_body/|g|
    # Since projected_gravity is already g_direction (unit vector pointing in gravity direction),
    # the IMU would measure: imu_gravity = projected_gravity - body_acc / |g|
    imu_gravity = projected_gravity - root_lin_acc_b / GRAVITY_MAGNITUDE
    
    return imu_gravity


# ============== Virtual Leg Observation Functions ==============

def virtual_leg_angle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg angles for both legs.
    
    Returns:
        Tensor [num_envs, 2]: [left_leg_angle, right_leg_angle]
        Angle is 0 when vertical (down), positive when forward.
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    angle_l, _ = _joint_to_virtual_leg(hip_l, knee_l)
    angle_r, _ = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([angle_l, angle_r], dim=1)


def virtual_leg_length(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg lengths for both legs.
    
    Returns:
        Tensor [num_envs, 2]: [left_leg_length, right_leg_length]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    _, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    _, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([length_l, length_r], dim=1)


def virtual_leg_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get full virtual leg state for both legs.
    
    Returns:
        Tensor [num_envs, 4]: [angle_l, angle_r, length_l, length_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    
    angle_l, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    angle_r, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    
    return torch.stack([angle_l, angle_r, length_l, length_r], dim=1)


def virtual_leg_angle_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg angle velocities using Jacobian.
    
    Returns:
        Tensor [num_envs, 2]: [angle_vel_l, angle_vel_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    # Numerical Jacobian (finite difference)
    eps = 1e-4
    
    # Left leg
    angle_l, _ = _joint_to_virtual_leg(hip_l, knee_l)
    angle_l_dh, _ = _joint_to_virtual_leg(hip_l + eps, knee_l)
    angle_l_dk, _ = _joint_to_virtual_leg(hip_l, knee_l + eps)
    
    d_angle_d_hip = (angle_l_dh - angle_l) / eps
    d_angle_d_knee = (angle_l_dk - angle_l) / eps
    angle_vel_l = d_angle_d_hip * hip_l_vel + d_angle_d_knee * knee_l_vel
    
    # Right leg
    angle_r, _ = _joint_to_virtual_leg(hip_r, knee_r)
    angle_r_dh, _ = _joint_to_virtual_leg(hip_r + eps, knee_r)
    angle_r_dk, _ = _joint_to_virtual_leg(hip_r, knee_r + eps)
    
    d_angle_d_hip = (angle_r_dh - angle_r) / eps
    d_angle_d_knee = (angle_r_dk - angle_r) / eps
    angle_vel_r = d_angle_d_hip * hip_r_vel + d_angle_d_knee * knee_r_vel
    
    return torch.stack([angle_vel_l, angle_vel_r], dim=1)


def virtual_leg_length_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get virtual leg length velocities using Jacobian.
    
    Returns:
        Tensor [num_envs, 2]: [length_vel_l, length_vel_r]
    """
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    eps = 1e-4
    
    # Left leg
    _, length_l = _joint_to_virtual_leg(hip_l, knee_l)
    _, length_l_dh = _joint_to_virtual_leg(hip_l + eps, knee_l)
    _, length_l_dk = _joint_to_virtual_leg(hip_l, knee_l + eps)
    
    d_length_d_hip = (length_l_dh - length_l) / eps
    d_length_d_knee = (length_l_dk - length_l) / eps
    length_vel_l = d_length_d_hip * hip_l_vel + d_length_d_knee * knee_l_vel
    
    # Right leg
    _, length_r = _joint_to_virtual_leg(hip_r, knee_r)
    _, length_r_dh = _joint_to_virtual_leg(hip_r + eps, knee_r)
    _, length_r_dk = _joint_to_virtual_leg(hip_r, knee_r + eps)
    
    d_length_d_hip = (length_r_dh - length_r) / eps
    d_length_d_knee = (length_r_dk - length_r) / eps
    length_vel_r = d_length_d_hip * hip_r_vel + d_length_d_knee * knee_r_vel
    
    return torch.stack([length_vel_l, length_vel_r], dim=1)
