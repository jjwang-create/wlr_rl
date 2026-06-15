# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def freeze_joint_effort_after_reset(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    hold_time_s: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Force joint effort limits to zero for a short window after reset.

    During ``hold_time_s`` after reset, selected joints cannot apply torque.
    After the window, default effort limits are restored.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids.numel() == 0:
        return

    cache_name = "_recovery_effort_limit_cache"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, {})
    effort_limit_cache = getattr(env, cache_name)

    cache_key = asset_cfg.name
    if cache_key not in effort_limit_cache:
        effort_limit_cache[cache_key] = {
            "actuator": {name: actuator.effort_limit.clone() for name, actuator in asset.actuators.items()},
        }

    elapsed_s = env.episode_length_buf[env_ids].to(dtype=torch.float32) * env.step_dt
    freeze_env_ids = env_ids[elapsed_s < hold_time_s]
    active_env_ids = env_ids[elapsed_s >= hold_time_s]

    if freeze_env_ids.numel() > 0:
        for actuator in asset.actuators.values():
            actuator.effort_limit[freeze_env_ids] = 0.0

    if active_env_ids.numel() > 0:
        for actuator_name, actuator in asset.actuators.items():
            actuator.effort_limit[active_env_ids] = effort_limit_cache[cache_key]["actuator"][actuator_name][active_env_ids]


def reset_root_state_from_weighted_points(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    points: list[tuple[float, float, float]],
    weights: list[float],
    position_std: tuple[float, float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    use_env_origins: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root state around weighted anchor points on the terrain.

    For each environment being reset, one anchor point is sampled according to ``weights``.
    The robot root position is then sampled from a Gaussian neighborhood around that anchor,
    while orientation and velocity perturbations follow the provided ranges.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids.numel() == 0:
        return

    if len(points) == 0:
        raise ValueError("'points' must contain at least one spawn anchor.")
    if len(points) != len(weights):
        raise ValueError("'points' and 'weights' must have the same length.")

    root_states = asset.data.default_root_state[env_ids].clone()

    anchor_points = torch.tensor(points, dtype=root_states.dtype, device=asset.device)
    anchor_weights = torch.tensor(weights, dtype=root_states.dtype, device=asset.device)
    anchor_weights = anchor_weights / anchor_weights.sum()
    anchor_indices = torch.multinomial(anchor_weights, num_samples=len(env_ids), replacement=True)
    anchors = anchor_points[anchor_indices]

    std = torch.tensor(position_std, dtype=root_states.dtype, device=asset.device)
    position_noise = torch.randn((len(env_ids), 3), device=asset.device, dtype=root_states.dtype) * std

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(range_list, device=asset.device, dtype=root_states.dtype)
    angle_noise = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 3), device=asset.device
    )

    positions = anchors + position_noise
    if use_env_origins:
        positions = positions + env.scene.env_origins[env_ids]
    orientations_delta = math_utils.quat_from_euler_xyz(angle_noise[:, 0], angle_noise[:, 1], angle_noise[:, 2])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(range_list, device=asset.device, dtype=root_states.dtype)
    velocity_noise = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_noise

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
