# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .observations import _env_origin_xy, _forward_back_sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_point_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    body_point: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when a local point on the root body drops below a world-frame height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    point_b = torch.tensor(body_point, dtype=asset.data.root_pos_w.dtype, device=env.device).repeat(env.num_envs, 1)
    point_w = asset.data.root_pos_w + quat_apply(asset.data.root_quat_w, point_b)
    return point_w[:, 2] < minimum_height


def roll_pitch_exceeds(
    env: ManagerBasedRLEnv,
    max_roll: float,
    max_pitch: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll or pitch exceeds separate limits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
    pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
    return torch.logical_or(torch.abs(roll) > max_roll, torch.abs(pitch) > max_pitch)


def forward_back_tracking_error_exceeds(
    env: ManagerBasedRLEnv,
    body_length: float,
    forward_duration: float,
    hold_duration: float,
    return_duration: float,
    max_x_error: float,
    max_y_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the root drifts too far from the scheduled 1D forward/back path."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_x, _, _ = _forward_back_sequence(env, body_length, forward_duration, hold_duration, return_duration)
    root_xy = asset.data.root_pos_w[:, :2] - _env_origin_xy(env)
    x_error = torch.abs(target_x - root_xy[:, 0])
    y_error = torch.abs(root_xy[:, 1])
    return torch.logical_or(x_error > max_x_error, y_error > max_y_error)


def yaw_exceeds(
    env: ManagerBasedRLEnv,
    max_yaw: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root heading drifts too far from the reset/world x-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    heading_w = quat_apply(
        asset.data.root_quat_w,
        torch.tensor([1.0, 0.0, 0.0], dtype=asset.data.root_quat_w.dtype, device=env.device).repeat(env.num_envs, 1),
    )
    yaw = torch.atan2(heading_w[:, 1], heading_w[:, 0])
    return torch.abs(yaw) > max_yaw


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
