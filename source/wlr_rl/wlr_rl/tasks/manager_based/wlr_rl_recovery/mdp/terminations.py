# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _upright_success_mask(
    env: "ManagerBasedRLEnv",
    ang_limit: float,
    ang_vel_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_error = torch.abs(torch.arccos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)))
    ang_ok = ang_error < ang_limit
    base_ang_vel_ok = torch.linalg.vector_norm(asset.data.root_ang_vel_b[:, :3], dim=1) < ang_vel_limit
    return ang_ok & base_ang_vel_ok


def upright_success_termination(
    env: "ManagerBasedRLEnv",
    ang_limit: float = 0.05,
    ang_vel_limit: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stage-1 success: body has recovered to an upright, low-angular-velocity state."""
    return _upright_success_mask(env, ang_limit=ang_limit, ang_vel_limit=ang_vel_limit, asset_cfg=asset_cfg)


def handoff_ready_success_termination(
    env: "ManagerBasedRLEnv",
    ang_limit: float = 0.05,
    ang_vel_limit: float = 0.5,
    leg_length_limit: float = 0.2,
    leg_angle_limit: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stage-2 success: upright plus handoff-ready leg geometry and low residual leg motion."""
    upright_ok = _upright_success_mask(env, ang_limit=ang_limit, ang_vel_limit=ang_vel_limit, asset_cfg=asset_cfg)

    from wlr_rl.tasks.manager_based.wlr_rl.mdp.observations import virtual_leg_angle, virtual_leg_length

    leg_angles = virtual_leg_angle(env)
    leg_lengths = virtual_leg_length(env)
    leg_angle_ok = torch.all(torch.abs(leg_angles) < leg_angle_limit, dim=1)
    leg_length_ok = torch.all(torch.abs(leg_lengths) < leg_length_limit, dim=1)

    return upright_ok & leg_angle_ok & leg_length_ok
