# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from wlr_rl.tasks.manager_based.wlr_rl.mdp.observations import virtual_leg_angle, virtual_leg_length

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _upright_success_condition(
    env: "ManagerBasedRLEnv",
    ang_limit: float = 0.2,
    ang_vel_limit: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_error = torch.abs(torch.arccos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)))
    ang_ok = ang_error < ang_limit
    base_ang_vel_ok = torch.linalg.vector_norm(asset.data.root_ang_vel_b[:, :3], dim=1) < ang_vel_limit
    return ang_ok & base_ang_vel_ok


def _recovery_vector_b(asset: Articulation) -> torch.Tensor:
    """Return a gated unit recovery vector in base frame from gravity-direction cross product.

    The vector is set to zero when the projected gravity z-component is negative to avoid
    oscillatory reward near the upright region.
    """
    actual_gravity_b = asset.data.projected_gravity_b
    target_gravity_b = torch.zeros_like(actual_gravity_b)
    target_gravity_b[:, 2] = -1.0

    recovery_vector_b = torch.cross(target_gravity_b, actual_gravity_b, dim=1)
    recovery_norm = torch.linalg.vector_norm(recovery_vector_b, dim=1, keepdim=True)
    recovery_unit_b = recovery_vector_b / recovery_norm.clamp_min(1.0e-6)

    gate = ((actual_gravity_b[:, 2] >= 0.0) & (recovery_norm[:, 0] > 1.0e-6)).unsqueeze(1)
    return torch.where(gate, recovery_unit_b, torch.zeros_like(recovery_unit_b))


def recovery_torque(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["hip.*", "knee.*", "wheel.*"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_ang_limit: float | None = None,
    upright_ang_vel_limit: float | None = None,
) -> torch.Tensor:
    """Reward contact resultant moment about the base_link origin along the recovery vector."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    base_body_id = asset.find_bodies("base_link")[0][0]
    base_pos_w = asset.data.body_pos_w[:, base_body_id].unsqueeze(1)

    body_pos_w = asset.data.body_pos_w[:, sensor_cfg.body_ids]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    lever_arm_w = body_pos_w - base_pos_w
    total_torque_w = torch.sum(torch.cross(lever_arm_w, net_forces_w, dim=-1), dim=1)

    total_torque_b = quat_apply_inverse(asset.data.root_quat_w, total_torque_w)
    recovery_vector_b = _recovery_vector_b(asset)
    reward = torch.sum(total_torque_b * recovery_vector_b, dim=1)
    if upright_ang_limit is not None and upright_ang_vel_limit is not None:
        pre_upright_mask = (~_upright_success_condition(env, ang_limit=upright_ang_limit, ang_vel_limit=upright_ang_vel_limit, asset_cfg=asset_cfg)).float()
        reward = reward * pre_upright_mask
    return reward


def desired_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    leg_body_name_groups: tuple[tuple[str, ...], tuple[str, ...]] = (
        ("hip_l_link", "knee_l_link", "wheel_l"),
        ("hip_r_link", "knee_r_link", "wheel_r"),
    ),
    threshold: float = 1.0,
    upright_ang_limit: float | None = None,
    upright_ang_vel_limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward desired contacts while counting at most one contact per leg group."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    leg_contact_count = torch.zeros(env.num_envs, device=env.device)

    for leg_body_names in leg_body_name_groups:
        leg_body_ids = contact_sensor.find_bodies(list(leg_body_names))[0]
        leg_contact_forces = contact_sensor.data.net_forces_w_history[:, :, leg_body_ids, :]
        leg_has_contact = torch.max(torch.norm(leg_contact_forces, dim=-1), dim=1)[0] > threshold
        leg_contact_count += torch.any(leg_has_contact, dim=1).float()

    if upright_ang_limit is not None and upright_ang_vel_limit is not None:
        pre_upright_mask = (~_upright_success_condition(env, ang_limit=upright_ang_limit, ang_vel_limit=upright_ang_vel_limit, asset_cfg=asset_cfg)).float()
        leg_contact_count = leg_contact_count * pre_upright_mask

    return leg_contact_count


def undesired_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    leg_body_name_groups: tuple[tuple[str, ...], tuple[str, ...]] = (
        ("hip_l_link", "knee_l_link"),
        ("hip_r_link", "knee_r_link"),
    ),
    threshold: float = 1.0,
    upright_ang_limit: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg-ground contact only after the body has basically recovered upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    leg_contact_count = torch.zeros(env.num_envs, device=env.device)

    for leg_body_names in leg_body_name_groups:
        leg_body_ids = contact_sensor.find_bodies(list(leg_body_names))[0]
        leg_contact_forces = contact_sensor.data.net_forces_w_history[:, :, leg_body_ids, :]
        leg_has_contact = torch.max(torch.norm(leg_contact_forces, dim=-1), dim=1)[0] > threshold
        leg_contact_count += torch.any(leg_has_contact, dim=1).float()

    ang_error = torch.abs(torch.arccos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)))
    post_upright_mask = (ang_error < upright_ang_limit).float()
    return leg_contact_count * post_upright_mask


def impact_force_rate_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base_link"]),
    deadzone: float = 100.0,
) -> torch.Tensor:
    """Penalize contact-force change rate on selected links with a small deadzone.

    Uses the most recent two samples from the contact-force history buffer and applies an
    L2-squared penalty only to the amount exceeding ``deadzone``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w_history = contact_sensor.data.net_forces_w_history
    if net_forces_w_history.shape[1] < 2:
        raise RuntimeError("impact_force_rate_l2 requires contact_forces history_length >= 2.")

    dt_sensor = contact_sensor.cfg.update_period if contact_sensor.cfg.update_period > 0.0 else env.physics_dt
    force_delta = net_forces_w_history[:, 0, sensor_cfg.body_ids, :] - net_forces_w_history[:, 1, sensor_cfg.body_ids, :]
    force_rate = torch.linalg.vector_norm(force_delta, dim=-1) / dt_sensor
    excess_force_rate = torch.clamp(force_rate - deadzone / dt_sensor, min=0.0)
    return torch.sum(torch.square(excess_force_rate), dim=1)


def projected_gravity_error_x(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward upright orientation using projected gravity in body frame."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])


def projected_gravity_error_y(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward upright orientation using projected gravity in body frame."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def projected_gravity_error_z(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward upright orientation using projected gravity in body frame."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 2] + 1.0)


def recovery_stability_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize high angular velocity after robot starts rising."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :3]), dim=1)


def joint_vel_l2(
    env: "ManagerBasedRLEnv",
    ang_limit: float = 0.2,
    ang_vel_limit: float = 5.0,
    vel_threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["wheel.*"]),
) -> torch.Tensor:
    """Penalize selected joint velocities above a threshold."""
    asset = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    excess_vel = torch.clamp(joint_vel - vel_threshold, min=0.0)
    vel_penalty = torch.sum(torch.square(excess_vel), dim=1)
    return vel_penalty


def action_edge_l2(
    env: "ManagerBasedRLEnv",
    action_name: str | None = None,
    action_names: Sequence[str] | None = None,
    soft_bound: float = 0.75,
    hard_bound: float = 1.0,
) -> torch.Tensor:
    """Penalize normalized actions that stay close to the clip boundary for one or more action terms."""
    if hard_bound <= soft_bound:
        raise ValueError("hard_bound must be greater than soft_bound.")

    names: list[str] = []
    if action_name is not None:
        names.append(action_name)
    if action_names is not None:
        names.extend(action_names)
    if not names:
        raise ValueError("action_edge_l2 requires action_name or action_names.")

    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in names:
        action_term = env.action_manager.get_term(name)
        action = action_term.raw_actions
        excess = torch.clamp(torch.abs(action) - soft_bound, min=0.0)
        normalized_excess = excess / max(hard_bound - soft_bound, 1.0e-6)
        penalty += torch.sum(torch.square(normalized_excess), dim=1)
    return penalty


class LatchedSuccessTerm(ManagerTermBase):
    """Latch a termination-style condition once it becomes true within an episode."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._latched[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        term_name: str = "upright_success",
        ang_limit: float = 0.2,
        ang_vel_limit: float = 5.0,
        use_termination_term: bool = False,
    ) -> torch.Tensor:
        if use_termination_term:
            current = env.termination_manager.get_term(term_name)
        else:
            current = _upright_success_condition(env, ang_limit=ang_limit, ang_vel_limit=ang_vel_limit)
        self._latched |= current
        return torch.zeros(env.num_envs, device=env.device)

    @property
    def latched(self) -> torch.Tensor:
        return self._latched


def virtual_leg_length_exp(
    env: "ManagerBasedRLEnv",
    gate_term_name: str | None = None,
    gate_reward_name: str | None = None,
    target_length: float = 0.0,
    std: float = 0.2,
) -> torch.Tensor:
    lengths = virtual_leg_length(env)
    reward = torch.exp(-torch.sum(torch.square(lengths - target_length), dim=1) / max(std**2, 1.0e-6))
    if gate_term_name is None and gate_reward_name is None:
        return reward
    if gate_reward_name is not None:
        gate_term = env.reward_manager._term_cfgs[env.reward_manager._term_names.index(gate_reward_name)].func
        gate_value = gate_term.latched.float()
    else:
        gate_value = env.termination_manager.get_term(gate_term_name).float()
    return reward * gate_value


def virtual_leg_angle_abs_exp(
    env: "ManagerBasedRLEnv",
    gate_term_name: str | None = None,
    gate_reward_name: str | None = None,
    leg_length_limit: float | None = None,
    target_angle: float = 0.0,
    std: float = 0.2,
) -> torch.Tensor:
    angles = virtual_leg_angle(env)
    lengths = virtual_leg_length(env)
    reward = torch.exp(-torch.sum(torch.square(torch.abs(angles) - target_angle), dim=1) / max(std**2, 1.0e-6))
    if gate_term_name is None and gate_reward_name is None:
        if leg_length_limit is not None:
            length_gate = torch.all(torch.abs(lengths) < leg_length_limit, dim=1).float()
            reward = reward * length_gate
        return reward
    if gate_reward_name is not None:
        gate_term = env.reward_manager._term_cfgs[env.reward_manager._term_names.index(gate_reward_name)].func
        gate_value = gate_term.latched.float()
    else:
        gate_value = env.termination_manager.get_term(gate_term_name).float()
    if leg_length_limit is not None:
        length_gate = torch.all(torch.abs(lengths) < leg_length_limit, dim=1).float()
        gate_value = gate_value * length_gate
    return reward * gate_value


def recovery_success_bonus(
    env: "ManagerBasedRLEnv",
    term_name: str = "handoff_ready_success",
) -> torch.Tensor:
    """One-shot terminal bonus when the recovery-success termination term fires.

    This matches the semantics of terminal penalty/reward terms (e.g., timeout penalty):
    reward is issued only on the termination step instead of accumulating every step
    while the success condition remains true.
    """
    return env.termination_manager.get_term(term_name).float()


def time_out_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot penalty on the exact step when an episode ends due to time-out."""
    return env.termination_manager.time_outs.float()


def recovery_success_time_bonus(
    env: "ManagerBasedRLEnv",
    term_name: str = "handoff_ready_success",
    freeze_time_s: float = 1.0,
    time_threshold_s: float = 3.0,
    time_decay_s: float = 3.0,
) -> torch.Tensor:
    """Extra success bonus that decays exponentially with recovery time after a grace threshold.

    Recovery time is measured from the end of the freeze window. Successes reached before
    ``time_threshold_s`` receive the full time bonus; after that the bonus decays exponentially.
    """
    success = env.termination_manager.get_term(term_name).float()
    elapsed_since_unfreeze = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt - freeze_time_s
    elapsed_since_unfreeze = torch.clamp(elapsed_since_unfreeze, min=0.0)
    late_time = torch.clamp(elapsed_since_unfreeze - time_threshold_s, min=0.0)
    time_factor = torch.exp(-late_time / max(time_decay_s, 1.0e-6))
    return success * time_factor


class success_once_bonus(ManagerTermBase):
    """Issue a one-shot bonus the first time a termination-style success term becomes true in an episode."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._issued = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._issued[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        term_name: str,
        ang_limit: float = 0.2,
        ang_vel_limit: float = 5.0,
        use_termination_term: bool = True,
    ) -> torch.Tensor:
        if use_termination_term:
            current = env.termination_manager.get_term(term_name)
        else:
            current = _upright_success_condition(env, ang_limit=ang_limit, ang_vel_limit=ang_vel_limit)
        new_success = current & (~self._issued)
        self._issued |= current
        return new_success.float()


class success_time_bonus_once(ManagerTermBase):
    """Issue a one-shot time bonus when a termination-style success term first becomes true."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._issued = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._issued[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        term_name: str,
        freeze_time_s: float = 1.0,
        time_threshold_s: float = 3.0,
        time_decay_s: float = 3.0,
        ang_limit: float = 0.2,
        ang_vel_limit: float = 5.0,
        use_termination_term: bool = True,
    ) -> torch.Tensor:
        if use_termination_term:
            current = env.termination_manager.get_term(term_name)
        else:
            current = _upright_success_condition(env, ang_limit=ang_limit, ang_vel_limit=ang_vel_limit)
        new_success = current & (~self._issued)
        self._issued |= current

        elapsed_since_unfreeze = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt - freeze_time_s
        elapsed_since_unfreeze = torch.clamp(elapsed_since_unfreeze, min=0.0)
        late_time = torch.clamp(elapsed_since_unfreeze - time_threshold_s, min=0.0)
        time_factor = torch.exp(-late_time / max(time_decay_s, 1.0e-6))
        return new_success.float() * time_factor


