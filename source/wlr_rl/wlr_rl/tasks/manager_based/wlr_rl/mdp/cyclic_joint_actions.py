from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.actions import joint_actions


class CyclicRelativeJointPositionAction(joint_actions.JointAction):
    """Relative position action for continuous joints using wrapped phase error."""

    cfg: "CyclicRelativeJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._offset = 0.0
        self._cycle_period = float(cfg.cycle_period)
        self._half_cycle = 0.5 * self._cycle_period
        self._phase_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._phase_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def _wrap_to_period(self, values: torch.Tensor) -> torch.Tensor:
        return torch.remainder(values + self._half_cycle, self._cycle_period) - self._half_cycle

    def apply_actions(self):
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._phase_targets[:] = self._wrap_to_period(self._phase_targets + self.processed_actions)
        phase_error = self._wrap_to_period(self._phase_targets - current_pos)
        targets = current_pos + phase_error
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)