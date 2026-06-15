from __future__ import annotations

import torch

from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class SpringDelayedPDActuator(DelayedPDActuator):
    cfg: "SpringDelayedPDActuatorCfg"

    def _compute_spring_torque(self, joint_pos: torch.Tensor) -> torch.Tensor:
        theta = joint_pos
        theta_min = self.cfg.theta_min
        theta_max = self.cfg.theta_max

        # User-requested normalization: (theta - theta_min) / (theta_max / theta_min)
        eps = 1e-6
        denom = theta_max / max(abs(theta_min), eps)
        x = (theta - theta_min) / max(abs(denom), eps)

        tau_spring = (
            self.cfg.spring_poly_c0
            + self.cfg.spring_poly_c1 * x
            + self.cfg.spring_poly_c2 * x**2
            + self.cfg.spring_poly_c3 * x**3
            + self.cfg.spring_poly_c4 * x**4
        )
        return torch.clamp(tau_spring, min=-self.cfg.spring_torque_limit, max=self.cfg.spring_torque_limit)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        spring_torque = self._compute_spring_torque(joint_pos)

        # Motor command compensates spring torque to match deployment behavior.
        # With motor limit ±T and spring torque tau_s, total torque range becomes [tau_s-T, tau_s+T].
        motor_cmd = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts - spring_torque
        motor_applied = self._clip_effort(motor_cmd)
        total_torque = motor_applied + spring_torque

        self.computed_effort = motor_cmd
        self.applied_effort = motor_applied
        control_action.joint_efforts = total_torque
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


@configclass
class SpringDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = SpringDelayedPDActuator

    theta_min: float = 0.416
    theta_max: float = 2.153
    spring_poly_c0: float = 0.0
    spring_poly_c1: float = 0.0
    spring_poly_c2: float = 0.0
    spring_poly_c3: float = 0.0
    spring_poly_c4: float = 0.0
    spring_torque_limit: float = 25.0
