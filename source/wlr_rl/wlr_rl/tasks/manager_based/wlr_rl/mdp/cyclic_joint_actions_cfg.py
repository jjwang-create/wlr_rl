from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from . import cyclic_joint_actions


@configclass
class CyclicRelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for cyclic relative joint position action term."""

    class_type: type[ActionTerm] = cyclic_joint_actions.CyclicRelativeJointPositionAction

    cycle_period: float = 2.0 * 3.141592653589793
    """Period of the continuous joint phase, defaults to 2π."""