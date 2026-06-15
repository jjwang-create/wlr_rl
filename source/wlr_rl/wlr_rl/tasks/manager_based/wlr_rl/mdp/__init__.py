# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401,F403
except Exception:
    # Fallback to generic env MDP (has a subset of functions)
    from isaaclab.envs.mdp import *  # noqa: F401,F403

from .curriculums import *  # noqa: F401, F403
from .cyclic_joint_actions import *  # noqa: F401, F403
from .cyclic_joint_actions_cfg import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
