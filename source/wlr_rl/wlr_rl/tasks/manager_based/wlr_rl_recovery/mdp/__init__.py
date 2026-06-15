# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reuse all default terms from the locomotion task and extend with recovery-specific ones.
from wlr_rl.tasks.manager_based.wlr_rl.mdp import *  # noqa: F401,F403

from .observations import *  # noqa: F401,F403
from .events import *  # noqa: F401,F403
from .rewards import *  # noqa: F401,F403
from .terminations import *  # noqa: F401,F403
