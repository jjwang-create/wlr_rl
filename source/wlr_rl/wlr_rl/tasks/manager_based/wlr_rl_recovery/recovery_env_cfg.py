# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import wlr_rl.tasks.manager_based.wlr_rl_recovery.mdp as mdp

from wlr_rl.assets.wheel_legged_robots import WHEEL_LEGGED_ROBOT_CFG

FREEZE_HOLD_TIME_S = 1.0

RECOVERY_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    color_scheme="none",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration for the recovery task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RECOVERY_FLAT_TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2, track_air_time=False)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the recovery task."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the recovery task."""

    hip_pos = mdp.CyclicRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["hip.*"], scale=0.1
    )
    knee_pos = mdp.JointPositionActionCfg(
        asset_name="robot",joint_names=["knee.*"],scale=1.5,use_default_offset=True
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=["wheel.*"], scale=75.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the recovery task."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_pos_sin_cos = ObsTerm(
            func=mdp.joint_pos_sin_cos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["knee.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_pos_sin_cos = ObsTerm(
            func=mdp.joint_pos_sin_cos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["knee.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Event specifications for the recovery task."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.4),
            "dynamic_friction_range": (0.2, 0.3),
            "restitution_range": (0.1, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi), "roll": (-math.pi, math.pi), "pitch": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-3, 3),
            "velocity_range": (0.0, 0.0),
        },
    )

    freeze_joint_effort_after_reset = EventTerm(
        func=mdp.freeze_joint_effort_after_reset,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "hold_time_s": FREEZE_HOLD_TIME_S,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the recovery task."""
# termination rewards
    timeout_penalty = RewTerm(
        func=mdp.time_out_penalty,
        weight=-20000.0,
    )
    upright_success_gate = RewTerm(
        func=mdp.LatchedSuccessTerm,
        weight=1.0,
        params={"term_name": "upright_success", "ang_limit": 0.2, "ang_vel_limit": 5.0, "use_termination_term": False},
    )
    upright_success_bonus = RewTerm(
        func=mdp.success_once_bonus,
        weight=10000.0,
        params={"term_name": "upright_success", "ang_limit": 0.2, "ang_vel_limit": 5.0, "use_termination_term": False},
    )
    upright_success_time_bonus = RewTerm(
        func=mdp.success_time_bonus_once,
        weight=10000.0,
        params={
            "freeze_time_s": FREEZE_HOLD_TIME_S,
            "time_threshold_s": 2.0,
            "time_decay_s": 4.0,
            "term_name": "upright_success",
            "ang_limit": 0.2,
            "ang_vel_limit": 5.0,
            "use_termination_term": False,
        },
    )
    handoff_ready_success_bonus = RewTerm(
        func=mdp.recovery_success_bonus,
        weight=10000.0,
        params={"term_name": "handoff_ready_success"},
    )
    handoff_ready_success_time_bonus = RewTerm(
        func=mdp.recovery_success_time_bonus,
        weight=10000.0,
        params={
            "freeze_time_s": FREEZE_HOLD_TIME_S,
            "time_threshold_s": 4.0,
            "time_decay_s": 3.0,
            "term_name": "handoff_ready_success",
        },
    )

# action rate and edge penalties
    leg_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*", "knee.*"])},
    )
    wheel_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])},
    )
    hip_action_edge_l2 = RewTerm(
        func=mdp.action_edge_l2,
        weight=-1.0,
        params={"action_name": "hip_pos", "soft_bound": 0.6},
    )
    knee_action_edge_l2 = RewTerm(
        func=mdp.action_edge_l2,
        weight=-1.0,
        params={"action_name": "knee_pos", "soft_bound": 0.6},
    )
    other_action_edge_l2 = RewTerm(
        func=mdp.action_edge_l2,
        weight=-1.0,
        params={"action_names": ["wheel_vel"], "soft_bound": 0.6},
    )

# joint torque penalty
    hip_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*"])})
    knee_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["knee.*"])})
    wheel_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])})

# joint and body velocity penalties
    hip_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1, params={"vel_threshold": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["hip.*"])})
    knee_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1, params={"vel_threshold": 5.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["knee.*"])})
    wheel_vel_l2 = RewTerm(func=mdp.joint_vel_l2,weight=-0.1,params={"vel_threshold": 10.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"])})
    recovery_stability = RewTerm(func=mdp.recovery_stability_l2, weight=-1.0)

# recovery rewards
    projected_gravity_error_z = RewTerm(func=mdp.projected_gravity_error_z, weight=-5.0)
    virtual_leg_length_exp = RewTerm(
        func=mdp.virtual_leg_length_exp,
        weight=50.0,
        params={"gate_reward_name": "upright_success_gate", "target_length": 0.1, "std": 0.3},
    )
    virtual_leg_angle_abs_exp = RewTerm(
        func=mdp.virtual_leg_angle_abs_exp,
        weight=50.0,
        params={"gate_reward_name": "upright_success_gate", "leg_length_limit": 0.2, "target_angle": 0.0, "std": 3.0},
    )
    recovery_torque = RewTerm(
        func=mdp.recovery_torque,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["hip.*", "knee.*", "wheel.*"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_ang_limit": 0.2,
            "upright_ang_vel_limit": 5.0,
        },
    )
    desired_contact = RewTerm(
        func=mdp.desired_contact,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "leg_body_name_groups": [("hip_l_link", "knee_l_link", "wheel_l"), ("hip_r_link", "knee_r_link", "wheel_r")],
            "threshold": 1.0,
            "upright_ang_limit": 0.2,
            "upright_ang_vel_limit": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contact,
        weight=-25.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "leg_body_name_groups": [("hip_l_link", "knee_l_link", "wheel_l"), ("hip_r_link", "knee_r_link", "wheel_r")],
            "threshold": 1.0,
            "upright_ang_limit": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

# contact force penalties
    # impact_force_rate_l2_leg = RewTerm(
    #     func=mdp.impact_force_rate_l2,
    #     weight=-2.0e-9,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["hip.*", "knee.*", "wheel.*"]),
    #         "deadzone": 50.0,
    #     },
    # )
    # impact_force_rate_l2_base = RewTerm(
    #     func=mdp.impact_force_rate_l2,
    #     weight=-1e-9,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
    #         "deadzone": 100.0,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the recovery task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    handoff_ready_success = DoneTerm(
        func=mdp.handoff_ready_success_termination,
        params={
            "ang_limit": 0.3,
            "ang_vel_limit": 5.0,
            "leg_length_limit": 0.15,
            "leg_angle_limit": 0.2,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the recovery task."""

    # curriculum can be extended later for staged reset generation
    terrain_levels = None


@configclass
class RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone recovery environment configuration."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 20.0
        # NOTE: reward-freeze gating depends on a local framework patch in
        # IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py.
        # RewardManager.compute() reads this attribute and zeros all reward terms while
        # episode_length_buf * step_dt < reward_freeze_time_s.
        self.reward_freeze_time_s = FREEZE_HOLD_TIME_S
        self.scene.robot = WHEEL_LEGGED_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt


@configclass
class RecoveryEnvCfg_PLAY(RecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 3
        self.episode_length_s = 10.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
