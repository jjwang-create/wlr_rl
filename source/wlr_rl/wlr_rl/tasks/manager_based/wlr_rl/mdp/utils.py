# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Virtual Leg Model for Wheel-Legged Infantry Robot.

Maps hip and knee joint angles to virtual leg parameters (angle and length).

Coordinate System:
    - Virtual leg angle: 0 when pointing straight down (vertical), 
      positive when swinging forward
    - Virtual leg length: distance from hip joint to wheel center

Kinematic Model:
    hip_joint ---(L1)--- knee_joint ---(L2)--- wheel_center
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============== Robot Parameters ==============
# Link lengths from the wheel_legged_robot URDF.
L1 = 0.210  # hip link length (hip_joint to knee_joint)
L2 = 0.248  # knee link length (knee_joint to wheel_center)

# Virtual leg length limits
LEG_LENGTH_MIN = abs(L1 - L2) + 0.01  # ~0.048m
LEG_LENGTH_MAX = L1 + L2 - 0.01       # ~0.448m


# ============== Core Conversion Functions ==============

def _get_joint_positions(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get hip and knee joint positions for both legs from the environment.
    
    Returns:
        hip_l, hip_r, knee_l, knee_r: Joint angles [num_envs]
    """
    asset: Articulation = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    
    # Get joint indices (cached after first call)
    hip_l_idx = asset.find_joints("hip_l_joint")[0][0]
    hip_r_idx = asset.find_joints("hip_r_joint")[0][0]
    knee_l_idx = asset.find_joints("knee_l_joint")[0][0]
    knee_r_idx = asset.find_joints("knee_r_joint")[0][0]
    
    return (
        joint_pos[:, hip_l_idx],
        joint_pos[:, hip_r_idx],
        joint_pos[:, knee_l_idx],
        joint_pos[:, knee_r_idx],
    )


def _get_joint_velocities(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get hip and knee joint velocities for both legs from the environment.
    
    Returns:
        hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel: Joint velocities [num_envs]
    """
    asset: Articulation = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    
    hip_l_idx = asset.find_joints("hip_l_joint")[0][0]
    hip_r_idx = asset.find_joints("hip_r_joint")[0][0]
    knee_l_idx = asset.find_joints("knee_l_joint")[0][0]
    knee_r_idx = asset.find_joints("knee_r_joint")[0][0]
    
    return (
        joint_vel[:, hip_l_idx],
        joint_vel[:, hip_r_idx],
        joint_vel[:, knee_l_idx],
        joint_vel[:, knee_r_idx],
    )


def _joint_to_virtual_leg(
    hip_angle: torch.Tensor, 
    knee_angle: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hip and knee joint angles to virtual leg angle and length.
    
    URDF Convention:
    - hip_joint has rpy="0 1.5708 0" (starts horizontal forward)
    - knee_joint has rpy="0 3.1416 0" (folds back)
    
    Args:
        hip_angle: Hip joint angles [num_envs]
        knee_angle: Knee joint angles [num_envs]
        
    Returns:
        leg_angle: Virtual leg angle from vertical (down=0, forward=positive) [num_envs]
        leg_length: Virtual leg length [num_envs]
    """
    # Upper leg angle from vertical (positive = forward swing)
    # In URDF: hip starts at pi/2 (horizontal), so angle from vertical = pi/2 - hip_angle
    alpha = -hip_angle
    
    # Lower leg angle from vertical
    # Due to knee joint convention (rpy pi rotation)
    gamma = alpha - knee_angle + torch.pi
    
    # Forward kinematics: wheel position relative to hip
    x1 = L1 * torch.sin(alpha)           # upper leg endpoint x
    z1 = -L1 * torch.cos(alpha)          # upper leg endpoint z (down is negative)
    
    x2 = x1 + L2 * torch.sin(gamma)      # wheel x
    z2 = z1 - L2 * torch.cos(gamma)      # wheel z
    
    # Virtual leg length
    leg_length = torch.sqrt(x2**2 + z2**2)
    
    # Virtual leg angle (from vertical down, positive forward)
    leg_angle = torch.atan2(x2, -z2)
    
    return leg_angle, leg_length


# ============== Debug Functions ==============

# Debug configuration - set which variables to observe
DEBUG_CONFIG = {
    "velocity_tracking": True,  # Command vs actual velocity comparison + orientation
    "gravity": True,            # Projected gravity vs IMU measured gravity
    "virtual_leg": False,       # Virtual leg angle and length
    "joint_torque": False,      # Joint applied torques
    "link_wrench": False,       # Link incoming joint wrench (force/torque from parent)
}


def _debug_velocity_tracking(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Compare commanded velocity vs actual velocity, and show orientation."""
    import math
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    
    asset: Articulation = env.scene["robot"]
    
    # Get actual velocities in robot's yaw-aligned frame
    root_quat = asset.data.root_quat_w[env_id:env_id+1]
    root_lin_vel_w = asset.data.root_lin_vel_w[env_id:env_id+1]
    root_ang_vel_w = asset.data.root_ang_vel_w[env_id]
    
    # Transform linear velocity to yaw-aligned frame (gravity-aligned, yaw from robot)
    vel_yaw = quat_apply_inverse(yaw_quat(root_quat), root_lin_vel_w)[0]
    vx_actual = vel_yaw[0].item()
    vy_actual = vel_yaw[1].item()
    wz_actual = root_ang_vel_w[2].item()
    
    # Get commanded velocities from observation (works for both normal and keyboard mode)
    try:
        # Find velocity_commands term in policy observation group
        obs_mgr = env.observation_manager
        term_names = obs_mgr._group_obs_term_names.get("policy", [])
        term_cfgs = obs_mgr._group_obs_term_cfgs.get("policy", [])
        
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0
        for name, cfg in zip(term_names, term_cfgs):
            if name == "velocity_commands":
                cmd = cfg.func(env, **cfg.params)[env_id]
                vx_cmd = cmd[0].item()
                vy_cmd = cmd[1].item()
                wz_cmd = cmd[2].item()
                break
    except Exception:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0
    
    # Get orientation (roll, pitch) from projected gravity
    proj_grav = asset.data.projected_gravity_b[env_id]
    pitch = torch.atan2(proj_grav[0], -proj_grav[2]).item()
    roll = torch.atan2(proj_grav[1], -proj_grav[2]).item()
    
    lines = [
        "--- Velocity Tracking ---",
        f"  Vx: cmd={vx_cmd:+.2f}, actual={vx_actual:+.2f}, err={vx_actual - vx_cmd:+.2f} m/s",
        f"  Vy: cmd={vy_cmd:+.2f}, actual={vy_actual:+.2f}, err={vy_actual - vy_cmd:+.2f} m/s",
        f"  Wz: cmd={wz_cmd:+.2f}, actual={wz_actual:+.2f}, err={wz_actual - wz_cmd:+.2f} rad/s",
        f"--- Orientation ---",
        f"  Roll: {math.degrees(roll):+.1f}°, Pitch: {math.degrees(pitch):+.1f}°",
    ]
    return "\n".join(lines)


def _debug_gravity(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Compare projected_gravity vs imu_measured_gravity.
    
    Directly calls observation functions to verify their correctness.
    
    Shows:
    - projected_gravity: True gravity direction from quaternion (ideal)
    - imu_measured_gravity: Simulated IMU measurement (corrupted by body acceleration)
    - difference: How much body acceleration affects the IMU reading
    """
    import math
    from isaaclab.envs.mdp import projected_gravity as projected_gravity_func
    from .observations import imu_measured_gravity as imu_measured_gravity_func
    
    # Call observation functions directly (returns [num_envs, 3])
    projected_gravity_all = projected_gravity_func(env)  # [num_envs, 3]
    imu_gravity_all = imu_measured_gravity_func(env)     # [num_envs, 3]
    
    # Extract single env
    projected_gravity = projected_gravity_all[env_id]  # [3]
    imu_gravity = imu_gravity_all[env_id]              # [3]
    
    # Compute difference
    diff = imu_gravity - projected_gravity
    
    # Extract values
    pg_x, pg_y, pg_z = projected_gravity[0].item(), projected_gravity[1].item(), projected_gravity[2].item()
    imu_x, imu_y, imu_z = imu_gravity[0].item(), imu_gravity[1].item(), imu_gravity[2].item()
    diff_x, diff_y, diff_z = diff[0].item(), diff[1].item(), diff[2].item()
    
    # Compute norms
    pg_norm = torch.norm(projected_gravity).item()
    imu_norm = torch.norm(imu_gravity).item()
    diff_norm = torch.norm(diff).item()
    
    # Compute roll/pitch from both sources
    pg_pitch = math.atan2(pg_x, -pg_z)
    pg_roll = math.atan2(pg_y, -pg_z)
    imu_pitch = math.atan2(imu_x, -imu_z)
    imu_roll = math.atan2(imu_y, -imu_z)
    
    lines = [
        "--- Gravity Comparison (from observation functions) ---",
        f"  Projected Gravity:    [{pg_x:+.4f}, {pg_y:+.4f}, {pg_z:+.4f}] (norm={pg_norm:.4f})",
        f"  IMU Measured Gravity: [{imu_x:+.4f}, {imu_y:+.4f}, {imu_z:+.4f}] (norm={imu_norm:.4f})",
        f"  Difference (IMU-PG):  [{diff_x:+.4f}, {diff_y:+.4f}, {diff_z:+.4f}] (norm={diff_norm:.4f})",
        f"  --- Orientation from Gravity ---",
        f"  From PG:  Roll={math.degrees(pg_roll):+.2f}°, Pitch={math.degrees(pg_pitch):+.2f}°",
        f"  From IMU: Roll={math.degrees(imu_roll):+.2f}°, Pitch={math.degrees(imu_pitch):+.2f}°",
        f"  Error:    Roll={math.degrees(imu_roll - pg_roll):+.2f}°, Pitch={math.degrees(imu_pitch - pg_pitch):+.2f}°",
    ]
    return "\n".join(lines)


def _debug_virtual_leg(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get virtual leg debug info."""
    import math
    hip_l, hip_r, knee_l, knee_r = _get_joint_positions(env)
    hip_l_vel, hip_r_vel, knee_l_vel, knee_r_vel = _get_joint_velocities(env)
    
    angle_l, length_l = _joint_to_virtual_leg(hip_l[env_id:env_id+1], knee_l[env_id:env_id+1])
    angle_r, length_r = _joint_to_virtual_leg(hip_r[env_id:env_id+1], knee_r[env_id:env_id+1])
    
    # Numerical Jacobian for velocities
    eps = 1e-4
    
    # Left leg velocity
    angle_l_dh, length_l_dh = _joint_to_virtual_leg(hip_l[env_id:env_id+1] + eps, knee_l[env_id:env_id+1])
    angle_l_dk, length_l_dk = _joint_to_virtual_leg(hip_l[env_id:env_id+1], knee_l[env_id:env_id+1] + eps)
    angle_vel_l = ((angle_l_dh - angle_l) / eps * hip_l_vel[env_id] + (angle_l_dk - angle_l) / eps * knee_l_vel[env_id]).item()
    length_vel_l = ((length_l_dh - length_l) / eps * hip_l_vel[env_id] + (length_l_dk - length_l) / eps * knee_l_vel[env_id]).item()
    
    # Right leg velocity
    angle_r_dh, length_r_dh = _joint_to_virtual_leg(hip_r[env_id:env_id+1] + eps, knee_r[env_id:env_id+1])
    angle_r_dk, length_r_dk = _joint_to_virtual_leg(hip_r[env_id:env_id+1], knee_r[env_id:env_id+1] + eps)
    angle_vel_r = ((angle_r_dh - angle_r) / eps * hip_r_vel[env_id] + (angle_r_dk - angle_r) / eps * knee_r_vel[env_id]).item()
    length_vel_r = ((length_r_dh - length_r) / eps * hip_r_vel[env_id] + (length_r_dk - length_r) / eps * knee_r_vel[env_id]).item()
    
    lines = [
        "--- Virtual Leg ---",
        f"  Angle L: {angle_l[0].item():+.3f} rad ({math.degrees(angle_l[0].item()):+.1f}°), vel={angle_vel_l:+.2f} rad/s",
        f"  Angle R: {angle_r[0].item():+.3f} rad ({math.degrees(angle_r[0].item()):+.1f}°), vel={angle_vel_r:+.2f} rad/s",
        f"  Length L: {length_l[0].item():.3f} m, vel={length_vel_l:+.3f} m/s",
        f"  Length R: {length_r[0].item():.3f} m, vel={length_vel_r:+.3f} m/s",
    ]
    return "\n".join(lines)



def _debug_joint_torque(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get joint torque debug info for all joints.
    
    Shows:
    - applied_torque: torque actually applied to simulation
    - computed_torque: torque computed by actuator model (before saturation/clipping)
    - For wheels: additional analysis of torque sources
    """
    asset: Articulation = env.scene["robot"]
    applied_torque = asset.data.applied_torque
    computed_torque = asset.data.computed_torque
    joint_names = asset.joint_names
    joint_vel = asset.data.joint_vel
    
    lines = ["--- Joint Torques ---"]
    for i, name in enumerate(joint_names):
        applied = applied_torque[env_id, i].item()
        computed = computed_torque[env_id, i].item()
        vel = joint_vel[env_id, i].item()
        lines.append(f"  {name}: applied={applied:+.3f} Nm, computed={computed:+.3f} Nm, vel={vel:+.2f} rad/s")
    
    
    return "\n".join(lines)


def _debug_link_wrench(env: "ManagerBasedRLEnv", env_id: int) -> str:
    """Get link incoming joint wrench debug info.
    
    Shows the 6-D wrench (force and torque) applied to each link by the parent joint.
    Format: [Fx, Fy, Fz, Tx, Ty, Tz] in body frame.
    
    Particularly useful for:
    - Analyzing torque around Y axis (pitch direction) on wheel links
    - Understanding reaction forces through the kinematic chain
    """
    asset: Articulation = env.scene["robot"]
    
    # Get link incoming joint wrench: shape (num_envs, num_bodies, 6)
    # Format: [Fx, Fy, Fz, Tx, Ty, Tz] in parent body frame
    wrench = asset.data.body_incoming_joint_wrench_b[env_id]  # (num_bodies, 6)
    body_names = asset.body_names
    
    lines = ["--- Link Incoming Joint Wrench (body frame) ---"]
    lines.append("  Format: [Fx, Fy, Fz | Tx, Ty, Tz] in Nm")
    
    for i, name in enumerate(body_names):
        fx, fy, fz = wrench[i, 0].item(), wrench[i, 1].item(), wrench[i, 2].item()
        tx, ty, tz = wrench[i, 3].item(), wrench[i, 4].item(), wrench[i, 5].item()
        
        # Highlight Y-axis torque (pitch torque)
        lines.append(
            f"  {name:12s}: F=[{fx:+6.2f}, {fy:+6.2f}, {fz:+6.2f}] "
            f"T=[{tx:+6.2f}, {ty:+6.2f}★, {tz:+6.2f}]"
        )
    
    
    return "\n".join(lines)


def _debug_policy_io(
    env: "ManagerBasedRLEnv",
    env_id: int,
    obs: torch.Tensor | dict | None,
    actions: torch.Tensor | None,
) -> str:
    """Get policy input/output debug info.
    
    Shows:
    - obs: Observation tensor fed into the policy network
    - actions: Action tensor output from the policy network
    
    Args:
        env: Environment instance
        env_id: Which environment to debug
        obs: Observation tensor, dict, or TensorDict from env.get_observations()
             If dict/TensorDict, expects "policy" key containing the tensor [num_envs, obs_dim]
        actions: Action tensor from policy(obs) [num_envs, action_dim]
    """
    lines = ["--- Policy I/O ---"]
    
    if obs is not None:
        # Handle TensorDict or dict observations (common in Isaac Lab)
        # TensorDict behaves like a dict, so we can use the same handling
        obs_tensor = None
        if hasattr(obs, "get") or isinstance(obs, dict):
            # It's a dict-like object (dict or TensorDict)
            obs_tensor = obs.get("policy", None)
            if obs_tensor is None:
                keys = list(obs.keys()) if hasattr(obs, "keys") else "unknown"
                lines.append(f"  Observation: dict-like with keys {keys}, 'policy' key not found")
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs
        
        if obs_tensor is not None and isinstance(obs_tensor, torch.Tensor) and obs_tensor.dim() >= 1:
            obs_single = obs_tensor[env_id]  # [obs_dim]
            obs_dim = obs_single.shape[0] if obs_single.dim() > 0 else 1
            lines.append(f"  Observation (dim={obs_dim}):")
            
            # Print observation values in a compact format (8 values per line)
            values_per_line = 8
            if obs_single.dim() > 0:
                for i in range(0, obs_dim, values_per_line):
                    end_idx = min(i + values_per_line, obs_dim)
                    vals = [f"{obs_single[j].item():+.4f}" for j in range(i, end_idx)]
                    indices = f"[{i:2d}-{end_idx-1:2d}]"
                    lines.append(f"    {indices}: {', '.join(vals)}")
            else:
                lines.append(f"    [{obs_single.item():+.4f}]")
        elif obs_tensor is not None:
            lines.append(f"  Observation: type={type(obs_tensor)}, shape={getattr(obs_tensor, 'shape', 'N/A')}")
    else:
        lines.append("  Observation: Not provided")
    
    if actions is not None:
        if isinstance(actions, torch.Tensor) and actions.dim() >= 1:
            actions_single = actions[env_id]  # [action_dim]
            action_dim = actions_single.shape[0] if actions_single.dim() > 0 else 1
            lines.append(f"  Actions (dim={action_dim}):")
            
            # Print action values
            if actions_single.dim() > 0:
                vals = [f"{actions_single[j].item():+.4f}" for j in range(action_dim)]
                lines.append(f"    [{', '.join(vals)}]")
            else:
                lines.append(f"    [{actions_single.item():+.4f}]")
        else:
            lines.append(f"  Actions: type={type(actions)}, shape={getattr(actions, 'shape', 'N/A')}")
    else:
        lines.append("  Actions: Not provided")
    
    return "\n".join(lines)




def debug_print(
    env: "ManagerBasedRLEnv",
    env_id: int = 0,
    print_interval: int = 100,
    config: dict | None = None,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
) -> None:
    """
    Main debug print function. Call this in play.py to observe variables.
    
    Args:
        env: Environment instance
        env_id: Which environment to debug (default 0)
        print_interval: How often to print (in simulation steps)
        config: Optional dict to override DEBUG_CONFIG. Keys:
            - "virtual_leg": bool
            - "base_state": bool
            - "joint_state": bool
            - "command": bool
            - "policy_io": bool  # Display policy observation input and action output
        obs: Observation tensor for policy_io debug [num_envs, obs_dim]
        actions: Action tensor for policy_io debug [num_envs, action_dim]
    
    Usage in play.py:
        from wlr_rl.tasks.manager_based.wlr_rl.mdp.utils import debug_print
        debug_print(env.unwrapped, env_id=0, print_interval=100, obs=obs, actions=actions)
    """
    if env.common_step_counter % print_interval != 0:
        return
    
    cfg = config if config is not None else DEBUG_CONFIG
    
    print(f"\n{'='*50}")
    print(f"DEBUG (Step {env.common_step_counter}, Env {env_id})")
    print('='*50)
    
    if cfg.get("velocity_tracking", False):
        print(_debug_velocity_tracking(env, env_id))
    
    if cfg.get("gravity", False):
        print(_debug_gravity(env, env_id))
    
    if cfg.get("virtual_leg", False):
        print(_debug_virtual_leg(env, env_id))
    
    if cfg.get("joint_torque", False):
        print(_debug_joint_torque(env, env_id))
    
    if cfg.get("link_wrench", False):
        print(_debug_link_wrench(env, env_id))
    
    if cfg.get("policy_io", False):
        print(_debug_policy_io(env, env_id, obs, actions))

    print('='*50)


# Legacy function for backwards compatibility
def debug_virtual_leg_kinematics(env: "ManagerBasedRLEnv", env_id: int = 0, print_interval: int = 100) -> None:
    """
    Legacy debug function. Use debug_print() instead.
    """
    debug_print(env, env_id, print_interval, config={"virtual_leg": True, "joint_state": True})


# ============== Play Logger ==============

class PlayLogger:
    """Record play-time telemetry to an Excel file for offline analysis.

    Logged columns (per step):
        step, time, cmd_vx, cmd_wz, actual_vx, actual_wz, pitch_deg

    Usage in play.py::

        from wlr_rl.tasks.manager_based.wlr_rl.mdp.utils import PlayLogger
        logger = PlayLogger(enabled=True)   # toggle with enabled=False
        # inside simulation loop:
        logger.log(env.unwrapped, env_id=0)
        # after loop ends:
        logger.save()
    """

    def __init__(self, enabled: bool = True, log_dir: str | None = None):
        """
        Args:
            enabled: If False, all operations are no-ops.
            log_dir: Directory to write xlsx files. Defaults to ``<wlr_rl_root>/play_log``.
        """
        self.enabled = enabled
        self._rows: list[dict] = []
        self._saved = False

        if not enabled:
            return

        import os, datetime, atexit, signal
        if log_dir is None:
            # wlr_rl project root: utils.py → mdp/ → wlr_rl/ → manager_based/ → tasks/ → wlr_rl/ → wlr_rl/ → source/ → wlr_rl/
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."]*7), "play_log"))
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._filepath = os.path.join(log_dir, f"{timestamp}.xlsx")
        print(f"[PlayLogger] Will save to: {self._filepath}")

        # Auto-save: register both atexit and SIGINT handler for robustness
        atexit.register(self.save)

        def _sigint_handler(signum, frame):
            self.save()
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, _sigint_handler)

    # ---- data collection ----

    def log(self, env: "ManagerBasedRLEnv", env_id: int = 0) -> None:
        """Collect one row of telemetry from the environment."""
        if not self.enabled:
            return

        # Print once to confirm logging is active
        if len(self._rows) == 0:
            print("[PlayLogger] First log() call — recording started.")

        import math
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        asset: Articulation = env.scene["robot"]
        step = env.common_step_counter

        # --- actual velocities (yaw-aligned frame) ---
        root_quat = asset.data.root_quat_w[env_id : env_id + 1]
        root_lin_vel_w = asset.data.root_lin_vel_w[env_id : env_id + 1]
        root_ang_vel_w = asset.data.root_ang_vel_w[env_id]
        vel_yaw = quat_apply_inverse(yaw_quat(root_quat), root_lin_vel_w)[0]
        actual_vx = vel_yaw[0].item()
        actual_wz = root_ang_vel_w[2].item()

        # --- pitch from projected gravity ---
        proj_grav = asset.data.projected_gravity_b[env_id]
        pitch_rad = torch.atan2(proj_grav[0], -proj_grav[2]).item()
        pitch_deg = math.degrees(pitch_rad)

        # --- commanded velocities ---
        cmd_vx, cmd_wz = 0.0, 0.0
        try:
            obs_mgr = env.observation_manager
            term_names = obs_mgr._group_obs_term_names.get("policy", [])
            term_cfgs = obs_mgr._group_obs_term_cfgs.get("policy", [])
            for name, cfg in zip(term_names, term_cfgs):
                if name == "velocity_commands":
                    cmd = cfg.func(env, **cfg.params)[env_id]
                    cmd_vx = cmd[0].item()
                    cmd_wz = cmd[2].item()
                    break
        except Exception:
            pass

        self._rows.append({
            "step": step,
            "time_s": step * env.step_dt,
            "cmd_vx": round(cmd_vx, 4),
            "cmd_wz": round(cmd_wz, 4),
            "actual_vx": round(actual_vx, 4),
            "actual_wz": round(actual_wz, 4),
            "pitch_deg": round(pitch_deg, 2),
        })

    # ---- persistence ----

    def save(self) -> None:
        """Write collected data to an Excel (.xlsx) file."""
        if not self.enabled or not self._rows or self._saved:
            return
        self._saved = True

        import pandas as pd
        df = pd.DataFrame(self._rows)
        df.to_excel(self._filepath, index=False, engine="openpyxl")
        print(f"[PlayLogger] Saved {len(self._rows)} rows → {self._filepath}")




