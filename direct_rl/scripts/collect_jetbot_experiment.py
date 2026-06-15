# Copyright (c) 2026
"""Collect a short JetBot trajectory from the direct_rl Isaac Lab environment."""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect a finite JetBot rollout and save it as an npz file.")
parser.add_argument("--task", type=str, default="Template-Direct-Rl-Direct-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=480, help="Number of environment steps to record.")
parser.add_argument("--output", type=str, default="outputs/jetbot_experiment.npz", help="Output npz path.")
parser.add_argument("--seed", type=int, default=7, help="Environment/action seed.")
parser.add_argument("--base_speed", type=float, default=7.0, help="Nominal forward wheel velocity target.")
parser.add_argument("--steer_gain", type=float, default=5.0, help="Wheel differential gain for heading correction.")
parser.add_argument("--max_steer", type=float, default=5.5, help="Maximum wheel differential steering command.")
parser.add_argument("--wheel_radius", type=float, default=0.055, help="Wheel radius used for rollout visualization.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import direct_rl.tasks  # noqa: F401


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw from Isaac Lab quaternions in w, x, y, z order."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _target_yaw(step: int, total_steps: int) -> float:
    """Piecewise target direction schedule that makes target changes visible."""
    schedule = (0.0, 0.9, -0.65, 1.35, 0.25)
    segment = min(len(schedule) - 1, int(step / max(1, total_steps) * len(schedule)))
    return schedule[segment]


def _set_command(env, target_yaw: float) -> None:
    command = torch.tensor(
        [[math.cos(target_yaw), math.sin(target_yaw), 0.0]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    env.unwrapped.commands[:] = command
    env.unwrapped.yaws[:] = target_yaw


def _make_action(env, target_yaw: float) -> tuple[torch.Tensor, float]:
    current_yaw = _yaw_from_quat_wxyz(env.unwrapped.robot.data.root_quat_w[:1])[0]
    heading_error = _wrap_to_pi(torch.tensor(target_yaw, dtype=torch.float32, device=env.unwrapped.device) - current_yaw)
    steer = torch.clamp(args_cli.steer_gain * heading_error, -args_cli.max_steer, args_cli.max_steer)
    left = args_cli.base_speed - steer
    right = args_cli.base_speed + steer
    action = torch.stack((left, right), dim=0).reshape(1, 2)
    return action, float(heading_error.detach().cpu().item())


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args_cli.seed)

    positions = []
    yaws = []
    lin_vel_world = []
    lin_vel_body = []
    actions = []
    rewards = []
    dones = []
    observations = []
    commands = []
    target_yaws = []
    heading_errors = []
    left_wheel_angles = []
    right_wheel_angles = []
    left_wheel_angle = 0.0
    right_wheel_angle = 0.0
    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            target_yaw = _target_yaw(step, args_cli.steps)
            _set_command(env, target_yaw)
            action, heading_error = _make_action(env, target_yaw)
            obs, reward, terminated, truncated, _ = env.step(action)
            robot_data = env.unwrapped.robot.data

            positions.append(robot_data.root_pos_w[0].detach().cpu().numpy())
            yaws.append(_yaw_from_quat_wxyz(robot_data.root_quat_w[:1]).detach().cpu().numpy()[0])
            lin_vel_world.append(robot_data.root_com_lin_vel_w[0].detach().cpu().numpy())
            lin_vel_body.append(robot_data.root_com_lin_vel_b[0].detach().cpu().numpy())
            actions.append(action[0].detach().cpu().numpy())
            left_wheel_angle += float(action[0, 0].detach().cpu().item()) * step_dt
            right_wheel_angle += float(action[0, 1].detach().cpu().item()) * step_dt
            left_wheel_angles.append(left_wheel_angle)
            right_wheel_angles.append(right_wheel_angle)
            rewards.append(float(reward[0].detach().cpu().item()))
            dones.append(bool((terminated[0] | truncated[0]).detach().cpu().item()))
            observations.append(obs["policy"][0].detach().cpu().numpy())
            commands.append(env.unwrapped.commands[0].detach().cpu().numpy())
            target_yaws.append(target_yaw)
            heading_errors.append(heading_error)

    positions_np = np.asarray(positions, dtype=np.float32)
    yaws_np = np.asarray(yaws, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    times_np = np.arange(args_cli.steps, dtype=np.float32) * float(env_cfg.sim.dt * env_cfg.decimation)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    np.savez(
        args_cli.output,
        task=args_cli.task,
        seed=np.asarray(args_cli.seed, dtype=np.int32),
        dt=np.asarray(float(env_cfg.sim.dt * env_cfg.decimation), dtype=np.float32),
        time=times_np,
        position=positions_np,
        yaw=yaws_np,
        lin_vel_world=np.asarray(lin_vel_world, dtype=np.float32),
        lin_vel_body=np.asarray(lin_vel_body, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        wheel_radius=np.asarray(args_cli.wheel_radius, dtype=np.float32),
        left_wheel_angle=np.asarray(left_wheel_angles, dtype=np.float32),
        right_wheel_angle=np.asarray(right_wheel_angles, dtype=np.float32),
        reward=rewards_np,
        done=np.asarray(dones, dtype=np.bool_),
        observation=np.asarray(observations, dtype=np.float32),
        command=np.asarray(commands, dtype=np.float32),
        target_yaw=np.asarray(target_yaws, dtype=np.float32),
        heading_error=np.asarray(heading_errors, dtype=np.float32),
        total_reward=np.asarray(float(np.sum(rewards_np)), dtype=np.float32),
        final_position=positions_np[-1],
        final_yaw=np.asarray(float(yaws_np[-1]), dtype=np.float32),
    )

    print(f"[INFO] Saved rollout: {args_cli.output}")
    print(f"[INFO] steps={args_cli.steps} total_reward={float(np.sum(rewards_np)):.6f}")
    print(f"[INFO] final_position={positions_np[-1].tolist()} final_yaw={float(yaws_np[-1]):.6f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
