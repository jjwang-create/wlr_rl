"""Render the direct_rl JetBot task with Isaac Lab/Isaac Sim and write an MP4."""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Render a finite JetBot rollout with Isaac Lab.")
parser.add_argument("--task", type=str, default="Template-Direct-Rl-Direct-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=240, help="Number of environment steps to render.")
parser.add_argument("--output", type=str, default="outputs/jetbot_isaaclab_render.mp4", help="Output MP4 path.")
parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
parser.add_argument("--seed", type=int, default=7, help="Environment/action seed.")
parser.add_argument("--base_speed", type=float, default=7.0, help="Nominal forward wheel velocity target.")
parser.add_argument("--steer_gain", type=float, default=5.0, help="Wheel differential gain for heading correction.")
parser.add_argument("--max_steer", type=float, default=5.5, help="Maximum wheel differential steering command.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import direct_rl.tasks  # noqa: F401


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _target_yaw(step: int, total_steps: int) -> float:
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


def _make_action(env, target_yaw: float) -> torch.Tensor:
    current_yaw = _yaw_from_quat_wxyz(env.unwrapped.robot.data.root_quat_w[:1])[0]
    target = torch.tensor(target_yaw, dtype=torch.float32, device=env.unwrapped.device)
    heading_error = _wrap_to_pi(target - current_yaw)
    steer = torch.clamp(args_cli.steer_gain * heading_error, -args_cli.max_steer, args_cli.max_steer)
    left = args_cli.base_speed - steer
    right = args_cli.base_speed + steer
    return torch.stack((left, right), dim=0).reshape(1, 2)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset(seed=args_cli.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    frames_written = 0
    first_shape = None

    try:
        with imageio.get_writer(args_cli.output, fps=args_cli.fps, codec="libx264", quality=8) as writer:
            with torch.inference_mode():
                for step in range(args_cli.steps):
                    target_yaw = _target_yaw(step, args_cli.steps)
                    _set_command(env, target_yaw)
                    env.step(_make_action(env, target_yaw))
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("env.render() returned None")
                    frame_np = np.asarray(frame)
                    if frame_np.ndim != 3 or frame_np.shape[-1] < 3:
                        raise RuntimeError(f"Unexpected frame shape: {frame_np.shape}")
                    rgb = frame_np[:, :, :3].astype(np.uint8, copy=False)
                    first_shape = first_shape or rgb.shape
                    writer.append_data(rgb)
                    frames_written += 1
    finally:
        env.close()
        simulation_app.close()

    print(f"[INFO] Wrote Isaac Lab render: {args_cli.output}")
    print(f"[INFO] frames={frames_written} fps={args_cli.fps} first_frame_shape={first_shape}")


if __name__ == "__main__":
    main()
