"""Render a saved JetBot trajectory with PyBullet and write an MP4."""

from __future__ import annotations

import argparse
import math
import os

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data


parser = argparse.ArgumentParser(description="Render a direct_rl JetBot rollout with PyBullet.")
parser.add_argument("--input", type=str, default="outputs/jetbot_experiment.npz", help="Input trajectory npz.")
parser.add_argument("--output", type=str, default="outputs/jetbot_pybullet.mp4", help="Output MP4 path.")
parser.add_argument("--width", type=int, default=960, help="Video width.")
parser.add_argument("--height", type=int, default=540, help="Video height.")
parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
parser.add_argument("--stride", type=int, default=2, help="Render every Nth trajectory sample.")
parser.add_argument("--no_egl", action="store_true", help="Disable EGL plugin loading.")
args = parser.parse_args()


def _connect() -> int:
    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")
    if not args.no_egl:
        try:
            plugin = p.loadPlugin("eglRendererPlugin")
            print(f"[INFO] Loaded PyBullet EGL renderer plugin: {plugin}")
        except Exception as exc:
            print(f"[WARN] EGL renderer plugin unavailable, falling back to TinyRenderer: {exc}")
    return cid


def _create_marker_robot() -> dict[str, int]:
    """Create a lightweight visual stand-in for the JetBot."""
    body_half_extents = [0.17, 0.10, 0.06]
    wheel_radius = 0.055
    wheel_length = 0.045

    base_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=body_half_extents,
        rgbaColor=[0.08, 0.33, 0.78, 1.0],
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=base_vis,
        basePosition=[0.0, 0.0, wheel_radius + body_half_extents[2]],
    )

    wheel_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=wheel_radius,
        length=wheel_length,
        rgbaColor=[0.03, 0.03, 0.03, 1.0],
    )
    wheel_spoke_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[wheel_radius * 0.85, 0.006, 0.004],
        rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )
    arrow_vis = p.createVisualShape(
        p.GEOM_MESH,
        vertices=[
            [-0.20, -0.025, 0.0],
            [0.11, -0.025, 0.0],
            [0.11, -0.08, 0.0],
            [0.26, 0.0, 0.0],
            [0.11, 0.08, 0.0],
            [0.11, 0.025, 0.0],
            [-0.20, 0.025, 0.0],
        ],
        indices=[
            0,
            1,
            5,
            0,
            5,
            6,
            1,
            2,
            3,
            1,
            3,
            4,
            1,
            4,
            5,
        ],
        rgbaColor=[0.95, 0.2, 0.12, 1.0],
    )

    wheels = {}
    for name, x, y in (
        ("front_left_wheel", 0.14, 0.175),
        ("front_right_wheel", 0.14, -0.175),
        ("rear_left_wheel", -0.14, 0.175),
        ("rear_right_wheel", -0.14, -0.175),
    ):
        wheels[name] = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=wheel_vis,
            basePosition=[x, y, wheel_radius],
            baseOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
        )
        wheels[f"{name}_spoke"] = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=wheel_spoke_vis,
            basePosition=[x, y, wheel_radius],
            baseOrientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
        )

    target_arrow = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=arrow_vis,
        basePosition=[0.0, 0.0, 0.34],
    )
    return {
        "body": body,
        **wheels,
        "target_arrow": target_arrow,
    }


def _transform_offset(base_pos: np.ndarray, yaw: float, offset: tuple[float, float, float]) -> list[float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x, y, z = offset
    return [
        float(base_pos[0] + cos_yaw * x - sin_yaw * y),
        float(base_pos[1] + sin_yaw * x + cos_yaw * y),
        float(base_pos[2] + z),
    ]


def _set_robot_pose(
    robot: dict[str, int],
    base_pos: np.ndarray,
    yaw: float,
    target_yaw: float,
    left_wheel_angle: float,
    right_wheel_angle: float,
) -> None:
    body_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    yaw_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    wheel_mount_quat = p.getQuaternionFromEuler([math.pi / 2, 0.0, 0.0])
    target_quat = p.getQuaternionFromEuler([0.0, 0.0, target_yaw])

    # All parts are kinematic visual bodies. Updating every part explicitly prevents constraint drift.
    p.resetBasePositionAndOrientation(robot["body"], _transform_offset(base_pos, yaw, (0.0, 0.0, 0.0)), body_quat)
    for name, offset in (
        ("front_left_wheel", (0.14, 0.175, -0.075)),
        ("front_right_wheel", (0.14, -0.175, -0.075)),
        ("rear_left_wheel", (-0.14, 0.175, -0.075)),
        ("rear_right_wheel", (-0.14, -0.175, -0.075)),
    ):
        wheel_angle = left_wheel_angle if offset[1] > 0.0 else right_wheel_angle
        # Cylinder local Z is the axle after the mount rotation. Spinning around local Z gives normal wheel rolling.
        wheel_spin_quat = p.getQuaternionFromEuler([0.0, 0.0, -wheel_angle])
        _, wheel_quat = p.multiplyTransforms([0, 0, 0], yaw_quat, [0, 0, 0], wheel_mount_quat)
        _, wheel_quat = p.multiplyTransforms([0, 0, 0], wheel_quat, [0, 0, 0], wheel_spin_quat)
        p.resetBasePositionAndOrientation(robot[name], _transform_offset(base_pos, yaw, offset), wheel_quat)
        side_sign = 1.0 if offset[1] > 0.0 else -1.0
        spoke_offset = (offset[0], offset[1] + side_sign * 0.03, offset[2])
        p.resetBasePositionAndOrientation(robot[f"{name}_spoke"], _transform_offset(base_pos, yaw, spoke_offset), wheel_quat)
    p.resetBasePositionAndOrientation(robot["target_arrow"], [float(base_pos[0]), float(base_pos[1]), float(base_pos[2] + 0.20)], target_quat)


def _make_camera(position: np.ndarray, yaw: float) -> tuple[list[float], list[float]]:
    target = [float(position[0]), float(position[1]), 0.10]
    distance = 2.0
    camera_yaw = 135.0
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=camera_yaw,
        pitch=-55.0,
        roll=0.0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=50.0, aspect=args.width / args.height, nearVal=0.02, farVal=20.0)
    return view, proj


def main() -> None:
    data = np.load(args.input)
    positions = np.asarray(data["position"], dtype=np.float32)
    yaws = np.asarray(data["yaw"], dtype=np.float32)
    rewards = np.asarray(data["reward"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    if "left_wheel_angle" in data and "right_wheel_angle" in data:
        left_wheel_angles = np.asarray(data["left_wheel_angle"], dtype=np.float32)
        right_wheel_angles = np.asarray(data["right_wheel_angle"], dtype=np.float32)
    else:
        dt = float(data["dt"]) if "dt" in data else 1.0 / args.fps
        left_wheel_angles = np.cumsum(actions[:, 0] * dt)
        right_wheel_angles = np.cumsum(actions[:, 1] * dt)
    if "target_yaw" in data:
        target_yaws = np.asarray(data["target_yaw"], dtype=np.float32)
    else:
        target_yaws = yaws.copy()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _connect()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot = _create_marker_robot()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    frame_indices = range(0, len(positions), max(1, args.stride))
    frame_indices_list = list(frame_indices)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame_no, idx in enumerate(frame_indices_list):
            pos = positions[idx].copy()
            pos[2] = max(float(pos[2]), 0.13)
            yaw = float(yaws[idx])
            target_yaw = float(target_yaws[idx])
            _set_robot_pose(
                robot,
                pos,
                yaw,
                target_yaw,
                float(left_wheel_angles[idx]),
                float(right_wheel_angles[idx]),
            )

            view, proj = _make_camera(pos, yaw)
            _, _, rgba, _, _ = p.getCameraImage(
                args.width,
                args.height,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            rgb = np.asarray(rgba, dtype=np.uint8).reshape(args.height, args.width, 4)[:, :, :3].copy()
            writer.append_data(rgb)

            if frame_no == 0:
                print(
                    "[INFO] first frame "
                    f"reward={float(rewards[idx]):.6f} "
                    f"target_yaw={target_yaw:.3f} "
                    f"action={actions[idx].round(3).tolist()}"
                )

    p.disconnect()
    print(f"[INFO] Wrote video: {args.output}")
    print(f"[INFO] frames={len(frame_indices_list)} total_reward={float(np.sum(rewards)):.6f}")


if __name__ == "__main__":
    main()
