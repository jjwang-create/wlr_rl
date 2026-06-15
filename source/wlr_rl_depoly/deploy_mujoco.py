"""
MuJoCo 部署脚本 — 轮足机器人

将 IsaacLab 训练的 RL 策略部署到 MuJoCo 仿真器中运行。

Observation 向量 (25 维):
  [0:3]   base_ang_vel        机体坐标系角速度
  [3:6]   projected_gravity   机体坐标系重力方向
  [6:8]   virtual_leg_angle   虚拟腿角度 [left, right]
  [8:10]  virtual_leg_length  虚拟腿长度 [left, right]
  [10:12] virtual_leg_angle_vel   虚拟腿角速度
  [12:14] virtual_leg_length_vel  虚拟腿长度速度
  [14:16] wheel_vel           轮子速度 [left, right]
  [16:19] cmd_vec             速度指令 [vx, vy, wz]
  [19:25] last_action         上一步动作

Action 向量 (6 维, 经 ActionsCfg scale + offset 映射):
  [0:2] hip position   → target = default + action * 3.14
  [2:4] knee position  → target = default + action * 1.2
  [4:6] wheel velocity → target = action * 60.0
"""

import time
import argparse
import os

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# 索引映射

# Action 索引 (ActionsCfg 顺序: hip_l, hip_r, knee_l, knee_r, wheel_l, wheel_r)
hip_indices = np.array([0, 1])
knee_indices = np.array([2, 3])
wheel_indices = np.array([4, 5])

# DOF 索引 (URDF 关节顺序: hip_l, knee_l, wheel_l, hip_r, knee_r, wheel_r)
hip_dof_indices = np.array([0, 3])
knee_dof_indices = np.array([1, 4])
wheel_dof_indices = np.array([2, 5])

# 虚拟腿运动学 (与训练端 mdp/utils.py 一致)

def forward_kinematics(hip_angle, knee_angle):
    """正运动学：关节角度 → 虚拟腿角度 & 长度
    
    与训练端 _joint_to_virtual_leg 一致，处理了 URDF 关节偏移：
      alpha = -hip_angle
      gamma = alpha - knee_angle + π
    
    Returns:
        leg_angle:  虚拟腿角度 (竖直向下=0, 向前=正)
        leg_length: 虚拟腿长度 (hip→wheel 距离)
    """
    alpha = -hip_angle
    gamma = alpha - knee_angle + np.pi
    
    x2 = l1 * np.sin(alpha) + l2 * np.sin(gamma)
    z2 = -l1 * np.cos(alpha) - l2 * np.cos(gamma)
    
    leg_length = np.sqrt(x2**2 + z2**2)
    leg_angle = np.arctan2(x2, -z2)
    return leg_angle, leg_length

def virtual_leg_velocity(hip_pos, hip_vel, knee_pos, knee_vel):
    """数值 Jacobian 法计算虚拟腿速度（与训练端一致，eps=1e-4）
    
    Returns:
        angle, length, angle_vel, length_vel  (各为 shape=[2] 的 ndarray)
    """
    eps = 1e-4
    
    angle, length = forward_kinematics(hip_pos, knee_pos)
    angle_dh, _ = forward_kinematics(hip_pos + eps, knee_pos)
    angle_dk, _ = forward_kinematics(hip_pos, knee_pos + eps)
    _, length_dh = forward_kinematics(hip_pos + eps, knee_pos)
    _, length_dk = forward_kinematics(hip_pos, knee_pos + eps)
    
    angle_vel = (angle_dh - angle) / eps * hip_vel + (angle_dk - angle) / eps * knee_vel
    length_vel = (length_dh - length) / eps * hip_vel + (length_dk - length) / eps * knee_vel
    
    return angle, length, angle_vel, length_vel


# PD 控制

def compute_torques(action_tmp, hip_pos_tmp, hip_vel_tmp, knee_pos_tmp, knee_vel_tmp, wheel_vel_tmp):
    """根据 ActionsCfg 将网络输出映射为关节扭矩
    
    hip/knee: PD 位置控制  torque = kp*(target - pos) - kd*vel
    wheel:    P 速度控制   torque = kd*(target_vel - vel)
    """
    # 目标 = 默认角度 + 网络输出 * scale
    hip_target = default_angles[hip_dof_indices] + action_tmp[hip_indices] * action_scale_hip
    knee_target = default_angles[knee_dof_indices] + action_tmp[knee_indices] * action_scale_knee
    wheel_vel_target = action_tmp[wheel_indices] * action_scale_wheel

    # PD 控制
    torque_hip = kp_joint * (hip_target - hip_pos_tmp) - kd_joint * hip_vel_tmp
    torque_knee = kp_joint * (knee_target - knee_pos_tmp) - kd_joint * knee_vel_tmp
    torque_wheel = kd_wheel * (wheel_vel_target - wheel_vel_tmp)

    return torque_hip, torque_knee, torque_wheel

# 传感器辅助函数

def get_gravity_orientation(quat):
    """重力方向投影到机体坐标系 (MuJoCo 四元数 [w,x,y,z] 输入)"""
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # mujoco [w,x,y,z] -> scipy [x,y,z,w]
    return r.inv().apply([0, 0, -1])

# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    # --- 命令行参数 ---
    parser = argparse.ArgumentParser(description="Deploy RL policy in MuJoCo")
    parser.add_argument("config_file", type=str, default="wheel_legged_robot.yaml", help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    # --- 加载配置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "configs", config_file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kp_joint = config["kp_joint"]
        kd_joint = config["kd_joint"]
        kd_wheel = config["kd_wheel"]

        default_angles = np.array(config["default_angles"], dtype=np.float64)

        action_scale_hip = config["action_scale_hip"]
        action_scale_knee = config["action_scale_knee"]
        action_scale_wheel = config["action_scale_wheel"]

        num_actions = config["num_actions"]
        clip_actions = config["clip_actions"]

        num_obs = config["num_obs"]

        l1 = config["l1"]
        l2 = config["l2"]

    # --- 初始化状态 ---
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # --- 加载 MuJoCo 模型 ---
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # --- 加载策略网络 ---
    policy = torch.jit.load(policy_path)

    # --- 仿真主循环 ---
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()

            # --- 1) 读取关节状态 (每个物理步) ---
            dof_pos = d.qpos[7:7+len(default_angles)].copy()
            dof_vel = d.qvel[6:6+len(default_angles)].copy()

            hip_pos = dof_pos[hip_dof_indices]
            hip_vel = dof_vel[hip_dof_indices]
            knee_pos = dof_pos[knee_dof_indices]
            knee_vel = dof_vel[knee_dof_indices]
            wheel_vel = dof_vel[wheel_dof_indices]

            # --- 2) PD 控制 → 扭矩 ---
            torque_hip, torque_knee, torque_wheel = compute_torques(
                action_tmp=action,
                hip_pos_tmp=hip_pos,
                hip_vel_tmp=hip_vel,
                knee_pos_tmp=knee_pos,
                knee_vel_tmp=knee_vel,
                wheel_vel_tmp=wheel_vel,
            )

            # --- 3) 扭矩限幅 & 施加 ---
            torques = np.array(
                [torque_hip[0], torque_knee[0], torque_wheel[0],
                torque_hip[1], torque_knee[1], torque_wheel[1]],
                dtype=np.float32
            )
            torques[[0, 1, 3, 4]] = np.clip(torques[[0, 1, 3, 4]], -80.0, 80.0)  # hip/knee
            torques[[2, 5]] = np.clip(torques[[2, 5]], -5, 5)                      # wheel

            d.ctrl[:len(torques)] = torques

            # --- 4) 物理步进 ---
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # --- 5) Decimation 完成后读取最新状态 & 推理 ---
                # IMU: 角速度 (世界系→机体系) + 重力投影
                base_quat = d.xquat[base_body_id]
                base_ang_vel_world = d.qvel[3:6]
                r_body = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
                base_ang_vel = r_body.inv().apply(base_ang_vel_world)
                projected_gravity = get_gravity_orientation(base_quat)

                # 关节状态 & 虚拟腿
                dof_pos = d.qpos[7:7+len(default_angles)].copy()
                dof_vel = d.qvel[6:6+len(default_angles)].copy()
                hip_pos = dof_pos[hip_dof_indices]
                hip_vel = dof_vel[hip_dof_indices]
                knee_pos = dof_pos[knee_dof_indices]
                knee_vel = dof_vel[knee_dof_indices]
                wheel_vel = dof_vel[wheel_dof_indices]

                theta0, l0, theta0_dot, l0_dot = virtual_leg_velocity(hip_pos, hip_vel, knee_pos, knee_vel)

                # 速度指令 (TODO: 接入外部输入)
                cmd_vec = [2.5, 0.0, 0.0]

                # 拼接观测 (顺序须与 ObservationsCfg.PolicyCfg 一致)
                obs[0:3] = base_ang_vel
                obs[3:6] = projected_gravity
                obs[6:8] = theta0
                obs[8:10] = l0
                obs[10:12] = theta0_dot
                obs[12:14] = l0_dot
                obs[14:16] = wheel_vel
                obs[16:19] = cmd_vec
                obs[19:25] = action

                # 策略推理
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze().clip(-clip_actions, clip_actions)

            # --- 6) 同步渲染 & 实时控制 ---
            # 让相机跟踪机器人
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance = 10.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

