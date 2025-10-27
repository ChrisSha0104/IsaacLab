# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: control module.

Imported by base, environment, and task classes. Not directly executed.
"""

import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
    ):
    """Compute Franka DOF torque to move fingertips towards target pose."""
    # References:
    # 1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # 2) Modern Robotics

    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # Offset task_wrench motion by random amount to simulate unreliability at low forces.
    # Check if absolute value is less than specified amount. If so, 0 out, otherwise, subtract.
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
    # roboticsproceedings.org/rss07/p31.pdf

    # useful tensors
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )  # ETH eq. 3.86; geometric Jacobian is assumed
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    # nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # TODO: Verify it's okay to no longer do gripper control here.
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
    ):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path using quaternion dot product.
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == "pinv":  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1  # 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
    ):
    """Interpret PD gains as task-space gains. Apply to task-space error."""

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rot error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench

def compute_dof_state(
    cfg,
    dof_pos, dof_vel,
    fingertip_midpoint_pos, fingertip_midpoint_quat,
    fingertip_midpoint_linvel, fingertip_midpoint_angvel,
    jacobian, arm_mass_matrix, # jacobian at fingertip
    ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat,
    task_prop_gains, task_deriv_gains,
    dt, F_ext, device,
    dead_zone_thresholds=None,
):
    """
    Compute Xarm DOF state using task-space admittance control.
    """
    B, _ = dof_pos.shape
    n = 7

    # 1) Pose error (pos + axis-angle)
    pos_err, aa_err = get_pose_error(
        fingertip_midpoint_pos,
        fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    e_task = torch.cat((pos_err, aa_err), dim=1)  # (B,6)
    xdot_now = torch.cat((fingertip_midpoint_linvel, fingertip_midpoint_angvel), dim=1)  # (B,6)

    # 2) Task-space inertia (Mx)
    Mq_inv = torch.inverse(arm_mass_matrix)  # (B,n,n) - M_inv in joint space
    JT = jacobian.transpose(1, 2)            # (B,n,6)
    Mx = torch.inverse(torch.bmm(jacobian, torch.bmm(Mq_inv, JT)))  # (B,6,6) - M in task space

    # 3) Spring-damper (admittance) term — separate linear/rotational parts
    spring_damper = torch.zeros_like(e_task)
    # linear components
    spring_damper[:, 0:3] = (
        task_prop_gains[:, 0:3] * e_task[:, 0:3]
        + task_deriv_gains[:, 0:3] * fingertip_midpoint_linvel
    )
    # rotational components
    spring_damper[:, 3:6] = (
        task_prop_gains[:, 3:6] * e_task[:, 3:6]
        + task_deriv_gains[:, 3:6] * fingertip_midpoint_angvel
    )

    # 4) Admittance control law: xddot = Mx^{-1} (F_ext - (K e + D xdot))
    if dead_zone_thresholds is not None:
        th = dead_zone_thresholds
        F_net = torch.where(
            (F_ext - spring_damper).abs() < th,
            torch.zeros_like(F_ext),
            (F_ext - spring_damper).sign() * ((F_ext - spring_damper).abs() - th),
        )
    else:
        F_net = F_ext - spring_damper # NOTE: F at eef, spring_damper at fingertip

    xddot = torch.bmm(torch.inverse(Mx), F_net.unsqueeze(-1)).squeeze(-1)  # (B,6)

    # 5) Integrate in task-space (semi-implicit)
    xdot_des = xdot_now + dt * xddot  # (B,6)

    # # 6) Map to joint velocities using Jacobian pseudoinverse
    # JJt_inv = torch.inverse(torch.bmm(jacobian, JT))  # (B,6,6)
    # J_pinv = torch.bmm(JT, JJt_inv)                   # (B,n,6)
    # qd_next = torch.bmm(J_pinv, xdot_des.unsqueeze(-1)).squeeze(-1)  # (B,n)

    # --- 6) Map to joint velocities with dynamic-consistent nullspace control ---

    # J_hash = M⁻¹ Jᵀ Mx
    J_hash = torch.bmm(Mq_inv, torch.bmm(JT, Mx))  # (B, n, 6)

    # Primary task joint velocity
    qd_task = torch.bmm(J_hash, xdot_des.unsqueeze(-1)).squeeze(-1)  # (B,n)

    # Nullspace projector N = I - J_hash * J
    I = torch.eye(n, device=device).unsqueeze(0).expand(B, -1, -1)   # (B,n,n)
    N = I - torch.bmm(J_hash, jacobian)                               # (B,n,n)

    # Secondary (null) objective: PD toward default posture with damping
    q_ref = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).unsqueeze(0).expand(B, -1)
    Kp_null = cfg.ctrl.kp_null
    Kd_null = cfg.ctrl.kd_null

    qd_null_des = Kp_null * (q_ref[:, :n] - dof_pos[:, :n]) - Kd_null * dof_vel[:, :n]   # (B,n)

    # Project null motion so it doesn't affect the task: J (N qd_null_des) ≈ 0
    qd_null = torch.bmm(N, qd_null_des.unsqueeze(-1)).squeeze(-1)     # (B,n)

    # Combine
    qd_next = qd_task + qd_null

    # 7) Integrate joint positions
    q_next = dof_pos[:, 0:7] + dt * qd_next

    return q_next, qd_next, xddot, e_task
