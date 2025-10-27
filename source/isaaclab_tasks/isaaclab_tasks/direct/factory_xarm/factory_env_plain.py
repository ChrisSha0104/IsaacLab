# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg


class FactoryEnvPlain(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        # factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        self.verbose = False

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device) # fixed obj pos in base frame
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device) # fixed obj pos noise

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("left_finger") 
        self.right_finger_body_idx = self._robot.body_names.index("right_finger")
        self.eef_body_idx = self._robot.body_names.index("link7") # TODO: change logic to fingertip == T(eef)
        self.fingertip2eef_offset = torch.tensor([self.cfg.fingertip2eef_offset], device=self.device).repeat(self.num_envs, 1)
        self.arm_dof_idx, _ = self._robot.find_joints("joint.*")
        self.gripper_dof_idx, _ = self._robot.find_joints("gripper")

        self.eef_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        self.eef_pos = self._robot.data.body_pos_w[:, self.eef_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_pos = torch_utils.tf_combine(
            self._robot.data.body_quat_w[:, self.eef_body_idx],
            self.eef_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            self.fingertip2eef_offset,
        )[1]

        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.eef_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.eef_body_idx] # NOTE: actually eef vels
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.eef_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians() # (num_envs, num_bodies, 6, num_dofs)
        self.eef_jacobian = jacobians[:, self.eef_body_idx - 1, 0:6, 0:7] # (num_envs, 6, arm_idx), origin at body idx

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7] # (num_envs, 6, arm_idx), origin at body idx
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.eef_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "joint_pos": self.joint_pos[:, 0:7],
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            # "held_pos": self.held_pos,
            # "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            # "held_quat": self.held_quat,
            # "fixed_pos": self.fixed_pos,
            # "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        # print("joint pos obs:", state_dict["joint_pos"][:3].cpu().numpy())
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order_no_task)
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.obs_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions
        self.goal_fingertip_pos = action[:, 0:3]
        self.goal_fingertip_quat = action[:, 3:7]

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # # Interpret actions as target pos displacements and set pos target
        # pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # # Interpret actions as target rot (axis-angle) displacements
        # rot_actions = self.actions[:, 3:6]
        # if self.cfg_task.unidirectional_rot:
        #     rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        # rot_actions = rot_actions * self.rot_threshold

        # ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # # To speed up learning, never allow the policy to move more than 5cm away from the base.
        # fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        # delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        # pos_error_clipped = torch.clip(
        #     delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        # )
        # ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # # Convert to quat and set rot target
        # angle = torch.norm(rot_actions, p=2, dim=-1)
        # axis = rot_actions / angle.unsqueeze(-1)

        # rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        # rot_actions_quat = torch.where(
        #     angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
        #     rot_actions_quat,
        #     torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        # )
        # ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        # target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        # target_euler_xyz[:, 1] = 0.0

        # ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
        #     roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        # )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self.goal_fingertip_pos,
            ctrl_target_fingertip_midpoint_quat=self.goal_fingertip_quat,
            ctrl_target_gripper_dof_pos=1.6,
        )

    def generate_ctrl_signals(
        self, 
        ctrl_target_fingertip_midpoint_pos, 
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos=0.0, # default open
        ):
        self.arm_joint_pose_target, self.joint_vel_target, x_acc, _, self.eef_vel = factory_control.compute_dof_state_admittance(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            eef_pos=self.eef_pos,
            eef_quat=self.fingertip_midpoint_quat,
            eef_linvel=self.fingertip_midpoint_linvel, # actually eef linvel
            eef_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.eef_jacobian,
            ctrl_target_eef_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_eef_quat=ctrl_target_fingertip_midpoint_quat,
            xdot_ref=self.eef_vel,
            dt=self.physics_dt * self.cfg.decimation,
            F_ext=None,
            device=self.device,
        )

        self._robot.set_joint_position_target(self.arm_joint_pose_target, joint_ids=self.arm_dof_idx)
        self._robot.set_joint_position_target(torch.tensor([[ctrl_target_gripper_dof_pos]], device=self.device).repeat(self.num_envs,1), joint_ids=self.gripper_dof_idx)
        # self._robot.set_joint_velocity_target(self.joint_vel_target)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, False

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        self.prev_actions = self.actions.clone()

        return torch.tensor([[0.0]], device=self.device).repeat(self.num_envs, 1)

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25 # 0.005 m
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        gripper_pos = 1.2
        # NOTE: has to set finger pos = 0.5 gripper pos by design of urdf
        joint_pos[:, 8:] = gripper_pos / 2.0 
        joint_pos[:, 7] = gripper_pos
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def log(self, string):
        if self.verbose:
            print(f"== {string} ==")
