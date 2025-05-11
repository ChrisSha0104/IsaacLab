# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Usd
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, combine_frame_transforms, quat_mul, matrix_from_quat, quat_inv, quat_conjugate
from omni.isaac.lab.assets import AssetBaseCfg, AssetBase
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from omni.isaac.lab.sensors import CameraCfg, Camera, TiledCameraCfg, TiledCamera
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils import convert_dict_to_backend
import cv2

from RRL.utilities import *
from RRL.rl_models.action_normalizer import ActionNormalizer

import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import time
import random
from typing import Callable
# import omni.replicator.core as rep
import os

"""
initial pose cfg
"""

@configclass
class XArmInsertionResidualTeacherEnvCfg(DirectRLEnvCfg):
    # env 
    episode_length_s = 11.6667 # eps_len_s = traj_len * (dt * decimation)
    decimation = 4
    action_space: int = 10                          # [position, 6D orientation, gripper qpos]
    observation_space = 10 + 10 + 120*120           # [robot state, teleop action, depth image]
    state_space = 10 + 10 + 9 + 120*120             # [robot state, teleop action, depth image, object state, demo idx]
    rerender_on_reset = False
    is_finite_horizon = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, # NOTE: 30 Hz
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=80,  # Important to avoid interpenetration.  
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.02,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation. # NOTE: THIS IS IMPORTANT 
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="RRL/robot/sapien_xarm7/xarm_urdf/xarm7_gripper.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,      
                solver_position_iteration_count=80, # TODO: check iteration count
                solver_velocity_iteration_count=1
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="acceleration"),              
        ), 
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={ 
                "joint1": 0.0 * np.pi / 180,
                "joint2": -7.2 * np.pi / 180, #-45
                "joint3": 0.2 * np.pi / 180,
                "joint4": 41.8 * np.pi / 180, # 30
                "joint5": -0.6 * np.pi / 180,
                "joint6": 49 * np.pi / 180, # 75
                "joint7": 0.1 * np.pi / 180,
                "gripper": 0.0, # 0.0 to 1.7
                "left_driver_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "left_finger_joint": 0.0,
                "right_driver_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                stiffness=200,#80.0, 
                damping=20,#10.0,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper"], 
                stiffness=1e4,#7500.0,
                damping=50,#173.0,
            ),
        },
    )

    # assets 
    quat = quat_from_euler_xyz(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(np.pi/2))
    quat = tuple(quat.tolist())
    base = RigidObjectCfg(
            prim_path="/World/envs/env_.*/base",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.25, 0.0, 0.0),
                rot=(quat),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/shuosha/Desktop/insertion_assets/base_smooth/insertion_base_smooth.usd", 
                scale=(0.80, 0.80, 0.80),
                rigid_props=RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
            ),
        )   

    nut = RigidObjectCfg(
            prim_path="/World/envs/env_.*/nut",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.40, 0.0, 0.0),
                rot=(0,1,0,0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/shuosha/Desktop/insertion_assets/nut_poly_wide_smooth/nut_poly_wide_smooth.usd", 
                scale=(1.1, 1.1, 1.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=80,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    angular_damping = 0.3,
                    linear_damping = 0.2,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    enable_gyroscopic_forces=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            ),
        ) 
    # cameras
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link7/cam", 
        offset=TiledCameraCfg.OffsetCfg(pos=(0.125, 0.0, 0.008), rot=(0.96815, 0.0, -0.25038, 0.0), convention="ros"), # z-down; x-forward # greater angle = towards gripper
        height=120,
        width=120,
        data_types=[
            "distance_to_image_plane",
            ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=733/6/100, focus_distance=400.0, horizontal_aperture=120/100, clipping_range=(0.01, 20) # NOTE: unit in cm
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -------- training options -------- 
    training_data_path = "RRL/tasks/insertion/training_set3"
    enable_residual = True
    apply_dmr = True           
    augment_real_data = True

    # -------- training params --------
    traj_length = 350
    num_demos = 20
    alpha = 0.1                 # residual scale
    num_samples = 3             # number of teleop samples to be used for training
    sample_interval = 5         # sample interval for teleop samples (NOTE: dt = 1/30)

    # -------- initialization --------
    # state normalization
    fingertip_init_pose_10D = [0.3998, 0.0051,  0.2797-0.155,                   # NOTE: initial fingertip pose in sim & real [m]
                                1.00, 0.00, 0.00, 0.00, -1.00, 0.00, 
                                -1.00]                                          # NOTE: 1 = close, -1 = open
    fingertip_lower_limit = [0.20, -0.4, 0.01, 
                            -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, 
                            -1.0]
    fingertip_upper_limit = [0.60, 0.4, 0.5, 
                            1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 
                            1.0]
    state_obs_lower_limit = fingertip_lower_limit * (num_samples+1) 
    state_obs_upper_limit = fingertip_upper_limit * (num_samples+1) 

    # privilaged obs normalization
    nut_lower_limit = [0.30, -0.1, 0.0, 
                        -1.05, -1.05, -1.05, -1.05, -1.05, -1.05]
    nut_upper_limit = [0.50, 0.1, 0.4, # 55 - 45
                        1.05, 1.05, 1.05, 1.05, 1.05, 1.05]
    base_lower_limit = [0.20, -0.05, 0.0,
                        -1.05, -1.05, -1.05, -1.05, -1.05, -1.05]
    base_upper_limit = [0.30, 0.05, 0.05,
                        1.05, 1.05, 1.05, 1.05, 1.05, 1.05]
    privilege_obs_lower_limit = [0.0, # fingertip nut dist
                                 0.0, # episode length
                                 0.0, # demo idx
                                 0.0] # reached nut
    privilege_obs_upper_limit = [1.0, 350.0, float(num_demos), 1.0]

    # -------- visualization options -------- 
    show_camera = False
    debug_intermediate_values = False
    order_demos = False
    log_success_rate = True

    # -------- sim2real options -------- 
    store_sim_trajectory = False
    storing_path = "RRL/tasks/nut/sim2real/traj1"

    # -------- rewards --------
    nres_penalty_scale = -1e-2
    nres_rate_scale = -1e-4
    completion_reward_scale = 10.0

class XArmInsertionResidualTeacherEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()

    # post-physics step calls
    #   |-- _get_dones()
    #   |  |-- _compute_intermediate_values()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XArmInsertionResidualTeacherEnvCfg

    def __init__(self, cfg: XArmInsertionResidualTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._print_training_info()
        self._init_buffers()
        if self.cfg.debug_intermediate_values:
            self._init_markers()
        self._init_controller()
        # self._set_default_dynamics_parameters()
        self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        # robot states
        self.fingertip_pos, self.fingertip_quat, self.fingertip_orn_6D, self.gripper_binary = self.get_fingertip_obs()
        self.fingertip_10D = torch.cat((self.fingertip_pos, self.fingertip_orn_6D, self.gripper_binary), dim=-1)
        self.ee_pos = self.fingertip_pos[:,:3].clone() + tf_vector(self.fingertip_quat, -1*self.fingertip2ee_offset_sim) 

        # teleop commands
        self.teleop_fingertip_pos, self.teleop_fingertip_quat, self.teleop_fingertip_orn_6D, self.teleop_gripper_binary = self.get_teleop_fingertip_comm()
        self.teleop_fingertip_10D = torch.cat((self.teleop_fingertip_pos, self.teleop_fingertip_orn_6D, self.teleop_gripper_binary), dim=-1)
        self.teleop_pos = self.teleop_fingertip_pos[:,:3].clone() + tf_vector(self.teleop_fingertip_quat, -1*self.fingertip2ee_offset_sim)

        # nut states
        robot_root_w = self._robot.data.root_state_w[:].clone()
        self.nut_pos, self.nut_quat = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._nut.data.body_com_state_w[:,0,:3], self._nut.data.body_com_state_w[:,0,3:7] # link state = coordinate frame, com state = actual object loc
        )
        self.nut_orn_6D = quat_to_6d(self.nut_quat)

        self.base_pos, self.base_quat = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._base.data.body_com_state_w[:,0,:3], self._base.data.body_com_state_w[:,0,3:7]
        )
        self.base_orn_6D = quat_to_6d(self.base_quat)

        self.nut_goal_dist = torch.norm(self.nut_pos - self.intended_goal_pose[:,:3], dim=-1)

        if self.cfg.debug_intermediate_values:
            # self.marker1.visualize(self.fingertip_pos + self.scene.env_origins, self.fingertip_quat)
            self.marker2.visualize(self.teleop_fingertip_pos + self.scene.env_origins, self.teleop_fingertip_quat)
            # self.marker3.visualize(self.nut_pos + self.scene.env_origins, self.nut_quat)
            # self.marker4.visualize(self.ee_pos + self.scene.env_origins, self.fingertip_quat)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._nut = RigidObject(self.cfg.nut)
        self._base = RigidObject(self.cfg.base)

        if self.cfg.show_camera:
            self._camera = TiledCamera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["nut"] = self._nut
        self.scene.rigid_objects["base"] = self._base
        if self.cfg.show_camera:
            self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) 
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # pre-physics step calls
    def _pre_physics_step(self, noutput: torch.Tensor):
        # store output
        self.last_nres = self.n_residual_10D.clone() if hasattr(self, "n_residual_10D") else noutput.clone()[:,:10]
        self.n_residual_10D = noutput.clone()[:,:10]
        self.last_fingertip_goal_10D = self.fingertip_goal_10D.clone() # last fingertip action
        
        # check if use residual policy
        if self.cfg.enable_residual:
            nbase = self.action_normalizer.normalize(self.teleop_fingertip_10D.clone())
            n_fingertip_goal_10D = nbase + self.cfg.alpha * self.n_residual_10D.clone()
            self.fingertip_goal_10D = self.action_normalizer.denormalize(n_fingertip_goal_10D.clone()) 
        else: 
            self.fingertip_goal_10D = self.teleop_fingertip_10D.clone()

        # clamp output
        fingertip_goal_10D_filtered = self.fingertip_goal_10D.clone()
        fingertip_goal_10D_filtered[:,:3] = torch.clamp(fingertip_goal_10D_filtered[:,:3], self.fingertip_low[:,:3], self.fingertip_high[:,:3])
        fingertip_goal_10D_filtered[:,-1] = torch.where(fingertip_goal_10D_filtered[:, -1] > 0.0, 
                                                        torch.tensor(1.0, device=fingertip_goal_10D_filtered.device),  # closed
                                                        torch.tensor(-1.0, device=fingertip_goal_10D_filtered.device)) # open

        # convert to qpos for control
        self.joint_pos = self.get_qpos_from_fingertip_10d(self.diff_ik_controller, fingertip_goal_10D_filtered, apply_smoothing=True)
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.qpos_lower_limits[:8], self.qpos_upper_limits[:8])       # (num_envs, 8)

        if self.cfg.store_sim_trajectory:
            self.residual_list.append(self.n_residual_10D.clone())

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets[:,:], joint_ids=[i for i in range(self.num_eff_joints)])

    def step(self, output):
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(output)
        if self.cfg.show_camera:
            self._camera.update(dt=self.dt)

        if self.reward_normalizer is not None:
            reward_buf = self.reward_normalizer(reward_buf)

        self.time_step_per_env += 1
        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras
    
    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        success = (self.nut_goal_dist < 0.02)
        self.extras["success"] = success.clone()
        self.ever_success |= success

        terminated = self.fingertip_pos[:,2] < 0.008
        terminated |= success

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.store_sim_trajectory:
            if terminated or truncated:
                os.makedirs(self.cfg.storing_path, exist_ok=True)
                state_tensor = torch.stack(self.state_list[3:], dim=0).reshape(-1, 10)
                torch.save(state_tensor, os.path.join(self.cfg.storing_path, "sim_state_obs.pt"))
                teleop_tensor = torch.stack(self.teleop_list[3:], dim=0).reshape(-1, self.cfg.num_samples*10)
                torch.save(teleop_tensor, os.path.join(self.cfg.storing_path, "sim_teleop_obs.pt"))
                nut_tensor = torch.stack(self.nut_list[3:], dim=0).reshape(-1, 9)
                torch.save(nut_tensor, os.path.join(self.cfg.storing_path, "sim_nut_obs.pt"))
                residual_tensor = torch.stack(self.residual_list[1:], dim=0).reshape(-1, 10)
                torch.save(residual_tensor, os.path.join(self.cfg.storing_path, "sim_residual_output.pt"))
                if self.cfg.show_camera:
                    depth_array = np.stack(self.depth_list, axis=0) 
                    np.save(os.path.join(self.cfg.storing_path, "sim_depth_obs.npy"), depth_array[3:])
                print(f"init nut pos: {nut_tensor[2, :3]}, nut tensor shape: {nut_tensor.shape}")
                print(f"init robot pos: {state_tensor[2]}, robot tensor shape: {state_tensor.shape}")
                print(f"init teleop pos: {teleop_tensor[2]}, teleop tensor shape: {teleop_tensor.shape}")
                print(f"sim data stored at {self.cfg.storing_path}")
                exit()

        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _compute_rewards(self):
        # log:
        self.fingertip_nut_dist = torch.norm(self.fingertip_pos - self.nut_pos, dim=-1) # (num_envs, )     
        reached_nut = torch.where((self.fingertip_nut_dist < 0.05), 1.0, 0.0)

        # completion reward
        inserted_nut = torch.where(((self.nut_goal_dist < 0.02)), 1.0, 0.0)
        completion_reward = inserted_nut

        # residual penalty
        nres_norm = torch.norm(self.n_residual_10D[:,:10], dim=-1)

        # residual rate
        nres_rate = torch.norm(self.n_residual_10D - self.last_nres, dim=-1) / self.dt

        rewards = (
            self.cfg.completion_reward_scale * completion_reward
        )

        low_finger = torch.where(self.fingertip_pos[:,2] < 0.008, -1.0, 0.0)

        self.extras["log"] = { # TODO: residual size
            "fingertip_nut_dist": self.fingertip_nut_dist.mean(),
            "reached_nut": reached_nut.mean(),
            "inserted_nut": inserted_nut.mean(),
            "low_finger": low_finger.mean(),
            "nres_norm": nres_norm.mean(),
            "teleop_misalignment": torch.norm(self.fingertip_goal_10D - self.teleop_fingertip_10D, dim=-1).mean(),
            "nres_penalty_rew": (self.cfg.nres_penalty_scale * nres_norm).mean(),
            "nres_rate_rew": (self.cfg.nres_rate_scale * nres_rate).mean(),
            "completion_rew": (self.cfg.completion_reward_scale * completion_reward).mean(),
        }

        return rewards
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if len(env_ids) > 0 and self.cfg.log_success_rate:
            print("success rate: ", self.ever_success.float().mean().item())
            self.ever_success.zero_()
        
        # reset time, demo counter, and training traj
        self._reset_buffers(env_ids, self.cfg.augment_real_data)
        
        # reset robot states
        self._reset_robot_state(env_ids, self.cfg.apply_dmr)

        # reset nut states based on new training traj
        self._reset_assets(env_ids, self.cfg.apply_dmr)

        # reset camera and controller
        if self.cfg.show_camera:
            self._camera.reset(env_ids)
        self.diff_ik_controller.reset(env_ids) # type: ignore

        # step through physics
        self._step_sim_no_action()

        # self._get_asset_offset(self._nut)
        
        # recompute / update training traj based on randomized robot state
        self._update_training_traj(env_ids)


    def _get_observations(self) -> dict:   
        if self.cfg.show_camera:
            depth_clean = self._camera.data.output["distance_to_image_plane"].permute(0,3,1,2)
            depth_filtered = filter_sim_depth(depth_clean) # (num_envs, 120*120) which should give same reading as real
            depth_noised = add_noise_in_depth_band(depth_filtered)
            normalized_depth = normalize_depth_01(depth_noised)
            # plt.imshow(depth_noised.detach().cpu().numpy().reshape(120,120), cmap='gray')
            # plt.show()
        
        if self.num_envs == 1 and self.cfg.show_camera and self.cfg.show_camera:
            depth_vis = filter_depth_for_visualization(depth_filtered.reshape(120,120).detach().cpu().numpy(), unit='m') # only works for 1 env
            cv2.imshow("depth_image", depth_vis)
            cv2.waitKey(1)
            # self.depth_list.append(depth_filtered.detach().cpu().numpy().reshape(120,120))

        self.teleop_hist.append(self.teleop_fingertip_10D.clone())
        teleop_hist = self.teleop_hist.get_history()                    # (num_envs, num_samples * 10)
        fingertip_nut_dist = torch.norm(self.fingertip_pos - self.nut_pos, dim=-1)
        reached_nut = torch.where((fingertip_nut_dist < 0.05), 1.0, 0.0)

        state_obs = torch.cat((
            self.fingertip_pos,
            self.fingertip_orn_6D,
            self.gripper_binary,
            teleop_hist
        ), dim=-1,)

        nut_obs = torch.cat((
            self.nut_pos,
            self.nut_orn_6D,
        ), dim=-1)

        base_obs = torch.cat((
            self.base_pos,
            self.base_orn_6D,
        ), dim=-1)

        privilege_obs = torch.cat((
            fingertip_nut_dist.unsqueeze(-1),
            self.episode_length_buf.float().unsqueeze(-1),
            self.demo_idx.float().unsqueeze(-1),
            reached_nut.unsqueeze(-1),
        ), dim=-1)

        normalized_state_obs = self.state_obs_normalizer.normalize(state_obs)
        normalized_nut_obs = self.nut_normalizer.normalize(nut_obs)
        normalized_base_obs = self.base_normalizer.normalize(base_obs)
        normalized_privledge_obs = self.privilege_obs_normalizer.normalize(privilege_obs)
        noramlized_intention_obs = self.intended_goal_binary.unsqueeze(-1)
        normalized_actor_obs = torch.cat((normalized_state_obs, normalized_nut_obs, normalized_privledge_obs), dim=-1)

        # -------------------------------- sim2real ------------------------------
        if self.cfg.store_sim_trajectory:
            self.state_list.append(self.fingertip_10D.clone())
            self.teleop_list.append(teleop_hist.clone())
            self.nut_list.append(torch.cat((self.nut_pos, self.nut_orn_6D), dim=-1))
            if self.cfg.show_camera:
                self.depth_list.append(depth_noised.detach().cpu().numpy().reshape(120,120))

        return {"policy": torch.clamp(normalized_actor_obs, -1.0, 1.0)}

    def _print_training_info(self):
        print(f"[Training Config]")
        print(f"Data path       : {self.cfg.training_data_path}")
        print(f"Show Camera     : {'Yes' if self.cfg.show_camera else 'No'}")
        print(f"Data augment    : {'Yes' if self.cfg.augment_real_data else f'No, using {self.cfg.num_demos} demos'}")
        print(f"DMR             : {'Enabled' if self.cfg.apply_dmr else 'Disabled'}")
        print(f"Residual policy : {'Enabled' if self.cfg.enable_residual else 'Disabled'}")
        # print(f"Asymmetric AC   : {'Yes' if self.cfg.use_privilege_obs else 'No'}\n")

    def _init_buffers(self):
        # simulation time
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot buffers
        self.qpos_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.qpos_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.fingertip2ee_offset_sim = torch.tensor([[0.0, 0.0, 0.16]], device=self.device).repeat(self.num_envs, 1) # const distance between fingertip and ee # NOTE: real - sim = 5 mm offset
        self.gripper_base_offset_ee_fr = torch.tensor([[0.0, 0.0, -0.034]], device=self.device).repeat(self.num_envs, 1) # offset between sim ee and real ee, (I.E, SAME QPOS PRODUCES SAME EE POSE IN SIM AND REAL)
        
        self.num_eff_joints = 8
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_eff_joints), device=self.device)
        self.joint_ids = list(range(self._robot.num_joints))

        # reading trajectories        
        self.time_step_per_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # (num_envs, )
        if self.cfg.order_demos:
            assert self.num_envs == self.cfg.num_demos, "Number of envs must be equal to number of demos"
            self.demo_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else: 
            self.demo_idx = torch.randint(
                                low=0,
                                high=self.cfg.num_demos,
                                size=(self.num_envs,),
                                device=self.device,
                                dtype=torch.long,
                            )
        self.env_idx = torch.arange(self.num_envs, device=self.device) # (num_envs, )

        self.traj_length = self.cfg.traj_length
        self.fingertip_init_pose_10D = torch.tensor([self.cfg.fingertip_init_pose_10D], device=self.device).repeat(self.num_envs, 1) # (num_envs, 10)

        # (num_envs, num_demos, traj_length, action_dim)
        self.clean_demo_trajs = self._resample_clean_demo(dir=self.cfg.training_data_path, num_demos=self.cfg.num_demos, init_pos=self.fingertip_init_pose_10D)
        self.training_demo_traj = self.clean_demo_trajs.clone()

        # normalize obs
        self.fingertip_low = torch.tensor([self.cfg.fingertip_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.fingertip_high = torch.tensor([self.cfg.fingertip_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.action_normalizer = ActionNormalizer(self.fingertip_low, self.fingertip_high)

        self.nut_low = torch.tensor([self.cfg.nut_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.nut_high = torch.tensor([self.cfg.nut_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.nut_normalizer = ActionNormalizer(self.nut_low, self.nut_high)

        self.base_low = torch.tensor([self.cfg.base_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.base_high = torch.tensor([self.cfg.base_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.base_normalizer = ActionNormalizer(self.base_low, self.base_high)

        obs_low = torch.tensor([self.cfg.state_obs_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        obs_high = torch.tensor([self.cfg.state_obs_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.state_obs_normalizer = ActionNormalizer(obs_low, obs_high)

        privilege_obs_low = torch.tensor([self.cfg.privilege_obs_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        privilege_obs_high = torch.tensor([self.cfg.privilege_obs_upper_limit], device=self.device).repeat(self.num_envs, 1)
        self.privilege_obs_normalizer = ActionNormalizer(privilege_obs_low, privilege_obs_high)

        self.reward_normalizer = (
            RunningMeanStdClip(shape=(1,), clip_value=5.0)
        )

        self.teleop_hist = HistoryBuffer(num_envs=self.num_envs, 
                                         hist_length=self.cfg.num_samples * self.cfg.sample_interval + 1,
                                         state_dim=10,
                                         num_samples=self.cfg.num_samples,
                                         sample_spacing=self.cfg.sample_interval,
                                         )
        self.teleop_hist.initialize(self.fingertip_init_pose_10D.clone()) # (num_envs, 10)

        self.fingertip_goal_10D = self.fingertip_init_pose_10D.clone() # (num_envs, 10)

        if self.cfg.store_sim_trajectory:
            assert self.num_envs == 1, "Storing observations only works for 1 env"
            self.state_list = []
            self.teleop_list = []
            self.nut_list = []
            self.depth_list = [] # NOTE: depth list is stored as np arrays

            self.residual_list = []

        self.ever_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        right_goal = torch.tensor([[0.25, -0.05, 0.02, 1.00, 0.00, 0.00, 0.00, -1.00, 0.00]], device=self.device)
        left_goal = torch.tensor([[0.25, 0.05, 0.02, 1.00, 0.00, 0.00, 0.00, -1.00, 0.00]], device=self.device)
        self.insertion_goals = torch.cat((right_goal, left_goal), dim=0) #(2, 9)
        self.intended_goal_pose = torch.zeros((self.num_envs, 9), device=self.device)

        self.nut_offset = torch.tensor([[-0.0855, -0.0257,  0.0150]], device=self.device).repeat(self.num_envs, 1), torch.tensor([[0.9728, 0.0012, 0.0027, 0.2314]], device=self.device).repeat(self.num_envs, 1) # (num_envs, 4)
        self.base_offset = torch.tensor([[0.0561, -0.0354,  0.0072]], device=self.device).repeat(self.num_envs, 1), torch.tensor([[1.0000e+00, 1.5101e-04, 2.4878e-04, 3.2966e-05]], device=self.device).repeat(self.num_envs, 1) # (num_envs, 4)

    def _init_markers(self):
        frame_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore

        self.markers = []
        for i in range(1, 8):
            marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/marker{i}"))  # type: ignore
            self.markers.append(marker)

        (self.marker1, self.marker2, self.marker3,
        self.marker4, self.marker5, self.marker6,
        self.marker7) = self.markers

    def _init_controller(self):
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

    def _resample_clean_demo(self, dir: str, num_demos: int, init_pos: torch.Tensor):
        all_demos = []

        for i in range(1, num_demos+1):
            traj: torch.Tensor = torch.load(os.path.join(dir, f"demo_traj{i}.pt"), weights_only=True).to(device=self.device).unsqueeze(0)                      # (1, traj_length, action_dim)
            traj = traj.repeat(self.num_envs, 1, 1)                                                                                         # (num_envs, traj_length, action_dim)
            if traj.shape[1] > self.traj_length:
                traj = traj[:,:self.traj_length,:]                                                                                      # (num_envs, traj_length, action_dim)
            else:
                traj = torch.cat((init_pos.unsqueeze(1).repeat(1, self.traj_length - traj.shape[1], 1), traj), dim=1)                      # (num_envs, traj_length, action_dim)
            # traj = resample_trajectory_10d(traj, self.traj_length, 10)                                                   # (num_envs, traj_length, action_dim)
            traj[..., -1] = torch.where(traj[..., -1] < 0.5, 
                                        torch.tensor(-1.0, device=traj[..., -1].device),     # open 
                                        torch.tensor(1.0, device=traj[..., -1].device))    # closed
            all_demos.append(traj)

        return torch.stack(all_demos, dim=1) # (num_envs, num_demos, traj_length, action_dim)
    
    def _get_nut_pick_up_poses(self, demo_traj: torch.Tensor):
        num_envs, timestep, action_dim = demo_traj.shape

        gripper_closed = demo_traj[..., -1] == 1.0  # shape: (num_envs, timestep)
        first_closed_idx = gripper_closed.float().argmax(dim=-1)  # shape: (num_envs,)

        env_idx = torch.arange(num_envs)  # (num_envs, 1)

        # Use advanced indexing
        selected_actions = demo_traj[env_idx, first_closed_idx]  # shape (num_envs, action_dim)
        
        return selected_actions

    def get_fingertip_obs(self):
        """
        return: 
            get current robot state in the base frame
            tuple: ee_pos, ee_orn_quat, ee_orn_6D, finger_status
        """
        # ee and root pose in world frame
        ee_pose_BIASED_w = self._robot.data.body_com_state_w[:,7,:].clone()
        ee_pose_CONST_w = ee_pose_BIASED_w.clone()
        ee_pose_CONST_w[:,:3] = ee_pose_BIASED_w[:,:3].clone() + tf_vector(ee_pose_BIASED_w[:,3:7], self.gripper_base_offset_ee_fr) 
        fingertip_w = ee_pose_CONST_w[:,:3].clone() + tf_vector(ee_pose_CONST_w[:,3:7], self.fingertip2ee_offset_sim)
        
        root_pose_w = self._robot.data.root_state_w[:]

        # ee pose in base (local) frame
        fingertip_pos_b, fingertip_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], fingertip_w[:, 0:3], ee_pose_CONST_w[:, 3:7]
            )
        orient_6d = quat_to_6d(fingertip_quat_b)

        finger_qpos = self._robot.data.joint_pos[:,7:8]#torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1) # TODO check mean here
        binary_gripper = torch.where(finger_qpos >= 0.5, 
                                     torch.tensor(1.0, device=finger_qpos.device), 
                                     torch.tensor(-1.0, device=finger_qpos.device))   # convert gripper qpos to binary

        return fingertip_pos_b, fingertip_quat_b, orient_6d, binary_gripper
    
    def get_fingertip_7D_b(self):
        fingertip_pos, fingertip_quat, _, _ = self.get_fingertip_obs()
        fingertip_7D_b = torch.cat((fingertip_pos, fingertip_quat), dim=-1) # (num_envs, 7)
        return fingertip_7D_b
    
    def get_fingertip_10D_b(self):
        fingertip_pos, _, fingertip_orn_6D, finger_status = self.get_fingertip_obs()
        fingertip_10D_b = torch.cat((fingertip_pos, fingertip_orn_6D, finger_status), dim=-1) # (num_envs, 10)
        return fingertip_10D_b
    
    def get_teleop_fingertip_comm(self):
        """
        return:
            get current teleop state in the base frame
            tuple: teleop_pos, teleop_quat, teleop_orn_6D, finger_status
        """
        teleop_fingertip_10D = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :]
        teleop_fingertip_pos = teleop_fingertip_10D[:,:3].clone()
        teleop_fingertip_quat = quat_from_6d(teleop_fingertip_10D[:,3:9].clone())
        teleop_fingertip_orn_6D = teleop_fingertip_10D[:,3:9].clone()
        teleop_finger_status = teleop_fingertip_10D[:,-1].unsqueeze(1).clone()

        return teleop_fingertip_pos, teleop_fingertip_quat, teleop_fingertip_orn_6D, teleop_finger_status

    def get_teleop_ee_10D_b(self):
        teleop_pos, _, teleop_orn_6D, finger_status = self.get_teleop_fingertip_comm()
        teleop_10D_b = torch.cat((teleop_pos, teleop_orn_6D, finger_status), dim=-1)
        return teleop_10D_b

    def get_teleop_ee_7D_b(self):
        teleop_pos, teleop_quat, _, _ = self.get_teleop_fingertip_comm()
        teleop_7D_b = torch.cat((teleop_pos, teleop_quat), dim=-1)
        return teleop_7D_b
    
    def get_qpos_from_fingertip_10d(self, controller: DifferentialIKController, fingertip_10D: torch.Tensor, apply_smoothing=False):
        ee_goal_quat = quat_from_6d(fingertip_10D[:,3:9])
        ee_goal_pos = fingertip_10D[:,:3].clone() + tf_vector(ee_goal_quat, -1 * self.fingertip2ee_offset_sim)
        ee_goal_7d = torch.cat((ee_goal_pos, ee_goal_quat), dim=-1)
        ik_commands = ee_goal_7d 
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link7")[0][0]-1
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(self.ee_pos, self.fingertip_quat, jacobian, joint_pos)

        gripper_status = fingertip_10D[:, -1:].clone() # binary gripper + residual output
        gripper_status[gripper_status > 0.0] = 1.7          # NOTE: close gripper in qpos
        gripper_status[gripper_status < 0.0] = 0.0          # NOTE: open gripper in qpos

        # if (gripper_status == 0.0).any() and self.time_step_per_env[0] > 250:
        #     print("ee dropped at", fingertip_10D[:,:3][(gripper_status == 0.0).repeat(1,3)])

        if apply_smoothing:
            delta = joint_pos_des_arm - joint_pos  # (E,13)

            # 1) compute per‐env joint norms over the first 7 dims
            joint_delta_norm = delta[:, :7].norm(dim=1)             # (E,)
            max_joint_delta  = delta[:, :7].abs().max(dim=1).values  # (E,)

            # 2) find which envs exceed the threshold
            max_delta_norm = 0.10
            mask = joint_delta_norm > max_delta_norm                # (E,) bool

            # # 3) optional logging for the offending envs
            # if mask.any():
            #     idxs = mask.nonzero(as_tuple=False).squeeze(1)
            #     for i in idxs.tolist():
            #         print(f"Env {i}: joint_delta_norm = {joint_delta_norm[i].item():.4f}, "
            #             f"max_joint_delta = {max_joint_delta[i].item():.4f}")

            # 4) scale only those rows back to max_delta_norm
            #    scale factor = max_delta_norm / joint_delta_norm
            scale = max_delta_norm / joint_delta_norm[mask]         # (num_bad,)
            delta[mask, :7] = delta[mask, :7] * scale.unsqueeze(1)   # broadcast to (num_bad,7)

            # 5) apply
            joint_pos_des_arm = joint_pos + delta  # (E,13)

        joint_pos_des = torch.cat((joint_pos_des_arm, gripper_status), dim=-1)

        return joint_pos_des # (num_envs, 8)
    
    def _step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        # self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _reset_robot_state(self, env_ids, apply_dmr=False):
        # Reset qpos
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)

        # DMR for initial qpos
        if apply_dmr:
            # print("Applying DMR")
            joint_pos[:,:7] += sample_uniform( 
                                -0.08,
                                0.08,
                                (len(env_ids), 7),
                                self.device,
                            )
            joint_pos = torch.clamp(joint_pos, self.qpos_lower_limits, self.qpos_upper_limits)

            randomized_stiffness = self._robot.data.default_joint_stiffness[env_ids, :7].clone() * sample_uniform( 
                                    0.95,
                                    1.05,
                                    (len(env_ids), 7),
                                    self.device,
                                )
            randomized_damping = self._robot.data.default_joint_damping[env_ids, :7].clone() * sample_uniform(
                                    0.95,
                                    1.05,
                                    (len(env_ids), 7),
                                    self.device,
                                )
            self._robot.write_joint_stiffness_to_sim(randomized_stiffness, [0,1,2,3,4,5,6], env_ids=env_ids)
            self._robot.write_joint_damping_to_sim(randomized_damping, [0,1,2,3,4,5,6], env_ids=env_ids)
        
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids=env_ids)

        self._step_sim_no_action()
        

    def _reset_assets(self, env_ids, apply_dmr=False):
        nut_root = self._nut.data.default_root_state.clone() # (num_envs, 13)
        pick_up_pose_10D = self._get_nut_pick_up_poses(self.training_demo_traj[env_ids, self.demo_idx[env_ids]]) # (env_ids, 10)
        nut_root[env_ids,:3] = pick_up_pose_10D.clone()[:,:3] + self.scene.env_origins[env_ids,:3]
        nut_root[env_ids,3:7] = quat_from_6d(pick_up_pose_10D.clone()[:,3:9])
        nut_root[env_ids, 2] += 0.01

        # visualize pick up pose
        # if self.cfg.debug_intermediate_values:
        #     self.marker7.visualize(nut_root[:,:3] + self.scene.env_origins, nut_root[:,3:7]) # visualize nut pick up pose

        base_root = self._base.data.default_root_state.clone()
        base_root[env_ids,:3] += self.scene.env_origins[env_ids,:3]
        base_root[env_ids,2] += 0.003

        if apply_dmr:
            nut_root[env_ids,0] += sample_uniform(-0.03, 0.03, len(env_ids), self.device) #x 
            nut_root[env_ids,1] += sample_uniform(-0.03, 0.03, len(env_ids), self.device) #y
            base_root[env_ids,0] += sample_uniform(-0.02, 0.02, len(env_ids), self.device) #x
            base_root[env_ids,1] += sample_uniform(-0.02, 0.02, len(env_ids), self.device) #y

        nut_root[env_ids,:3] = torch.clamp(nut_root[env_ids,:3], self.nut_low[env_ids,:3], self.nut_high[env_ids,:3])
        base_root[env_ids,:3] = torch.clamp(base_root[env_ids,:3], self.base_low[env_ids,:3], self.base_high[env_ids,:3])
        
        self._nut.write_root_state_to_sim(root_state=nut_root, env_ids=env_ids) #NOTE: no render on reset
        self._nut.reset(env_ids)
        self._base.write_root_state_to_sim(root_state=base_root, env_ids=env_ids) #NOTE: no render on reset
        self._base.reset(env_ids)

    def _reset_buffers(self, env_ids, augment_data = False):
        # Reset time step
        self.time_step_per_env[env_ids] = 0

        # Reset demo idx
        self.demo_idx[env_ids] = (self.demo_idx[env_ids] + 1) % self.cfg.num_demos 

        if not augment_data:
            self.training_demo_traj = self.clean_demo_trajs
        else:
            # 1) sample burst [s,e]
            min_len, max_len = 10, self.cfg.traj_length // 2
            length      = int(torch.randint(min_len, max_len+1, (1,), device=self.device).item())
            noise_start = int(torch.randint(0, self.traj_length - length + 1, (1,), device=self.device).item())
            noise_end   = noise_start + length - 1

            # 2) sample noise params
            noise_level = float(torch.rand(1).item() * (0.04 - 0.02) + 0.02)  # smaller
            beta_filter = float(torch.rand(1).item() * (0.9 - 0.5) + 0.5)     # stays <1

            # 3) sample step_interval ∈ [5,30] but ≤ length
            low, high = 10, 30
            hi = min(high, length)
            lo = low if length >= low else 1
            step_interval = int(torch.randint(lo, hi+1, (1,), device=self.device).item())
            self.training_demo_traj = add_local_correlated_noise(trajectories=self.clean_demo_trajs, 
                                                                env_ids=env_ids, 
                                                                step_interval=step_interval, 
                                                                noise_level=noise_level, 
                                                                beta=beta_filter,
                                                                noise_start=noise_start,
                                                                noise_end=noise_end,
                                                                )

    def _update_training_traj(self, env_ids):
        # # update initial pose of demo trajs
        # ee after dmr
        initial_fingertip = self.fingertip_10D.clone() # (num_envs, 10)
        # interpolated from randomized ee to demo traj            
        initial_traj = interpolate_10d_ee_trajectory(initial_fingertip[env_ids], self.training_demo_traj[env_ids, self.demo_idx[env_ids], 20, :], 20) # (env_ids, 1, 20, 10)
        # fill in initial traj
        self.training_demo_traj[env_ids, self.demo_idx[env_ids], :20, :] = initial_traj.clone()

        # compute intended goals
        self.intended_goal_binary = self.detect_goal_from_gripper_release_binary(self.training_demo_traj[env_ids, self.demo_idx[env_ids]], self.insertion_goals)
        self.intended_goal_pose = self.insertion_goals.clone()[self.intended_goal_binary.long()] # (num_envs, 9)

        if self.cfg.debug_intermediate_values:
            self.marker5.visualize(self.intended_goal_pose[:,:3] + self.scene.env_origins[env_ids,:3], self.intended_goal_pose[:,3:7]) # visualize nut pick up pose

        # reset teleop hist and initialize with initial pose
        self.teleop_hist.initialize(initial_fingertip.clone()[env_ids], env_ids=env_ids) # (num_envs, 10)


    def _get_asset_offset(self, object: RigidObject):
        """
        call this method after reseting the asset into default root state to get the 7D offset
            T_intended = T_offset * T_actual
        """

        intended_pose = object.data.body_link_state_w.clone()[:, 0, :7]
        actual_pose = object.data.body_com_state_w.clone()[:, 0, :7]

        print("link pose: ", intended_pose)
        print("actual pose: ", actual_pose)

        offset_p, offset_q = compute_offset(intended_pose, actual_pose) # (num_envs, 3), (num_envs, 4)
        print("offset_p, offset_q: ", offset_p, offset_q)

    def detect_goal_from_gripper_release_binary(
        self,
        trajectories: torch.Tensor,
        goals: torch.Tensor,
        env_ids: torch.Tensor = None,  # type: ignore
    ) -> torch.Tensor:
        """
        Args:
            trajectories: Tensor of shape (num_envs, timesteps, action_dim)
            goals: Tensor of shape (2, 9) — [x, y, z, 6D orientation]
            env_ids: Optional (num_selected_envs,) — indices of environments to compute

        Returns:
            Tensor of shape (num_envs,), containing:
            - 0 for envs where release was closer to goals[0]
            - 1 for closer to goals[1]
            - -1 if no release happened
        """
        num_envs, timesteps, action_dim = trajectories.shape
        gripper_action = trajectories[..., -1]  # (num_envs, timesteps)

        if env_ids is None:
            env_ids = torch.arange(num_envs, device=trajectories.device)

        selected_gripper = gripper_action[env_ids]       # (N, T)
        selected_actions = trajectories[env_ids]         # (N, T, D)

        # Detect release transition: 1 → -1
        gripper_prev = selected_gripper[:, :-1]
        gripper_next = selected_gripper[:, 1:]
        release_mask = (gripper_prev == 1) & (gripper_next == -1)
        release_idx = torch.argmax(release_mask.int(), dim=1)  # (N,)
        has_release = release_mask.any(dim=1)

        # Position at release
        release_pos = selected_actions[torch.arange(env_ids.shape[0]), release_idx, :3]  # (N, 3)

        goal_positions = goals[:, :3]  # (2, 3)
        dists = torch.norm(release_pos[:, None, :] - goal_positions[None, :, :], dim=-1)  # (N, 2)
        closer_to_goal0 = dists[:, 0] < dists[:, 1]  # (N,)

        result = -1 * torch.ones(num_envs, device=trajectories.device, dtype=torch.long)
        result[env_ids[has_release]] = torch.where(closer_to_goal0[has_release], 0, 1)

        if (result == -1).any():
            print("[Warning] Some environments had no gripper release — result contains -1s")

        return result.float()

    
    def detect_goal_from_gripper_release_9d(
        self,
        trajectories: torch.Tensor,
        goals: torch.Tensor,
        env_ids: torch.Tensor = None, # type: ignore
    ) -> torch.Tensor:
        """
        Args:
            trajectories: Tensor of shape (num_envs, timesteps, action_dim)
            goals: Tensor of shape (2, 9) — [x, y, z, 6D orientation]
            env_ids: Optional (num_selected_envs,) — indices of environments to compute
        
        Returns:
            Tensor of shape (num_envs, 9), where non-selected envs are filled with zeros.
        """
        num_envs, timesteps, action_dim = trajectories.shape
        gripper_action = trajectories[..., -1]  # (num_envs, timesteps)

        if env_ids is None:
            env_ids = torch.arange(num_envs, device=trajectories.device)

        # Select relevant envs
        selected_gripper = gripper_action[env_ids]           # (N, T)
        selected_actions = trajectories[env_ids]             # (N, T, D)

        # Detect release transition: 1 → -1
        gripper_prev = selected_gripper[:, :-1]
        gripper_next = selected_gripper[:, 1:]
        release_mask = (gripper_prev == 1) & (gripper_next == -1)

        # First timestep of release
        release_idx = torch.argmax(release_mask.int(), dim=1)  # (N,)
        has_release = release_mask.any(dim=1)

        # Position at release: take the first 3 dimensions
        release_pos = selected_actions[torch.arange(env_ids.shape[0]), release_idx, :3]  # (N, 3)

        # release_pose_whole = selected_actions[torch.arange(env_ids.shape[0]), release_idx-1, :9]  # (N, 10)
        # print("release_pos: ", release_pose_whole)

        # Compare to both goal positions (first 3 dims of each goal)
        goal_positions = goals[:, :3]  # (2, 3)
        dists = torch.norm(release_pos[:, None, :] - goal_positions[None, :, :], dim=-1)  # (N, 2)
        closest_goal_idx = torch.argmin(dists, dim=-1)  # (N,)
        selected_goal_vals = goals[closest_goal_idx]  # (N, 9)

        # Zero out for environments that never released
        selected_goal_vals[~has_release] = 0.0

        # Populate full result
        output_goals = torch.zeros((num_envs, 9), device=trajectories.device)
        output_goals[env_ids] = selected_goal_vals

        return output_goals
