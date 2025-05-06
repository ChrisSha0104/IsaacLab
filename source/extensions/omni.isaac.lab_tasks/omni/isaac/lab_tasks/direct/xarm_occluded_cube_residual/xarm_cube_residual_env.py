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
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, combine_frame_transforms, quat_mul, matrix_from_quat
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
class XArmCubeResidualEnvCfg(DirectRLEnvCfg):
    # env 
    episode_length_s = 13.3333 # eps_len_s = traj_len * (dt * decimation)
    decimation = 4
    action_space: int = 10                          # [position, 6D orientation, gripper qpos]
    observation_space = 10 + 10 + 120*120           # [robot state, teleop action, depth image]
    state_space = 10 + 10 + 9 + 120*120             # [robot state, teleop action, depth image, object state, demo idx]
    rerender_on_reset = False
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, # NOTE: 30 Hz
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=12,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            # bounce_threshold_velocity=0.2,
            # friction_offset_threshold=0.01,
            # friction_correlation_distance=0.00625,
            # gpu_max_rigid_contact_count=2**23,
            # gpu_max_rigid_patch_count=2**23,
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
                max_depenetration_velocity=5.0,
                # linear_damping=0.0,
                # angular_damping=0.0,
                # max_linear_velocity=1000.0,
                # max_angular_velocity=3666.0,
                # enable_gyroscopic_forces=True,
                # solver_position_iteration_count=16,
                # solver_velocity_iteration_count=1,
                # max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,      # TODO: check if this caused the problem 
                solver_position_iteration_count=12, # TODO: check iteration count
                solver_velocity_iteration_count=1
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="acceleration"),                # TODO: check difference force vs acc   
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ), 
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={ # TODO: change to the initial pose corresponding to teleop initial EE
                "joint1": 0.0,
                "joint2": -np.pi / 4, #-45
                "joint3": 0.0,
                "joint4": np.pi / 6, # 30
                "joint5": 0.0,
                "joint6": np.pi / 12 * 5, # 75
                "joint7": 0.0,
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
                # effort_limit=50.0,
                # velocity_limit=3.14,
                stiffness=200,#80.0, 
                damping=20,#10.0,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper"], 
                # effort_limit=40.0,
                # velocity_limit=0.3,
                stiffness=1e4,#7500.0,
                damping=1e2,#173.0,
            ),
        },
    )

    # assets 
    cube = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.0, 0.0),
                rot=(0, 1, 0, 0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd", 
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    # enable_gyroscopic_forces=True,
                ),
                # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
        )   
    # cameras
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link7/cam", # TODO: ADD NOISE
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
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -------- training options -------- 
    training_data_path = "RRL/tasks/cube/training_set4"
    enable_vision = True
    enable_residual = True
    use_privilege_obs = True 
    apply_dmr = True           
    augment_real_data = True

    # -------- training params --------
    traj_length = 400
    num_demos = 50
    alpha = 0.25    # residual scale
    tilde = 1.0     # low pass filter
    num_samples = 5 # number of teleop samples to be used for training
    sample_interval = 4 # sample interval for teleop samples # TODO: increase to 5

    # -------- initialization --------
    fingertip_init_pose_10D = [0.2568, 0.005,  0.245, # unit: m         # NOTE: initial fingertip pose in sim & real
                                1.00, 0.00, 0.00, 0.00, -1.00, 0.00, 
                                -1.00]                                  # NOTE: 1 = open, -1 = close
    fingertip_lower_limit = [0.15, -0.4, 0.03, 
                            -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, 
                            -1.0]
    fingertip_upper_limit = [0.65, 0.4, 0.5, 
                            1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 
                            1.0]
    cube_lower_limit = [0.25, -0.2, 0.0, 
                        -1.05, -1.05, -1.05, -1.05, -1.05, -1.05]
    cube_upper_limit = [0.55, 0.2, 0.4, 
                        1.05, 1.05, 1.05, 1.05, 1.05, 1.05]
    state_obs_lower_limit = fingertip_lower_limit * (num_samples+1) 
    state_obs_upper_limit = fingertip_upper_limit * (num_samples+1) 
    minimum_height= 0.1

    privilege_obs_lower_limit = [0.0, # fingertip cube dist
                                 0.0, # episode length
                                 0.0, # demo idx
                                 0.0] # reached cube
    privilege_obs_upper_limit = [1.0, 400.0, float(num_demos), 1.0]

    # -------- visualization options -------- 
    show_camera = False
    debug_intermediate_values = False
    show_success_rate = False   
    order_demos = False

    # -------- sim2real options -------- 
    store_sim_trajectory = False
    storing_path = "RRL/tasks/cube/sim2real/traj4"

    # -------- rewards --------
    nres_penalty_scale = -1e-5 # implies desired nres norm = 1.0
    completion_reward_scale = 10.0

class XArmCubeResidualEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()

    # post-physics step calls
    #   |-- _get_dones()
    #   |  |-- _compute_intermediate_values()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XArmCubeResidualEnvCfg

    def __init__(self, cfg: XArmCubeResidualEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.teleop_fingertip_10D = torch.cat((self.teleop_fingertip_pos, self.teleop_fingertip_orn_6D, self.teleop_gripper_binary), dim=-1) # (num_envs, 10)
        self.teleop_pos = self.teleop_fingertip_pos[:,:3].clone() + tf_vector(self.teleop_fingertip_quat, -1*self.fingertip2ee_offset_sim) # (num_envs, 3)

        # cube states
        robot_root_w = self._robot.data.root_state_w[:].clone()
        self.cube_pos, self.cube_quat = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._cube.data.body_com_state_w[:,0,:3], self._cube.data.body_com_state_w[:,0,3:7]
        )
        self.cube_orn_6D = quat_to_6d(self.cube_quat)

        if self.cfg.debug_intermediate_values:
            self.marker1.visualize(self.fingertip_pos + self.scene.env_origins, self.fingertip_quat)
            self.marker4.visualize(self.ee_pos + self.scene.env_origins, self.fingertip_quat)
            self.marker2.visualize(self.teleop_fingertip_pos + self.scene.env_origins, self.teleop_fingertip_quat)
            self.marker3.visualize(self.cube_pos + self.scene.env_origins, self.cube_quat)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)

        if self.cfg.enable_vision:
            self._camera = TiledCamera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube
        if self.cfg.enable_vision:
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
        self.n_residual_10D = noutput.clone()[:,:10]

        if self.cfg.store_sim_trajectory:
            self.residual_list.append(self.n_residual_10D.clone())

        self.last_fingertip_goal_10D = self.fingertip_goal_10D.clone() # last fingertip action

        if self.cfg.enable_residual:
            nbase = self.action_normalizer.normalize(self.teleop_fingertip_10D.clone())
            n_fingertip_goal_10D = nbase + self.cfg.alpha * self.n_residual_10D.clone()
            self.fingertip_goal_10D = self.action_normalizer.denormalize(n_fingertip_goal_10D.clone()) 
        else: 
            self.fingertip_goal_10D = self.teleop_fingertip_10D.clone()

        fingertip_goal_10D_filtered = self.cfg.tilde * self.fingertip_goal_10D.clone() + (1-self.cfg.tilde) * self.last_fingertip_goal_10D.clone()
        fingertip_goal_10D_filtered[:,:3] = torch.clamp(fingertip_goal_10D_filtered[:,:3], self.action_low[:,:3], self.action_high[:,:3])
        fingertip_goal_10D_filtered[:,-1] = torch.where(fingertip_goal_10D_filtered[:, -1] > 0.0, 
                                                        torch.tensor(1.0, device=fingertip_goal_10D_filtered.device),  # closed
                                                        torch.tensor(-1.0, device=fingertip_goal_10D_filtered.device)) # open

        self.joint_pos = self.get_qpos_from_fingertip_10d(self.diff_ik_controller, fingertip_goal_10D_filtered, apply_smoothing=True)                                  # ee_goal always abs coordinates
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.qpos_lower_limits[:8], self.qpos_upper_limits[:8])       # (num_envs, 8)

    def _apply_action(self): # TODO: check this
        self._robot.set_joint_position_target(self.robot_dof_targets[:,:], joint_ids=[i for i in range(self.num_eff_joints)])

    def step(self, output):
        _return = super().step(output)
        if self.cfg.enable_vision:
            self._camera.update(dt=self.dt)

        self.time_step_per_env += 1
        return _return
    
    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        success = (self.cube_pos[:,2] > self.cfg.minimum_height)
        self.extras["success"] = success

        terminated = self.fingertip_pos[:,2] < 0.02
        terminated |= success

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | truncated | success
        self.episode_count += (done).long()
        self.success_count += (success).long()

        if self.cfg.store_sim_trajectory:
            if self.success_count.sum() > 0:
                os.makedirs(self.cfg.storing_path, exist_ok=True)
                state_tensor = torch.stack(self.state_list, dim=0).reshape(-1, 10)
                torch.save(state_tensor[1:], os.path.join(self.cfg.storing_path, "sim_state_obs.pt"))
                teleop_tensor = torch.stack(self.teleop_list, dim=0).reshape(-1, 10)
                torch.save(teleop_tensor[1:], os.path.join(self.cfg.storing_path, "sim_teleop_obs.pt"))
                cube_tensor = torch.stack(self.cube_list, dim=0).reshape(-1, 9)
                torch.save(cube_tensor[1:], os.path.join(self.cfg.storing_path, "sim_cube_obs.pt"))
                residual_tensor = torch.stack(self.residual_list, dim=0).reshape(-1, 10)
                torch.save(residual_tensor, os.path.join(self.cfg.storing_path, "sim_residual_output.pt"))
                if self.cfg.enable_vision:
                    depth_array = np.stack(self.depth_list, axis=0) 
                    np.save(os.path.join(self.cfg.storing_path, "sim_depth_obs.npy"), depth_array[3:])
                print(f"init cube pos: {cube_tensor[1, :3]}")
                print(f"init robot pos: {state_tensor[1]}")
                print(f"init teleop pos: {teleop_tensor[1]}")
                print(f"sim data stored at {self.cfg.storing_path}")
                exit()

        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _compute_rewards(self):
        # log:
        self.fingertip_cube_dist = torch.norm(self.fingertip_pos - self.cube_pos, dim=-1) # (num_envs, )     
        reached_cube = torch.where((self.fingertip_cube_dist < 0.05), 1.0, 0.0)

        # completion reward
        picked_cube = torch.where((self.cube_pos[:,2] > self.cfg.minimum_height), 1.0, 0.0)
        completion_reward = picked_cube

        # residual penalty
        nres_norm = torch.norm(self.n_residual_10D[:,:10], dim=-1)

        rewards = (
            self.cfg.completion_reward_scale * completion_reward
            + self.cfg.nres_penalty_scale * nres_norm
        )

        low_finger = torch.where(self.fingertip_pos[:,2] < 0.02, -1.0, 0.0)

        self.extras["log"] = { # TODO: residual size
            "fingertip_cube_dist": self.fingertip_cube_dist.mean(),
            "reached_cube": reached_cube.mean(),
            "picked_cube": picked_cube.mean(),
            # "low_finger": low_finger.mean(),
            "nres_norm": nres_norm.mean(),
            "nres_penalty_rew": self.cfg.nres_penalty_scale * nres_norm.mean(),
            "completion_reward": self.cfg.completion_reward_scale * completion_reward.mean(),
            "teleop_misalignment": torch.norm(self.fingertip_goal_10D - self.teleop_fingertip_10D, dim=-1).mean(),
            "cummulative_success_rate": self.success_count.sum().float() / self.episode_count.sum().clamp(min=1).float(),
        }

        if self.cfg.show_success_rate and self.episode_count.sum() % self.num_envs == 0:
            if self.episode_count.sum() > 0:
                print(f"episode count: {self.episode_count.sum()}")
                print(f"success count: {self.success_count.sum()}")
                print(f"overall success rate: {self.success_count.sum().float() / self.episode_count.sum().clamp(min=1).float()}")
                print(f"failed demos: {self.demo_idx[self.success_count == 0]}")

        return rewards
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        
        # reset time, demo counter, and training traj
        self._reset_buffers(env_ids, self.cfg.augment_real_data)
        
        # reset robot states
        self._reset_robot_state(env_ids, self.cfg.apply_dmr)

        # reset cube states based on new training traj
        self._reset_assets(env_ids, self.cfg.apply_dmr)

        # reset camera and controller
        if self.cfg.enable_vision:
            self._camera.reset(env_ids)
        self.diff_ik_controller.reset(env_ids) # type: ignore

        # step through physics
        self._step_sim_no_action()
        
        # recompute / update training traj based on randomized robot state
        self._update_training_traj(env_ids, self.cfg.augment_real_data)


    def _get_observations(self) -> dict:   
        if self.cfg.enable_vision:
            depth_clean = self._camera.data.output["distance_to_image_plane"].permute(0,3,1,2)
            depth_filtered = filter_sim_depth(depth_clean) # (num_envs, 120*120) which should give same reading as real
            depth_noised = add_noise_in_depth_band(depth_filtered)
            normalized_depth = normalize_depth_01(depth_noised)
            # plt.imshow(depth_noised.detach().cpu().numpy().reshape(120,120), cmap='gray')
            # plt.show()
        
        if self.num_envs == 1 and self.cfg.show_camera and self.cfg.enable_vision:
            depth_vis = filter_depth_for_visualization(depth_filtered.reshape(120,120).detach().cpu().numpy(), unit='m') # only works for 1 env
            cv2.imshow("depth_image", depth_vis)
            cv2.waitKey(1)
            # self.depth_list.append(depth_filtered.detach().cpu().numpy().reshape(120,120))

        self.teleop_hist.append(self.teleop_fingertip_10D.clone())
        teleop_hist = self.teleop_hist.get_history() # (num_envs, num_samples, 10)
        num_actor_obs = 10 * (1+self.cfg.num_samples)
        
        if self.cfg.enable_vision:
            actor_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.fingertip_orn_6D,
                    self.gripper_binary,
                    teleop_hist,
                ),
                dim=-1,
            )
            normalized_actor_obs = self.state_obs_normalizer.normalize(actor_obs) # TODO: clean this up !!!!
            normalized_actor_obs = torch.cat((normalized_actor_obs, normalized_depth), dim=-1)
        else:
            actor_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.fingertip_orn_6D,
                    self.gripper_binary,
                    teleop_hist,
                    self.cube_pos,
                    self.cube_orn_6D,
                ),
                dim=-1,
            )
            normalized_state_obs = self.state_obs_normalizer.normalize(actor_obs[:,:num_actor_obs])
            normalized_cube_obs = self.cube_normalizer.normalize(actor_obs[:,num_actor_obs:].clone())
            normalized_actor_obs = torch.cat((normalized_state_obs, normalized_cube_obs), dim=-1)

        if self.cfg.use_privilege_obs:
            fingertip_cube_dist = torch.norm(self.fingertip_pos - self.cube_pos, dim=-1)
            reached_cube = torch.where((fingertip_cube_dist < 0.05), 1.0, 0.0)
            critic_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.fingertip_orn_6D,
                    self.gripper_binary,
                    teleop_hist,
                    self.cube_pos,
                    self.cube_orn_6D,
                    fingertip_cube_dist.unsqueeze(-1),
                    self.episode_length_buf.float().unsqueeze(-1),
                    self.demo_idx.float().unsqueeze(-1),
                    reached_cube.unsqueeze(-1),
                ),
                dim=-1,
            )
            normalized_base_obs = self.state_obs_normalizer.normalize(critic_obs[:,:num_actor_obs].clone())
            normalized_cube_obs = self.cube_normalizer.normalize(critic_obs[:,num_actor_obs:-4].clone())
            normalized_privledge_obs = self.privilege_obs_normalizer.normalize(critic_obs[:,-4:].clone())

            if self.cfg.enable_vision:
                normalized_critic_obs = torch.cat((normalized_base_obs, normalized_cube_obs, normalized_privledge_obs, normalized_depth), dim=-1)
            else:
                normalized_critic_obs = torch.cat((normalized_base_obs, normalized_cube_obs, normalized_privledge_obs), dim=-1)
            
        # -------------------------------- sim2real ------------------------------
        if self.cfg.store_sim_trajectory:
            self.state_list.append(actor_obs[:,:10].clone())
            self.teleop_list.append(teleop_hist.clone())
            self.cube_list.append(torch.cat((self.cube_pos, self.cube_orn_6D), dim=-1).clone())
            if self.cfg.enable_vision:
                self.depth_list.append(depth_noised.detach().cpu().numpy().reshape(120,120))

        if self.cfg.use_privilege_obs:
            return {"policy": torch.clamp(normalized_actor_obs, -1.0, 1.0),
                    "critic": torch.clamp(normalized_critic_obs, -1.0, 1.0)}
        else:
            return {"policy": torch.clamp(normalized_actor_obs, -1.0, 1.0)}

    def _print_training_info(self):
        print(f"[Training Config]")
        print(f"Data path       : {self.cfg.training_data_path}")
        print(f"Policy type     : {'Vision' if self.cfg.enable_vision else 'State-based'}")
        print(f"Data augment    : {'Yes' if self.cfg.augment_real_data else f'No, using {self.cfg.num_demos} demos'}")
        print(f"DMR             : {'Enabled' if self.cfg.apply_dmr else 'Disabled'}")
        print(f"Residual policy : {'Enabled' if self.cfg.enable_residual else 'Disabled'}")
        print(f"Asymmetric AC   : {'Yes' if self.cfg.use_privilege_obs else 'No'}\n")

    def _init_buffers(self):
        # simulation time
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot buffers
        self.qpos_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.qpos_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.fingertip2ee_offset_sim = torch.tensor([[0.0, 0.0, 0.15]], device=self.device).repeat(self.num_envs, 1) # const distance between fingertip and ee # NOTE: real - sim = 5 mm offset
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
        self.episode_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # (num_envs, )
        self.success_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # (num_envs, )

        self.traj_length = self.cfg.traj_length
        self.fingertip_init_pose_10D = torch.tensor([self.cfg.fingertip_init_pose_10D], device=self.device).repeat(self.num_envs, 1) # (num_envs, 10)

        # (num_envs, num_demos, traj_length, action_dim)
        self.clean_demo_trajs = self._generate_clean_demo_traj(dir=self.cfg.training_data_path, num_demos=self.cfg.num_demos, init_pos=self.fingertip_init_pose_10D)
        self.training_demo_traj = self.clean_demo_trajs.clone()

        # normalize obs
        self.action_low = torch.tensor([self.cfg.fingertip_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.action_high = torch.tensor([self.cfg.fingertip_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.action_normalizer = ActionNormalizer(self.action_low, self.action_high)

        double_action_low = self.action_low.clone().repeat(1, 2)
        double_action_high = self.action_high.clone().repeat(1, 2)
        self.double_action_normalizer = ActionNormalizer(double_action_low, double_action_high)

        self.cube_low = torch.tensor([self.cfg.cube_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.cube_high = torch.tensor([self.cfg.cube_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.cube_normalizer = ActionNormalizer(self.cube_low, self.cube_high)

        obs_low = torch.tensor([self.cfg.state_obs_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        obs_high = torch.tensor([self.cfg.state_obs_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.state_obs_normalizer = ActionNormalizer(obs_low, obs_high)

        privilege_obs_low = torch.tensor([self.cfg.privilege_obs_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        privilege_obs_high = torch.tensor([self.cfg.privilege_obs_upper_limit], device=self.device).repeat(self.num_envs, 1)
        self.privilege_obs_normalizer = ActionNormalizer(privilege_obs_low, privilege_obs_high)

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
            self.cube_list = []
            self.depth_list = [] # NOTE: depth list is stored as np arrays

            self.residual_list = []

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

    def _generate_clean_demo_traj(self, dir: str, num_demos: int, init_pos: torch.Tensor):
        all_demos = []

        for i in range(1, num_demos+1):
            traj: torch.Tensor = torch.load(os.path.join(dir, f"demo_traj{i}.pt"), weights_only=True).to(device=self.device).unsqueeze(0)                      # (1, traj_length, action_dim)
            traj = traj.repeat(self.num_envs, 1, 1)                                                                                         # (num_envs, traj_length, action_dim)
            traj = resample_trajectory_10d(traj, self.traj_length, 10)                                                   # (num_envs, traj_length, action_dim)
            traj[..., -1] = torch.where(traj[..., -1] > 0.0, 
                                        torch.tensor(1.0, device=traj[..., -1].device),     # closed 
                                        torch.tensor(-1.0, device=traj[..., -1].device))    # open
            all_demos.append(traj)

        return torch.stack(all_demos, dim=1) # (num_envs, num_demos, traj_length, action_dim)
    
    def _get_cube_pick_up_poses(self, demo_traj: torch.Tensor):
        gripper_closed = demo_traj[..., -1] == 1.0  # shape: (num_envs, num_demos, timestep)
        first_closed_idx = gripper_closed.float().argmax(dim=-1)  # shape: (num_envs, num_demos)
        num_envs, num_demos, timestep, action_dim = demo_traj.shape

        env_idx = torch.arange(num_envs).unsqueeze(1).expand(-1, num_demos)  # (num_envs, num_demos)
        demo_idx = torch.arange(num_demos).unsqueeze(0).expand(num_envs, -1)  # (num_envs, num_demos)

        # Use advanced indexing
        selected_actions = demo_traj[env_idx, demo_idx, first_closed_idx]  # shape (num_envs, num_demos, action_dim)
        
        return selected_actions

    # def _create_filter_pairs(self, prim1: str, prim2: str):
    #     stage = get_current_stage()
    #     filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1)) # type: ignore
    #     filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
    #     filteredpairs_rel.AddTarget(prim2)
    #     stage.Save()

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

        finger_qpos = self._robot.data.joint_pos[:,8:9]#torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1) # TODO check mean here
        binary_gripper = torch.where(finger_qpos > 0.2, 
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


        if apply_smoothing:
            delta = joint_pos_des_arm - joint_pos  # (E,13)

            # 1) compute per‐env joint norms over the first 7 dims
            joint_delta_norm = delta[:, :7].norm(dim=1)             # (E,)
            max_joint_delta  = delta[:, :7].abs().max(dim=1).values  # (E,)

            # 2) find which envs exceed the threshold
            max_delta_norm = 0.10 # TODO: decide on this!!!
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
                                -0.125, # NOTE originally 0.125
                                0.125,
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
        cube_root = self._cube.data.default_root_state.clone()
        pick_up_pose_10D = self._get_cube_pick_up_poses(self.training_demo_traj)
        pick_pose = pick_up_pose_10D[torch.arange(self.num_envs, device=self.device), self.demo_idx, :9].clone()

        cube_root[:,:2] = pick_pose.clone()[:,:2]
        cube_root[:,3:7] = quat_from_6d(pick_pose.clone()[:,3:9])

        if apply_dmr:
            cube_root[env_ids,0] += sample_uniform(-0.03, 0.03, len(env_ids), self.device) #x 
            cube_root[env_ids,1] += sample_uniform(-0.03, 0.03, len(env_ids), self.device) #y

        cube_root[:,:3] = torch.clamp(cube_root[:,:3], self.cube_low[:,:3], self.cube_high[:,:3]) # (num_envs, 3)
        cube_root[env_ids,:3] += self.scene.env_origins[env_ids,:3]

        self._cube.write_root_state_to_sim(root_state=cube_root, env_ids=env_ids) #NOTE: no render on reset
        self._cube.reset(env_ids)

    def _reset_buffers(self, env_ids, augment_data = False):
        # Reset time step
        self.time_step_per_env[env_ids] = 0

        # Reset demo idx
        self.demo_idx[env_ids] = (self.demo_idx[env_ids] + 1) % self.cfg.num_demos 

        # Reset episode count
        if self.episode_count.sum() >= self.num_envs*2:
            self.episode_count[:] = 0
            self.success_count[:] = 0

        if not augment_data:
            self.training_demo_traj = self.clean_demo_trajs
        else:
            step_interval = int(torch.randint(20, 41, (1,)).item())
            pos_noise = torch.rand(1).item() * (0.06 - 0.04) + 0.05
            beta_filter = torch.rand(1).item() * (0.9 - 0.7) + 0.7
            self.training_demo_traj = add_correlated_noise_vectorized(trajectories=self.clean_demo_trajs, 
                                                                        env_ids=env_ids, 
                                                                        step_interval=step_interval, 
                                                                        noise_level=pos_noise, 
                                                                        beta=beta_filter) # ee traj

    def _update_training_traj(self, env_ids, add_noise=True):
        # # update initial pose of demo trajs
        # ee after dmr
        initial_fingertip = self.fingertip_10D.clone() # (num_envs, 10)
        # interpolated from randomized ee to demo traj            
        initial_traj = interpolate_10d_ee_trajectory(initial_fingertip[env_ids], self.training_demo_traj[env_ids, self.demo_idx[env_ids], 50, :], 50) # (env_ids, 1, 50, 10)
        # fill in initial traj
        self.training_demo_traj[env_ids, self.demo_idx[env_ids], :50, :] = initial_traj.clone()

        # reset teleop hist and initialize with initial pose
        self.teleop_hist.initialize(initial_fingertip.clone()[env_ids], env_ids=env_ids) # (num_envs, 10)

