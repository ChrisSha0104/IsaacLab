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
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
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
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=12, # TODO: check iteration count
                solver_velocity_iteration_count=1
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="acceleration"),                  
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
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
                # effort_limit=200.0,
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
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )   

    # cameras
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link7/cam", # TODO: TUNE OFFSET # TODO: ADD NOISE
        offset=TiledCameraCfg.OffsetCfg(pos=(0.13, 0.0, 0.008), rot=(0.9744,0,-0.2334,0), convention="ros"), # z-down; x-forward
        height=120,
        width=120,
        data_types=[
            "distance_to_image_plane",
            ],
        spawn=sim_utils.PinholeCameraCfg( # TODO: TUNE CAMERA PARAMETERS
            focal_length=21.315, focus_distance=400.0, horizontal_aperture=24, clipping_range=(0.01, 20) # NOTE: unit in cm
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
    training_data_path = "RRL/tasks/cube/training_set1"
    enable_vision = False
    enable_residual = False
    use_privilege_obs = False
    apply_dmr = False
    augment_real_data = False

    # -------- training params --------
    traj_length = 400
    num_demos = 5
    alpha = 0.1 # residual scale
    tilde = 1.0 # low pass filter

    # -------- initialization --------
    init_10D_pose = [0.2568, 0.005,  0.4005, # unit: m
                     1.00, 0.00, 0.00, 0.00, -1.00, 0.00, 
                     0.00] # TODO: change gripper to -1 and 1
    action_lower_limit = [0.15, -0.4, 0.03, 
                          -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, 
                          0.0]
    action_upper_limit = [0.55, 0.4, 0.5, 
                          1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 
                          1.0]
    cube_lower_limit = [0.15, -0.3, 0.0, 
                        -1.05, -1.05, -1.05, -1.05, -1.05, -1.05]
    cube_upper_limit = [0.55, 0.3, 0.5, 
                        1.15, 1.15, 1.15, 1.15, 1.15, 1.15]
    obs_lower_limit = action_lower_limit + action_lower_limit + cube_lower_limit
    obs_upper_limit = action_upper_limit + action_upper_limit + cube_upper_limit
    minimum_height= 0.1
    

    # -------- actor critic params -------- 
    use_visual_encoder = enable_vision
    learn_std = True
    visual_idx_actor = [20,20+120*120]
    visual_idx_critic = [20,20+120*120]

    # -------- visualization options -------- 
    show_camera = True
    debug_intermediate_values = True
    order_demos = False

    # -------- sim2real options -------- 
    store_sim_trajectory = False
    storing_path = "RRL/tasks/cube/sim2real/traj1"

    # -------- rewards --------
    completion_reward_scale = 1.0

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
        self.ee_pos, self.ee_quat, self.ee_orn_6D, self.gripper_binary = self.get_robot_state_info_b()
        self.ee_10D = torch.cat((self.ee_pos, self.ee_orn_6D, self.gripper_binary), dim=-1)

        # teleop commands
        self.teleop_pos, self.teleop_quat, self.teleop_orn_6D, self.teleop_gripper_binary = self.get_teleop_state_info_b()
        self.teleop_10D = torch.cat((self.teleop_pos, self.teleop_orn_6D, self.teleop_gripper_binary), dim=-1) # (num_envs, 10)

        # print("ee 10d")
        # format_tensor(self.ee_10D)
        # print("teleop 10d")
        # format_tensor(self.teleop_10D)

        # fingertip positions
        self.fingertip_pos = self.ee_pos + tf_vector(self.ee_quat, self.finger_offset_ee_fr) # TODO: check if this is correct
        self.fingertip_10D = torch.cat((self.fingertip_pos, self.ee_orn_6D, self.gripper_binary), dim=-1) # (num_envs, 10)

        self.teleop_fingertip_pos = self.teleop_pos + tf_vector(self.teleop_quat, self.finger_offset_ee_fr)
        self.teleop_fingertip_10D = torch.cat((self.teleop_fingertip_pos, self.teleop_orn_6D, self.teleop_gripper_binary), dim=-1) # (num_envs, 10)

        # print("fingertip 10d")
        # format_tensor(self.fingertip_10D)
        # print("teleop fingertip 10d")
        # format_tensor(self.teleop_fingertip_10D)

        # cube states
        robot_root_w = self._robot.data.root_state_w[:].clone()
        self.cube_pos, self.cube_quat = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._cube.data.body_com_state_w[:,0,:3], self._cube.data.body_com_state_w[:,0,3:7]
        )
        self.cube_orn_6D = quat_to_6d(self.cube_quat)

        if self.cfg.debug_intermediate_values:
            self.marker1.visualize(self.fingertip_pos + self.scene.env_origins, self.ee_quat)
            self.marker4.visualize(self.ee_pos + self.scene.env_origins, self.ee_quat)
            # self.marker2.visualize(self.teleop_fingertip_pos + self.scene.env_origins, self.teleop_quat)
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

        # filter finger collision # TODO debug gripper
        self._create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_outer_knuckle")
        self._create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_outer_knuckle")
        self._create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_finger")
        self._create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_finger")

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) 
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # pre-physics step calls
    def _pre_physics_step(self, noutput: torch.Tensor):
        n_residual_10D = noutput.clone()[:,:10]

        if self.cfg.store_sim_trajectory:
            self.residual_list.append(n_residual_10D.clone())

        # TODO: autonomy_param = noutput.clone()[:,-1:] # TODO: check if 10 dim is better

        self.last_fingertip_goal_10D = self.fingertip_goal_10D.clone() # last fingertip action

        if self.cfg.enable_residual:
            nbase = self.action_normalizer.normalize(self.teleop_fingertip_10D.clone())
            n_fingertip_goal_10D = nbase + self.cfg.alpha * n_residual_10D
            self.fingertip_goal_10D = self.action_normalizer.denormalize(n_fingertip_goal_10D.clone()) # negative finger goal!!!!
        else: 
            self.fingertip_goal_10D = self.teleop_fingertip_10D.clone()

        # print("noutput: ", noutput)
        # print("teleop finger", self.teleop_fingertip_10D)
        # print("clean goal: ", self.fingertip_goal_10D)

        # print("base action: ", self.teleop_fingertip_10D)
        # print("action with residual: ", self.fingertip_goal_10D)

        fingertip_goal_filtered_10D = self.cfg.tilde * self.fingertip_goal_10D.clone() + (1-self.cfg.tilde) * self.last_fingertip_goal_10D.clone()
        fingertip_goal_filtered_10D[:,:3] = torch.clamp(fingertip_goal_filtered_10D[:,:3], self.action_low[:,:3], self.action_high[:,:3])
        fingertip_goal_filtered_10D[:,-1] = (fingertip_goal_filtered_10D[:,-1] > 0.5).float() # TODO: change to -1,1, i,e gripper > 0
        # print("actual goal: ", fingertip_goal_filtered_10D)

        self.joint_pos = self.get_qpos_from_fingertip_10d(self.diff_ik_controller, fingertip_goal_filtered_10D, apply_smoothing=True)                                  # ee_goal always abs coordinates
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.robot_dof_lower_limits[:8], self.robot_dof_upper_limits[:8])       # (num_envs, 8)

        # print("qpos", self.robot_dof_targets)    

    def _apply_action(self): # TODO: check this
        # apply action for arm and drive joint
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
        terminated = self.fingertip_pos[:,2] < 0.02

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.store_sim_trajectory:
            if len(self.state_list) == self.traj_length + 2:
                os.makedirs(self.cfg.storing_path, exist_ok=True)
                state_tensor = torch.stack(self.state_list, dim=0).reshape(-1, 10)
                torch.save(state_tensor[2:], os.path.join(self.cfg.storing_path, "sim_state_obs.pt"))
                teleop_tensor = torch.stack(self.teleop_list, dim=0).reshape(-1, 10)
                torch.save(teleop_tensor[2:], os.path.join(self.cfg.storing_path, "sim_teleop_obs.pt"))
                cube_tensor = torch.stack(self.cube_list, dim=0).reshape(-1, 9)
                torch.save(cube_tensor[2:], os.path.join(self.cfg.storing_path, "sim_cube_obs.pt"))
                residual_tensor = torch.stack(self.residual_list, dim=0).reshape(-1, 10)
                torch.save(residual_tensor, os.path.join(self.cfg.storing_path, "sim_residual_output.pt"))
                if self.cfg.enable_vision:
                    depth_tensor = torch.stack(self.depth_list, dim=0).reshape(-1, 120*120)
                    np.save(os.path.join(self.cfg.storing_path, "sim_depth_obs.npy"), depth_tensor.detach().cpu().numpy())
                print(f"init cube pos: {cube_tensor[2, :3]}")
                print(f"init robot pos: {state_tensor[2]}")
                print(f"init teleop pos: {teleop_tensor[2]}")
                print(f"sim data stored at {self.cfg.storing_path}")
                exit()

        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _compute_rewards(self):
        self.fingertip_cube_dist = torch.norm(self.fingertip_pos - self.cube_pos, dim=-1) # (num_envs, )     
        reached_cube = torch.where((self.fingertip_cube_dist < 0.05)
                                  & (self.episode_length_buf >= self.max_episode_length - 1), 1.0, 0.0) # extend for longer time
        picked_cube = torch.where((self.cube_pos[:,2] > self.cfg.minimum_height)
                                  & (self.episode_length_buf >= self.max_episode_length - 1), 1.0, 0.0) 
        completion_reward = reached_cube + 2 * picked_cube

        rewards = (
            self.cfg.completion_reward_scale * completion_reward
        )

        low_finger = torch.where(self.fingertip_pos[:,2] < 0.02, -1.0, 0.0)

        self.extras["log"] = {
            "fingertip_cube_dist": self.fingertip_cube_dist.mean(),
            "reached_cube": reached_cube.mean(),
            "picked_cube": picked_cube.mean(),
            "low_finger": low_finger.mean(),
        }

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
            normalized_depth = normalize_depth_01(depth_filtered)
        
        if self.num_envs == 1 and self.cfg.show_camera and self.cfg.enable_vision:
            depth_vis = filter_depth_for_visualization(depth_filtered.reshape(120,120).detach().cpu().numpy(), unit='m') # only works for 1 env
            cv2.imshow("depth_image", depth_vis)
            cv2.waitKey(1)
            # self.depth_list.append(depth_filtered.detach().cpu().numpy().reshape(120,120))
        
        if self.cfg.enable_vision:
            actor_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.ee_orn_6D,
                    self.gripper_binary,
                    self.teleop_fingertip_pos,
                    self.teleop_orn_6D,
                    self.teleop_gripper_binary,
                ),
                dim=-1,
            )
            normalized_actor_obs = self.double_action_normalizer.normalize(actor_obs)
            normalized_actor_obs = torch.cat((normalized_actor_obs, normalized_depth), dim=-1)
        else:
            actor_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.ee_orn_6D,
                    self.gripper_binary,
                    self.teleop_fingertip_pos,
                    self.teleop_orn_6D,
                    self.teleop_gripper_binary,
                    self.cube_pos,
                    self.cube_orn_6D,
                ),
                dim=-1,
            )
            normalized_actor_obs = self.state_obs_normalizer.normalize(actor_obs)

        if self.cfg.use_privilege_obs and self.cfg.enable_vision:
            critic_obs = torch.cat(
                (
                    self.fingertip_pos,
                    self.ee_orn_6D,
                    self.gripper_binary,
                    self.teleop_fingertip_pos,
                    self.teleop_orn_6D,
                    self.teleop_gripper_binary,
                    self.cube_pos,
                    self.cube_orn_6D,
                ),
                dim=-1,
            )
            normalized_state_obs = self.double_action_normalizer.normalize(critic_obs[:,:20].clone())
            normalized_cube_obs = self.cube_normalizer.normalize(critic_obs[:,20:].clone())
            normalized_critic_obs = torch.cat((normalized_state_obs, normalized_cube_obs, normalized_depth), dim=-1)
            
        # -------------------------------- sim2real ------------------------------
        if self.cfg.store_sim_trajectory:
            self.state_list.append(actor_obs[:,:10].clone())
            self.teleop_list.append(actor_obs[:,10:20].clone())
            self.cube_list.append(actor_obs[:,20:].clone())
            if self.cfg.enable_vision:
                self.depth_list.append(depth_filtered.detach().cpu().numpy().reshape(120,120))

        if self.cfg.use_privilege_obs and self.cfg.enable_vision:
            return {"policy": torch.clamp(normalized_actor_obs, -1.0, 1.0),
                    "critic": torch.clamp(normalized_critic_obs, -1.0, 1.0)}
        else:
            return {"policy": torch.clamp(normalized_actor_obs, -1.0, 1.0)}

    def check_obs_bounds(self, obs: torch.Tensor, threshold: float = 1.0, tol: float = 1e-5):
        """
        obs: (num_envs, obs_dim) — normalized observations
        threshold: the clipping threshold (e.g., 1.0)
        tol: floating point tolerance (e.g., 1e-5)
        """
        over_max = obs > (threshold + tol)
        under_min = obs < (-threshold - tol)
        mask = over_max | under_min  # (num_envs, obs_dim)

        if mask.any():
            env_ids, dim_ids = torch.where(mask)
            for env, dim in zip(env_ids.tolist(), dim_ids.tolist()):
                print(f"Env {env}, Obs dim {dim}, Value: {obs[env, dim].item():.6f}")

    def _print_training_info(self):
        print(f"Loading data from {self.cfg.training_data_path} \n")
        if self.cfg.enable_vision:
            print(f"Using vision policy \n")
        else:
            print(f"Using state-based policy \n")
        
        if self.cfg.augment_real_data:
            print(f"Adding noise to demonstrations \n")
        else:
            print(f"Using clean demonstrations \n")
        
        if self.cfg.apply_dmr:
            print(f"Applying DMR \n")
        else:
            print(f"Not applying DMR \n")

        if self.cfg.enable_residual:
            print(f"Enabled residual \n")
        else:
            print(f"Disabled residual\n")

    def _init_buffers(self):
        # simulation time
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot buffers
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # self.robot_position_upper_limits = torch.tensor([[0.65, 0.5, 0.5]], device=self.device)
        # self.robot_position_lower_limits = torch.tensor([[0.15, -0.5, 0.18]], device=self.device)
        self.finger_offset_ee_fr = torch.tensor([[0.0, 0.0, 0.15]], device=self.device).repeat(self.num_envs, 1) # const distance between gripper and ee
        self.gripper_base_offset_ee_fr = torch.tensor([[0.0, 0.0, -0.026]], device=self.device).repeat(self.num_envs, 1) # offset between sim ee and real ee, (I.E, SAME QPOS PRODUCES SAME EE POSE IN SIM AND REAL)

        self.num_eff_joints = 8#self._robot.num_joints - 5
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
        self.init_10D_pose = torch.tensor([self.cfg.init_10D_pose], device=self.device).repeat(self.num_envs, 1) # (num_envs, 10)

        # (num_envs, num_demos, traj_length, action_dim)
        self.demo_traj = self._generate_clean_demo_traj(dir=self.cfg.training_data_path, num_demos=self.cfg.num_demos, init_pos=self.init_10D_pose)
        self.training_demo_traj = self.demo_traj.clone()

        # normalize obs
        self.action_low = torch.tensor([self.cfg.action_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.action_high = torch.tensor([self.cfg.action_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.action_normalizer = ActionNormalizer(self.action_low, self.action_high)

        double_action_low = self.action_low.clone().repeat(1, 2)
        double_action_high = self.action_high.clone().repeat(1, 2)
        self.double_action_normalizer = ActionNormalizer(double_action_low, double_action_high)

        self.cube_low = torch.tensor([self.cfg.cube_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.cube_high = torch.tensor([self.cfg.cube_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.cube_normalizer = ActionNormalizer(self.cube_low, self.cube_high)

        obs_low = torch.tensor([self.cfg.obs_lower_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        obs_high = torch.tensor([self.cfg.obs_upper_limit], device=self.device).repeat(self.num_envs, 1) # (num_envs, action_dim)
        self.state_obs_normalizer = ActionNormalizer(obs_low, obs_high)

        self.fingertip_goal_10D = self.init_10D_pose.clone() # (num_envs, 10)

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
            traj: torch.Tensor = load_from_txt(os.path.join(dir, f"demo_traj{i}.txt"), return_type="torch").unsqueeze(0).to(self.device) # (num_lens, action_dim) # type: ignore
            traj = traj.repeat(self.num_envs, 1, 1) # (num_envs, traj_length, action_dim)
            initial_traj = interpolate_10d_ee_trajectory(init_pos, traj[:,0,:], 50)
            traj = torch.cat((initial_traj, traj), dim=1)
            traj = resample_trajectory_10d(traj, self.traj_length, self.cfg.action_space) # (num_envs, traj_length, action_dim)
            traj[..., -1] = (traj[..., -1] > 0.2).float() # convert gripper qpos to binary
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

    def _create_filter_pairs(self, prim1: str, prim2: str):
        stage = get_current_stage()
        filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1)) # type: ignore
        filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
        filteredpairs_rel.AddTarget(prim2)
        stage.Save()

    def get_robot_state_info_b(self):
        """
        return: 
            get current robot state in the base frame
            tuple: ee_pos, ee_orn_quat, ee_orn_6D, finger_status
        """
        # ee and root pose in world frame
        ee_pose_w = self._robot.data.body_com_state_w[:,7,:].clone()
        ee_pose_w[:,:3] = ee_pose_w[:,:3] + tf_vector(ee_pose_w[:,3:7], self.gripper_base_offset_ee_fr)
        
        root_pose_w = self._robot.data.root_state_w[:]

        # ee pose in base (local) frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
        orient_6d = quat_to_6d(ee_quat_b)

        finger_status = torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1) # TODO check mean here
        binary_gripper = (finger_status > 0.2).float() # convert gripper qpos to binary # TODO: use same metric as IK

        return ee_pos_b, ee_quat_b, orient_6d, binary_gripper
    
    def get_robot_ee_7D_b(self):
        ee_pos, ee_quat, _, _ = self.get_robot_state_info_b()
        ee_7D_b = torch.cat((ee_pos, ee_quat), dim=-1) # (num_envs, 7)
        return ee_7D_b
    
    def get_robot_ee_10D_b(self):
        ee_pos, _, ee_orn_6D, finger_status = self.get_robot_state_info_b()
        ee_10D_b = torch.cat((ee_pos, ee_orn_6D, finger_status), dim=-1) # (num_envs, 10)
        return ee_10D_b
    
    def get_teleop_state_info_b(self):
        """
        return:
            get current teleop state in the base frame
            tuple: teleop_pos, teleop_quat, teleop_orn_6D, finger_status
        """
        teleop_10D = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :]
        teleop_pos = teleop_10D[:,:3].clone()
        teleop_quat = quat_from_6d(teleop_10D[:,3:9].clone())
        teleop_orn_6D = teleop_10D[:,3:9].clone()
        teleop_finger_status = teleop_10D[:,-1].unsqueeze(1).clone()

        return teleop_pos, teleop_quat, teleop_orn_6D, teleop_finger_status

    def get_teleop_ee_10D_b(self):
        teleop_pos, _, teleop_orn_6D, finger_status = self.get_teleop_state_info_b()
        teleop_10D_b = torch.cat((teleop_pos, teleop_orn_6D, finger_status), dim=-1)
        return teleop_10D_b

    def get_teleop_ee_7D_b(self):
        teleop_pos, teleop_quat, _, _ = self.get_teleop_state_info_b()
        teleop_7D_b = torch.cat((teleop_pos, teleop_quat), dim=-1)
        return teleop_7D_b
    
    def get_qpos_from_fingertip_10d(self, controller: DifferentialIKController, fingertip_10D: torch.Tensor, apply_smoothing=False):
        ee_goal_quat = quat_from_6d(fingertip_10D[:,3:9])
        ee_goal_pos = fingertip_10D[:,:3].clone() + tf_vector(ee_goal_quat, -1 * self.finger_offset_ee_fr)
        ee_goal_7d = torch.cat((ee_goal_pos, ee_goal_quat), dim=-1) # (num_envs, 7)
        ik_commands = ee_goal_7d #(num_envs, 7)
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link7")[0][0]-1 # or link7?
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(self.ee_pos, self.ee_quat, jacobian, joint_pos)

        gripper_status = fingertip_10D[:, -1:].clone() # binary gripper + residual output
        gripper_status[gripper_status > 0.5] = 1.7          # NOTE: close gripper in qpos
        gripper_status[gripper_status < 0.5] = 0.0          # NOTE: open gripper in qpos


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
                                -0.125, # NOTE originally 0.125
                                0.125,
                                (len(env_ids), 7),
                                self.device,
                            )
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

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
        reset_cube_pos = pick_pose[:,:3].clone() + tf_vector(quat_from_6d(pick_pose[:,3:9]), self.finger_offset_ee_fr) # (num_envs, 3)

        cube_root[:,:3] = reset_cube_pos[:,:3]

        if apply_dmr:
            cube_root[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            cube_root[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y

        cube_root[:,:3] = torch.clamp(cube_root[:,:3], self.cube_low[:,:3], self.cube_high[:,:3]) # (num_envs, 3)
        cube_root[env_ids,:3] += self.scene.env_origins[env_ids,:3]

        self._cube.write_root_state_to_sim(root_state=cube_root, env_ids=env_ids) #NOTE: no render on reset
        self._cube.reset(env_ids)

    def _reset_buffers(self, env_ids, augment_data = False):
        # Reset time step
        self.time_step_per_env[env_ids] = 0

        # Reset demo idx
        self.demo_idx[env_ids] = (self.demo_idx[env_ids] + 1) % self.cfg.num_demos 

        if not augment_data:
            self.training_demo_traj = self.demo_traj
        else:
            step_interval = int(torch.randint(20, 41, (1,)).item())
            pos_noise = torch.rand(1).item() * (0.04 - 0.02) + 0.02
            beta_filter = torch.rand(1).item() * (0.9 - 0.7) + 0.7
            self.training_demo_traj = add_correlated_noise_vectorized(trajectories=self.demo_traj, 
                                                                        env_ids=env_ids, 
                                                                        step_interval=step_interval, 
                                                                        noise_level=pos_noise, 
                                                                        beta=beta_filter) # ee traj

    def _update_training_traj(self, env_ids, add_noise=True):
        # # update initial pose of demo trajs
        # ee after dmr
        initial_ee = self.teleop_10D.clone() # (num_envs, 10)
        # interpolated from randomized ee to demo traj            
        initial_traj = interpolate_10d_ee_trajectory(initial_ee[env_ids], self.training_demo_traj[env_ids, self.demo_idx[env_ids], 50, :], 50) # (env_ids, 1, 50, 10)
        # fill in initial traj
        self.training_demo_traj[env_ids, self.demo_idx[env_ids], :50, :] = initial_traj.clone()

