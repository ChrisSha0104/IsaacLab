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
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, quat_mul, matrix_from_quat
from omni.isaac.lab.assets import AssetBaseCfg, AssetBase
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors import CameraCfg, Camera, TiledCameraCfg, TiledCamera
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils import convert_dict_to_backend
import cv2

from RRL.utilities import *

import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
import time
import random
from typing import Callable
# import omni.replicator.core as rep
import os



@configclass
class XArmCubeResidualCamLocalBinaryV4EnvCfg(DirectRLEnvCfg):
    # env 
    episode_length_s = 13.33333 # eps_len_s = traj_len * (dt * decimation)
    decimation = 4
    action_space: int = 10                          # [position, 6D orientation, gripper qpos]
    observation_space = 16 + 10 + 120*120           # [robot state, teleop action, depth image]
    state_space = 16 + 10 + 120*120 + 7 + 1       # [robot state, teleop action, depth image, object state, demo idx]
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
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/shuosha/projects/IsaacLab/RRL/robot/xarm/xarm7_with_gripper.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, # TODO: whether enable gravity
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),                  
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": -0.785, #-45
                "joint3": 0.0,
                "joint4": 0.52, # 30
                "joint5": 0.0,
                "joint6": 1.31, # 75
                "joint7": 0.0,
                "drive_joint": 0.0,
                "left_finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit=50.0,
                velocity_limit=3.14,
                stiffness=80.0, 
                damping=10.0,
            ),
            "upper_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-5]"],
                effort_limit=30.0,
                velocity_limit=3.14,
                stiffness=30.0,
                damping=5.0,
            ),
            "forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint[6-7]"],
                effort_limit=20.0,
                velocity_limit=3.14,
                stiffness=10.0,
                damping=2.0,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"], 
                # effort_limit=200.0,
                # velocity_limit=0.2,
                stiffness=2e3, # TODO
                damping=1e2,
            ),
        },
    )
    

    # cube
    cube = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.0, 0.0), # rel traj v2 m{1]
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
        prim_path="/World/envs/env_.*/Robot/link_eef/cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.09, 0.0, 0.05), rot=(0.866, 0.0, -0.3, 0.0), convention="ros"), # z-down; x-forward
        height=120,
        width=120,
        data_types=[
            # "rgb",
            "distance_to_image_plane",
            # "normals",
            # "semantic_segmentation",
            # "instance_segmentation_fast",
            # "instance_id_segmentation_fast",
            ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.315, focus_distance=400.0, horizontal_aperture=24, clipping_range=(0.01, 20) # NOTE: unit in cm
        ),
    )

    # camera_intrinsics_info = [
    #     21.315/24*120,
    #     21.315/24*120,
    #     120/2,
    #     120/2,
    # ]
    # cam_int = torch.zeros((3, 3), device="cuda")
    # cam_int[0, 0] = camera_intrinsics_info[0]
    # cam_int[1, 1] = camera_intrinsics_info[1]
    # cam_int[0, 2] = camera_intrinsics_info[2]
    # cam_int[1, 2] = camera_intrinsics_info[3]
    # cam_int[2, 2] = 1.0

    # cam2eef = torch.eye(4, device="cuda")
    # R = matrix_from_quat(torch.tensor([0.866, 0.0, -0.3, 0.0], device="cuda"))
    # pos = torch.tensor([0.09, 0.0, 0.05], device="cuda")
    # cam2eef[:3, :3] = R
    # cam2eef[:3, 3] = pos

    # print("camera transform")
    # print(eef2camera)

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

    # visualization options:
    show_demo = True
    clean_demo = True
    debug_ee = False
    debug_joint_pos = False
    mark_ee = False
    mark_demo = False
    show_camera = True                  # option only used for play
    store_obs = False
    
    # training options:
    learn_std = True
    use_privilege_obs = True
    apply_dmr = False
    num_demos = 2
    state_history_length = 50
    use_relative_coordinates = False
    train_encoder = True
    
    alpha = 0.2 # residual scale
    tilde = 0.5 # low pass filter
    rel_action_scale = 5
    dof_velocity_scale = 0.1
    minimal_height = 0.05
    std = 0.1

    # reward scale RESIDUAL
    # --- task-completion ---
    completion_scale = 30.0
    ee_dist_reward_scale = 0.3 
    height_reward_scale = 0.1

    # --- auxiliary ---
    residual_penalty_scale = -0.05
    # action_penalty_scale = -0.3
    ee_rate_scale = -0.2
    residual_rate_scale = -0.1
    velocity_penalty_scale = -0.05
    jerk_penalty_scale = -0.2 # -0.5
    gripper_height_scale = -10.0
    # contact_force_scale = -0.001 # reduce?

class XArmCubeResidualCamLocalBinaryV4Env(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XArmCubeResidualCamLocalBinaryV4EnvCfg

    def __init__(self, cfg: XArmCubeResidualCamLocalBinaryV4EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        if self.cfg.debug_ee or self.cfg.debug_joint_pos or self.cfg.mark_demo or self.cfg.show_demo:
            print("--------DEBUG MODE--------")
            if self.cfg.show_demo:
                if self.cfg.clean_demo:
                    print("SHOWING CLEAN DEMO TRAJECTORY")
                else:
                    print("SHOWING NOISY DEMO TRAJECTORY")
        else:
            print("--------TRAINING MODE--------")
        
        frame_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)                                           # type: ignore
        self.demo_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/finger"))   # type: ignore
        self.ee_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object"))   # type: ignore

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_position_upper_limits = torch.tensor([[0.65, 0.5, 0.5]], device=self.device)
        self.robot_position_lower_limits = torch.tensor([[0.15, -0.5, 0.17]], device=self.device)

        self.num_eff_joints = self._robot.num_joints - 5
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_eff_joints), device=self.device)

        self.joint_ids = list(range(self._robot.num_joints))
        self.body_ids = list(range(self._robot.num_bodies))
        self.last_action = self._robot.data.default_joint_pos[:, :8].clone()
        self.last_dacc = torch.zeros((self.num_envs, 10), device=self.device)
        self.ee_residual = torch.zeros((self.num_envs, 10), device=self.device)
        self.last_ee_residual = self.ee_residual.clone()

        self.action_list = []

        self.teleop_comm_hist = HistoryBuffer(self.num_envs, self.cfg.state_history_length, self.cfg.action_space, device=self.device) # (num_envs, state_history_length, action_space)
        self.robot_state_hist = HistoryBuffer(self.num_envs, self.cfg.state_history_length, self.cfg.action_space+6, device=self.device) # (num_envs, state_history_length, action_space+6)
        self.ee_hist = HistoryBuffer(self.num_envs, 3, self.cfg.action_space, device=self.device) # (num_envs, 3, action_space)

        # create time-based indexing for demo trajectories
        self.time_step_per_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # (num_envs, )
        self.demo_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # which demo is each env current on, shaped (num_envs, )
        self.env_idx = torch.arange(self.num_envs, device=self.device) # (num_envs, )

        # load demo traj
        self.traj_length = 400
        self.demo_traj = torch.zeros((self.num_envs, self.cfg.num_demos, self.traj_length, self.cfg.action_space), device=self.device) # (num_envs, num_demos, traj_length, action_space)
        self.training_demo_traj = self.demo_traj.clone()
        self.init_pos = torch.tensor([[0.256, 0.00,  0.399, 1.00, 0.00, 0.00, 0.00, -1.00, 0.00, 0.00]], device=self.device).repeat(self.num_envs, 1) # (num_envs, 10)
        self.init_state = torch.tensor([[0.256, 0.00,  0.399, 1.00, 0.00, 0.00, 0.00, -1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]], device=self.device).repeat(self.num_envs, 1) # (num_envs, 10)

        self.teleop_comm_obs = self.init_pos.clone() # most recent rel teleop ee command for each env (num_envs, 10)
        self.last_ee = self.init_pos.clone()
        self.ee_goal = self.init_pos.clone()

        self.obs_hist = HistoryBuffer(self.num_envs, 60, 16+10+120*120, device=self.device) # (num_envs, state_history_length, obs_space_size)
 
        real_t_sim = matrix_from_quat(torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device))
        all_demos = []
        for i in range(1,self.cfg.num_demos+1):
            # traj = self.postprocess_real_demo_trajectory(f"/home/shuosha/Desktop/rl_cube_demos_real/rel_traj_mode{i}/robot")
            traj = postprocess_real_demo_trajectory_6D(real_t_sim, f"/home/shuosha/projects/IsaacLab/RRL/demo_trajs/rel_traj_v2_m{i}/robot") # (1, traj_length, action_dim)
            traj = traj.repeat(self.num_envs, 1, 1) # (num_envs, traj_length, action_dim)
            initial_traj = interpolate_10d_ee_trajectory(self.init_pos, traj[:,0,:], 50)
            traj = torch.cat((initial_traj, traj), dim=1)
            traj = resample_trajectory_10d(traj, self.traj_length, self.cfg.action_space)
            traj[..., -1] = (traj[..., -1] > 0.2).float() # convert gripper qpos to binary
            all_demos.append(traj)
            
        self.demo_traj = torch.stack(all_demos, dim=1)

        # traj = postprocess_real_demo_trajectory_test(real_t_sim, f"/home/shuosha/Desktop/robot") # (1, traj_length, action_dim)
        # traj = traj.repeat(self.num_envs, 1, 1) # (num_envs, traj_length, action_dim)
        # initial_traj = interpolate_10d_ee_trajectory(self.init_pos, traj[:,0,:], 50)
        # traj = torch.cat((initial_traj, traj), dim=1)
        # traj = resample_trajectory_10d(traj, self.traj_length, self.cfg.action_space)
        # traj[..., -1] = (traj[..., -1] > 0.2).float() # convert gripper qpos to binary
        # all_demos.append(traj)
            
        # self.demo_traj = torch.stack(all_demos, dim=1)

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        self.collected = []

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)
        self._camera = TiledCamera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # filter finger collision
        self.create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_outer_knuckle")
        self.create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_outer_knuckle")
        self.create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_finger")
        self.create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_finger")

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) 
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, ee_residual: torch.Tensor):
        # for RP, the arg actions is the residual!
        self.last_ee_residual = self.ee_residual.clone()
        self.ee_residual = ee_residual.clone()
        # print("ee residual")
        # format_tensor(self.ee_residual)
        # print("demo ee")
        # format_tensor(self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx] + self.init_ee)
        
        self.last_ee = self.ee_goal.clone()

        if self.cfg.show_demo:
            self.ee_goal = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :]
        else: 
            self.ee_goal = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :] + self.cfg.alpha * self.ee_residual

        ee_goal_filtered = self.cfg.tilde * self.ee_goal.clone() + (1-self.cfg.tilde) * self.last_ee.clone()

        ee_goal_filtered[:,:3] = torch.clamp(ee_goal_filtered[:,:3], self.robot_position_lower_limits, self.robot_position_upper_limits)
        ee_goal_filtered[:,-1] = (ee_goal_filtered[:,-1] > 0.5).float()

        # print(ee_goal_filtered)

        self.ee_hist.append(ee_goal_filtered.clone())

        if self.cfg.debug_ee:
                print("ee goal")
                format_tensor(ee_goal_filtered)
                print("current ee")
                format_tensor(self.get_curr_waypoint_b())
        if self.cfg.mark_ee:
                quat = quat_from_6d(ee_goal_filtered[:,3:9])
                self.ee_goal_marker.visualize(ee_goal_filtered[:,:3] + self.scene.env_origins[:,:3], quat)          

        # print("gripper status")
        # format_tensor(ee_goal_filtered[:,7])
        # import pdb; pdb.set_trace()
        self.joint_pos = self.get_joint_pos_from_ee_pos_10d(self.diff_ik_controller, ee_goal_filtered)                           # ee_goal always abs coordinates
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.robot_dof_lower_limits[:8], self.robot_dof_upper_limits[:8])   # (num_envs, 8)
        self.time_step_per_env += 1

        self.action_list.append(ee_goal_filtered.clone())
        # if self.time_step_per_env == 399:
        #     actions = np.concatenate([t.detach().cpu().numpy() for t in self.action_list], axis=0)  # shape: (100, 7)

        #     time_steps = np.arange(399)

        #     # Create 10 subplots (one for each acti dim)
        #     fig, axs = plt.subplots(10, 1, figsize=(10, 14), sharex=True)

        #     # Loop through each acti dim (0 to 9)
        #     for dim in range(10):
        #         axs[dim].plot(time_steps, actions[:, dim], label=f'action over time')
        #         axs[dim].set_ylabel(f'action dim {dim+1}')
        #         if dim == 0:
        #             axs[dim].legend(loc='upper right')
                    
        #     axs[-1].set_xlabel('Time Step')
        #     plt.tight_layout()
        #     plt.suptitle('Actions over Time w/ Low-pass Filter lambda=7 & smoothing reward', fontsize=16)
        #     plt.show()

    def _apply_action(self): # TODO: check this
        # apply action for arm and drive joint
        self._robot.set_joint_position_target(self.robot_dof_targets[:,:-1], joint_ids=[i for i in range(self.num_eff_joints-1)])

        # self.gripper_force = torch.norm(self._contact_sensor1.data.net_forces_w - self._contact_sensor2.data.net_forces_w)
        self.finger_joint_dif = self._robot.data.joint_pos[:,7:].max(dim=1).values - self._robot.data.joint_pos[:,7:].min(dim=1).values

        finger_target = self.robot_dof_targets[:,-1].unsqueeze(1)#.repeat(1,6)

        # modify envs with invalid finger commands
        invalid_env_ids =  self.finger_joint_dif > 0.01 #| self.gripper_force > 30 #NOTE: stick to 0.05, smaller value -> cube slips
        finger_target[invalid_env_ids] = torch.mean(self._robot.data.joint_pos[invalid_env_ids,7:], dim=1).unsqueeze(1)#.repeat(1,6)
    
        self._robot.set_joint_position_target(finger_target, joint_ids=[7])

        if self.cfg.debug_joint_pos:
            print("robot joint pose")
            format_tensor(self._robot.data.joint_pos)
            print("robot joint pose target")
            format_tensor(self._robot.data.joint_pos_target)        

    def step(self, ee_residual):
        _return = super().step(ee_residual)

        # update demo wp
        self.teleop_comm_obs = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :] # (num_envs, 10)

        if self.cfg.mark_demo:
            quat = quat_from_6d(self.teleop_comm_obs[:,3:9])
            self.demo_ee_marker.visualize(self.teleop_comm_obs[:, :3] - self.scene.env_origins[:,:3], quat)

        self._camera.update(dt=self.dt)
        return _return

    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_y = self._cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        cube_x = self._cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        terminated = (cube_y > 0.2) | (cube_y < -0.2) | (cube_x > 0.6) | (cube_x < 0.1)
        # terminated |= (self._robot.data.joint_pos[:,7:].min(dim=1).values < 0)

        self.finger_joint_dif = self._robot.data.joint_pos[:,7:].max(dim=1).values - self._robot.data.joint_pos[:,7:].min(dim=1).values

        terminated = self._robot.data.body_com_state_w[:, 9, 2] < 0.165 #NOTE: 9th body is link_eef
        terminated |= self.finger_joint_dif > 0.25 #TODO: can't be a termination condition 

        # if self.finger_joint_dif > 0.05:
        #     self.cfg.debug_actions = True
        # else:
        #     self.cfg.debug_actions = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)            
        if len(self.collected) > 0:
            data = torch.cat(self.collected, dim=0)  # shape: (N,26)

            # Compute column-wise minimum and maximum.
            min_vals = torch.min(data, dim=0).values  # shape: (26,)
            max_vals = torch.max(data, dim=0).values  # shape: (26,)

            print("Minimum values for each dimension:", min_vals)
            print("Maximum values for each dimension:", max_vals)

        joint_pos = self._robot.data.default_joint_pos[env_ids, :] 
        if self.cfg.apply_dmr:
            print("Applying DMR")
            joint_pos[:,:7] += sample_uniform( 
                                -0.125,
                                0.125,
                                (len(env_ids), self._robot.num_joints-6),
                                self.device,
                            )
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self.cfg.clean_demo:
            self.training_demo_traj = self.demo_traj
        else:
            step_interval = int(torch.randint(20, 31, (1,)).item())
            noise_level = torch.rand(1).item() * (0.04 - 0.02) + 0.02
            beta_filter = torch.rand(1).item() * (0.7 - 0.5) + 0.5
            self.training_demo_traj = smooth_noisy_trajectory(self.demo_traj, env_ids, step_interval=step_interval, noise_level=noise_level, beta_filter=beta_filter)

        # cube state
        reseted_root_states = self._cube.data.default_root_state.clone()
        if self.cfg.apply_dmr:
            reseted_root_states[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            reseted_root_states[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y
        
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:3]

        self._cube.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids) #NOTE: no render on reset
        self._cube.reset(env_ids)

        # camera
        self._camera.reset(env_ids)

        # controller
        self.diff_ik_controller.reset(env_ids)

        self.demo_idx[env_ids] = (self.demo_idx[env_ids] + 1) % self.cfg.num_demos 
        self.time_step_per_env[env_ids] = 0
        self.robot_state_hist.clear_envs(env_ids)
        self.teleop_comm_hist.clear_envs(env_ids)

        # go through physics
        self.scene.update(dt=self.physics_dt)

        # import pdb; pdb.set_trace()

        self.teleop_comm_obs[env_ids,:] = self.training_demo_traj[env_ids, self.demo_idx[env_ids], 0, :]

        # self.init_ee = torch.concat((
        #     self._robot.data.body_com_state_w[env_ids, 9, :7],  torch.mean(self._robot.data.joint_pos[env_ids,7:], dim=1).unsqueeze(1)
        # ), dim=-1)
        # self.init_ee[env_ids,:3] -= self.scene.env_origins[env_ids,:3]
        # print("init ee")
        # format_tensor(self.init_ee)
        # import pdb; pdb.set_trace() 


    def _get_observations(self) -> dict:
        depth_clean = self._camera.data.output["distance_to_image_plane"].permute(0,3,1,2) # From (B, H, W, 1) to (B, 1, H, W)
        # rotated_depth = torch.rot90(depth_clean.reshape(-1,1,120,120), k=1, dims=(-2, -1))

        depth_rotated = torch.rot90(depth_clean, k=1, dims=(-2, -1))
        # save_tensor_as_txt(depth_rotated, "depth_clean")

        depth_noised = simulate_depth_noise(depth_rotated)
        depth_clamped = clamp_depth_01(depth_noised)

        depth_input = normalize_depth_01(depth_noised)
        # save_tensor_as_txt(depth_input.reshape(-1,1,120,120), "depth_input")

        # if True:
        #     depth = depth_noised.reshape(120,120).detach().cpu().numpy()              # convert to np array
        #     depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha = 255/0.5), cv2.COLORMAP_JET)
        #     depth_vis = cv2.resize(depth_vis,(480,480))
        #     cv2.imshow("depth_image",depth_vis)
        #     cv2.waitKey(1)

        robot_state_obs = self.get_robot_state_b() # reset issues
        self.robot_state_hist.append(robot_state_obs)
        prev_robot_state_obs = self.robot_state_hist.get_oldest_obs() 
        relative_robot_state = compute_relative_state(prev_robot_state_obs, robot_state_obs)

        teleop_comm_obs = self.teleop_comm_obs.clone()
        self.teleop_comm_hist.append(teleop_comm_obs)
        prev_teleop_comm_obs = self.teleop_comm_hist.get_oldest_obs()
        relative_teleop_comm = compute_relative_state(prev_teleop_comm_obs, teleop_comm_obs)

        print(robot_state_obs)
        print(teleop_comm_obs)
        import pdb; pdb.set_trace()

        # if self.time_step_per_env % 50 == 0:
        #     self.latest_depth = depth_clean.reshape(-1,1,120,120)
        
        # if True:
        #     depth = self.latest_depth.reshape(120,120).detach().cpu().numpy()
        #     depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / depth[depth < 15].max().item()), cv2.COLORMAP_JET)
        #     pts_3d_cam = transform_point_eef_to_cam(robot_state_obs[:, :3], self.cfg.eef2camera)
        #     pts_2d = project_points(pts_3d_cam, self.cfg.cam_int)
        #     img_with_points = visualize_points_on_image(pts_2d, depth_vis.copy(), color=(0, 255, 0), radius=4)
        #     img_with_points = cv2.resize(img_with_points, (480, 480))
        #     img_with_points = cv2.rotate(img_with_points, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     cv2.imshow("Projected 2D Points on Depth Map", img_with_points)
        #     cv2.waitKey(1)
        #     time.sleep(0.02)

        # if self.cfg.store_obs and self.num_envs == 1:
        #     obs_raw = torch.cat((robot_state_obs, teleop_comm_obs, depth_clean.reshape(1, -1)), dim=-1)
        #     self.obs_hist.append(obs_raw)
        #     if self.obs_hist.is_full():
        #         latest_obs = self.obs_hist.get_oldest_obs()
        #         obs_hist = self.obs_hist.get_history()
        #         depth = latest_obs[:, 26:].reshape(120,120).detach().cpu().numpy()
        #         depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / depth[depth < 15].max().item()), cv2.COLORMAP_JET)
                
        #         init_state_w = latest_obs[:, :16] # eef_0 in world

        #         for i in range(60):
        #             curr_state = obs_hist[:, i, :16] # eef_i in world
        #             curr_state_eef = compute_relative_state(init_state_w, curr_state) # eef_i in eef_0
        #             R_eef_i2eef_0 = sixd_to_rotation_matrix(curr_state_eef[:, 3:9])
        #             p_eef_i2eef_0 = curr_state_eef[:, :3]
        #             T_eef_i2eef_0 = torch.eye(4, device=self.device)
        #             T_eef_i2eef_0[:3, :3] = R_eef_i2eef_0
        #             T_eef_i2eef_0[:3, 3] = p_eef_i2eef_0
        #             T_eef_i2eef_0[3,3] = 1.0

        #             cam2eef = self.cfg.cam2eef.clone() # cam_0 in eef_0
                    
        #             # concat with 1
        #             pts_3d_eef0 = torch.cat((curr_state_eef[:,:3].clone(), torch.ones((1,1), device=self.device)), dim=1)
        #             pts_3d_cam = cam2eef.inverse() @ pts_3d_eef0.T # eef_i in cam_0
        #             pts_2d = project_points(pts_3d_cam.reshape(1,4)[:,:3], self.cfg.cam_int)

        #             # import pdb; pdb.set_trace()
        #             depth_vis = visualize_points_on_image(pts_2d, depth_vis, color=(0, 255, 0), radius=4)
                
        #         depth_vis = cv2.resize(depth_vis, (480, 480))
        #         depth_vis = cv2.rotate(depth_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #         cv2.imshow("Projected 2D Points on Depth Map", depth_vis)
        #         cv2.waitKey(1)
        #         time.sleep(0.02)
                
        # print("robot state")
        # format_tensor(robot_state_obs)
        # print("relative robot state")
        # format_tensor(relative_robot_state)

        # print("teleop comm")
        # format_tensor(teleop_comm_obs)
        # print("relative teleop comm")
        # format_tensor(relative_teleop_comm)
        
        curr_root_state = self._robot.data.root_state_w[:]
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            curr_root_state[:, 0:3], curr_root_state[:, 3:7], self._cube.data.body_com_state_w[:,0,:3], self._cube.data.body_com_state_w[:,0,3:7]
        )

        robot_state_min = torch.tensor([-0.1, -0.1, -0.1,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation 
                                        -0.5, -0.5, -0.5, # lin vel
                                        -1.0, -1.0, -1.0, # ang vel
                                        0.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) # TODO: check why x is so large
        

        robot_state_max = torch.tensor([0.1, 0.1, 0.1,  # position
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation 
                                        0.5, 0.5, 0.5, # lin vel
                                        1.0, 1.0, 1.0, # ang vel
                                        1.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) # TODO: check why x is so large
        
        teleop_comm_min = torch.tensor([-0.1, -0.1, -0.1,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation 
                                        0.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) # TODO: check why x is so large
        

        teleop_comm_max = torch.tensor([0.1, 0.1, 0.1,  # position
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation 
                                        1.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) # TODO: check why x is so large

        # if relative
        # - state
        #   - curr_rel_robot_state = curr_robot_state <--> hist[0]
        # - teleop cmd
        #   - curr_rel_teleop_cmd = curr_teleop_cmd <--> hist[0]
        # - norm
        #   - option 1: find min and max of the whole dataset
        #   - option 2: define min and max
        #   - option 3: scale [risky]

        # to compute transformation
        # - world_t_curr: 4x4
        # - world_t_init: 4x4
        # - init_t_curr = np.linalg.inv(world_t_init) @ world_t_curr

        # visualize inputs
        # - img (hist[0])
        # - camera extrinsic and intrinsic (hist[0])
        # - overlay with robot state (like DP)
        #   - robot states are in wrist frame
        # - overlay with teleop command (like DP)
        # - overlay with lin vel [hist[0]]
        # - a_t_b: b's pose in a frame
        # - a_t_c = a_t_b @ b_t_c

        standardized_robot_state_obs = (relative_robot_state - robot_state_min) / (robot_state_max - robot_state_min)
        standardized_teleop_comm_obs = (relative_teleop_comm - teleop_comm_min) / (teleop_comm_max - teleop_comm_min)

        actor_obs = torch.cat(
            (
                standardized_robot_state_obs,
                standardized_teleop_comm_obs,
                depth_input,
            ),
            dim=-1,
        )

        # self.collected.append(actor_obs.clone()[:,:26])

        # print("normalized robot state")
        # format_tensor(standardized_robot_state_obs)
        # print("normalized teleop comm")
        # format_tensor(standardized_teleop_comm_obs)

        if self.cfg.use_privilege_obs:
            critic_obs = torch.cat(
                (
                    standardized_robot_state_obs,
                    standardized_teleop_comm_obs,
                    cube_pos_b,
                    cube_quat_b,
                    self.demo_idx.unsqueeze(1),
                    depth_input,
                ),
                dim=-1,
            )

            return {"policy": torch.clamp(actor_obs, -5.0, 5.0),
                    "critic": torch.clamp(critic_obs, -5.0, 5.0)}
        else: 
            return {"policy": torch.clamp(actor_obs, -5.0, 5.0)}

    def _compute_rewards(self):
        # height of the cube
        height_reward = torch.where(self._cube.data.root_pos_w[:, 2] > self.cfg.minimal_height, 1.0, 0.0) #* (1+self._cube.data.root_pos_w[:, 2].clamp(0,0.3))
        # print(self._cube.data.root_pos_w[:, 2])
        # height_reward = self._cube.data.root_pos_w[:, 2].clamp(0.0,0.3) + (self._cube.data.root_pos_w[:, 2] > self.cfg.minimal_height)
        cube_y = self._cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        cube_x = self._cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        height_reward *= (cube_y < 0.3) * (cube_y > -0.3) * (cube_x < 0.6) * (cube_x > 0.1)

        # regularization on residuals
        residual_penalty = torch.norm(self.ee_residual, dim=-1) #TODO: change to norm?

        # reward for reaching cube
        curr_ee_pose_w = self._robot.data.body_com_state_w[:,9,0:7]
        curr_root_pose_w = self._robot.data.root_state_w[:, 0:7]

        # ee pose in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_pose_w[:, 0:3], curr_root_pose_w[:, 3:7], curr_ee_pose_w[:, 0:3], curr_ee_pose_w[:, 3:7]
            )
        
        ee_b = torch.cat([curr_ee_pos_b, curr_ee_quat_b], dim=-1)
        offset_gripper_position = subtract_z_in_baseframe_batch(ee_b, z_offset=0.14)
        ee_gripper_b = torch.cat([offset_gripper_position, curr_ee_quat_b], dim=-1)

        cube_position_b, cube_quaternion_b = subtract_frame_transforms(
            curr_root_pose_w[:, 0:3], curr_root_pose_w[:, 3:7], self._cube.data.body_com_state_w[:,0,:3], self._cube.data.body_com_state_w[:,0,3:7]
        )
        cube_pos_b = torch.cat([cube_position_b, cube_quaternion_b], dim=-1) # (num_envs, 7)

        # self.demo_ee_marker.visualize(cube_pos_b[:,:3], cube_pos_b[:,3:7])
        # self.ee_goal_marker.visualize(ee_gripper_b[:,:3], ee_gripper_b[:,3:7])

        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(cube_pos_b - ee_gripper_b, dim=1)
        ee_dist_reward = torch.exp(-object_ee_distance / self.cfg.std)
   
        # action smoothing
        delta_actions = torch.norm(self.joint_pos - self.last_action, p=2, dim=-1)
        self.last_action = self.joint_pos.clone()
        action_penalty = delta_actions #torch.sum(self.actions**2, dim=-1)

        delta_ee = torch.norm(self.ee_goal - self.last_ee, p=2, dim=-1)
        # self.last_ee = self.ee_goal.clone()

        # residual rate
        delta_residual = torch.norm((self.ee_residual.clone() - self.last_ee_residual.clone()), p=2, dim=-1)

        # jerk
        ee_hist = self.ee_hist.get_history()
        d_acc = ee_hist[:, 2, :] - 2*ee_hist[:, 1, :] + ee_hist[:, 0, :]
        jerk = torch.norm(d_acc[:,:9]-self.last_dacc[:,:9], p=2, dim=-1)
        jerk[self.time_step_per_env < 4] = 0.0
        self.last_dacc = d_acc.clone()
        # print("jerk", jerk)

        gripper_height = torch.where(self.ee_goal[:,2] < 0.17, 1.0, 0.0)
        # print(self.ee_goal[:,2])

        overgrasp = (self.finger_joint_dif > 0.05) * self.ee_goal[:,7]

        # gripper closing reward
        closing_gripper = torch.exp(self._robot.data.joint_pos[:, 7]).clamp(0.0,50.0)
        closing_gripper *= ((self._robot.data.body_com_state_w[:,9,2] < 0.20) | (self._cube.data.root_pos_w[:, 2] > self.cfg.minimal_height) | (self.time_step_per_env > 200))

        # completion reward at end of episode
        completion_condition = (self._cube.data.root_pos_w[:, 2] > 0.15) & (self.episode_length_buf >= self.max_episode_length - 1)
        completion = torch.where(completion_condition, 1.0, 0.0)
        # print("completion", completion)

        # velocity penalty
        velocity_penalty = torch.norm(self._robot.data.body_com_state_w[:,9,7:13], p=2, dim=1)

        # print(self.cfg.velocity_penalty_scale * velocity_penalty)

        link_incoming_forces = self._robot.root_physx_view.get_link_incoming_joint_force()[:, -6:, :3] # forces in gripper frame
        contact_force = torch.norm((link_incoming_forces[:, -2, :3] + link_incoming_forces[:, -3, :3]), dim=-1)  # NOTE: proper grasp has values around 0.1

        rewards = (
            + self.cfg.height_reward_scale * height_reward
            + self.cfg.residual_penalty_scale * residual_penalty
            + self.cfg.ee_dist_reward_scale * ee_dist_reward
            # + self.cfg.action_penalty_scale * action_penalty
            + self.cfg.ee_rate_scale * delta_ee
            + self.cfg.gripper_height_scale * gripper_height
            + self.cfg.completion_scale * completion
            + self.cfg.velocity_penalty_scale * velocity_penalty
            # + self.cfg.contact_force_scale * contact_force
            + self.cfg.jerk_penalty_scale * jerk
            + self.cfg.residual_rate_scale * delta_residual
            # + self.cfg.overgrasp * overgrasp
            # + self.cfg.closing_gripper * closing_gripper
        )

        # print("jerk", jerk)
        # print("ee rate", delta_ee)
        # print("velocity penalty", velocity_penalty)

        # rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.05, rewards + 0.25, rewards)
        # rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.1, rewards + 0.25, rewards)
        # rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.15, rewards + 1.0, rewards)

        self.extras["log"] = {
            "height_reward": (self.cfg.height_reward_scale * height_reward).mean(),
            "residual_penalty": (self.cfg.residual_penalty_scale * residual_penalty).mean(),
            "ee_dist_reward": (self.cfg.ee_dist_reward_scale * ee_dist_reward).mean(),
            # "action_penalty": (self.cfg.action_penalty_scale * action_penalty).mean(),
            "ee_rate": (self.cfg.ee_rate_scale * delta_ee).mean(),
            "gripper_height": (self.cfg.gripper_height_scale * gripper_height).mean(),
            "completion": (self.cfg.completion_scale * completion).mean(),
            "velocity_penalty": (self.cfg.velocity_penalty_scale * velocity_penalty).mean(),
            "jerk_penalty": (self.cfg.jerk_penalty_scale * jerk).mean(),
            "delta_residual": (self.cfg.residual_rate_scale * delta_residual).mean(),
            # "contact_force": (self.cfg.contact_force_scale * contact_force).mean(),
        }

        # print("ee rate", self.extras["log"]["ee_rate"])
        # print("vel pen", self.extras["log"]["velocity_penalty"])
        
        return rewards
    
    def create_filter_pairs(self, prim1: str, prim2: str):
        stage = get_current_stage()
        filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1))
        filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
        filteredpairs_rel.AddTarget(prim2)
        stage.Save()
    
    def get_curr_waypoint_b(self):
        # ee and root pose in world frame
        curr_ee_pos_w = self._robot.data.body_com_state_w[:,9,0:7]
        curr_root_pos_w = self._robot.data.root_state_w[:, 0:7]

        # import pdb; pdb.set_trace()

        # ee pos in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_pos_w[:, 0:3], curr_root_pos_w[:, 3:7], curr_ee_pos_w[:, 0:3], curr_ee_pos_w[:, 3:7]
            )
        curr_orient_6d = quat_to_6d(curr_ee_quat_b)
        
        # finger status
        curr_finger_status = torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1)
        curr_finger_status = (curr_finger_status > 0.2).float() # convert gripper qpos to binary
        
        curr_ee_pos_combined_b = torch.cat((curr_ee_pos_b, curr_orient_6d, curr_finger_status), dim=-1)
        return curr_ee_pos_combined_b
    
    def get_robot_state_b(self):
        # ee and root pose in world frame
        curr_state_w = self._robot.data.body_com_state_w[:,9,:]
        curr_root_state = self._robot.data.root_state_w[:]

        # ee pose in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_state[:, 0:3], curr_root_state[:, 3:7], curr_state_w[:, 0:3], curr_state_w[:, 3:7]
            )
        # import pdb; pdb.set_trace()
        curr_orient_6d = quat_to_6d(curr_ee_quat_b)
        
        # ee pose in base (local) frame
        curr_lin_vel_b, _ = subtract_frame_transforms(
                curr_root_state[:, 7:10], curr_root_state[:, 3:7], curr_state_w[:, 7:10], curr_state_w[:, 3:7]
            )
        
        curr_ang_vel_b, _ = subtract_frame_transforms(
                curr_root_state[:, 10:13], curr_root_state[:, 3:7], curr_state_w[:, 10:13], curr_state_w[:, 3:7]
            )

        curr_finger_status = torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1)
        curr_finger_status = (curr_finger_status > 0.2).float() # convert gripper qpos to binary

        curr_robot_state_b = torch.cat((curr_ee_pos_b, curr_orient_6d, curr_lin_vel_b, curr_ang_vel_b, curr_finger_status), dim=-1)

        return curr_robot_state_b

    def get_joint_pos_from_ee_pos(self, controller: DifferentialIKController, ee_goal):
        ee_abs_b = self.get_curr_waypoint_b() # (num_envs, 8)

        ik_commands = ee_goal[:, :-1] #(num_envs, 8)
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link_eef")[0][0]-1
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(ee_abs_b[:,:3], ee_abs_b[:,3:7], jacobian, joint_pos)

        finger_joint_pos_des = ee_goal[:, -1].unsqueeze(1).clone()
        
        joint_pos_des = torch.cat((joint_pos_des_arm, finger_joint_pos_des), dim=-1)

        return joint_pos_des
    

    def get_joint_pos_from_ee_pos_binary(self, controller: DifferentialIKController, ee_goal):
        ee_abs_b = self.get_curr_waypoint_b() # (num_envs, 8)

        ik_commands = ee_goal[:, :-1] #(num_envs, 8)
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link_eef")[0][0]-1
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(ee_abs_b[:,:3], ee_abs_b[:,3:7], jacobian, joint_pos)

        finger_joint_pos_des = ee_goal[:, -1].unsqueeze(1).clone()
        
        finger_joint_pos_des[finger_joint_pos_des > 0.42] = 0.5          # NOTE: close gripper
        finger_joint_pos_des[finger_joint_pos_des < 0.42] = 0.0          # NOTE: open gripper

        joint_pos_des = torch.cat((joint_pos_des_arm, finger_joint_pos_des), dim=-1)

        return joint_pos_des
    
    def get_joint_pos_from_ee_pos_10d(self, controller: DifferentialIKController, ee_goal):
        curr_ee_b = self.get_curr_waypoint_b() # (num_envs, 10)
        curr_ee_quat_b = quat_from_6d(curr_ee_b[:,3:9])

        ee_goal_quat = quat_from_6d(ee_goal[:,3:9])
        ee_goal_7d = torch.cat((ee_goal[:,:3], ee_goal_quat), dim=-1) # (num_envs, 7)
        ik_commands = ee_goal_7d #(num_envs, 7)
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link_eef")[0][0]-1
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(curr_ee_b[:,:3], curr_ee_quat_b, jacobian, joint_pos)

        gripper_status = ee_goal[:, -1].unsqueeze(1).clone()
        gripper_status[gripper_status > 0.5] = 0.5          # NOTE: close gripper
        gripper_status[gripper_status < 0.5] = 0.0          # NOTE: open gripper

        joint_pos_des = torch.cat((joint_pos_des_arm, gripper_status), dim=-1)

        return joint_pos_des