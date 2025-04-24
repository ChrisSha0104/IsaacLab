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
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, quat_mul
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
import copy
import pdb
import time
import random
from typing import Callable
# import omni.replicator.core as rep
import os



@configclass
class XArmCubeResidualCamLocalBinaryEnvCfg(DirectRLEnvCfg):
    # env 
    episode_length_s = 6.6666 #3.31 ~ 200 timesteps  # 8.3333 = 500 timesteps #TODO: reduce episode length if needed
    decimation = 2
    action_space: int = 8 # ee waypoint
    observation_space = 3*8 + 4*8 + 120*120 # last 3 wp + last 3 expert wp
    state_space = 3*8 + 4*8 + 120*120 + 7 + 1
    rerender_on_reset = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
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
                disable_gravity=True,
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
            # pos=(0.0, 0.0, 0.0),
            # rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                # effort_limit=87.0,
                # velocity_limit=2.175,
                stiffness=2000.0,
                damping=573,
            ),
            "upper_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-5]"],
                # effort_limit=87.0,
                # velocity_limit=2.175,
                stiffness=1000.0,
                damping=286.5,
            ),
            "forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint[6-7]"],
                # effort_limit=12.0,
                # velocity_limit=2.61,
                stiffness=500.0,
                damping=114.6,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"], 
                # effort_limit=200.0,
                # velocity _limit=0.2,
                stiffness=1000.0, # TODO
                damping=100.0,
            ),
        },
    )

    # cube
    cube = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                # pos=(0.36, -0.01, 0.0), # rel traj mode i
                pos=(0.34, 0.0, 0.0), # rel traj v2 m{1]
                # pos=(0.22, -0.24, 0),  # eps 0000
                # pos=(0.43, -0.19, 0),  # real_demo_traj1
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
    show_demo = False
    clean_demo = False
    debug_ee = False
    debug_joint_pos = False
    mark_ee = False
    mark_demo = False
    show_camera = True # option only used for play
    
    # training options:
    use_privilege_obs = True
    intervene_step = 0 # NOTE: not used
    apply_dmr = True # NOTE: DMR turned off during play
    num_demos = 2
    use_relative_coordinates = True
    train_encoder = True

    alpha = 0.1
    rel_action_scale = 5
    dof_velocity_scale = 0.1
    minimal_height = 0.05
    std = 0.1

    # reward scales stage 1
    # --- task-completion ---
    ee_dist_reward_scale = 0.2 
    height_reward_scale = 1.0
    gripper_height = -10.0

    # --- auxiliary ---
    residual_penalty_scale = -5.0
    action_penalty = -0.25
    # overgrasp = -0.5

    # # reward scales stage 2
    # # --- task-completion ---
    # ee_dist_reward_scale = 0.2 
    # height_reward_scale = 1.0
    # gripper_height = -10.0

    # # --- auxiliary ---
    # residual_penalty_scale = -5.0
    # action_penalty = -0.5
    # overgrasp = -0.5

class XArmCubeResidualCamLocalBinaryEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XArmCubeResidualCamLocalBinaryEnvCfg

    def __init__(self, cfg: XArmCubeResidualCamLocalBinaryEnvCfg, render_mode: str | None = None, **kwargs):
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
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # type: ignore
        self.demo_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/finger")) # type: ignore
        self.ee_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object")) # type: ignore

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

        self.robot_position_upper_limits = torch.tensor([[0.55, 0.5, 0.5]], device=self.device)
        self.robot_position_lower_limits = torch.tensor([[0.15, -0.5, 0.17]], device=self.device)

        self.num_eff_joints = self._robot.num_joints - 5
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_eff_joints), device=self.device)

        self.joint_ids = list(range(self._robot.num_joints))
        self.body_ids = list(range(self._robot.num_bodies))
        self.last_action = self._robot.data.default_joint_pos[:, :8].clone()

        # init obs buffer for robot and demo history waypoints
        self.robot_wp_obs = torch.zeros((self.num_envs, 8), device=self.device).repeat(1,3)
        self.demo_wp_obs = torch.zeros((self.num_envs, 8), device=self.device).repeat(1,4)

        # create time-based indexing for demo trajectories
        self.time_step_per_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.demo_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long).unsqueeze(1).repeat(1, 8)
        self.offsets = torch.arange(8, device=self.device)
        self.env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).repeat(1, 8)

        # compute frame transform from real to sim
        quat_real = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)  # Real-world quaternion
        quat_sim = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32, device=self.device)  # Simulator quaternion
        t_real = torch.zeros((1, 3), device=self.device)  # No translation
        t_sim = torch.zeros((1, 3), device=self.device)
        _, self.quat_diff = subtract_frame_transforms(t_real, quat_real, t_sim, quat_sim)

        # load demo traj
        self.traj_length = 400
        self.demo_traj = torch.zeros((self.num_envs, self.cfg.action_space*self.traj_length, self.cfg.num_demos), device=self.device)
        self.init_ee = torch.tensor([[ 0.2558, -0.0054,  0.3961,  0.0250,  0.9995, -0.0018,  0.0171, 0.0]], device=self.device).repeat(self.num_envs, 1) # NOTE: Hardcoded: obtained after initialization

        all_demos = []
        for i in range(1,self.cfg.num_demos+1):
            # traj = self.postprocess_real_demo_trajectory(f"/home/shuosha/Desktop/rl_cube_demos_real/rel_traj_mode{i}/robot")
            traj = postprocess_real_demo_trajectory(self.quat_diff, f"/home/shuosha/projects/IsaacLab/RRL/demo_trajs/rel_traj_v2_m{i}/robot")
            traj = traj.repeat(self.num_envs, 1)
            initial_traj = interpolate_ee_trajectory(self.init_ee, traj[:,:8], 50)
            traj = torch.cat((initial_traj, traj), dim=1)
            traj = resample_trajectory(traj, self.traj_length)
            all_demos.append(traj)
            
        self.demo_traj_abs = torch.stack(all_demos, dim=-1)
        self.demo_traj_rel = abs_to_rel_traj(self.init_ee, self.demo_traj_abs)
        # export_tensor_to_txt(self.demo_traj_rel[0,:,0].reshape(1,400,8), "/home/shuosha/Desktop/new_traj.txt")
        # exit()

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

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
        self.action_idx = self.time_step_per_env.unsqueeze(1)*8 + self.offsets
        # for RP, the arg actions is the residual!
        self.ee_residual = self.cfg.alpha * ee_residual.clone()
        # print("ee residual")
        # format_tensor(ee_residual)
        
        if self.cfg.show_demo:# or self.time_step_per_env < self.cfg.intervene_step:
            if self.cfg.use_relative_coordinates:
                self.ee_goal = self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx] + self.init_ee 
            else:
                self.ee_goal = self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx]
        else: 
            if self.cfg.use_relative_coordinates:
                self.ee_goal = self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx] + self.init_ee + self.ee_residual
            else:
                self.ee_goal = self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx] + self.ee_residual

        ee_goal_filtered = self.ee_goal.clone() # NOTE: ee_goal always abs

        ee_goal_filtered[:,:3] = torch.clamp(ee_goal_filtered[:,:3], self.robot_position_lower_limits, self.robot_position_upper_limits)
        ee_goal_filtered[:,7] = torch.clamp(ee_goal_filtered[:,7], self.robot_dof_lower_limits[7], self.robot_dof_upper_limits[7])
        ee_goal_filtered[:,3:7] /= torch.norm(ee_goal_filtered[:,3:7], p=2, dim=1, keepdim=True)

        if self.cfg.debug_ee:
                print("ee goal")
                format_tensor(ee_goal_filtered)
                print("current ee")
                format_tensor(self.get_curr_waypoint_b()[:,:])
        if self.cfg.mark_ee:
                self.ee_goal_marker.visualize(ee_goal_filtered[:,:3] + self.scene.env_origins[:,:3], ee_goal_filtered[:,3:7])          

        # print("gripper status")
        # format_tensor(ee_goal_filtered[:,7])
          
        
        self.joint_pos = self.get_joint_pos_from_ee_pos_binary(self.diff_ik_controller, ee_goal_filtered) # ee_goal always abs coordinates
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.robot_dof_lower_limits[:8], self.robot_dof_upper_limits[:8]) # (num_envs, 8)
        self.time_step_per_env += 1

    def _apply_action(self):
        # apply action for arm and drive joint
        self._robot.set_joint_position_target(self.robot_dof_targets[:,:-1], joint_ids=[i for i in range(self.num_eff_joints-1)])

        # self.gripper_force = torch.norm(self._contact_sensor1.data.net_forces_w - self._contact_sensor2.data.net_forces_w)
        self.finger_joint_dif = self._robot.data.joint_pos[:,7:].max(dim=1).values - self._robot.data.joint_pos[:,7:].min(dim=1).values

        finger_target = self.robot_dof_targets[:,-1].unsqueeze(1)#.repeat(1,6)

        # modify envs with invalid finger commands
        invalid_env_ids =  self.finger_joint_dif>0.01 #| self.gripper_force > 30 #NOTE: stick to 0.05, smaller value -> cube slips
        finger_target[invalid_env_ids] = torch.mean(self._robot.data.joint_pos[invalid_env_ids,7:], dim=1).unsqueeze(1)#.repeat(1,6)
    
        self._robot.set_joint_position_target(finger_target, joint_ids=[7])

        if self.cfg.debug_joint_pos:
            print("robot joint pose")
            format_tensor(self._robot.data.joint_pos)
            print("robot joint pose target")
            format_tensor(self._robot.data.joint_pos_target)        

    def step(self, ee_residual):
        _return = super().step(ee_residual)

        # update robot wp
        self.robot_wp_obs = torch.roll(self.robot_wp_obs, shifts = 8, dims = 1)

        if self.cfg.use_relative_coordinates:
            self.robot_wp_obs[:, :8] = self.get_curr_waypoint_b() - self.init_ee
        else:
            self.robot_wp_obs[:, :8] = self.get_curr_waypoint_b()

        # update demo wp
        self.action_idx = self.time_step_per_env.unsqueeze(1)*8 + self.offsets
        self.demo_wp_obs = torch.roll(self.demo_wp_obs, shifts = 8, dims = 1)
        self.demo_wp_obs[:, :8] = self.training_demo_traj[self.env_idx, self.action_idx, self.demo_idx]

        if self.cfg.mark_demo:
            if self.cfg.use_relative_coordinates:
                self.demo_ee_marker.visualize(self.demo_wp_obs[:, :3] + self.init_ee[:,:3] + self.scene.env_origins[:,:3], 
                                              (self.demo_wp_obs[:,3:7] + self.init_ee[:,3:7])/torch.norm((self.demo_wp_obs[:,3:7] + self.init_ee[:,3:7]), keepdim=True))                
            else:
                self.demo_ee_marker.visualize(self.demo_wp_obs[:, :3] - self.scene.env_origins[:,:3], self.demo_wp_obs[:,3:7])

        self._camera.update(dt=self.dt)
        return _return


    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_y = self._cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        cube_x = self._cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        terminated = (cube_y > 0.2) | (cube_y < -0.2) | (cube_x > 0.6) | (cube_x < 0.1)
        # terminated |= (self._robot.data.joint_pos[:,7:].min(dim=1).values < 0)

        self.finger_joint_dif = self._robot.data.joint_pos[:,7:].max(dim=1).values - self._robot.data.joint_pos[:,7:].min(dim=1).values

        terminated = self._robot.data.body_link_state_w[:, 9, 2] < 0.165 #NOTE: 9th body is link_eef
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
        # robot state -> static reset #TODO: add randomized init robot joint pos
        joint_pos = self._robot.data.default_joint_pos[env_ids, :] 
        if self.cfg.apply_dmr:
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

        # cube state
        reseted_root_states = self._cube.data.default_root_state.clone()
        if self.cfg.apply_dmr:
            reseted_root_states[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            reseted_root_states[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y

        if self.cfg.use_relative_coordinates:
            self.demo_traj = self.demo_traj_rel
        else:
            self.demo_traj = self.demo_traj_abs

        if self.cfg.show_demo and self.cfg.clean_demo:
            self.training_demo_traj = self.demo_traj
        else:
            if self.cfg.use_relative_coordinates:
                step_interval = int(torch.randint(20, 31, (1,)).item())
                noise_level = torch.rand(1).item() * (0.04 - 0.02) + 0.02
                beta_filter = torch.rand(1).item() * (0.7 - 0.5) + 0.5
                noised_traj_abs = smooth_noisy_trajectory(self.demo_traj_abs, env_ids, step_interval=step_interval, noise_level=noise_level, beta_filter=beta_filter) #TODO: randomize step_interval, noise_level, and beta_filter
                self.training_demo_traj = abs_to_rel_traj(self.init_ee, noised_traj_abs)
            else:
                self.training_demo_traj = smooth_noisy_trajectory(self.demo_traj, env_ids, step_interval=30, noise_level=0.05, beta_filter=0.7) #TODO: randomize step_interval, noise_level, and beta_filter
        
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:3]

        self._cube.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids) #NOTE: no render on reset
        self._cube.reset(env_ids)

        # camera
        self._camera.reset(env_ids)

        # controller
        self.diff_ik_controller.reset(env_ids)

        self.time_step_per_env[env_ids] = 0
        self.robot_wp_obs[env_ids, :] = 0
        self.demo_wp_obs[env_ids, :] = 0
        self.demo_wp_obs[env_ids,:8] = self.training_demo_traj[env_ids,:8, 0]

        self.demo_idx[env_ids, :] = (self.demo_idx[env_ids, :] + 1) % self.cfg.num_demos 
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        depth_clean = self._camera.data.output["distance_to_image_plane"].permute(0,3,1,2) # From (B, H, W, 1) to (B, 1, H, W)
        vis = cv2.resize(depth_clean[0,0].cpu().numpy(), (20, 20), interpolation=cv2.INTER_NEAREST)
        # print(np.round(vis, 2))
        # print(self.demo_wp_obs)
        # print(self.robot_wp_obs)
        # import pdb; pdb.set_trace()
        depth_noised = simulate_depth_noise(depth_clean)
        depth_noised = normalize_depth_01(depth_noised)

        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self._robot.data.body_link_state_w[:,9,:3], self._robot.data.body_link_state_w[:,9,3:7], self._cube.data.body_link_state_w[:,0,:3], self._cube.data.body_link_state_w[:,0,3:7]
        )

        actor_obs = torch.cat(
            (
                self.robot_wp_obs*self.cfg.rel_action_scale, 
                self.demo_wp_obs*self.cfg.rel_action_scale, #TODO change to abs
                depth_noised,
            ),
            dim=-1,
        )

        if self.cfg.use_privilege_obs:
            critic_obs = torch.cat(
                (
                    self.robot_wp_obs*self.cfg.rel_action_scale,
                    self.demo_wp_obs*self.cfg.rel_action_scale,
                    cube_pos_b,
                    cube_quat_b,
                    self.demo_idx[:,:1],
                    depth_noised,
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
        residual_penalty = torch.sum(self.ee_residual**2, dim=-1) #TODO: change to norm?

        # reward for reaching cube
        cube_pos_w = self._cube.data.root_pos_w                 # Target object position: (num_envs, 3)
        ee_w = self._robot.data.body_link_state_w[:,9,0:3].clone()   # End-effector position: (num_envs, 3)
        ee_w[:, 2] -= 0.15                                      # Offset for the end-effector

        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
        ee_dist_reward = torch.exp(-object_ee_distance / self.cfg.std)
   
        # action smoothing
        delta_actions = torch.norm(self.joint_pos - self.last_action, p=2, dim=-1)
        self.last_action = self.joint_pos.clone()
        action_penalty = delta_actions #torch.sum(self.actions**2, dim=-1)

        gripper_height = torch.where(self.ee_goal[:,2] < 0.17, 1.0, 0.0)

        overgrasp = (self.finger_joint_dif > 0.05) * self.ee_goal[:,7]

        # gripper closing reward
        closing_gripper = torch.exp(self._robot.data.joint_pos[:, 7]).clamp(0.0,50.0)
        closing_gripper *= ((self._robot.data.body_link_state_w[:,9,2] < 0.20) | (self._cube.data.root_pos_w[:, 2] > self.cfg.minimal_height) | (self.time_step_per_env > 200))

        rewards = (
            + self.cfg.height_reward_scale * height_reward
            + self.cfg.residual_penalty_scale * residual_penalty
            + self.cfg.ee_dist_reward_scale * ee_dist_reward
            + self.cfg.action_penalty * action_penalty
            + self.cfg.gripper_height * gripper_height
            # + self.cfg.overgrasp * overgrasp
            # + self.cfg.closing_gripper * closing_gripper
        )

        rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.05, rewards + 0.25, rewards)
        rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.1, rewards + 0.25, rewards)
        # rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.15, rewards + 1.0, rewards)

        self.extras["log"] = {
            "height_reward": (self.cfg.height_reward_scale * height_reward).mean(),
            "residual_penalty": (self.cfg.residual_penalty_scale * residual_penalty).mean(),
            "ee_dist_reward": (self.cfg.ee_dist_reward_scale * ee_dist_reward).mean(),
            "action_penalty": (self.cfg.action_penalty * action_penalty).mean(),
            "gripper_height": (self.cfg.gripper_height * gripper_height).mean(),
            # "overgrasp": (self.cfg.overgrasp * overgrasp).mean(),
            # "closing_gripper": (self.cfg.closing_gripper * closing_gripper).mean(),
        }

        return rewards
    
    def create_filter_pairs(self, prim1: str, prim2: str):
        stage = get_current_stage()
        filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1))
        filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
        filteredpairs_rel.AddTarget(prim2)
        stage.Save()
    
    def get_curr_waypoint_b(self):
        # ee and root pose in world frame
        curr_ee_pose_w = self._robot.data.body_link_state_w[:,9,0:7]
        curr_root_pose_w = self._robot.data.root_state_w[:, 0:7]

        # ee pose in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_pose_w[:, 0:3], curr_root_pose_w[:, 3:7], curr_ee_pose_w[:, 0:3], curr_ee_pose_w[:, 3:7]
            )

        curr_finger_status = torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1)
        curr_ee_pos_combined_b = torch.cat((curr_ee_pos_b, curr_ee_quat_b, curr_finger_status), dim=-1)

        return curr_ee_pos_combined_b

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