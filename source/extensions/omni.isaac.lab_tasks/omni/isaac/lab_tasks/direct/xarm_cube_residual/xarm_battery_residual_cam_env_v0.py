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
TODO: 
- isolate scene cfgsd


"""

@configclass
class XArmBatteryResidualCamLocalBinaryV0EnvCfg(DirectRLEnvCfg):
    # env 
    episode_length_s = 13.33333 # eps_len_s = traj_len * (dt * decimation)
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
            max_position_iteration_count=40,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
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
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=40,
                solver_velocity_iteration_count=1,
                # max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=192, 
                solver_velocity_iteration_count=1
            ), 
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),                  
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
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    robot_friction = 0.75

    battery_R = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Battery_R",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.495, -0.0542, 0.0),
                rot=(0.707, 0.707, 0, 0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/shuosha/Desktop/battery_assets/battery/battery_d.usd", 
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),

        )

    battery_L = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Battery_L",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, 0.08, 0.0), 
                rot=(0.707, 0.707, 0, 0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/shuosha/Desktop/battery_assets/battery/battery_d.usd", 
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
        )
    
    battery_box = RigidObjectCfg(
            prim_path="/World/envs/env_.*/BatteryBox",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.2, 0.0, 0.0),
                rot=(0.707, 0.0, 0, 0.707),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/shuosha/Desktop/battery_assets/battery_box/battery_box.usd", 
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    # solver_position_iteration_count=16,
                    # solver_velocity_iteration_count=1,
                    # max_angular_velocity=1000.0,
                    # max_linear_velocity=1000.0,
                    # max_depenetration_velocity=5.0,
                    # disable_gravity=False,
                    kinematic_enabled=True
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
        )
    
    # cameras
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link_eef/cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.13, 0.0, 0.008), rot=(0.9744,0,-0.2334,0), convention="ros"), # z-down; x-forward
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

    # training path
    training_data_path = "RRL/tasks/battery/training_set1"

    # visualization options: (All turned to false during training)
    play_training_demo = False
    play_real_demo = False
    mark_obs = False

    # play option:
    show_camera = False                  # option only used for play

    # debug options
    print_all_intermediate_value = False
    debug_ee = False
    debug_joint_pos = False
    store_obs = False
    
    # actor critic args
    use_visual_encoder = True
    visual_idx_actor = [20,20+120*120]
    visual_idx_critic = [20,20+120*120]

    # training options:
    add_noise_to_demo = True
    learn_std = True
    use_privilege_obs = True
    apply_dmr = False
    num_demos = 10
    state_history_length = 50 # 1.5s ago

    # parameters
    pos_std = 5e-3 # dmr scales
    rot_std = 1e-3
    alpha = 0.1 # residual scale
    tilde = 0.5 # low pass filter
    minimal_height = 0.05
    fingertip_dist_std = 0.1

    # reward scale RESIDUAL
    # --- task-completion ---
    completion_reward_scale = 1.0
    fingertip_dist_reward_scale = 0.1
    battery_goal_dist_reward_scale = 0.1

    # --- auxiliary ---
    residual_penalty_scale = -0.1
    residual_rate_scale = -0.1
    velocity_penalty_scale = -0.05
    collision_penalty_scale = -10.0
    gripper_height_penalty_scale = -10.0

class XArmBatteryResidualCamLocalBinaryV0Env(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()

    # post-physics step calls
    #   |-- _get_dones()
    #   |  |-- _compute_intermediate_values()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XArmBatteryResidualCamLocalBinaryV0EnvCfg

    def __init__(self, cfg: XArmBatteryResidualCamLocalBinaryV0EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        if self.cfg.debug_ee or self.cfg.debug_joint_pos or self.cfg.play_training_demo:
            print("--------DEBUG MODE--------")
            if self.cfg.play_training_demo:
                if self.cfg.add_noise_to_demo:
                    print("SHOWING NOISED DEMO TRAJECTORY")
                else:
                    print("SHOWING CLEAN DEMO TRAJECTORY")
        else:
            print("--------TRAINING MODE--------")

        if self.cfg.mark_obs:
            frame_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # type: ignore
            self.marker1 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/marker1")) # type: ignore
            self.marker2 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/marker2")) # type: ignore
            self.marker3 = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/marker3")) # type: ignore

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_position_upper_limits = torch.tensor([[0.65, 0.5, 0.5]], device=self.device)
        self.robot_position_lower_limits = torch.tensor([[0.15, -0.5, 0.175]], device=self.device)

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
        self.robot_state_hist = HistoryBuffer(self.num_envs, self.cfg.state_history_length, self.cfg.action_space, device=self.device) # (num_envs, state_history_length, action_space+6)
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

        self.teleop_comm_obs = self.init_pos.clone() # most recent rel teleop ee command for each env (num_envs, 10)
        self.last_ee_goal = self.init_pos.clone()
        self.ee_goal_b = self.init_pos.clone()

        self.demo_traj = self._generate_clean_demo_traj(dir=self.cfg.training_data_path, num_demos=self.cfg.num_demos, init_pos=self.init_pos)
            
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        self.collected = []
        self.collect_min_max_obs_values = False
        self.sim_teleop_base_fr = []

        self._set_default_dynamics_parameters()

        # tensors to reward for aligning with human intentions
        self.intended_targets = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long) # pick up target {0,1} and placing target {0,1}. 0 for left, 1 for right
        self._set_goal_targets() # hardcoded
        self._set_box_region()

        self._compute_intermediate_values()

    def _set_goal_targets(self):
        """get hardcoded placing goals for battery L and R"""
        left_placing_target_pose_7D = torch.tensor([[0.2429, 0.0678, 0.0056, 0.6939,  0.7198,  0.0031, -0.0193]], device=self.device).repeat(self.num_envs, 1) # target pose for battery L
        right_placing_target_pose_7D = torch.tensor([[0.2430, -0.0679,  0.0057, 0.6940,  0.7198,  0.0113, -0.0112]], device=self.device).repeat(self.num_envs, 1) # target pose for battery R
        self.placing_targets_7D_b = torch.stack((left_placing_target_pose_7D, right_placing_target_pose_7D), dim=1) # (num_envs, 2, 7)

    def _set_box_region(self):
        self.box_lower_limits = torch.tensor([[0.12, -0.1, 0.0]], device=self.device).repeat(self.num_envs, 1)
        self.box_upper_limits = torch.tensor([[0.28, 0.1, 0.10]], device=self.device).repeat(self.num_envs, 1)

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        pass
        # Set masses and frictions.
        # self._set_friction(self._held_asset, self.cfg.held_asset_cfg.friction)
        # self._set_friction(self._fixed_asset, self.cfg.fixed_asset_cfg.friction)
        # self._set_friction(self._robot, self.cfg.robot_friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

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
    
    def _compute_intermediate_values(self):
        """
        get intermediate values computed from raw tensors
        """
        # compute battery poses in base frame
        robot_root_w = self._robot.data.root_state_w[:]
        battery_L_pos_b, battery_R_pos_b = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._battery_L.data.body_com_state_w[:,0,:3], self._battery_L.data.body_com_state_w[:,0,3:7]
        )
        battery_L_b_7D = torch.cat([battery_L_pos_b, battery_R_pos_b], dim=-1)
        battery_R_pos_b, battery_L_pos_b = subtract_frame_transforms(
            robot_root_w[:, 0:3], robot_root_w[:, 3:7], self._battery_R.data.body_com_state_w[:,0,:3], self._battery_R.data.body_com_state_w[:,0,3:7]
        )
        battery_R_b_7D = torch.cat([battery_R_pos_b, battery_L_pos_b], dim=-1)
        self.battery_poses_b = torch.stack((battery_L_b_7D, battery_R_b_7D), dim=1) # (num_envs, 2, 7), 0 for left, 1 for right

        # compute intended battery's poses in base frame
        battery_targets_exp = self.intended_targets[:,0].view(self.num_envs, 1, 1).expand(-1, 1, 7) # (num_envs, 1, 7)
        self.intended_battery_pose_b = torch.gather(self.battery_poses_b, dim=1, index=battery_targets_exp).squeeze(1) # (num_envs, 7)

        # compute intended battery's goal poses in base frame
        placing_targets_exp = self.intended_targets[:,0].view(self.num_envs, 1, 1).expand(-1, 1, 7) # (num_envs, 1, 7)
        self.intended_battery_goal_pose_b = torch.gather(self.placing_targets_7D_b, dim=1, index=placing_targets_exp).squeeze(1) # (num_envs, 7)

        # compute fingertip 7D pose in base frame
        robot_ee_pose_w = self._robot.data.body_com_state_w[:,9,0:7]
        robot_root_pose_w = self._robot.data.root_state_w[:, 0:7]
        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
                robot_root_pose_w[:, 0:3], robot_root_pose_w[:, 3:7], robot_ee_pose_w[:, 0:3], robot_ee_pose_w[:, 3:7]
            )        
        ee_b = torch.cat([robot_ee_pos_b, robot_ee_quat_b], dim=-1)
        offset_gripper_position = subtract_z_in_baseframe_batch(ee_b, z_offset=0.14)
        self.ee_fingertip_b_7D = torch.cat([offset_gripper_position, robot_ee_quat_b], dim=-1)

        # compute 10D robot state in base frame
        self.curr_robot_state_b_10D = self.get_robot_state_b() # (num_envs, 10)
        
        # get curr teleop command 10D
        self.teleop_comm_obs = self.training_demo_traj[self.env_idx, self.demo_idx, self.time_step_per_env, :] # (num_envs, 10)

        # self.marker1.visualize(self.ee_fingertip_b_7D[:1,:3], self.ee_fingertip_b_7D[:1,3:7])
        # self.marker2.visualize(self.intended_battery_goal_pose_b[:1,:3], self.intended_battery_goal_pose_b[:1,3:7])
        # self.marker3.visualize(self.intended_battery_pose_b[:1,:3], self.intended_battery_pose_b[:1,3:7])
    
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._battery_L = RigidObject(self.cfg.battery_L)
        self._battery_R = RigidObject(self.cfg.battery_R)
        self._battery_box = RigidObject(self.cfg.battery_box)
        self._camera = TiledCamera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["battery_box"] = self._battery_box
        self.scene.rigid_objects["battery_L"] = self._battery_L
        self.scene.rigid_objects["battery_R"] = self._battery_R
        self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # filter finger collision
        # self.create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_outer_knuckle")
        # self.create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_outer_knuckle")
        # self.create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_finger")
        # self.create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_finger")

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
        
        self.last_ee_goal = self.ee_goal_b.clone()

        if self.cfg.play_training_demo:
            goal_in_ee_fr = self.curr_comm_in_curr_ee_fr
            ee_10D_b = self.curr_robot_ee_b.clone() # shape (num_envs, 10)
            self.ee_goal_b = combine_frame_transforms_10D(ee_10D_b, goal_in_ee_fr)
        else: 
            goal_in_ee_fr = self.curr_comm_in_curr_ee_fr + self.cfg.alpha * self.ee_residual
            ee_10D_b = self.curr_robot_ee_b.clone()
            self.ee_goal_b = combine_frame_transforms_10D(ee_10D_b, goal_in_ee_fr)

            if self.cfg.print_all_intermediate_value:
                print(">>>>>>>>>>>>>>> POLICY OUTPUT <<<<<<<<<<<<<<<<")
                print("base action (comm in ee fr): ")
                format_tensor(self.curr_comm_in_curr_ee_fr)
                print("residual: ")
                format_tensor(self.ee_residual)
                print("ee goal clean b: ")
                format_tensor(self.ee_goal_b)

        ee_goal_filtered = self.cfg.tilde * self.ee_goal_b.clone() + (1-self.cfg.tilde) * self.last_ee_goal.clone()
        ee_goal_filtered[:,:3] = torch.clamp(ee_goal_filtered[:,:3], self.robot_position_lower_limits, self.robot_position_upper_limits)
        ee_goal_filtered[:,-1] = (ee_goal_filtered[:,-1] > 0.5).float()

        if self.cfg.print_all_intermediate_value:
            print("ee goal filtered: ", ee_goal_filtered)

        self.ee_hist.append(ee_goal_filtered.clone())

        if self.cfg.debug_ee:
                print("ee goal")
                format_tensor(ee_goal_filtered)
                print("current ee")
                format_tensor(self.get_robot_state_b())       

        self.joint_pos = self.get_joint_pos_from_ee_pos_10d(self.diff_ik_controller, ee_goal_filtered)                                  # ee_goal always abs coordinates
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.robot_dof_lower_limits[:8], self.robot_dof_upper_limits[:8])       # (num_envs, 8)
        self.time_step_per_env += 1
        if self.cfg.print_all_intermediate_value:
            print("robot joint pose target")
            format_tensor(self.robot_dof_targets)

        # self.action_list.append(ee_goal_filtered.clone())
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

        finger_target = self.robot_dof_targets[:,-1].unsqueeze(1)

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

        self._camera.update(dt=self.dt)
        return _return

    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # cube_y = self._cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
        # cube_x = self._cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        # terminated = (cube_y > 0.2) | (cube_y < -0.2) | (cube_x > 0.6) | (cube_x < 0.1)
        terminated = None
        # terminated |= (self._robot.data.joint_pos[:,7:].min(dim=1).values < 0)

        self.finger_joint_dif = self._robot.data.joint_pos[:,7:].max(dim=1).values - self._robot.data.joint_pos[:,7:].min(dim=1).values

        terminated = self._robot.data.body_com_state_w[:, 9, 2] < 0.165 #NOTE: 9th body is link_eef
        # print("actual ee height: ", self._robot.data.body_com_state_w[:, 9, 2])

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _collect_obs_data_on_termination(self, env_ids):
        """collect obs data and compute min/max for manual normalization"""
        data = torch.cat(self.collected, dim=0)  # shape: (N,26)

        # Compute column-wise minimum and maximum.
        min_vals = torch.min(data, dim=0).values  # shape: (26,)
        max_vals = torch.max(data, dim=0).values  # shape: (26,)

        print("Minimum values for each dimension:", min_vals)
        print("Maximum values for each dimension:", max_vals)

    def _reset_buffers(self, env_ids):
        # Reset time step
        self.time_step_per_env[env_ids] = 0

        # Reset demo idx
        self.demo_idx[env_ids] = (self.demo_idx[env_ids] + 1) % self.cfg.num_demos 

        self._reset_demo_traj(env_ids, self.cfg.apply_dmr, self.cfg.add_noise_to_demo)
        self._update_human_intentions(env_ids)

        # clear obs buffers
        self.robot_state_hist.clear_envs(env_ids)
        self.teleop_comm_hist.clear_envs(env_ids)
        self.teleop_comm_obs[env_ids,:] = self.training_demo_traj[env_ids, self.demo_idx[env_ids], 0, :]

    def _update_human_intentions(self, env_ids: torch.Tensor):
        """
        compute / update the self.intended_targets tensor for the resetted envs
        """
        reset_trajs = self.training_demo_traj[env_ids, self.demo_idx[env_ids]]
        N, T, D = reset_trajs.shape
        
        # 1) compute the frame‐wise status diff: from t->t+1
        status = reset_trajs[..., -1]                    # (N, T)
        diff   = status[:, 1:] - status[:, :-1]      # (N, T-1)

        # 2) find first 0->1 (close/grasp) and first 1->0 (open/release)
        #   we add +1 because diff[i, t]==1 means status went 0->1 at t -> t+1
        mask_close = diff ==  1                      # (N, T-1)
        mask_open  = diff == -1                      # (N, T-1)

        # for each env, get the first occurrence index; fallback to T-1 if never happens
        # (you can adjust fallback behavior as needed)
        idxs = torch.arange(N, device=reset_trajs.device)

        def first_idx(mask, default):
            # mask: (N, T-1) bool
            has = mask.any(dim=1)
            # argmax gives the first max; but if no True then returns 0 
            pos = mask.float().argmax(dim=1)
            return torch.where(has, pos + 1, default)  # +1 to map to actions timestep
        
        close_t = first_idx(mask_close, default=T-1)  # (N,)
        open_t  = first_idx(mask_open,  default=T-1)  # (N,)

        # 3) extract gripper‐position at those timesteps
        pos           = reset_trajs[..., :3]              # (N, T, 3)
        grasp_pos_b   = pos[idxs, close_t]            # (N, 3)
        release_pos_b = pos[idxs, open_t]             # (N, 3)

        # 4) compute Euclidean dists to each object, at each event
        d1_grasp   = (grasp_pos_b   - self.battery_poses_b[env_ids, 0, :3]).norm(dim=1)  # (N,)
        d2_grasp   = (grasp_pos_b   - self.battery_poses_b[env_ids, 1, :3]).norm(dim=1)  # (N,)
        d1_release = (release_pos_b - self.placing_targets_7D_b[env_ids, 0, :3]).norm(dim=1)  # (N,)
        d2_release = (release_pos_b - self.placing_targets_7D_b[env_ids, 1, :3]).norm(dim=1)  # (N,)

        # 5) decide which is closer: 0 if obj1, 1 if obj2
        intent_grasp   = (d2_grasp   < d1_grasp).long()    # (N,)
        intent_release = (d2_release < d1_release).long()  # (N,)

        # 6) stack into final (N,2) tensor
        #    [:,0] = intention at grasp, [:,1] = at release
        self.intended_targets[env_ids] = torch.stack([intent_grasp, intent_release], dim=1)  # (N, 2)
        # print("intended targets: ", self.intended_targets[env_ids])

    def _reset_robot_state(self, env_ids, apply_dmr=False):
        # Reset qpos
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)

        # DMR for initial qpos
        if apply_dmr:
            # print("Applying DMR")
            joint_pos[:,:7] += sample_uniform( 
                                -0.125,
                                0.125,
                                (len(env_ids), self._robot.num_joints-6),
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
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids=env_ids)
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self._step_sim_no_action()

    def _reset_demo_traj(self, env_ids, apply_dmr=False, add_noise=True):
        # # update initial pose of demo trajs
        if apply_dmr: # TODO: check this!!!! Learn indexing
            # ee after dmr
            initial_ee = self.get_robot_state_b().clone()
            # interpolated from randomized ee to demo traj            
            initial_traj = interpolate_10d_ee_trajectory(initial_ee[env_ids], self.demo_traj[env_ids, self.demo_idx[env_ids], 20, :], 20) # (env_ids, 1, 50, 10)
            # fill in initial traj
            self.demo_traj[env_ids, self.demo_idx[env_ids], :20, :] = initial_traj.clone()
        
        # add noise to demo traj
        if not add_noise:
            self.training_demo_traj = self.demo_traj
        else:
            step_interval = int(torch.randint(20, 41, (1,)).item())
            noise_level = torch.rand(1).item() * (0.04 - 0.02) + 0.02
            beta_filter = torch.rand(1).item() * (0.7 - 0.5) + 0.5
            self.training_demo_traj = smooth_noisy_trajectory(self.demo_traj, env_ids, step_interval=step_interval, noise_level=noise_level, beta_filter=beta_filter)

        # 

    def _step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # DEBUG: collect obs data and compute min/max            
        if self.collect_min_max_obs_values and len(self.collected) > 0: 
            self._collect_obs_data_on_termination(env_ids)
        
        self._reset_buffers(env_ids)
        self._reset_robot_state(env_ids, self.cfg.apply_dmr)
        self._reset_assets(env_ids, self.cfg.apply_dmr)
        
        self._camera.reset(env_ids)
        self.diff_ik_controller.reset(env_ids) # type: ignore

        self._step_sim_no_action()

    def _reset_assets(self, env_ids, apply_dmr=False):
        reseted_root_states = self._battery_L.data.default_root_state.clone()
        if self.cfg.apply_dmr:
            reseted_root_states[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            reseted_root_states[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y
        
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:3]
        self._battery_L.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids) #NOTE: no render on reset
        self._battery_L.reset(env_ids)

        reseted_root_states = self._battery_R.data.default_root_state.clone()
        if self.cfg.apply_dmr:
            reseted_root_states[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            reseted_root_states[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y
        
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:3]
        self._battery_R.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids) #NOTE: no render on reset
        self._battery_R.reset(env_ids)

        reseted_root_states = self._battery_box.data.default_root_state.clone()
        if self.cfg.apply_dmr:
            reseted_root_states[env_ids,0] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #x 
            reseted_root_states[env_ids,1] += sample_uniform(-0.05, 0.05, len(env_ids), self.device) #y
        
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:3]
        self._battery_box.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids) #NOTE: no render on reset
        self._battery_box.reset(env_ids)

    def _get_observations(self) -> dict:
        self.curr_robot_ee_b = self.get_robot_state_b() # reset issues
        self.robot_state_hist.append(self.curr_robot_ee_b)
        prev_robot_ee_b = self.robot_state_hist.get_oldest_obs() 
        prev_ee_in_curr_ee_fr = subtract_frame_transforms_10D(self.curr_robot_ee_b, prev_robot_ee_b) 

        curr_teleop_comm_b = self.teleop_comm_obs.clone()
        self.curr_comm_in_curr_ee_fr = subtract_frame_transforms_10D(self.curr_robot_ee_b, curr_teleop_comm_b)
        # self.sim_teleop_ee_fr.append(self.curr_comm_in_curr_ee_fr.clone())
        # if len(self.sim_teleop_ee_fr) > 400:
        #     save_to_txt(self.sim_teleop_ee_fr, "RRL/sim2real/sim_traj1/teleop_comm_ee_fr.txt")
        #     print("teleop comm ee fr saved")
        #     exit()
        # self.sim_teleop_base_fr.append(curr_teleop_comm_b.clone())
        # if len(self.sim_teleop_base_fr) == 400:
        #     save_to_txt(self.sim_teleop_base_fr, "RRL/sim2real/sim_traj1/teleop_comm_base_fr.txt")
        #     print("teleop comm base fr saved")
        #     exit()

        # print("curr comm in curr ee fr: ", self.curr_comm_in_curr_ee_fr)

        # cube_8D_w = self._robot.data.root_state_w[:]
        # cube_pos_b, cube_quat_b = subtract_frame_transforms(
        #     cube_8D_w[:, 0:3], cube_8D_w[:, 3:7], self._cube.data.body_com_state_w[:,0,:3], self._cube.data.body_com_state_w[:,0,3:7]
        # )
        # cube_7D_b = torch.cat([cube_pos_b, cube_quat_b], dim=-1)
        # cube_10D_b = ee_7D_to_10D(cube_7D_b)
        # cube_pose_in_curr_ee_fr = subtract_frame_transforms_10D(self.curr_robot_ee_b, cube_10D_b)[:,:9]
        
        if self.cfg.print_all_intermediate_value:
            print(">>>>>>>>>>>>>>> POLICY INPUTS <<<<<<<<<<<<<<<<")
            print("-------------- 1. 10D observations in base frame --------------")
            print("curr robot ee b: ")
            format_tensor(self.curr_robot_ee_b)
            print("curr teleop comm b: ")
            format_tensor(curr_teleop_comm_b)
            # print("cube pose b: ")
            # format_tensor(cube_10D_b)

        # visualize obs in base fr
        # if self.cfg.mark_obs:
            # self.marker1.visualize(prev_robot_ee_b[:,:3], ee_10D_to_8D(prev_robot_ee_b)[:,3:7])
            # self.marker2.visualize(self.ee_goal_b[:,:3], ee_10D_to_8D(self.ee_goal_b)[:,3:7])
            # self.marker3.visualize(self._robot.data.body_com_pos_w[:,9,:3])

        robot_state_min = torch.tensor([-0.1, -0.1, -0.1,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation 
                                        0.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) 
        

        robot_state_max = torch.tensor([0.1, 0.1, 0.1,  # position
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation 
                                        1.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) 
        
        teleop_comm_min = torch.tensor([-0.05, -0.05, -0.05,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation 
                                        0.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) 
        

        teleop_comm_max = torch.tensor([0.05, 0.05, 0.05,  # position
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation 
                                        1.0, # gripper
                                        ], device=self.device).repeat(self.num_envs, 1) 

        cube_state_min = torch.tensor([-0.1, -0.1, -0.0,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation
                                        ], device=self.device).repeat(self.num_envs, 1)
        
        cube_state_max = torch.tensor([0.2, 0.1, 0.5,  # position
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation
                                        ], device=self.device).repeat(self.num_envs, 1)
                                        
        normalized_robot_state_obs = (prev_ee_in_curr_ee_fr - robot_state_min) / (robot_state_max - robot_state_min)
        normalized_teleop_comm_obs = (self.curr_comm_in_curr_ee_fr - teleop_comm_min) / (teleop_comm_max - teleop_comm_min)
        # normalized_cube_pose_obs = (cube_pose_in_curr_ee_fr - cube_state_min) / (cube_state_max - cube_state_min)

        depth_clean = self._camera.data.output["distance_to_image_plane"].permute(0,3,1,2)
        depth_filtered = filter_sim_depth(depth_clean) # (num_envs, 120*120) which should give same reading as real
        normalized_depth = normalize_depth_01(depth_filtered)

        if self.num_envs == 1: 
            depth_vis = filter_depth_for_visualization(depth_filtered.reshape(120,120).detach().cpu().numpy(), unit='m') # only works for 1 env
            cv2.imshow("depth_image", depth_vis)
            cv2.waitKey(1)
        #     save_tensor_as_txt(depth_filtered, "RRL/sim2real/vision_gap/raw/visual_raw_obs_sim")
        #     import pdb; pdb.set_trace()

        # save_tensor_as_txt(normalized_depth, "RRL/sim2real/vision_gap/input/visual_input_obs_sim")
        # import pdb; pdb.set_trace()
    
        actor_obs = torch.cat(
            (
                normalized_robot_state_obs,
                normalized_teleop_comm_obs,
                normalized_depth,
            ),
            dim=-1,
        )
        if self.collect_min_max_obs_values:
            self.collected.append(actor_obs.clone())

        if self.cfg.print_all_intermediate_value:
            print("------------ 2. 10D normalized rel obs (actual input) ------------")
            print("normalized robot state obs")
            format_tensor(normalized_robot_state_obs)
            print("normalized teleop comm obs")
            format_tensor(normalized_teleop_comm_obs)
            # print("normalized cube pose obs")
            # format_tensor(normalized_cube_pose_obs)

        # if self.cfg.use_privilege_obs:
        #     critic_obs = torch.cat(
        #         (
        #             normalized_robot_state_obs,
        #             normalized_teleop_comm_obs,
        #             normalized_cube_pose_obs,
        #             # self.demo_idx.unsqueeze(1),
        #             normalized_depth,
        #         ),
        #         dim=-1,
        #     )

        #     return {"policy": torch.clamp(actor_obs, -5.0, 5.0),
        #             "critic": torch.clamp(critic_obs, -5.0, 5.0)}
        # else: 
        return {"policy": torch.clamp(actor_obs, -5.0, 5.0)}

    def _compute_rewards(self):        
        # RESIDUAL PENALTY
        residual_penalty = torch.norm(self.ee_residual, dim=-1) 

        # residual rate
        residual_rate = torch.norm((self.ee_residual.clone() - self.last_ee_residual.clone()), p=2, dim=-1)

        # gripper height
        gripper_height_penalty = torch.where(self.ee_goal_b[:,2] < 0.17, 1.0, 0.0)

        # gripper box collision
        collided = (self.ee_fingertip_b_7D[:,:3] >= self.box_lower_limits) & (self.ee_fingertip_b_7D[:,:3] <= self.box_upper_limits)
        collided = collided.all(dim=1)
        collision_penalty = torch.where(collided, 1.0, 0.0)

        # fingertip battery dist
        fingertip_batter_dist = torch.norm(self.intended_battery_pose_b[:,:3] - self.ee_fingertip_b_7D[:,:3], dim=1)
        fingertip_battery_dist_reward = torch.exp(-fingertip_batter_dist / self.cfg.fingertip_dist_std)

        # battery goal dist
        baterry_goal_dist = torch.norm(self.intended_battery_goal_pose_b[:,:3] - self.intended_battery_pose_b[:,:3], dim=1)
        baterry_goal_dist_reward = torch.exp(-baterry_goal_dist / self.cfg.fingertip_dist_std)

        # completion reward at end of episode
        baterry_goal_dist = torch.norm(self.intended_battery_goal_pose_b[:,:3] - self.intended_battery_pose_b[:,:3], dim=1)
        completion_condition = (baterry_goal_dist < 0.05) & (self.episode_length_buf >= self.max_episode_length - 1 - 50) # reward for completion in the last 50 time steps!
        completion_reward = torch.where(completion_condition, 1.0, 0.0)

        # velocity penalty
        velocity_penalty = torch.norm(self._robot.data.body_com_state_w[:,9,7:13], p=2, dim=1)

        rewards = (
            # task completion rewards
            self.cfg.fingertip_dist_reward_scale * fingertip_battery_dist_reward
            + self.cfg.battery_goal_dist_reward_scale * baterry_goal_dist_reward
            + self.cfg.completion_reward_scale * completion_reward
            # smoothness rewards
            + self.cfg.residual_penalty_scale * residual_penalty
            + self.cfg.velocity_penalty_scale * velocity_penalty
            + self.cfg.residual_rate_scale * residual_rate
            # safety rewards
            + self.cfg.gripper_height_penalty_scale * gripper_height_penalty
            + self.cfg.collision_penalty_scale * collision_penalty
            
        )

        self.extras["log"] = {
            "completion": (self.cfg.completion_reward_scale * completion_reward).mean(),
            "fingertip_battery_dist": (self.cfg.fingertip_dist_reward_scale * fingertip_battery_dist_reward).mean(),
            "battery_goal_dist": (self.cfg.battery_goal_dist_reward_scale * baterry_goal_dist_reward).mean(),
            "residual_penalty": (self.cfg.residual_penalty_scale * residual_penalty).mean(),
            "residual_rate": (self.cfg.residual_rate_scale * residual_rate).mean(),
            "velocity_penalty": (self.cfg.velocity_penalty_scale * velocity_penalty).mean(),
            "gripper_height_penalty": (self.cfg.gripper_height_penalty_scale * gripper_height_penalty).mean(),
            "collision_penalty": (self.cfg.collision_penalty_scale * collision_penalty).mean(),
        }
        
        return rewards
    
    def create_filter_pairs(self, prim1: str, prim2: str):
        stage = get_current_stage()
        filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1)) # type: ignore
        filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
        filteredpairs_rel.AddTarget(prim2)
        stage.Save()
    
    def get_robot_state_b(self):
        """
        return: 
            get current robot state in the base frame, (num_envs, 10)
        """

        # ee and root pose in world frame
        curr_ee_pos_w = self._robot.data.body_com_state_w[:,9,:]
        curr_root_w = self._robot.data.root_state_w[:]

        # ee pose in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_w[:, 0:3], curr_root_w[:, 3:7], curr_ee_pos_w[:, 0:3], curr_ee_pos_w[:, 3:7]
            )
        curr_orient_6d = quat_to_6d(curr_ee_quat_b)

        curr_finger_status = torch.mean(self._robot.data.joint_pos[:,7:], dim=1).unsqueeze(1)
        curr_finger_status = (curr_finger_status > 0.2).float() # convert gripper qpos to binary

        curr_robot_state_b = torch.cat((curr_ee_pos_b, curr_orient_6d, curr_finger_status), dim=-1)

        return curr_robot_state_b
    
    def get_joint_pos_from_ee_pos_10d(self, controller: DifferentialIKController, ee_goal):
        curr_ee_b = self.get_robot_state_b() # (num_envs, 10)
        curr_ee_quat_b = quat_from_6d(curr_ee_b[:,3:9])

        ee_goal_quat = quat_from_6d(ee_goal[:,3:9])
        ee_goal_7d = torch.cat((ee_goal[:,:3], ee_goal_quat), dim=-1) # (num_envs, 7)
        ik_commands = ee_goal_7d #(num_envs, 7)
        controller.set_command(ik_commands)
        
        ee_jacobi_idx = self._robot.find_bodies("link_eef")[0][0]-1
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        joint_pos_des_arm = controller.compute(curr_ee_b[:,:3], curr_ee_quat_b, jacobian, joint_pos)

        gripper_status = ee_goal[:, -1].unsqueeze(1).clone() # binary gripper + residual output
        gripper_status[gripper_status > 0.5] = 0.6          # NOTE: close gripper in qpos
        gripper_status[gripper_status < 0.5] = 0.0          # NOTE: open gripper in qpos

        joint_pos_des = torch.cat((joint_pos_des_arm, gripper_status), dim=-1)

        return joint_pos_des