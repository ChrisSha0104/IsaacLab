# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, quat_mul, matrix_from_quat

from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Usd
from RRL.utilities import *
import pdb

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip




@configclass
class XArmCubeScene(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

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
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0, # 30
                "joint5": 0.0,
                "joint6": 0.0, # 75
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot: Articulation = scene["robot"]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint.*"], body_names=["xarm_gripper_base_link"]) # TODO: check
    robot_entity_cfg.resolve(scene)

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 # type: ignore

    kin_helper = KinHelper(robot_name='xarm7')

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    decimation = 4
    sim_step_counter = 0
    count = 0

    qpos_arm_goals = [
        [0.3,-0.2,0.4,0.24,0.1,-0.3, -0.2],
        [0.2,-0.1,0.12,0.43,0.3,-0.12, 0.324],
        [-0.45,-0.3,0.5,-0.24,0.1,-0.3, -0.2],
        [0.1,0.34,0.21,0.24,0.3,-0.2, -0.7],
        ]

    # reset robot
    for i in range(len(qpos_arm_goals)):
        joint_pos_arm = torch.tensor([qpos_arm_goals[i]]).to('cuda:0')
        joint_pos = torch.cat((joint_pos_arm, torch.zeros((1,6)).to('cuda:0')), dim=1)  # ensure it's a 2D tensor
        joint_vel = robot.data.default_joint_vel.clone()
        
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        print("qpos goal: ", joint_pos[:, :7])
        print("robot ee: ", robot.data.body_state_w[:, 9, 0:7])

        fk = kin_helper.compute_fk_sapien_links(joint_pos_arm.detach().cpu().numpy().reshape(-1,), [kin_helper.sapien_eef_idx])[0]
        pos = torch.from_numpy(fk[:3, 3]).reshape(1,-1).to('cuda:0')
        mat = torch.from_numpy(fk[:3, :3]).to('cuda:0')
        quat = quat_from_matrix(mat).reshape(1,-1)

        ee_pos_sapien = torch.cat((pos, quat), dim=-1)
        print("sapien ee: ", ee_pos_sapien)

    exit()    

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0, 0, 0]) # type: ignore
    # Design scene
    scene_cfg = XArmCubeScene(num_envs=args_cli.num_envs, env_spacing=3.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_outer_knuckle")
    create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_outer_knuckle")
    create_filter_pairs("/World/envs/env_0/Robot/right_inner_knuckle", "/World/envs/env_0/Robot/right_finger")
    create_filter_pairs("/World/envs/env_0/Robot/left_inner_knuckle", "/World/envs/env_0/Robot/left_finger")

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)  # type: ignore

def create_filter_pairs(prim1: str, prim2: str):
    stage = get_current_stage()
    filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1))  # type: ignore
    filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
    filteredpairs_rel.AddTarget(prim2)
    stage.Save()

if __name__ == "__main__":
    torch.set_printoptions(precision=4)

    # run the main function
    main()
    # close sim app
    simulation_app.close()