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

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


@configclass
class FrankaCabinetScene(InteractiveSceneCfg):
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
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
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
                stiffness=2e3, # TODO
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
    traj_length = 10

    # reset robot
    init_qpos = torch.tensor([[-0.0537, -0.6888, -0.0261,  0.4878, -0.0491,  1.2839, -0.0107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device='cuda:0')
    joint_pos = init_qpos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    
    # random ee trajectory from start traj
    init_ee_pos = torch.tensor([[ 0.2644, -0.0247,  0.3620,  0.0126,  0.9982, -0.0229, -0.0542]], device='cuda:0')
    # end_ee_pos = torch.tensor([[0.4, 0.1, 0.18, 0.707, 0.707, 0.0, 0.0]], device='cuda:0')
    end_ee_pos = torch.tensor([[0.1, 0.3, 0.64, -0.707, 0.707, 0.0, 0.0]], device='cuda:0')

    ee_traj = interpolate_7d_ee_trajectory(init_ee_pos, end_ee_pos, num_steps=traj_length)

    # reset controller
    ik_commands = init_ee_pos
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    jpos_diff_ik = []    
    jpos_sapien = []

    # Simulation loop
    while simulation_app.is_running(): 
        for i in range(traj_length):
            ee_goal = ee_traj[:, i, :]
            ik_commands = ee_goal
            diff_ik_controller.set_command(ik_commands)

            # import pdb; pdb.set_trace()
            jacobian = robot.root_physx_view.get_jacobians()[:, 8, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, 9, 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # compute jpos using ik
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            print("IK goal: ", ee_goal)
            print("diff ik soln: ", joint_pos_des)
            jpos_diff_ik.append(joint_pos_des.clone())

            r, p, y = euler_xyz_from_quat(ee_goal[:,3:7]) # type: ignore
            ee_sim = np.concatenate((ee_goal[:, :3].cpu().numpy().reshape(-1,), r.cpu().numpy(), p.cpu().numpy(), y.cpu().numpy())) # type: ignore
            print("ee_sim", ee_sim)
            # compute jpos using ik
            ik_qpos = kin_helper.compute_ik_sapien(joint_pos.cpu().numpy().reshape(-1,), ee_sim.astype(np.float32)) # prev -> curr qpos
            print("ik_qpos", ik_qpos)
            jpos_sapien.append(torch.from_numpy(ik_qpos).to(device=sim.device).float())

            # import pdb; pdb.set_trace()

            for _ in range(decimation):
                # apply jpos action
                # import pdb; pdb.set_trace()
                robot.set_joint_position_target(joint_pos_des, joint_ids=[0,1,2,3,4,5,6])
                # set actions into simulator
                scene.write_data_to_sim()
                # simulate
                sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if sim_step_counter % decimation == 0:
                    sim.render()
                # update buffers at sim dt
                scene.update(dt=sim_dt)
        
        break
    
    # import pdb; pdb.set_trace()
    data1 = np.concatenate([t.detach().cpu().numpy() for t in jpos_diff_ik], axis=0) 
    data2 = np.concatenate([t.detach().cpu().numpy().reshape(1,-1) for t in jpos_sapien], axis=0)

    time_steps = np.arange(traj_length)

    # Create 7 subplots (one for each joint)
    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)

    # Loop through each joint index (0 to 6)
    for joint in range(7):
        axs[joint].plot(time_steps, data1[:, joint], label='differential IK')
        axs[joint].plot(time_steps, data2[:, joint], label='sapein IK')
        axs[joint].set_ylabel(f'Joint {joint+1}')
        if joint == 0:
            axs[joint].legend(loc='upper right')
            
    axs[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()
    # with open('sim_jpos_traj.txt', 'w') as f:
    #     for arr in new_jpos_sim:
    #         arr = arr.cpu().numpy()
    #         arr_flat = arr.flatten()  # Flatten in case it's multi-dimensional
    #         line = ' '.join(map(str, arr_flat))
    #         f.write(line + '\n')

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
    scene_cfg = FrankaCabinetScene(num_envs=args_cli.num_envs, env_spacing=3.0, replicate_physics=True)
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
    filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1)) # type: ignore
    filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
    filteredpairs_rel.AddTarget(prim2)
    stage.Save()

if __name__ == "__main__":
    torch.set_printoptions(precision=4)

    # run the main function
    main()
    # close sim app
    simulation_app.close()