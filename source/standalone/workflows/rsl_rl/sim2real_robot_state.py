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

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    decimation = 4
    sim_step_counter = 0
    count = 0

    qpos_traj = torch.from_numpy(np.loadtxt("RRL/sim2real_data/qpos_goal.txt")).float() # without gripper pose
    qpos_traj = qpos_traj.to('cuda:0')[:, :7]
    teleop_obs_traj_real = torch.from_numpy(np.loadtxt("RRL/sim2real_data/teleop_obs.txt")).float()  # load teleop observations
    teleop_obs_traj_real = teleop_obs_traj_real.to('cuda:0')  # ensure it's on the same device as qpos_traj
    robot_obs_traj_real = torch.from_numpy(np.loadtxt("RRL/sim2real_data/robot_obs.txt")).float()  # load robot observations
    robot_obs_traj_real = robot_obs_traj_real.to('cuda:0')  # ensure it's on the same device as qpos_traj

    # reset robot
    joint_pos = qpos_traj[0,:].clone().unsqueeze(0) # (1, 7)
    joint_pos = torch.cat((joint_pos, torch.zeros((1,6)).to('cuda:0')), dim=1)  # ensure it's a 2D tensor
    joint_vel = robot.data.default_joint_vel.clone()
    
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    print("initial qpos goal: ", joint_pos[:, :7])
    print("robot qpos: ", robot.data.joint_pos[:, :7])  
    
    # reset controller
    # ik_commands[:] = torch.from_numpy(ee_poses[0, :].reshape(1,-1)).to(device='cuda')
    diff_ik_controller.reset()
    # diff_ik_controller.set_command(ik_commands)

    robot_obs_traj_sim = []    
    robot_obs_hist = HistoryBuffer(1, 50, 16, device='cuda:0') # type: ignore

    # Simulation loop
    while simulation_app.is_running(): 
        for i in range(qpos_traj.shape[0]):
            qpos_goal = qpos_traj[i, :7].clone()  # get the goal from teleop observations

            # apply qpos goal to the robot
            for _ in range(decimation):
                # apply jpos action
                robot.set_joint_position_target(qpos_goal.reshape(1,-1), joint_ids=[0,1,2,3,4,5,6])
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

            print("qpos_goal: ", qpos_goal)
            print("robot qpos: ", robot.data.joint_pos[:, :7])  
            
            # robot state obs in np
            robot_state_obs_sim = get_robot_state_obs(robot, robot_obs_hist)
            robot_obs_traj_sim.append(robot_state_obs_sim.clone())

            print(f"robot obs sim at step: {i+1}: ", robot_state_obs_sim)
            print(f"robot obs real at step: {i+1}: ", robot_obs_traj_real[i, :]) 

            # pdb.set_trace()  # Debugging breakpoint to inspect variables
        
        print("finished running traj")
        break
    
    # import pdb; pdb.set_trace()
    robot_obs_sim = np.concatenate([t.detach().cpu().numpy() for t in robot_obs_traj_sim], axis=0) 
    robot_obs_real = np.concatenate([t.detach().cpu().numpy().reshape(1,-1) for t in robot_obs_traj_real], axis=0)

    time_steps = np.arange(len(robot_obs_traj_sim))

    # Create 7 subplots (one for each joint)
    fig, axs = plt.subplots(16, 1, figsize=(10, 14), sharex=True)

    # Loop through each joint index (0 to 6)
    for dim in range(16):
        axs[dim].plot(time_steps, robot_obs_sim[:, dim], label='robot_obs_sim')
        axs[dim].plot(time_steps, robot_obs_real[:, dim], label='robot_obs_real')
        axs[dim].set_ylabel(f'Dim {dim+1}')
        if dim == 0:
            axs[dim].legend(loc='upper right')
            
    axs[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()
    # with open('sim_jpos_traj.txt', 'w') as f:
    #     for arr in new_jpos_sim:
    #         arr = arr.cpu().numpy()
    #         arr_flat = arr.flatten()  # Flatten in case it's multi-dimensional
    #         line = ' '.join(map(str, arr_flat))
    #         f.write(line + '\n')

def get_robot_state_obs(robot: Articulation, robot_state_hist: HistoryBuffer) -> torch.Tensor:
    """Get the teleop and robot observations from the robot."""

    robot_state_obs = get_robot_state_b(robot) # reset issues
    robot_state_hist.append(robot_state_obs)
    prev_robot_state_obs = robot_state_hist.get_oldest_obs() 
    relative_robot_state = compute_relative_state(prev_robot_state_obs, robot_state_obs)

    robot_state_min = torch.tensor([-0.1, -0.1, -0.1,  # position
                                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, # orientation 
                                        -0.5, -0.5, -0.5, # lin vel
                                        -1.0, -1.0, -1.0, # ang vel
                                        0.0, # gripper
                                        ], device='cuda:0').repeat(1, 1) # TODO: check why x is so large
        

    robot_state_max = torch.tensor([0.1, 0.1, 0.1,  # position
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # orientation 
                                    0.5, 0.5, 0.5, # lin vel
                                    1.0, 1.0, 1.0, # ang vel
                                    1.0, # gripper
                                    ], device='cuda:0').repeat(1, 1) # TODO: check why x is so large
        
    standardized_robot_state_obs = (relative_robot_state - robot_state_min) / (robot_state_max - robot_state_min)

    return standardized_robot_state_obs

def get_robot_state_b(robot: Articulation) -> torch.Tensor:
    curr_state_w = robot.data.body_com_state_w[:,9,:]
    curr_root_state = robot.data.root_state_w[:]

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

    curr_finger_status = torch.mean(robot.data.joint_pos[:,7:], dim=1).unsqueeze(1)
    curr_finger_status = (curr_finger_status > 0.2).float() # convert gripper qpos to binary

    curr_robot_state_b = torch.cat((curr_ee_pos_b, curr_orient_6d, curr_lin_vel_b, curr_ang_vel_b, curr_finger_status), dim=-1)

    return curr_robot_state_b

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