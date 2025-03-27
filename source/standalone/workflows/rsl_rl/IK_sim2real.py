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

from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Usd

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
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),                  
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -0.0037,
                "joint2": -0.781,
                "joint3": 0.00095,
                "joint4": 0.5280, # 30
                "joint5": -0.003766,
                "joint6": 1.31369, # 75
                "joint7": -0.0016879,
                "drive_joint": 0.00625,
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

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    decimation = 4
    sim_step_counter = 0
    count = 0

    # Simulation loop
    curr_jpos_list = torch.from_numpy(np.loadtxt("/home/shuosha/projects/IsaacLab/RRL/demo_trajs/real_jpos_trajs/curr_jpos.txt")).float()
    curr_jpos_list = curr_jpos_list.to('cuda:0')[60:, :7]
    delta_jpos_list = torch.from_numpy(np.loadtxt("/home/shuosha/projects/IsaacLab/RRL/demo_trajs/real_jpos_trajs/delta_jpos.txt")).float()
    delta_jpos_list = delta_jpos_list.to('cuda:0')[60:, :7]
    new_jpos_list = torch.from_numpy(np.loadtxt("/home/shuosha/projects/IsaacLab/RRL/demo_trajs/real_jpos_trajs/new_jpos.txt")).float()
    new_jpos_list = new_jpos_list.to('cuda:0')[60:, :7]

    joint_pos_init = torch.cat((curr_jpos_list[0, :].clone().unsqueeze(0), torch.zeros((1,6)).to('cuda:0')), dim=1)
    joint_vel_init = robot.data.default_joint_vel.clone()

    robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
    robot.reset()
    print("done resetting")
    
    new_jpos_sim = []

    jpos_traj_sim = []
    jpos_traj_real = []
    jpos_goal_traj = []

    while simulation_app.is_running(): 
        # match init pos at every step
        # joint_pos_init = torch.cat((curr_jpos_list[sim_step_counter, :].clone().unsqueeze(0), torch.zeros((1,6)).to('cuda:0')), dim=1)
        # joint_vel_init = robot.data.default_joint_vel.clone()
        # robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
        # robot.reset() 

        print("curr jpos sim: ", robot.data.joint_pos[:, :7].clone())
        jpos_traj_sim.append(robot.data.joint_pos[:, :7].clone())
        print("curr jpos real: ", curr_jpos_list[sim_step_counter, :])
        jpos_traj_real.append(curr_jpos_list[sim_step_counter, :].clone().unsqueeze(0))

        joint_pos_des = curr_jpos_list[sim_step_counter, :] + delta_jpos_list[sim_step_counter, :]
        print("goal jpos: ", joint_pos_des)
        jpos_goal_traj.append(joint_pos_des.clone().unsqueeze(0))

        for _ in range(decimation):
            # apply jpos action
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

        # import pdb; pdb.set_trace()
        new_jpos = robot.data.joint_pos[:, :7].clone()
        print("new jpos sim: ", new_jpos)
        new_jpos_sim.append(new_jpos)

        print("new jpos real: ", new_jpos_list[sim_step_counter, :])
        print("jpos diff: ", torch.norm(new_jpos - new_jpos_list[sim_step_counter, :]))

        # import pdb; pdb.set_trace()

        if sim_step_counter == 99:
            break

        sim_step_counter += 1

    data1 = np.concatenate([t.detach().cpu().numpy() for t in jpos_traj_sim], axis=0)  # shape: (100, 7)
    data2 = np.concatenate([t.detach().cpu().numpy() for t in jpos_traj_real], axis=0)
    data3 = np.concatenate([t.detach().cpu().numpy() for t in jpos_goal_traj], axis=0)

    time_steps = np.arange(100)

    # Create 7 subplots (one for each joint)
    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)

    # Loop through each joint index (0 to 6)
    for joint in range(7):
        axs[joint].plot(time_steps, data1[:, joint], label='sim jpos traj')
        axs[joint].plot(time_steps, data2[:, joint], label='real jpos traj')
        axs[joint].plot(time_steps, data3[:, joint], label='goal jpos traj')
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
    run_simulator(sim, scene)

def create_filter_pairs(prim1: str, prim2: str):
    stage = get_current_stage()
    filteredpairs_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim1))
    filteredpairs_rel = filteredpairs_api.CreateFilteredPairsRel()
    filteredpairs_rel.AddTarget(prim2)
    stage.Save()

if __name__ == "__main__":
    torch.set_printoptions(precision=4)

    # run the main function
    main()
    # close sim app
    simulation_app.close()