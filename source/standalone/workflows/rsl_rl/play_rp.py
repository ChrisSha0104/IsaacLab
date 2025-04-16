# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
import cv2
import numpy as np
import matplotlib.pyplot as plt

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser) # type: ignore
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time

from rsl_rl.runners import OnPolicyRunner
from RRL.rl_models import OnPolicyRunnerResidual

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def depth2fgpcd(depth, intrinsic_matrix):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic_matrix).T
    return points

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli) # type: ignore

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env) # type: ignore

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) #NOTE: initial env reset occurs in the initialization of Vectorized Env # type: ignore

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner = OnPolicyRunnerResidual(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device) # type: ignore
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    #TODO: debug why encoder shape error
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    obs, _ = env.get_observations()
    # print("initial obs", obs[0,:8])
    # print("initial obs", obs[0,24:32])
    # print(obs)
    timestep = 0
    slowly = False
    
    # turn off dmr during play?
    apply_dmr = False
    setattr(env.cfg, "apply_dmr", apply_dmr) # NOTE: only applies after initial reset!!!

    mark_demo = True
    mark_ee = True
    setattr(env.cfg, "mark_demo", mark_demo)
    setattr(env.cfg, "mark_ee", mark_ee)

    slow_start = False
    i = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if slow_start:
                time.sleep(2)
                slow_start = False
            if slowly:
                time.sleep(0.2)
            # agent stepping
            print("robot obs: ", obs[:,:10])
            print("teleop obs: ", obs[:,10:20])
            print("cube obs: ", obs[:,20:])
            actions = policy(obs) # output residual
            # print("output: ", actions)
            # print("output norm: ", torch.norm(actions))

            # env stepping
            obs, rew, dones, extras = env.step(actions)
            if getattr(env.cfg, "show_camera", False): 
                raw_depth = obs[0,20:].reshape(120,120).detach().cpu().numpy()              # convert to np array
                # rotated_depth = cv2.rotate(raw_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(raw_depth, alpha=255 / raw_depth[raw_depth < 15].max().item()), cv2.COLORMAP_JET)
                depth_vis = cv2.resize(depth_vis,(480,480))
                cv2.imshow("depth_image",depth_vis)
                cv2.waitKey(1)
            
            # import pdb; pdb.set_trace()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    
    # close the simulator
    env.close()
    cv2.destroyAllWindows()

def format_tensor(tensor, precision=3):
    if tensor.dim() == 0:  # Handle scalar tensor
        print(f"{tensor.item():.{precision}f}")
    elif tensor.dim() == 1:  # Handle 1D tensor
        formatted = [f"{value:.{precision}f}" for value in tensor.cpu().tolist()]
        print(formatted)
    elif tensor.dim() == 2:  # Handle 2D tensor
        formatted_rows = [
            [f"{value:.{precision}f}" for value in row.cpu().tolist()] for row in tensor
        ]
        for row in formatted_rows:
            print(row)
    else:
        print("Tensor with more than 2 dimensions is not supported.")

def trim_and_downsample(depth_tensor):
    """
    Trims the sides of the depth tensor to keep only the middle 480 pixels in dim=2,
    then downsamples the last two dimensions by keeping every 4th pixel.

    Args:
        depth_tensor (torch.Tensor): Shape (num_envs, 1, H=848, W=480)

    Returns:
        torch.Tensor: Shape (num_envs, 1, H=120, W=120)
    """
    H, W = 848, 480
    depth_tensor = depth_tensor.reshape(H, W)
    device = depth_tensor.device

    # Step 1: Trim the middle 480 pixels in dim=2 (H dimension)
    start_idx = (H - 480) // 2
    end_idx = start_idx + 480
    depth_tensor_trimmed = depth_tensor[start_idx:end_idx, :]

    # Step 2: Downsample by taking every 4th pixel in both H and W
    depth_tensor_downsampled = depth_tensor_trimmed[::4, ::4]

    return depth_tensor_downsampled

if __name__ == "__main__":
    # run the main function
    main() # type: ignore
    # close sim app
    simulation_app.close()
