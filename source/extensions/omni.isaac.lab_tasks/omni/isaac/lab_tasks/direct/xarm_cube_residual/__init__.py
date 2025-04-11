# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .xarm_cube_residual_cam_env_v0 import XArmCubeResidualCamLocalEnv, XArmCubeResidualCamLocalEnvCfg
from .xarm_cube_residual_cam_env_v1 import XArmCubeResidualCamLocalBinaryEnv, XArmCubeResidualCamLocalBinaryEnvCfg
from .xarm_cube_residual_cam_env_v2 import XArmCubeResidualCamLocalBinaryNewEnv, XArmCubeResidualCamLocalBinaryNewEnvCfg
from .xarm_cube_residual_cam_env_v3 import XArmCubeResidualCamLocalBinaryV3Env, XArmCubeResidualCamLocalBinaryV3EnvCfg
from .xarm_cube_residual_cam_env_v4 import XArmCubeResidualCamLocalBinaryV4Env, XArmCubeResidualCamLocalBinaryV4EnvCfg

from .xarm_cube_residual_state_env_v2 import XArmCubeResidualStateLocalBinaryNewEnv, XArmCubeResidualStateLocalBinaryNewEnvCfg
from .xarm_cube_residual_state_env_v3 import XArmCubeResidualStateLocalBinaryV3Env, XArmCubeResidualStateLocalBinaryV3EnvCfg


##
# Register Gym environments.
##

gym.register(
    id="XArm-Residual-Cube-v0", #NOTE: local policy with relative coordinates, deployable on real setup
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualCamLocalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualCamLocalEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-v1", #NOTE: v0 + binary gripper status
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualCamLocalBinaryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualCamLocalBinaryEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-v2", #NOTE: v1 + follow ResiP designs (network structures & obs space)
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualCamLocalBinaryNewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualCamLocalBinaryNewEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamV2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-v3", #NOTE: v2 + formalize observations + 10dim action space
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualCamLocalBinaryV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualCamLocalBinaryV3EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamV2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-v4", #NOTE: v3 + dmr + 10dim state space
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualCamLocalBinaryV4Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualCamLocalBinaryV4EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamV2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="XArm-Residual-Cube-State", #NOTE: v1 + follow ResiP designs (network structures & obs space)
#     entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualStateLocalBinaryNewEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": XArmCubeResidualStateLocalBinaryNewEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamV2Cfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="XArm-Residual-Cube-State-v2", #NOTE: (cam v4) + relative_obs + dmr
    entry_point="omni.isaac.lab_tasks.direct.xarm_cube_residual:XArmCubeResidualStateLocalBinaryV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualStateLocalBinaryV3EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubePPORunnerCamV2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)