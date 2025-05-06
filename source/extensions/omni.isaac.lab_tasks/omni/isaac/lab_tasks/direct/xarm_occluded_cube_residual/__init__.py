# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .xarm_cube_residual_env import XArmCubeResidualEnvCfg, XArmCubeResidualEnv
from .xarm_cube_residual_teacher_env import XArmCubeResidualTeacherEnvCfg, XArmCubeResidualTeacherEnv
from .xarm_cube_residual_student_env import XArmCubeResidualStudentEnvCfg, XArmCubeResidualStudentEnv

##
# Register Gym environments.
##

gym.register(
    id="XArm-Residual-Cube-Teacher", # state-based
    entry_point="omni.isaac.lab_tasks.direct.xarm_occluded_cube_residual:XArmCubeResidualTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualTeacherEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubeStatePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-Student", # vision distilled
    entry_point="omni.isaac.lab_tasks.direct.xarm_occluded_cube_residual:XArmCubeResidualStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualStudentEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubeDistillationRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Cube-Vision-From-Scratch", # vision from scratch
    entry_point="omni.isaac.lab_tasks.direct.xarm_occluded_cube_residual:XArmCubeResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmCubeResidualEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualCubeVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

