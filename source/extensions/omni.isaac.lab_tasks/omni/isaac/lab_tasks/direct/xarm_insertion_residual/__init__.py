# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
# from .xarm_insertion_residual_env import XArmInsertionResidualEnvCfg, XArmInsertionResidualEnv
from .xarm_insertion_residual_teacher_env import XArmInsertionResidualTeacherEnvCfg, XArmInsertionResidualTeacherEnv
from .xarm_insertion_residual_student_env import XArmInsertionResidualStudentEnvCfg, XArmInsertionResidualStudentEnv

##
# Register Gym environments.
##

gym.register(
    id="XArm-Residual-Insertion-Teacher", # state-based
    entry_point="omni.isaac.lab_tasks.direct.xarm_insertion_residual:XArmInsertionResidualTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmInsertionResidualTeacherEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualInsertionStatePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-Residual-Insertion-Student", # vision distilled
    entry_point="omni.isaac.lab_tasks.direct.xarm_insertion_residual:XArmInsertionResidualStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmInsertionResidualStudentEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualInsertionDistillationRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="XArm-Residual-Insertion-Vision-From-Scratch", # vision from scratch
#     entry_point="omni.isaac.lab_tasks.direct.xarm_occluded_insertion_residual:XArmInsertionResidualEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": XArmInsertionResidualEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:XArmResidualInsertionVisionPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

