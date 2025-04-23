# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlResidualPpoActorCriticCfg,
)

@configclass
class XArmResidualCubePPORunnerCamCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 40000
    save_interval = 100
    experiment_name = "xarm-cube-residual"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticVisual",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOWithResNet",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule= "adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

@configclass
class XArmResidualCubePPORunnerCamV2Cfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 40000
    save_interval = 100
    experiment_name = "xarm-cube-residual"
    empirical_normalization = False
    policy = RslRlResidualPpoActorCriticCfg(
        class_name="ResidualActorCriticVisual",
        init_logstd=-3.0,
        actor_hidden_size=512,
        actor_num_layers=2,
        actor_activation="SiLU",
        critic_hidden_size=512,
        critic_num_layers=2,
        critic_activation="SiLU",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOWithResNet",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule= "adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )