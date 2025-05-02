# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlResidualPpoActorCriticCfg
)

# from RRL.rl_models.residual_rl_cfg import RslRlResidualPpoActorCriticCfg # TODO debug this import


@configclass
class XArmResidualCubeStatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 50
    experiment_name = "xarm-cube-residual-state" 
    empirical_normalization = False
    policy = RslRlResidualPpoActorCriticCfg(
        class_name="ResidualActorCritic",
        init_logstd=-1.5,
        actor_hidden_size=256,
        actor_num_layers=2,
        actor_activation="ReLU",
        critic_hidden_size=256,
        critic_num_layers=2,
        critic_activation="ReLU",
        action_head_std=0.0,
        action_scale=0.1,
        critic_last_layer_bias_const=0.25,
        critic_last_layer_std=0.25,
        use_visual_encoder=False,
        visual_idx_actor=[20,20+120*120],
        visual_idx_critic=[29,29+120*120],
        encoder_output_dim=128,
        learn_std=True, 
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="ResidualPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4, #REDUCE LEARNING RATE
        schedule= "cosine", 
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

@configclass
class XArmResidualCubeVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 50
    experiment_name = "xarm-cube-residual-vision"
    empirical_normalization = False
    policy = RslRlResidualPpoActorCriticCfg(
        class_name="ResidualActorCritic",
        init_logstd=-1.5,
        actor_hidden_size=256, # TODO: maybe larger MLP for vision based policy
        actor_num_layers=2,
        actor_activation="ReLU",
        critic_hidden_size=256,
        critic_num_layers=2,
        critic_activation="ReLU",
        action_head_std=0.0,
        action_scale=0.1,
        critic_last_layer_bias_const=0.25,
        critic_last_layer_std=0.25,
        use_visual_encoder=True,
        visual_idx_actor=[20,20+120*120],
        visual_idx_critic=[29,29+120*120],
        encoder_output_dim=128,
        learn_std=True, 
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="ResidualPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4, #REDUCE LEARNING RATE
        schedule= "cosine", 
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )