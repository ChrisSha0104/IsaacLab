from omni.isaac.lab.utils import configclass
from dataclasses import MISSING
from typing import Literal

@configclass
class RslRlResidualPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_logstd: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_size: int = MISSING
    """The hidden dimensions of the actor network."""

    actor_num_layers: int = MISSING
    """The number of layers in the actor network."""

    critic_hidden_size: int = MISSING
    """The hidden dimensions of the critic network."""

    critic_num_layers: int = MISSING
    """The number of layers in the critic network."""

    actor_activation: str = MISSING
    """The activation function for the actor network."""

    critic_activation: str = MISSING
    """The activation function for the critic network."""

    action_head_std: float = MISSING
    """The initialization gain of the last layer of the action head."""

    action_scale: float = MISSING
    """The scale of the action residual in the environment."""

    critic_last_layer_bias_const: float = MISSING
    """The constant bias for the last layer of the critic network."""
    
    critic_last_layer_std: float = MISSING
    """The standard deviation for the last layer of the critic network."""

    use_visual_encoder: bool = MISSING
    """Whether to use a visual encoder."""

    visual_idx_actor: list[int] = MISSING
    """The indices for the visual encoder in the actor network."""

    visual_idx_critic: list[int] = MISSING
    """The indices for the visual encoder in the critic network."""

    encoder_output_dim: int = MISSING
    """The output dimension of the visual encoder."""