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