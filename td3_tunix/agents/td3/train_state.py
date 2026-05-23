from dataclasses import dataclass
from typing import Any

import optax


@dataclass
class TD3TrainState:
    actor_params: Any
    critic_params: Any
    target_actor_params: Any
    target_critic_params: Any

    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation

    total_it: int = 0
