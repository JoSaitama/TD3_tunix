from typing import Any

import optax
from flax import struct


@struct.dataclass
class TD3TrainState:
    actor_params: Any
    critic_params: Any
    target_actor_params: Any
    target_critic_params: Any

    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    # Optimizer objects contain Python functions, so they should not be traced by JAX.
    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    total_it: Any = 0
