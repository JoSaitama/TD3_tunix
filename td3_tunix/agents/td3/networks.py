from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class Actor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        xu = jnp.concatenate([obs, action], axis=-1)

        q1 = xu
        for h in self.hidden_dims:
            q1 = nn.Dense(h)(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        q2 = xu
        for h in self.hidden_dims:
            q2 = nn.Dense(h)(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)

        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)


def soft_update(params, target_params, tau: float):
    return jax_tree_map(
        lambda p, tp: tau * p + (1.0 - tau) * tp,
        params,
        target_params,
    )


def jax_tree_map(fn, tree_a, tree_b):
    import jax
    return jax.tree_util.tree_map(fn, tree_a, tree_b)
