import jax
import jax.numpy as jnp


def select_action(
    actor_apply,
    actor_params,
    obs: jnp.ndarray,
    key,
    exploration_noise: float = 0.0,
):
    """Select one action from the deterministic actor with optional Gaussian noise."""
    obs_batch = jnp.expand_dims(obs, axis=0)
    action = actor_apply(actor_params, obs_batch)[0]

    if exploration_noise > 0.0:
        noise = jax.random.normal(key, action.shape) * exploration_noise
        action = action + noise

    return jnp.clip(action, -1.0, 1.0)


def random_action(key, action_dim: int):
    """Uniform random action in [-1, 1]."""
    return jax.random.uniform(
        key,
        shape=(action_dim,),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
