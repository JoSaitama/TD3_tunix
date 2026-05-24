import numpy as np
import jax
import jax.numpy as jnp


def evaluate_policy(
    env,
    actor_apply,
    actor_params,
    key,
    num_episodes: int = 5,
    max_episode_steps: int = 1000,
):
    """Run deterministic evaluation episodes."""
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    returns = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = reset_fn(reset_key)

        episode_return = 0.0

        for _ in range(max_episode_steps):
            obs_batch = jnp.expand_dims(state.obs, axis=0)
            action = actor_apply(actor_params, obs_batch)[0]
            action = jnp.clip(action, -1.0, 1.0)

            state = step_fn(state, action)
            episode_return += float(jax.device_get(state.reward))

            done = bool(jax.device_get(state.done))
            if done:
                break

        returns.append(episode_return)

    return float(np.mean(returns)), key
