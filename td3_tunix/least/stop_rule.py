import jax
import jax.numpy as jnp


def masked_lower_median(values: jnp.ndarray, mask: jnp.ndarray):
    """Compute a lower median over valid values only.

    This avoids dynamic-shape indexing and is friendly to JAX/XLA.

    If values = [1, 3, 5, 7], lower median = 3.
    That is enough for a robust LEAST-Q threshold in the first version.
    """
    valid_count = jnp.sum(mask.astype(jnp.int32))

    safe_values = jnp.where(mask, values, jnp.inf)
    sorted_values = jnp.sort(safe_values)

    median_index = jnp.maximum((valid_count - 1) // 2, 0)
    median_value = sorted_values[median_index]

    return median_value, valid_count


def least_q_decision(
    q_current,
    q_history: jnp.ndarray,
    valid_history: jnp.ndarray,
    episode_step: int,
    global_step: int,
    least_start_steps: int,
    min_ref_episodes: int,
):
    """Return whether LEAST-Q should stop the current episode.

    Stop rule:
      stop if q_current < median(BQ[:, episode_step])

    Only activated after:
      global_step >= least_start_steps
      and enough valid reference episodes exist at this episode step.
    """
    step = jnp.minimum(
        jnp.array(episode_step, dtype=jnp.int32),
        q_history.shape[1] - 1,
    )

    q_column = q_history[:, step]
    valid_column = valid_history[:, step]

    threshold, valid_count = masked_lower_median(
        q_column,
        valid_column,
    )

    enough_history = valid_count >= min_ref_episodes
    after_start = global_step >= least_start_steps

    stop = jnp.logical_and(
        jnp.logical_and(after_start, enough_history),
        q_current < threshold,
    )

    return stop, threshold, valid_count


def compute_current_min_q(
    critic_apply,
    critic_params,
    obs,
    action,
):
    """Compute min(Q1, Q2) for one transition."""
    obs_batch = jnp.expand_dims(obs, axis=0)
    action_batch = jnp.expand_dims(action, axis=0)

    q1, q2 = critic_apply(
        critic_params,
        obs_batch,
        action_batch,
    )

    return jnp.minimum(q1[0], q2[0])
