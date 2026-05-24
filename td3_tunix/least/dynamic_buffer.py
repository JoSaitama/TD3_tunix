import jax
import jax.numpy as jnp


def make_active_slot_mask(current_slot, buffer_size: int, active_size):
    """Return mask for the most recent active_size completed episode slots.

    current_slot is the row currently being written, so it is excluded.
    Previous completed episode slots are:
      current_slot - 1, current_slot - 2, ...
    """
    idx = jnp.arange(buffer_size)
    distance = (current_slot - idx) % buffer_size

    return jnp.logical_and(
        distance >= 1,
        distance <= active_size,
    )


def estimate_q_entropy(
    q_values,
    valid,
    active_slot_mask,
    num_bins: int = 20,
    eps: float = 1e-8,
):
    """Estimate normalized entropy H(B_Q) over the active local buffer.

    q_values: [K, H]
    valid: [K, H]
    active_slot_mask: [K]

    Returns entropy normalized to [0, 1].
    """
    active_valid = jnp.logical_and(
        valid,
        active_slot_mask[:, None],
    )

    count = jnp.sum(active_valid.astype(jnp.float32))

    safe_min_values = jnp.where(active_valid, q_values, jnp.inf)
    safe_max_values = jnp.where(active_valid, q_values, -jnp.inf)

    q_min = jnp.min(safe_min_values)
    q_max = jnp.max(safe_max_values)

    def non_empty_entropy():
        width = (q_max - q_min + eps) / num_bins

        bin_idx = jnp.floor((q_values - q_min) / width).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)

        one_hot = jax.nn.one_hot(bin_idx, num_bins)
        weighted = one_hot * active_valid[..., None].astype(jnp.float32)

        counts = jnp.sum(weighted, axis=(0, 1))
        probs = counts / (jnp.sum(counts) + eps)

        entropy = -jnp.sum(
            jnp.where(
                probs > 0.0,
                probs * jnp.log(probs + eps),
                0.0,
            )
        )

        return entropy / jnp.log(float(num_bins))

    def empty_entropy():
        return jnp.array(0.0, dtype=jnp.float32)

    valid_entropy = jnp.logical_and(count > 1.0, q_max > q_min)

    return jax.lax.cond(
        valid_entropy,
        non_empty_entropy,
        empty_entropy,
    )


def update_active_size(
    active_size,
    current_entropy,
    baseline_entropy,
    overflow_rate: float,
    adjust_scale: int,
    min_active_size: int,
    max_active_size: int,
):
    """Update active local reflection-set size.

    If entropy is high, shrink the active local buffer.
    If entropy is low, expand it slowly.
    """
    high_entropy = current_entropy > (1.0 + overflow_rate) * baseline_entropy
    low_entropy = current_entropy <= baseline_entropy

    shrink_size = jnp.maximum(active_size - adjust_scale, min_active_size)
    expand_size = jnp.minimum(active_size + adjust_scale, max_active_size)

    new_size = jnp.where(
        high_entropy,
        shrink_size,
        jnp.where(low_entropy, expand_size, active_size),
    )

    return new_size
