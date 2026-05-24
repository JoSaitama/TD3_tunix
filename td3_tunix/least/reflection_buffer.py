from typing import Any

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LeastQBuffer:
    """Reflection buffer for LEAST-Q.

    q_values: [K, H]
      K = number of recent episodes
      H = maximum episode length

    valid: [K, H]
      valid[k, t] = whether q_values[k, t] is a real recorded Q value.

    current_slot:
      Which episode row is currently being written.
    """
    q_values: Any
    valid: Any
    current_slot: Any
    buffer_size: int = struct.field(pytree_node=False)
    max_episode_steps: int = struct.field(pytree_node=False)


def create_least_q_buffer(
    buffer_size: int,
    max_episode_steps: int,
) -> LeastQBuffer:
    return LeastQBuffer(
        q_values=jnp.zeros(
            (buffer_size, max_episode_steps),
            dtype=jnp.float32,
        ),
        valid=jnp.zeros(
            (buffer_size, max_episode_steps),
            dtype=bool,
        ),
        current_slot=jnp.array(0, dtype=jnp.int32),
        buffer_size=buffer_size,
        max_episode_steps=max_episode_steps,
    )


def record_q_value(
    buffer: LeastQBuffer,
    episode_step: int,
    q_value,
) -> LeastQBuffer:
    """Record current Q value at the current episode slot and step."""
    step = jnp.minimum(
        jnp.array(episode_step, dtype=jnp.int32),
        buffer.max_episode_steps - 1,
    )

    q_values = buffer.q_values.at[buffer.current_slot, step].set(q_value)
    valid = buffer.valid.at[buffer.current_slot, step].set(True)

    return LeastQBuffer(
        q_values=q_values,
        valid=valid,
        current_slot=buffer.current_slot,
        buffer_size=buffer.buffer_size,
        max_episode_steps=buffer.max_episode_steps,
    )


def advance_episode_slot(buffer: LeastQBuffer) -> LeastQBuffer:
    """Move to the next episode slot and clear that row."""
    next_slot = (buffer.current_slot + 1) % buffer.buffer_size

    q_values = buffer.q_values.at[next_slot, :].set(0.0)
    valid = buffer.valid.at[next_slot, :].set(False)

    return LeastQBuffer(
        q_values=q_values,
        valid=valid,
        current_slot=next_slot,
        buffer_size=buffer.buffer_size,
        max_episode_steps=buffer.max_episode_steps,
    )
