from typing import Any

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LeastQBuffer:
    """Reflection buffer for LEAST-Q.

    q_values: [K, H]
    valid: [K, H]
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
        q_values=jnp.zeros((buffer_size, max_episode_steps), dtype=jnp.float32),
        valid=jnp.zeros((buffer_size, max_episode_steps), dtype=bool),
        current_slot=jnp.array(0, dtype=jnp.int32),
        buffer_size=buffer_size,
        max_episode_steps=max_episode_steps,
    )


def record_q_value(
    buffer: LeastQBuffer,
    episode_step: int,
    q_value,
) -> LeastQBuffer:
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


@struct.dataclass
class LeastQLossBuffer:
    """Reflection buffer for LEAST-Q-loss.

    This corresponds to Algorithm 1:

      B_Q stores min(Q1, Q2)
      B_G stores L_i, here implemented as absolute TD error

    q_values:    [K, H]
    loss_values: [K, H]
    valid:       [K, H]
    """
    q_values: Any
    loss_values: Any
    valid: Any
    current_slot: Any
    buffer_size: int = struct.field(pytree_node=False)
    max_episode_steps: int = struct.field(pytree_node=False)


def create_least_qloss_buffer(
    buffer_size: int,
    max_episode_steps: int,
) -> LeastQLossBuffer:
    return LeastQLossBuffer(
        q_values=jnp.zeros((buffer_size, max_episode_steps), dtype=jnp.float32),
        loss_values=jnp.zeros((buffer_size, max_episode_steps), dtype=jnp.float32),
        valid=jnp.zeros((buffer_size, max_episode_steps), dtype=bool),
        current_slot=jnp.array(0, dtype=jnp.int32),
        buffer_size=buffer_size,
        max_episode_steps=max_episode_steps,
    )


def record_q_loss_value(
    buffer: LeastQLossBuffer,
    episode_step: int,
    q_value,
    loss_value,
) -> LeastQLossBuffer:
    step = jnp.minimum(
        jnp.array(episode_step, dtype=jnp.int32),
        buffer.max_episode_steps - 1,
    )

    q_values = buffer.q_values.at[buffer.current_slot, step].set(q_value)
    loss_values = buffer.loss_values.at[buffer.current_slot, step].set(loss_value)
    valid = buffer.valid.at[buffer.current_slot, step].set(True)

    return LeastQLossBuffer(
        q_values=q_values,
        loss_values=loss_values,
        valid=valid,
        current_slot=buffer.current_slot,
        buffer_size=buffer.buffer_size,
        max_episode_steps=buffer.max_episode_steps,
    )


def advance_qloss_episode_slot(buffer: LeastQLossBuffer) -> LeastQLossBuffer:
    next_slot = (buffer.current_slot + 1) % buffer.buffer_size

    q_values = buffer.q_values.at[next_slot, :].set(0.0)
    loss_values = buffer.loss_values.at[next_slot, :].set(0.0)
    valid = buffer.valid.at[next_slot, :].set(False)

    return LeastQLossBuffer(
        q_values=q_values,
        loss_values=loss_values,
        valid=valid,
        current_slot=next_slot,
        buffer_size=buffer.buffer_size,
        max_episode_steps=buffer.max_episode_steps,
    )
