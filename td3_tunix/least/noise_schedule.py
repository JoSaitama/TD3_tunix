from typing import Any

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LeastNoiseState:
    """Adaptive exploration noise state.

    stop_flags:
      Circular window storing recent episode endings.
      1.0 means the episode ended by LEAST stop.
      0.0 means natural end or timeout.

    current_noise:
      Current Gaussian exploration noise std used by TD3 actor.
    """
    stop_flags: Any
    ptr: Any
    size: Any
    current_noise: Any


def create_noise_state(
    window_size: int,
    base_noise: float,
) -> LeastNoiseState:
    return LeastNoiseState(
        stop_flags=jnp.zeros((window_size,), dtype=jnp.float32),
        ptr=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        current_noise=jnp.array(base_noise, dtype=jnp.float32),
    )


def record_episode_stop(
    state: LeastNoiseState,
    is_least_stop,
) -> LeastNoiseState:
    """Record whether the latest episode ended by LEAST stop."""
    idx = state.ptr % state.stop_flags.shape[0]
    flag = jnp.asarray(is_least_stop, dtype=jnp.float32)

    stop_flags = state.stop_flags.at[idx].set(flag)
    ptr = state.ptr + 1
    size = jnp.minimum(state.size + 1, state.stop_flags.shape[0])

    return LeastNoiseState(
        stop_flags=stop_flags,
        ptr=ptr,
        size=size,
        current_noise=state.current_noise,
    )


def recent_stop_rate(state: LeastNoiseState):
    denom = jnp.maximum(state.size, 1)
    return jnp.sum(state.stop_flags) / denom


def update_exploration_noise(
    state: LeastNoiseState,
    base_noise: float,
    max_noise: float,
    target_stop_rate: float,
    smoothing: float,
    min_events: int,
):
    """Update Gaussian exploration noise from recent LEAST stop frequency.

    If recent LEAST stop rate is higher than target_stop_rate,
    increase exploration noise toward max_noise. Otherwise decay toward base_noise.

    This is a TPU/JAX-friendly implementation of the paper's adaptive
    exploration-noise idea.
    """
    rate = recent_stop_rate(state)

    enough_events = state.size >= min_events

    excess = (rate - target_stop_rate) / (1.0 - target_stop_rate + 1e-8)
    excess = jnp.clip(excess, 0.0, 1.0)

    target_noise = base_noise + excess * (max_noise - base_noise)
    target_noise = jnp.where(enough_events, target_noise, base_noise)

    new_noise = (1.0 - smoothing) * state.current_noise + smoothing * target_noise

    new_state = LeastNoiseState(
        stop_flags=state.stop_flags,
        ptr=state.ptr,
        size=state.size,
        current_noise=new_noise,
    )

    return new_state, rate
