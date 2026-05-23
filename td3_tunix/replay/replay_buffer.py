from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class ReplayBuffer:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    ptr: jnp.ndarray
    size: jnp.ndarray
    capacity: int


def create_replay_buffer(
    capacity: int,
    obs_dim: int,
    action_dim: int,
) -> ReplayBuffer:
    return ReplayBuffer(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        dones=jnp.zeros((capacity,), dtype=jnp.float32),
        ptr=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        capacity=capacity,
    )


def add_transition(
    buffer: ReplayBuffer,
    obs,
    action,
    reward,
    next_obs,
    done,
) -> ReplayBuffer:
    idx = buffer.ptr % buffer.capacity

    return ReplayBuffer(
        obs=buffer.obs.at[idx].set(obs),
        actions=buffer.actions.at[idx].set(action),
        rewards=buffer.rewards.at[idx].set(reward),
        next_obs=buffer.next_obs.at[idx].set(next_obs),
        dones=buffer.dones.at[idx].set(done),
        ptr=buffer.ptr + 1,
        size=jnp.minimum(buffer.size + 1, buffer.capacity),
        capacity=buffer.capacity,
    )


def sample_batch(buffer: ReplayBuffer, key, batch_size: int):
    idxs = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=buffer.size,
    )

    return {
        "obs": buffer.obs[idxs],
        "actions": buffer.actions[idxs],
        "rewards": buffer.rewards[idxs],
        "next_obs": buffer.next_obs[idxs],
        "dones": buffer.dones[idxs],
    }
