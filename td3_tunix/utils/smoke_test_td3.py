import jax
import jax.numpy as jnp
import optax

from configs.td3_brax_halfcheetah import get_config
from td3_tunix.agents.td3.networks import Actor, Critic
from td3_tunix.agents.td3.train_state import TD3TrainState
from td3_tunix.replay.replay_buffer import (
    create_replay_buffer,
    add_transition,
    sample_batch,
)


def main():
    cfg = get_config()

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    key = jax.random.PRNGKey(cfg.seed)
    key_actor, key_critic, key_sample = jax.random.split(key, 3)

    # Approximate HalfCheetah dimensions.
    # Later, we will read obs_dim/action_dim from Brax automatically.
    obs_dim = 17
    action_dim = 6

    actor = Actor(
        action_dim=action_dim,
        hidden_dims=(cfg.hidden_dim, cfg.hidden_dim),
    )
    critic = Critic(
        hidden_dims=(cfg.hidden_dim, cfg.hidden_dim),
    )

    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

    actor_params = actor.init(key_actor, dummy_obs)
    critic_params = critic.init(key_critic, dummy_obs, dummy_action)

    action = actor.apply(actor_params, dummy_obs)
    q1, q2 = critic.apply(critic_params, dummy_obs, action)

    print("Action shape:", action.shape)
    print("Q1 shape:", q1.shape)
    print("Q2 shape:", q2.shape)

    actor_optimizer = optax.adam(cfg.actor_lr)
    critic_optimizer = optax.adam(cfg.critic_lr)

    _ = TD3TrainState(
        actor_params=actor_params,
        critic_params=critic_params,
        target_actor_params=actor_params,
        target_critic_params=critic_params,
        actor_opt_state=actor_optimizer.init(actor_params),
        critic_opt_state=critic_optimizer.init(critic_params),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        total_it=0,
    )

    buffer = create_replay_buffer(
        capacity=1000,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
    act = jnp.zeros((action_dim,), dtype=jnp.float32)
    reward = jnp.array(1.0, dtype=jnp.float32)
    next_obs = jnp.ones((obs_dim,), dtype=jnp.float32)
    done = jnp.array(0.0, dtype=jnp.float32)

    for _ in range(300):
        buffer = add_transition(buffer, obs, act, reward, next_obs, done)

    batch = sample_batch(buffer, key_sample, cfg.batch_size)

    print("Replay size:", buffer.size)
    print("Batch obs shape:", batch["obs"].shape)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
