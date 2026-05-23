import jax
import jax.numpy as jnp
import optax

from configs.td3_brax_halfcheetah import get_config
from td3_tunix.agents.td3.networks import Actor, Critic
from td3_tunix.agents.td3.train_state import TD3TrainState
from td3_tunix.agents.td3.update import make_td3_update
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
    key_actor, key_critic, key_sample, key_update = jax.random.split(key, 4)

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

    actor_optimizer = optax.adam(cfg.actor_lr)
    critic_optimizer = optax.adam(cfg.critic_lr)

    state = TD3TrainState(
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
    action = jnp.zeros((action_dim,), dtype=jnp.float32)
    reward = jnp.array(1.0, dtype=jnp.float32)
    next_obs = jnp.ones((obs_dim,), dtype=jnp.float32)
    done = jnp.array(0.0, dtype=jnp.float32)

    for _ in range(300):
        buffer = add_transition(buffer, obs, action, reward, next_obs, done)

    batch = sample_batch(buffer, key_sample, cfg.batch_size)

    update_step = make_td3_update(
        actor.apply,
        critic.apply,
        cfg,
    )

    new_state, metrics = update_step(state, batch, key_update)

    print("Old total_it:", state.total_it)
    print("New total_it:", new_state.total_it)
    print("Critic loss:", metrics["critic_loss"])
    print("Actor loss:", metrics["actor_loss"])
    print("TD3 update smoke test passed.")


if __name__ == "__main__":
    main()
