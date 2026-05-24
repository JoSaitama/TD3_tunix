import argparse
import time

import jax
import jax.numpy as jnp
import optax

from configs.td3_brax_halfcheetah import get_config
from td3_tunix.envs.make_env import make_brax_env
from td3_tunix.agents.td3.networks import Actor, Critic
from td3_tunix.agents.td3.train_state import TD3TrainState
from td3_tunix.agents.td3.update import make_td3_update
from td3_tunix.agents.td3.agent import select_action, random_action
from td3_tunix.replay.replay_buffer import (
    create_replay_buffer,
    add_transition,
    sample_batch,
)
from td3_tunix.training.eval import evaluate_policy
from td3_tunix.least.reflection_buffer import (
    create_least_q_buffer,
    record_q_value,
    advance_episode_slot,
)
from td3_tunix.least.stop_rule import (
    compute_current_min_q,
    least_q_decision,
)
from td3_tunix.least.metrics import (
    LeastEpisodeStats,
    format_least_episode_log,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # TD3 / environment arguments
    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--start_timesteps", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--replay_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--num_eval_episodes", type=int, default=3)

    # LEAST-Q arguments
    parser.add_argument("--least_start_steps", type=int, default=10000)
    parser.add_argument("--least_buffer_size", type=int, default=20)
    parser.add_argument("--least_min_ref_episodes", type=int, default=5)
    parser.add_argument("--least_min_episode_steps", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    cfg.env_name = args.env_name
    cfg.seed = args.seed
    cfg.total_steps = args.total_steps
    cfg.start_timesteps = args.start_timesteps
    cfg.eval_freq = args.eval_freq
    cfg.replay_size = args.replay_size
    cfg.batch_size = args.batch_size
    cfg.max_episode_steps = args.max_episode_steps
    cfg.num_eval_episodes = args.num_eval_episodes

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("Config:", cfg)
    print(
        "LEAST-Q config:",
        {
            "least_start_steps": args.least_start_steps,
            "least_buffer_size": args.least_buffer_size,
            "least_min_ref_episodes": args.least_min_ref_episodes,
            "least_min_episode_steps": args.least_min_episode_steps,
        },
    )

    key = jax.random.PRNGKey(cfg.seed)

    env = make_brax_env(cfg.env_name)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    obs_dim = env.observation_size
    action_dim = env.action_size

    print("Env:", cfg.env_name)
    print("Observation dim:", obs_dim)
    print("Action dim:", action_dim)

    key, key_actor, key_critic, key_env = jax.random.split(key, 4)

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

    td3_state = TD3TrainState(
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

    replay_buffer = create_replay_buffer(
        capacity=cfg.replay_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    least_buffer = create_least_q_buffer(
        buffer_size=args.least_buffer_size,
        max_episode_steps=cfg.max_episode_steps,
    )

    update_step = make_td3_update(
        actor.apply,
        critic.apply,
        cfg,
    )

    env_state = reset_fn(key_env)

    episode_return = 0.0
    episode_length = 0
    episode_count = 0

    least_stop_count = 0
    natural_or_timeout_count = 0

    start_time = time.time()

    for t in range(cfg.total_steps):
        key, action_key, sample_key, update_key, reset_key, eval_key = jax.random.split(
            key,
            6,
        )

        obs = env_state.obs
        current_episode_step = episode_length

        if t < cfg.start_timesteps:
            action = random_action(action_key, action_dim)
        else:
            action = select_action(
                actor.apply,
                td3_state.actor_params,
                obs,
                action_key,
                cfg.exploration_noise,
            )

        next_env_state = step_fn(env_state, action)

        reward = next_env_state.reward
        env_done = next_env_state.done

        # Compute current Q before TD3 update.
        current_q = compute_current_min_q(
            critic.apply,
            td3_state.critic_params,
            obs,
            action,
        )

        # Decide LEAST-Q before recording current step into the reflection buffer.
        least_stop, threshold, valid_count = least_q_decision(
            q_current=current_q,
            q_history=least_buffer.q_values,
            valid_history=least_buffer.valid,
            episode_step=current_episode_step,
            global_step=t,
            least_start_steps=args.least_start_steps,
            min_ref_episodes=args.least_min_ref_episodes,
            least_min_episode_steps=args.least_min_episode_steps,
        )

        # Record current Q after decision, so threshold uses previous episodes only.
        least_buffer = record_q_value(
            least_buffer,
            episode_step=current_episode_step,
            q_value=current_q,
        )

        # Store transition.
        # Important: LEAST stop is an artificial truncation, not a natural MDP terminal.
        # Therefore, done for TD target uses env_done only.
        replay_buffer = add_transition(
            replay_buffer,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_env_state.obs,
            done=env_done.astype(jnp.float32),
        )

        episode_return += float(jax.device_get(reward))
        episode_length += 1

        done_bool = bool(jax.device_get(env_done))
        least_stop_bool = bool(jax.device_get(least_stop))
        timeout_bool = episode_length >= cfg.max_episode_steps

        if done_bool or least_stop_bool or timeout_bool:
            episode_count += 1

            if least_stop_bool:
                least_stop_count += 1
            else:
                natural_or_timeout_count += 1

            stats = LeastEpisodeStats(
                episode_return=episode_return,
                episode_length=episode_length,
                least_stop=least_stop_bool,
                stop_step=current_episode_step,
                stop_q=float(jax.device_get(current_q)),
                stop_threshold=float(jax.device_get(threshold)),
            )

            print(
                format_least_episode_log(
                    episode_id=episode_count,
                    global_step=t + 1,
                    stats=stats,
                )
            )

            least_buffer = advance_episode_slot(least_buffer)
            env_state = reset_fn(reset_key)

            episode_return = 0.0
            episode_length = 0
        else:
            env_state = next_env_state

        if t >= cfg.start_timesteps and replay_buffer.size >= cfg.batch_size:
            batch = sample_batch(replay_buffer, sample_key, cfg.batch_size)
            td3_state, metrics = update_step(td3_state, batch, update_key)

        if (t + 1) % cfg.eval_freq == 0:
            avg_return, key = evaluate_policy(
                env=env,
                actor_apply=actor.apply,
                actor_params=td3_state.actor_params,
                key=eval_key,
                num_episodes=cfg.num_eval_episodes,
                max_episode_steps=cfg.max_episode_steps,
            )

            elapsed = time.time() - start_time
            total_ended = least_stop_count + natural_or_timeout_count
            least_stop_rate = (
                least_stop_count / total_ended if total_ended > 0 else 0.0
            )

            print(
                f"[eval] step={t + 1}, "
                f"avg_return={avg_return:.2f}, "
                f"buffer_size={int(jax.device_get(replay_buffer.size))}, "
                f"td3_updates={td3_state.total_it}, "
                f"least_stops={least_stop_count}, "
                f"episode_ends={total_ended}, "
                f"least_stop_rate={least_stop_rate:.3f}, "
                f"elapsed_sec={elapsed:.1f}"
            )

    print("TD3 + LEAST-Q training run finished.")


if __name__ == "__main__":
    main()
