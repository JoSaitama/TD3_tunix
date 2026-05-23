import jax
import jax.numpy as jnp


def critic_loss_fn(
    critic_params,
    target_critic_params,
    target_actor_params,
    critic_apply,
    actor_apply,
    batch,
    key,
    gamma: float,
    policy_noise: float,
    noise_clip: float,
):
    next_action = actor_apply(target_actor_params, batch["next_obs"])

    noise = jax.random.normal(key, next_action.shape) * policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)

    next_action = jnp.clip(next_action + noise, -1.0, 1.0)

    target_q1, target_q2 = critic_apply(
        target_critic_params,
        batch["next_obs"],
        next_action,
    )
    target_q = jnp.minimum(target_q1, target_q2)

    not_done = 1.0 - batch["dones"]
    target = batch["rewards"] + gamma * not_done * target_q
    target = jax.lax.stop_gradient(target)

    current_q1, current_q2 = critic_apply(
        critic_params,
        batch["obs"],
        batch["actions"],
    )

    loss_q1 = jnp.mean((current_q1 - target) ** 2)
    loss_q2 = jnp.mean((current_q2 - target) ** 2)

    return loss_q1 + loss_q2


def actor_loss_fn(
    actor_params,
    critic_params,
    actor_apply,
    critic_apply,
    obs,
):
    action = actor_apply(actor_params, obs)
    q1, _ = critic_apply(critic_params, obs, action)
    return -jnp.mean(q1)
