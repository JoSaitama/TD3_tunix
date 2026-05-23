import jax
import optax

from td3_tunix.agents.td3.losses import critic_loss_fn, actor_loss_fn
from td3_tunix.agents.td3.networks import soft_update
from td3_tunix.agents.td3.train_state import TD3TrainState


def make_td3_update(actor_apply, critic_apply):
    @jax.jit
    def update_step(state: TD3TrainState, batch, key, cfg):
        key_critic, key_actor = jax.random.split(key)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            state.critic_params,
            state.target_critic_params,
            state.target_actor_params,
            critic_apply,
            actor_apply,
            batch,
            key_critic,
            cfg.gamma,
            cfg.policy_noise,
            cfg.noise_clip,
        )

        critic_updates, new_critic_opt_state = state.critic_optimizer.update(
            critic_grads,
            state.critic_opt_state,
            state.critic_params,
        )
        new_critic_params = optax.apply_updates(
            state.critic_params,
            critic_updates,
        )

        def update_actor_and_targets(_):
            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
                state.actor_params,
                new_critic_params,
                actor_apply,
                critic_apply,
                batch["obs"],
            )

            actor_updates, new_actor_opt_state = state.actor_optimizer.update(
                actor_grads,
                state.actor_opt_state,
                state.actor_params,
            )
            new_actor_params = optax.apply_updates(
                state.actor_params,
                actor_updates,
            )

            new_target_actor_params = soft_update(
                new_actor_params,
                state.target_actor_params,
                cfg.tau,
            )
            new_target_critic_params = soft_update(
                new_critic_params,
                state.target_critic_params,
                cfg.tau,
            )

            new_state = TD3TrainState(
                actor_params=new_actor_params,
                critic_params=new_critic_params,
                target_actor_params=new_target_actor_params,
                target_critic_params=new_target_critic_params,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                actor_optimizer=state.actor_optimizer,
                critic_optimizer=state.critic_optimizer,
                total_it=state.total_it + 1,
            )

            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
            }
            return new_state, metrics

        def update_critic_only(_):
            new_state = TD3TrainState(
                actor_params=state.actor_params,
                critic_params=new_critic_params,
                target_actor_params=state.target_actor_params,
                target_critic_params=state.target_critic_params,
                actor_opt_state=state.actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                actor_optimizer=state.actor_optimizer,
                critic_optimizer=state.critic_optimizer,
                total_it=state.total_it + 1,
            )

            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": 0.0,
            }
            return new_state, metrics

        do_actor_update = state.total_it % cfg.policy_freq == 0

        return jax.lax.cond(
            do_actor_update,
            update_actor_and_targets,
            update_critic_only,
            operand=None,
        )

    return update_step
