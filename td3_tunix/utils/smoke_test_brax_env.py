import jax
import jax.numpy as jnp

from td3_tunix.envs.make_env import make_brax_env


def main():
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    env_name = "halfcheetah"
    env = make_brax_env(env_name)

    print("Env name:", env_name)
    print("Observation size:", env.observation_size)
    print("Action size:", env.action_size)

    key = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(key)
    print("Initial obs shape:", state.obs.shape)
    print("Initial reward:", state.reward)
    print("Initial done:", state.done)

    action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    next_state = step_fn(state, action)

    print("Next obs shape:", next_state.obs.shape)
    print("Next reward:", next_state.reward)
    print("Next done:", next_state.done)
    print("Brax env smoke test passed.")


if __name__ == "__main__":
    main()
