from brax import envs


def make_brax_env(env_name: str):
    """Create a Brax environment by name.

    Example env_name:
      - "halfcheetah"
      - "ant"
      - "walker2d"
      - "hopper"
    """
    return envs.get_environment(env_name=env_name)
