from brax import envs


def make_brax_env(env_name: str, backend: str = "generalized"):
    """Create a Brax environment.

    We keep this wrapper small so the training code does not depend on
    Brax API details directly.
    """
    if hasattr(envs, "get_environment"):
        return envs.get_environment(env_name=env_name, backend=backend)

    return envs.create(env_name=env_name, backend=backend)
