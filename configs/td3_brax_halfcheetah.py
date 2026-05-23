from dataclasses import dataclass


@dataclass
class TD3Config:
    env_name: str = "halfcheetah"

    seed: int = 0
    total_steps: int = 1_000_000
    start_timesteps: int = 25_000
    eval_freq: int = 5_000
    max_episode_steps: int = 1000

    replay_size: int = 1_000_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    policy_freq: int = 2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    hidden_dim: int = 256

    num_eval_episodes: int = 10


def get_config() -> TD3Config:
    return TD3Config()
