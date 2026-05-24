from dataclasses import dataclass


@dataclass
class LeastEpisodeStats:
    episode_return: float
    episode_length: int
    least_stop: bool
    stop_step: int
    stop_q: float
    stop_threshold: float


def format_least_episode_log(
    episode_id: int,
    global_step: int,
    stats: LeastEpisodeStats,
) -> str:
    stop_tag = "LEAST_STOP" if stats.least_stop else "ENV_OR_TIMEOUT"

    return (
        f"[episode {episode_id}] "
        f"step={global_step}, "
        f"return={stats.episode_return:.2f}, "
        f"length={stats.episode_length}, "
        f"end={stop_tag}, "
        f"stop_step={stats.stop_step}, "
        f"stop_q={stats.stop_q:.4f}, "
        f"threshold={stats.stop_threshold:.4f}"
    )
