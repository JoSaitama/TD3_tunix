import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EVAL_RE = re.compile(
    r"\[eval\] step=(?P<step>\d+), "
    r"avg_return=(?P<avg_return>[-+]?\d+\.\d+), "
    r"buffer_size=(?P<buffer_size>\d+), "
    r"td3_updates=(?P<td3_updates>\d+)"
    r"(?:, least_stops=(?P<least_stops>\d+), "
    r"episode_ends=(?P<episode_ends>\d+), "
    r"least_stop_rate=(?P<least_stop_rate>[-+]?\d+\.\d+), "
    r"active_size=(?P<active_size>\d+), "
    r"entropy=(?P<entropy>[-+]?\d+\.\d+), "
    r"exploration_noise=(?P<exploration_noise>[-+]?\d+\.\d+), "
    r"recent_stop_rate=(?P<recent_stop_rate>[-+]?\d+\.\d+))?"
)


def parse_seed_from_name(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.name)
    if not m:
        return -1
    return int(m.group(1))


def parse_logs(log_dir: str, method: str):
    rows = []
    for path in sorted(Path(log_dir).glob("*.log")):
        seed = parse_seed_from_name(path)
        text = path.read_text(errors="ignore")
        for m in EVAL_RE.finditer(text):
            row = {
                "method": method,
                "seed": seed,
                "step": int(m.group("step")),
                "avg_return": float(m.group("avg_return")),
                "buffer_size": int(m.group("buffer_size")),
                "td3_updates": int(m.group("td3_updates")),
            }
            for key in [
                "least_stops",
                "episode_ends",
                "least_stop_rate",
                "active_size",
                "entropy",
                "exploration_noise",
                "recent_stop_rate",
            ]:
                val = m.groupdict().get(key)
                if val is None:
                    row[key] = np.nan
                else:
                    row[key] = float(val)
            rows.append(row)
    return rows


def mean_std(df, value_col):
    grouped = df.groupby(["method", "step"])[value_col]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_learning_curve(df, out_path):
    stats = mean_std(df, "avg_return")

    plt.figure(figsize=(8, 5))
    for method in stats["method"].unique():
        sub = stats[stats["method"] == method].sort_values("step")
        x = sub["step"].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()
        plt.plot(x, y, label=method)
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Training steps")
    plt.ylabel("Evaluation average return")
    plt.title("HalfCheetah: TD3 vs TD3 + LEAST, 50k steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_final_box(df, out_path):
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step]

    methods = list(final_df["method"].unique())
    data = [final_df[final_df["method"] == m]["avg_return"].to_numpy() for m in methods]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=methods, showmeans=True)
    plt.ylabel("Final evaluation average return")
    plt.title(f"Final performance at {final_step} steps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_least_diagnostics(df, out_dir):
    least_df = df[df["method"].str.contains("LEAST")].copy()
    if least_df.empty:
        return

    for col, ylabel, fname in [
        ("least_stop_rate", "LEAST stop rate", "least_stop_rate_curve.png"),
        ("exploration_noise", "Exploration noise", "exploration_noise_curve.png"),
        ("active_size", "Active reflection-set size", "active_size_curve.png"),
    ]:
        stats = mean_std(least_df.dropna(subset=[col]), col)
        if stats.empty:
            continue

        plt.figure(figsize=(8, 5))
        for method in stats["method"].unique():
            sub = stats[stats["method"] == method].sort_values("step")
            x = sub["step"].to_numpy()
            y = sub["mean"].to_numpy()
            s = sub["std"].to_numpy()
            plt.plot(x, y, label=method)
            plt.fill_between(x, y - s, y + s, alpha=0.2)

        plt.xlabel("Training steps")
        plt.ylabel(ylabel)
        plt.title(ylabel + " over training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / fname, dpi=200)
        plt.close()


def make_summary(df):
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step]

    summary = final_df.groupby("method").agg(
        final_return_mean=("avg_return", "mean"),
        final_return_std=("avg_return", "std"),
        seeds=("seed", "nunique"),
        final_least_stop_rate_mean=("least_stop_rate", "mean"),
        final_exploration_noise_mean=("exploration_noise", "mean"),
        final_active_size_mean=("active_size", "mean"),
    ).reset_index()

    auc_rows = []
    for (method, seed), sub in df.groupby(["method", "seed"]):
        sub = sub.sort_values("step")
        auc = np.trapz(sub["avg_return"].to_numpy(), sub["step"].to_numpy())
        auc_rows.append({"method": method, "seed": seed, "auc_return": auc})
    auc_df = pd.DataFrame(auc_rows)

    auc_summary = auc_df.groupby("method").agg(
        auc_return_mean=("auc_return", "mean"),
        auc_return_std=("auc_return", "std"),
    ).reset_index()

    return summary.merge(auc_summary, on="method", how="left")


def main():
    out_dir = Path("results/halfcheetah_50k")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += parse_logs("logs/baseline_50k", "TD3 baseline")
    rows += parse_logs("logs/least_full_50k", "TD3 + LEAST")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No eval rows found. Check log paths and file names.")

    df.to_csv(out_dir / "eval_results.csv", index=False)

    summary = make_summary(df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    plot_learning_curve(df, out_dir / "learning_curve_mean_std.png")
    plot_final_box(df, out_dir / "final_return_boxplot.png")
    plot_least_diagnostics(df, out_dir)

    print("Saved:")
    print(out_dir / "eval_results.csv")
    print(out_dir / "summary.csv")
    print(out_dir / "learning_curve_mean_std.png")
    print(out_dir / "final_return_boxplot.png")
    print(summary)


if __name__ == "__main__":
    main()
