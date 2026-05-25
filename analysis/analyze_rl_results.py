import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_RE = re.compile(
    r"\[eval\] step=(?P<step>\d+), "
    r"avg_return=(?P<avg_return>[-+]?\d+(?:\.\d+)?), "
    r"buffer_size=(?P<buffer_size>\d+), "
    r"td3_updates=(?P<td3_updates>\d+)"
    r"(?:, least_stops=(?P<least_stops>\d+), "
    r"episode_ends=(?P<episode_ends>\d+), "
    r"least_stop_rate=(?P<least_stop_rate>[-+]?\d+(?:\.\d+)?), "
    r"active_size=(?P<active_size>\d+), "
    r"entropy=(?P<entropy>[-+]?\d+(?:\.\d+)?), "
    r"exploration_noise=(?P<exploration_noise>[-+]?\d+(?:\.\d+)?), "
    r"recent_stop_rate=(?P<recent_stop_rate>[-+]?\d+(?:\.\d+)?))?"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--least_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--baseline_label", type=str, default="TD3 baseline")
    parser.add_argument("--least_label", type=str, default="TD3 + LEAST")

    parser.add_argument("--env_name", type=str, default="HalfCheetah")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--expected_seeds", type=int, default=None)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed filter, e.g. --seeds 0 1 2",
    )

    return parser.parse_args()


def parse_seed_from_name(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.name)
    if not m:
        return -1
    return int(m.group(1))


def parse_logs(log_dir: str, method: str, seeds=None):
    rows = []
    log_paths = sorted(Path(log_dir).glob("*.log"))

    for path in log_paths:
        seed = parse_seed_from_name(path)
        if seeds is not None and seed not in seeds:
            continue

        text = path.read_text(errors="ignore")

        for m in EVAL_RE.finditer(text):
            gd = m.groupdict()

            row = {
                "method": method,
                "seed": seed,
                "log_file": str(path),
                "step": int(gd["step"]),
                "avg_return": float(gd["avg_return"]),
                "buffer_size": int(gd["buffer_size"]),
                "td3_updates": int(gd["td3_updates"]),
            }

            optional_float_cols = [
                "least_stop_rate",
                "entropy",
                "exploration_noise",
                "recent_stop_rate",
            ]
            optional_int_cols = [
                "least_stops",
                "episode_ends",
                "active_size",
            ]

            for key in optional_float_cols:
                row[key] = float(gd[key]) if gd.get(key) is not None else np.nan

            for key in optional_int_cols:
                row[key] = int(gd[key]) if gd.get(key) is not None else np.nan

            rows.append(row)

    return rows


def aggregate_mean_std(df: pd.DataFrame, value_col: str):
    out = (
        df.groupby(["method", "step"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def safe_auc(x, y):
    if len(x) < 2:
        return np.nan
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def make_summary(df: pd.DataFrame, final_step=None):
    if final_step is None:
        final_step = int(df["step"].max())

    final_df = df[df["step"] == final_step].copy()

    final_summary = (
        final_df.groupby("method")
        .agg(
            final_step=("step", "max"),
            final_return_mean=("avg_return", "mean"),
            final_return_std=("avg_return", "std"),
            final_return_min=("avg_return", "min"),
            final_return_max=("avg_return", "max"),
            seeds=("seed", "nunique"),
            final_least_stop_rate_mean=("least_stop_rate", "mean"),
            final_exploration_noise_mean=("exploration_noise", "mean"),
            final_active_size_mean=("active_size", "mean"),
            final_entropy_mean=("entropy", "mean"),
        )
        .reset_index()
    )

    auc_rows = []
    best_rows = []

    for (method, seed), sub in df.groupby(["method", "seed"]):
        sub = sub.sort_values("step")

        x = sub["step"].to_numpy()
        y = sub["avg_return"].to_numpy()

        auc_rows.append(
            {
                "method": method,
                "seed": seed,
                "auc_return": safe_auc(x, y),
                "mean_eval_return": float(np.mean(y)),
            }
        )

        best_rows.append(
            {
                "method": method,
                "seed": seed,
                "best_eval_return": float(np.max(y)),
                "best_eval_step": int(sub.iloc[int(np.argmax(y))]["step"]),
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    best_df = pd.DataFrame(best_rows)

    auc_summary = (
        auc_df.groupby("method")
        .agg(
            auc_return_mean=("auc_return", "mean"),
            auc_return_std=("auc_return", "std"),
            mean_eval_return_mean=("mean_eval_return", "mean"),
            mean_eval_return_std=("mean_eval_return", "std"),
        )
        .reset_index()
    )

    best_summary = (
        best_df.groupby("method")
        .agg(
            best_eval_return_mean=("best_eval_return", "mean"),
            best_eval_return_std=("best_eval_return", "std"),
            best_eval_step_mean=("best_eval_step", "mean"),
        )
        .reset_index()
    )

    summary = final_summary.merge(auc_summary, on="method", how="left")
    summary = summary.merge(best_summary, on="method", how="left")

    return summary, auc_df, best_df


def plot_learning_curve(df, out_path, title):
    stats = aggregate_mean_std(df, "avg_return")

    plt.figure(figsize=(9, 5.5))

    for method in stats["method"].unique():
        sub = stats[stats["method"] == method].sort_values("step")

        x = sub["step"].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()

        plt.plot(x, y, label=method, linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("Training steps")
    plt.ylabel("Evaluation average return")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_individual_curves(df, out_path, title):
    plt.figure(figsize=(9, 5.5))

    for (method, seed), sub in df.groupby(["method", "seed"]):
        sub = sub.sort_values("step")
        plt.plot(
            sub["step"],
            sub["avg_return"],
            linewidth=1.2,
            alpha=0.7,
            label=f"{method}, seed {seed}",
        )

    plt.xlabel("Training steps")
    plt.ylabel("Evaluation average return")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_final_return(df, out_path, title, final_step=None):
    if final_step is None:
        final_step = int(df["step"].max())

    final_df = df[df["step"] == final_step].copy()
    methods = list(final_df["method"].unique())

    data = [
        final_df[final_df["method"] == method]["avg_return"].to_numpy()
        for method in methods
    ]

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot(data, labels=methods, showmeans=True)

    for idx, method in enumerate(methods, start=1):
        vals = final_df[final_df["method"] == method]["avg_return"].to_numpy()
        x = np.full_like(vals, idx, dtype=float)
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else np.array([0.0])
        plt.scatter(x + jitter, vals, alpha=0.8)

    plt.ylabel("Final evaluation average return")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_diagnostic_curve(df, value_col, ylabel, out_path, title):
    sub_df = df.dropna(subset=[value_col]).copy()
    if sub_df.empty:
        return False

    stats = aggregate_mean_std(sub_df, value_col)

    plt.figure(figsize=(9, 5.5))

    for method in stats["method"].unique():
        sub = stats[stats["method"] == method].sort_values("step")

        x = sub["step"].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()

        plt.plot(x, y, label=method, linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

    return True


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_filter = set(args.seeds) if args.seeds is not None else None

    rows = []
    rows += parse_logs(args.baseline_dir, args.baseline_label, seeds=seed_filter)
    rows += parse_logs(args.least_dir, args.least_label, seeds=seed_filter)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No eval rows found. Check --baseline_dir, --least_dir, and log format."
        )

    df = df.sort_values(["method", "seed", "step"]).reset_index(drop=True)

    if args.total_steps is not None:
        final_step = args.total_steps
    else:
        final_step = int(df["step"].max())

    eval_csv = out_dir / "eval_results.csv"
    summary_csv = out_dir / "summary.csv"
    auc_csv = out_dir / "auc_by_seed.csv"
    best_csv = out_dir / "best_by_seed.csv"
    mean_std_csv = out_dir / "mean_std_by_step.csv"

    df.to_csv(eval_csv, index=False)

    mean_std_df = aggregate_mean_std(df, "avg_return")
    mean_std_df.to_csv(mean_std_csv, index=False)

    summary, auc_df, best_df = make_summary(df, final_step=final_step)
    summary.to_csv(summary_csv, index=False)
    auc_df.to_csv(auc_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    steps_label = f"{final_step // 1000}k" if final_step >= 1000 else str(final_step)
    seed_count = df.groupby("method")["seed"].nunique().to_dict()

    title_suffix = (
        f"{args.env_name}: {args.baseline_label} vs {args.least_label}, "
        f"{steps_label} steps"
    )

    plot_learning_curve(
        df,
        out_dir / "learning_curve_mean_std.png",
        title=title_suffix,
    )

    plot_individual_curves(
        df,
        out_dir / "individual_seed_curves.png",
        title=f"{title_suffix} individual seeds",
    )

    plot_final_return(
        df,
        out_dir / "final_return_boxplot.png",
        title=f"Final return at {steps_label} steps",
        final_step=final_step,
    )

    least_df = df[df["method"] == args.least_label].copy()

    diagnostics = [
        ("least_stop_rate", "LEAST stop rate", "least_stop_rate_curve.png"),
        ("exploration_noise", "Exploration noise", "exploration_noise_curve.png"),
        ("active_size", "Active reflection-set size", "active_size_curve.png"),
        ("entropy", "Reflection-set entropy", "entropy_curve.png"),
        ("recent_stop_rate", "Recent stop rate", "recent_stop_rate_curve.png"),
    ]

    for col, ylabel, fname in diagnostics:
        plot_diagnostic_curve(
            least_df,
            value_col=col,
            ylabel=ylabel,
            out_path=out_dir / fname,
            title=f"{args.env_name}: {ylabel}",
        )

    print("Saved outputs to:", out_dir)
    print("Eval CSV:", eval_csv)
    print("Summary CSV:", summary_csv)
    print("Mean/std CSV:", mean_std_csv)
    print("AUC by seed CSV:", auc_csv)
    print("Best by seed CSV:", best_csv)
    print()
    print("Detected seeds per method:")
    for method, n in seed_count.items():
        print(f"  {method}: {n} seeds")
    print()
    print("Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
