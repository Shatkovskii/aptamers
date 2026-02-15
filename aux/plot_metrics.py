#!/usr/bin/env python3
"""
Visualize training metrics from a run directory.

Reads metrics.csv produced by core.train and plots train/val curves
side by side for every tracked metric.

Usage:
    python -m aux.plot_metrics checkpoints/v1.4_213030
    python -m aux.plot_metrics checkpoints/v1.4_213030 --save
    python -m aux.plot_metrics checkpoints/v1.4_213030 --save --format pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Metric groups: (display_name, train_col, val_col, lower_is_better)
METRIC_GROUPS = [
    ("Loss",                    "train_loss",       "val_loss",       True),
    ("Perplexity",              "train_perplexity", "val_perplexity", True),
    ("Exact Match",             "train_em",         "val_em",         False),
    ("Edit Distance (TF)",      "train_ed_tf",      "val_ed_tf",     True),
    ("Edit Distance (AR)",      None,               "val_ed_ar",      True),
]


def _best_marker(series: pd.Series, lower_is_better: bool) -> int:
    """Return the index (position) of the best value in series."""
    if lower_is_better:
        return int(series.idxmin())
    return int(series.idxmax())


def plot_run(run_dir: Path, save: bool = False, fmt: str = "png") -> None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")

    df = pd.read_csv(metrics_path)
    epochs = df["epoch"]

    available = [
        g for g in METRIC_GROUPS
        if (g[1] is None or g[1] in df.columns) and (g[2] is None or g[2] in df.columns)
    ]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.2), squeeze=False)
    axes = axes.ravel()

    palette = {"train": "#2563eb", "val": "#dc2626"}

    for ax, (name, train_col, val_col, lower_better) in zip(axes, available):
        has_train = train_col is not None and train_col in df.columns
        has_val = val_col is not None and val_col in df.columns

        if has_train:
            ax.plot(epochs, df[train_col], marker="o", markersize=4,
                    color=palette["train"], linewidth=1.8, label="train")
            best_idx = _best_marker(df[train_col], lower_better)
            ax.scatter(epochs.iloc[best_idx], df[train_col].iloc[best_idx],
                       s=110, zorder=5, facecolors="none",
                       edgecolors=palette["train"], linewidths=2)

        if has_val:
            ax.plot(epochs, df[val_col], marker="s", markersize=4,
                    color=palette["val"], linewidth=1.8, label="val")
            best_idx = _best_marker(df[val_col], lower_better)
            ax.scatter(epochs.iloc[best_idx], df[val_col].iloc[best_idx],
                       s=110, zorder=5, facecolors="none",
                       edgecolors=palette["val"], linewidths=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    desc_path = run_dir / "description.txt"
    subtitle = ""
    if desc_path.exists():
        lines = desc_path.read_text().strip().splitlines()
        kv = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        parts = []
        for key in ["Split mode", "lr", "Train size", "Val size"]:
            if key in kv:
                parts.append(f"{key}={kv[key]}")
        subtitle = "  |  ".join(parts)

    fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=3.0)
    fig.suptitle(f"Run: {run_dir.name}", fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=9, color="gray")

    if save:
        out = run_dir / f"metrics.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot training metrics")
    p.add_argument("run_dir", type=Path, help="Path to run directory with metrics.csv")
    p.add_argument("--save", action="store_true", help="Save figure instead of showing")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="Image format when --save is used")
    args = p.parse_args()

    plot_run(args.run_dir, save=args.save, fmt=args.format)


if __name__ == "__main__":
    main()
