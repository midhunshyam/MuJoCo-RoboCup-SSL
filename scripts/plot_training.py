"""Plot the training curve from the Monitor CSV files written by scripts/train.py.

Episodes from all environments are put in order of their end time and grouped into windows
of training steps that widen geometrically, so the fast start and the long tail both show on
a log axis. Writes the binned table (results/training_curve.csv) and the figure
(docs/images/training-curve.png). The scripted baseline's evaluation result, read from
results/eval.json if present, is drawn as a reference line.

    python scripts/plot_training.py --monitor-dir runs/ppo_go_to_ball/monitor
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_ssl import GoToBallConfig  # noqa: E402

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PPO_COLOUR, BASELINE_COLOUR = "#2a78d6", "#eb6834"


def load_episodes(monitor_dir: Path) -> np.ndarray:
    """Rows of (end time, length in steps, return, success) for every finished episode."""
    rows = []
    for path in sorted(monitor_dir.glob("*.monitor.csv")):
        with path.open() as handle:
            handle.readline()  # JSON header line
            for record in csv.DictReader(handle):
                success = record["is_success"] == "True"
                rows.append((float(record["t"]), int(record["l"]), float(record["r"]), success))
    if not rows:
        raise SystemExit(f"no *.monitor.csv files in {monitor_dir}")
    episodes = np.array(rows, dtype=float)
    return episodes[np.argsort(episodes[:, 0], kind="stable")]


def bin_episodes(episodes: np.ndarray, bins: int, first_bin: int, dt: float) -> list[dict]:
    """Group episodes into windows of training steps (counted over all environments)."""
    steps_done = np.cumsum(episodes[:, 1])
    edges = np.unique(np.geomspace(first_bin, steps_done[-1], bins).round().astype(int))
    rows = []
    for low, high in zip(np.concatenate([[0], edges[:-1]]), edges):
        window = episodes[(steps_done > low) & (steps_done <= high)]
        if len(window) == 0:
            continue
        solved = window[window[:, 3] == 1]
        rows.append({
            "steps_start": int(low),
            "steps_end": int(high),
            "episodes": len(window),
            "success_rate": round(float(window[:, 3].mean()), 4),
            "time_to_ball_s": round(float(solved[:, 1].mean() * dt), 3) if len(solved) else "",
            "mean_return": round(float(window[:, 2].mean()), 3),
        })
    return rows


def style_axis(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6, colors=MUTED, labelcolor=INK_2)


def plot(bins: list[dict], baseline: dict | None, out: Path) -> None:
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10.5})
    steps = np.array([b["steps_end"] for b in bins])
    success = np.array([b["success_rate"] for b in bins]) * 100
    times = np.array([np.nan if b["time_to_ball_s"] == "" else b["time_to_ball_s"] for b in bins])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    panels = [
        (left, success, "Success rate (%)", (0, 103), "success_rate", 100),
        (right, times, "Mean time to the ball (s)", None, "time_to_ball_s", 1),
    ]
    for ax, values, label, limits, key, scale in panels:
        style_axis(ax)
        ax.plot(steps, values, color=PPO_COLOUR, linewidth=2, solid_capstyle="round",
                label="PPO, training episodes (sampled actions)")
        if baseline is not None:
            level = baseline[key]["mean"] if key == "time_to_ball_s" else baseline[key]
            ax.axhline(level * scale, color=BASELINE_COLOUR, linewidth=1.5, linestyle=(0, (4, 3)),
                       label="Scripted baseline, evaluation (200 episodes)")
        ax.set_title(label, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
        ax.set_xscale("log")
        ticks = [t for t in (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7) if steps[0] <= t <= steps[-1]]
        ax.set_xticks(ticks, [f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Training steps (log scale)", color=INK_2)
        ax.set_xlim(steps[0] * 0.9, steps[-1] * 1.1)
        if limits:
            ax.set_ylim(*limits)
    if baseline is not None:
        lowest = np.nanmin(times[len(times) // 10:])
        floor = min(lowest, baseline["time_to_ball_s"]["mean"]) * 0.8
        right.set_ylim(floor, np.nanmax(times) * 1.05)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=2,
               frameon=False, fontsize=10, labelcolor=INK_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monitor-dir", type=Path, default=Path("runs/ppo_go_to_ball/monitor"))
    parser.add_argument("--bins", type=int, default=32, help="number of points on the curve")
    parser.add_argument("--first-bin", type=int, default=50_000, help="steps in the first point")
    parser.add_argument("--eval", type=Path, default=Path("results/eval.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("results/training_curve.csv"))
    parser.add_argument("--png-out", type=Path, default=Path("docs/images/training-curve.png"))
    args = parser.parse_args()

    dt = 1.0 / GoToBallConfig().control_frequency  # seconds per environment step
    bins = bin_episodes(load_episodes(args.monitor_dir), args.bins, args.first_bin, dt)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)

    baseline = None
    if args.eval.exists():
        baseline = json.loads(args.eval.read_text())["summary"]["Scripted baseline"]
    plot(bins, baseline, args.png_out)
    print(f"wrote {args.csv_out} ({len(bins)} rows) and {args.png_out}")


if __name__ == "__main__":
    main()
