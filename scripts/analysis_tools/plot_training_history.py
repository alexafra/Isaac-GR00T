#!/usr/bin/env python3
"""Plot GR00T training loss, gradient norm, and learning-rate history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def find_complete_history(run_dir: Path) -> tuple[Path, list[dict]]:
    candidates = []
    for path in run_dir.rglob("trainer_state.json"):
        with path.open(encoding="utf-8") as file:
            history = json.load(file).get("log_history", [])
        maximum_step = max((int(row.get("step", -1)) for row in history), default=-1)
        candidates.append((maximum_step, path.stat().st_mtime_ns, path, history))

    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found beneath {run_dir}")

    _, _, path, history = max(candidates, key=lambda item: (item[0], item[1]))
    return path, history


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path, history = find_complete_history(args.run_dir)
    train_rows = [row for row in history if "step" in row and "loss" in row]
    if not train_rows:
        raise RuntimeError(f"No training-loss records found in {state_path}")

    frame = pd.DataFrame(train_rows).sort_values("step").drop_duplicates("step", keep="last")
    window = max(1, args.smooth_window)
    frame["loss_smoothed"] = frame["loss"].rolling(window, min_periods=1).mean()
    frame.to_csv(output_dir / "training_history.csv", index=False)

    validation_rows = [row for row in history if "step" in row and "eval_loss" in row]
    validation_frame = (
        pd.DataFrame(validation_rows).sort_values("step").drop_duplicates("step", keep="last")
        if validation_rows
        else None
    )

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(
        frame["step"],
        frame["loss"],
        color="tab:gray",
        linestyle="--",
        alpha=0.35,
        linewidth=1,
        label="Training loss",
    )
    axes[0].plot(
        frame["step"],
        frame["loss_smoothed"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label=f"Training {window}-point mean",
    )
    if validation_frame is not None:
        axes[0].plot(
            validation_frame["step"],
            validation_frame["eval_loss"],
            color="tab:orange",
            linestyle="-",
            marker="o",
            linewidth=1.8,
            label="Validation loss",
        )
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "grad_norm" in frame:
        axes[1].plot(frame["step"], frame["grad_norm"], color="tab:orange", linewidth=1.0, alpha=0.65)
    axes[1].set_ylabel("Gradient norm")

    if "learning_rate" in frame:
        axes[2].plot(frame["step"], frame["learning_rate"], color="tab:green", linewidth=1)
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Optimizer step")

    for axis in axes:
        axis.grid(alpha=0.25)

    figure.suptitle(args.run_dir.name)
    figure.tight_layout()
    plot_path = output_dir / "training_history.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    minimum = frame.loc[frame["loss"].idxmin()]
    final = frame.iloc[-1]
    print(f"History source: {state_path}")
    print(f"Records:        {len(frame)}")
    print(f"Final:          step={int(final['step'])}, loss={final['loss']:.6f}")
    print(f"Minimum:        step={int(minimum['step'])}, loss={minimum['loss']:.6f}")
    print(f"CSV:            {output_dir / 'training_history.csv'}")
    print(f"Plot:           {plot_path}")


if __name__ == "__main__":
    main()
