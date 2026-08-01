#!/usr/bin/env python3
"""Evaluate GR00T checkpoints and plot aggregate and per-joint errors."""

from __future__ import annotations

import argparse
import gc
import logging
from math import ceil
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import open_loop_eval
from gr00t.policy.gr00t_policy import Gr00tPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--traj-ids", type=int, nargs="*")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument("--steps", type=int, default=0, help="0 evaluates each complete episode")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--modality-keys", nargs="+", default=None)
    parser.add_argument("--skip-trajectory-plots", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def find_checkpoints(run_dir: Path, selected_steps: list[int] | None) -> list[Path]:
    if re.fullmatch(r"checkpoint-\d+", run_dir.name):
        checkpoints = [run_dir]
    else:
        checkpoints = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    checkpoints.sort(key=checkpoint_step)

    if selected_steps:
        selected = set(selected_steps)
        checkpoints = [path for path in checkpoints if checkpoint_step(path) in selected]

    if not checkpoints:
        raise FileNotFoundError(f"No matching checkpoint directories found in {run_dir}")
    return checkpoints


def action_labels(trajectory: pd.DataFrame, action_keys: list[str]) -> list[str]:
    labels = []
    for key in action_keys:
        width = np.asarray(trajectory.iloc[0][f"action.{key}"]).reshape(-1).size
        labels.extend(f"{key}[{index}]" for index in range(width))
    return labels


def plot_trajectory(gt: np.ndarray, pred: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    columns = 4
    rows = ceil(len(labels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 3.1 * rows), sharex=True)
    flat_axes = np.asarray(axes).reshape(-1)

    for index, label in enumerate(labels):
        axis = flat_axes[index]
        axis.plot(gt[:, index], linewidth=1.2, label="ground truth")
        axis.plot(pred[:, index], linewidth=1.0, alpha=0.85, label="prediction")
        axis.set_title(label, fontsize=9)
        axis.grid(alpha=0.2)

    for axis in flat_axes[len(labels):]:
        axis.set_visible(False)

    flat_axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_error_heatmap(error: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 8))
    image = axis.imshow(np.abs(error).T, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Action step")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Absolute error (joint units)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_checkpoint_progress(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(summary["checkpoint_step"], summary["mae"], marker="o", label="MAE")
    axis.plot(summary["checkpoint_step"], summary["rmse"], marker="o", label="RMSE")
    axis.set_xlabel("Checkpoint step")
    axis.set_ylabel("Unnormalized action error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_joint_checkpoint_heatmap(joints: pd.DataFrame, labels: list[str], path: Path) -> None:
    table = joints.pivot(index="checkpoint_step", columns="joint", values="mae").reindex(columns=labels)
    figure, axis = plt.subplots(figsize=(15, max(4, 0.65 * len(table))))
    image = axis.imshow(table.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axis.set_yticks(range(len(table.index)))
    axis.set_yticklabels(table.index)
    axis.set_xlabel("Joint")
    axis.set_ylabel("Checkpoint step")
    axis.set_title("Per-joint MAE across checkpoints")
    figure.colorbar(image, ax=axis, label="MAE (joint units)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_best_checkpoint_joints(joints: pd.DataFrame, best_step: int, labels: list[str], path: Path) -> None:
    selected = joints[joints["checkpoint_step"] == best_step].set_index("joint").reindex(labels)
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.bar(range(len(labels)), selected["mae"])
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    axis.set_ylabel("MAE (joint units)")
    axis.set_title(f"Per-joint MAE at checkpoint {best_step}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    output_dir = args.output_dir or args.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints(args.run_dir, args.checkpoint_steps)
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    episode_rows = []
    checkpoint_rows = []
    joint_rows = []
    canonical_labels = None

    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        checkpoint_dir = output_dir / checkpoint.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Loading %s", checkpoint)

        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(checkpoint),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        modality = policy.get_modality_config()
        loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset_path), modality_configs=modality)
        traj_ids = args.traj_ids if args.traj_ids else list(range(len(loader)))
        action_keys = modality["action"].modality_keys if args.modality_keys is None else args.modality_keys

        checkpoint_errors = []
        labels = None

        for traj_id in traj_ids:
            if not 0 <= traj_id < len(loader):
                raise IndexError(f"Trajectory {traj_id} is outside dataset range 0..{len(loader) - 1}")

            trajectory = loader[traj_id]
            labels = action_labels(trajectory, action_keys)
            if canonical_labels is None:
                canonical_labels = labels
            elif labels != canonical_labels:
                raise RuntimeError("Action dimensions changed between checkpoints or trajectories")

            captured: dict[str, np.ndarray] = {}

            def capture_plot(**kwargs) -> None:
                captured["ground_truth"] = np.asarray(kwargs["gt_action_across_time"])
                captured["prediction"] = np.asarray(kwargs["pred_action_across_time"])

            original_plotter = open_loop_eval.plot_trajectory_results
            open_loop_eval.plot_trajectory_results = capture_plot
            try:
                evaluation_steps = args.steps if args.steps > 0 else len(trajectory)
                mse, mae = open_loop_eval.evaluate_single_trajectory(
                    policy=policy,
                    loader=loader,
                    traj_id=traj_id,
                    embodiment_tag=embodiment_tag,
                    modality_keys=args.modality_keys,
                    steps=evaluation_steps,
                    execution_horizon=args.execution_horizon,
                    save_plot_path=None,
                )
            finally:
                open_loop_eval.plot_trajectory_results = original_plotter

            gt = captured["ground_truth"]
            pred = captured["prediction"]
            error = pred - gt
            checkpoint_errors.append(error)
            episode_rows.append(
                {
                    "checkpoint_step": step,
                    "trajectory": traj_id,
                    "frames": len(error),
                    "mae": mae,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "bias": float(np.mean(error)),
                    "max_absolute_error": float(np.max(np.abs(error))),
                }
            )

            if not args.skip_trajectory_plots:
                plot_trajectory(
                    gt,
                    pred,
                    labels,
                    f"Checkpoint {step}, trajectory {traj_id}",
                    checkpoint_dir / f"trajectory_{traj_id:04d}_joints.png",
                )
                plot_error_heatmap(
                    error,
                    labels,
                    f"Absolute error: checkpoint {step}, trajectory {traj_id}",
                    checkpoint_dir / f"trajectory_{traj_id:04d}_error_heatmap.png",
                )

        combined = np.concatenate(checkpoint_errors, axis=0)
        checkpoint_mse = float(np.mean(combined**2))
        checkpoint_rows.append(
            {
                "checkpoint_step": step,
                "episodes": len(checkpoint_errors),
                "frames": len(combined),
                "mae": float(np.mean(np.abs(combined))),
                "mse": checkpoint_mse,
                "rmse": float(np.sqrt(checkpoint_mse)),
                "bias": float(np.mean(combined)),
                "max_absolute_error": float(np.max(np.abs(combined))),
            }
        )

        for index, label in enumerate(labels or []):
            values = combined[:, index]
            mse = float(np.mean(values**2))
            joint_rows.append(
                {
                    "checkpoint_step": step,
                    "joint": label,
                    "mae": float(np.mean(np.abs(values))),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "bias": float(np.mean(values)),
                    "max_absolute_error": float(np.max(np.abs(values))),
                }
            )

        del loader, policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    episodes = pd.DataFrame(episode_rows)
    summary = pd.DataFrame(checkpoint_rows).sort_values("checkpoint_step")
    joints = pd.DataFrame(joint_rows)
    episodes.to_csv(output_dir / "metrics_per_episode.csv", index=False)
    summary.to_csv(output_dir / "metrics_by_checkpoint.csv", index=False)
    joints.to_csv(output_dir / "metrics_per_joint.csv", index=False)

    plot_checkpoint_progress(summary, output_dir / "checkpoint_error_progress.png")
    plot_joint_checkpoint_heatmap(joints, canonical_labels or [], output_dir / "joint_error_by_checkpoint.png")
    best_step = int(summary.loc[summary["mae"].idxmin(), "checkpoint_step"])
    plot_best_checkpoint_joints(
        joints,
        best_step,
        canonical_labels or [],
        output_dir / "best_checkpoint_joint_mae.png",
    )

    print(summary.to_string(index=False))
    print(f"\nLowest held-out MAE: checkpoint {best_step}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()