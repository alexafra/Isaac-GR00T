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
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Held-out validation dataset path.",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        help="Optional dataset containing training episodes to use as a fixed train probe.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--traj-ids", type=int, nargs="*")
    parser.add_argument(
        "--train-traj-ids",
        type=int,
        nargs="*",
        help="Specific episode indices from --train-dataset-path to use for the train probe.",
    )
    parser.add_argument(
        "--train-probe-episodes",
        type=int,
        default=0,
        help="Seeded number of train episodes to probe; 0 uses all unless --train-traj-ids is set.",
    )
    parser.add_argument("--train-probe-seed", type=int, default=42)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument("--steps", type=int, default=0, help="0 evaluates each complete episode")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--modality-keys", nargs="+", default=None)
    parser.add_argument("--skip-trajectory-plots", action="store_true")
    args = parser.parse_args()
    if args.train_dataset_path is None and (args.train_traj_ids or args.train_probe_episodes):
        parser.error(
            "--train-traj-ids and --train-probe-episodes require --train-dataset-path"
        )
    return args


def select_trajectory_ids(
    dataset_size: int,
    selected_ids: list[int] | None,
    episode_count: int = 0,
    seed: int = 42,
) -> list[int]:
    if episode_count < 0:
        raise ValueError(f"Episode count must be non-negative, got {episode_count}")

    if selected_ids:
        trajectory_ids = list(selected_ids)
    elif episode_count:
        if episode_count > dataset_size:
            raise ValueError(
                f"Requested {episode_count} probe episodes from a dataset with {dataset_size} episodes"
            )
        rng = np.random.default_rng(seed)
        trajectory_ids = sorted(rng.choice(dataset_size, size=episode_count, replace=False).tolist())
    else:
        trajectory_ids = list(range(dataset_size))

    invalid_ids = [traj_id for traj_id in trajectory_ids if not 0 <= traj_id < dataset_size]
    if invalid_ids:
        raise IndexError(
            f"Trajectory IDs {invalid_ids} are outside dataset range 0..{dataset_size - 1}"
        )
    return trajectory_ids


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


def plot_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    execution_horizon: int,
) -> None:
    from collections import defaultdict
    from matplotlib.ticker import MultipleLocator

    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, label in enumerate(labels):
        key = label.split("[")[0]
        groups[key].append((index, label))

    preferred = ["left_arm", "left_hand", "right_arm", "right_hand"]
    ordered_keys = [key for key in preferred if key in groups]
    ordered_keys += [key for key in groups if key not in ordered_keys]

    n_cols = len(ordered_keys)
    max_rows = max(len(groups[key]) for key in ordered_keys)

    # Find the data range required by each joint.
    joint_minimums = np.minimum(
        np.nanmin(gt, axis=0),
        np.nanmin(pred, axis=0),
    )
    joint_maximums = np.maximum(
        np.nanmax(gt, axis=0),
        np.nanmax(pred, axis=0),
    )
    joint_ranges = joint_maximums - joint_minimums

    # Give every subplot the range required by the widest-ranging joint.
    common_y_span = float(np.nanmax(joint_ranges))

    if not np.isfinite(common_y_span) or common_y_span <= 0:
        common_y_span = 1.0

    # Add 10% vertical padding.
    common_y_span *= 1.10

    # Choose one readable tick interval for every subplot.
    raw_tick_step = common_y_span / 6.0
    exponent = np.floor(np.log10(raw_tick_step))
    fraction = raw_tick_step / (10**exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    tick_step = 0.1
    #tick_step = float(nice_fraction * (10**exponent))


    col_width = 4.5
    row_height = 8.5
    # row_height = 2.7

    figure, axes = plt.subplots(
        max_rows,
        n_cols,
        figsize=(col_width * n_cols, row_height * max_rows),
        sharex=True,
        squeeze=False,
    )

    for column, key in enumerate(ordered_keys):
        items = groups[key]

        for row in range(max_rows):
            axis = axes[row, column]

            if row >= len(items):
                axis.set_visible(False)
                continue

            joint_index, label = items[row]

            axis.plot(
                gt[:, joint_index],
                linewidth=1.2,
                label="ground truth",
            )
            axis.plot(
                pred[:, joint_index],
                linewidth=1.0,
                alpha=0.85,
                label="prediction",
            )

            inference_steps = np.arange(
                0,
                len(gt),
                execution_horizon,
            )

            # Dashed vertical line marks the beginning of each predicted chunk.
            for inference_step in inference_steps:
                axis.axvline(
                    inference_step,
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.30,
                    label="chunk boundary" if inference_step == 0 else None,
                )

            # Red dots reproduce the inference markers from open_loop_eval.py.
            axis.scatter(
                inference_steps,
                gt[inference_steps, joint_index],
                color="red",
                s=16,
                zorder=5,
                label="inference point",
            )

            # Centre each joint independently but use the same total span.
            joint_centre = (
                joint_minimums[joint_index] + joint_maximums[joint_index]
            ) / 2.0

            axis.set_ylim(
                joint_centre - common_y_span / 2.0,
                joint_centre + common_y_span / 2.0,
            )
            axis.yaxis.set_major_locator(MultipleLocator(tick_step))

            axis.set_title(label, fontsize=9)
            axis.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{title}\nCommon y-axis span: {common_y_span:.3f} joint units",
    )
    figure.subplots_adjust(hspace=0.38, wspace=0.30)
    figure.savefig(path, dpi=150, bbox_inches="tight")
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


def summary_splits(summary: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "split" not in summary.columns:
        return [("validation", summary)]
    return [
        (str(split), split_summary.sort_values("checkpoint_step"))
        for split, split_summary in summary.groupby("split", sort=False)
    ]


def split_label(split: str) -> str:
    return split.replace("_", " ").title()


def plot_checkpoint_progress(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for split, split_summary in summary_splits(summary):
        label = split_label(split)
        axis.plot(
            split_summary["checkpoint_step"],
            split_summary["mae"],
            marker="o",
            label=f"{label} MAE",
        )
        axis.plot(
            split_summary["checkpoint_step"],
            split_summary["rmse"],
            marker="o",
            linestyle="--",
            label=f"{label} RMSE",
        )
    axis.set_xlabel("Checkpoint step")
    axis.set_ylabel("Unnormalized action error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_checkpoint_metric_summary(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
    )

    magnitude_axis = axes[0, 0]
    mse_axis = axes[0, 1]
    bias_axis = axes[1, 0]
    maximum_axis = axes[1, 1]

    all_steps = sorted(summary["checkpoint_step"].unique())
    for split, split_summary in summary_splits(summary):
        steps = split_summary["checkpoint_step"]
        label = split_label(split)
        magnitude_axis.plot(steps, split_summary["mae"], marker="o", label=f"{label} MAE")
        magnitude_axis.plot(
            steps,
            split_summary["rmse"],
            marker="o",
            linestyle="--",
            label=f"{label} RMSE",
        )
        magnitude_axis.plot(
            steps,
            split_summary["median_absolute_error"],
            marker="o",
            linestyle=":",
            label=f"{label} median absolute error",
        )
        magnitude_axis.plot(
            steps,
            split_summary["p95_absolute_error"],
            marker="o",
            linestyle="-.",
            label=f"{label} 95th-percentile absolute error",
        )
        mse_axis.plot(steps, split_summary["mse"], marker="o", label=label)
        bias_axis.plot(steps, split_summary["bias"], marker="o", label=label)
        maximum_axis.plot(
            steps,
            split_summary["max_absolute_error"],
            marker="o",
            label=label,
        )
    magnitude_axis.set_ylabel("Action error")
    magnitude_axis.set_title("Aggregate error magnitude")
    magnitude_axis.legend(fontsize=8)

    mse_axis.set_ylabel("Squared action error")
    mse_axis.set_title("Aggregate MSE")
    mse_axis.legend()

    bias_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    bias_axis.set_ylabel("Signed action error")
    bias_axis.set_title("Aggregate prediction bias")
    bias_axis.legend()

    maximum_axis.set_ylabel("Action error")
    maximum_axis.set_title("Worst observed error")
    maximum_axis.legend()

    for axis in axes.flat:
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(all_steps)
        axis.grid(alpha=0.25)

    figure.suptitle("Aggregate train-probe and validation metrics across checkpoints")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
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


def evaluate_probe(
    *,
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    trajectory_ids: list[int],
    split: str,
    checkpoint_step_value: int,
    plot_dir: Path,
    embodiment_tag: EmbodimentTag,
    action_keys: list[str],
    modality_keys: list[str] | None,
    steps: int,
    execution_horizon: int,
    skip_trajectory_plots: bool,
    canonical_labels: list[str] | None,
) -> tuple[list[dict], dict, list[dict], list[str]]:
    episode_rows = []
    joint_rows = []
    checkpoint_errors = []
    labels = None

    for traj_id in trajectory_ids:
        trajectory = loader[traj_id]
        labels = action_labels(trajectory, action_keys)
        if canonical_labels is None:
            canonical_labels = labels
        elif labels != canonical_labels:
            raise RuntimeError(
                "Action dimensions changed between datasets, checkpoints, or trajectories"
            )

        captured: dict[str, np.ndarray] = {}

        def capture_plot(**kwargs) -> None:
            captured["ground_truth"] = np.asarray(kwargs["gt_action_across_time"])
            captured["prediction"] = np.asarray(kwargs["pred_action_across_time"])

        original_plotter = open_loop_eval.plot_trajectory_results
        open_loop_eval.plot_trajectory_results = capture_plot
        try:
            evaluation_steps = steps if steps > 0 else len(trajectory)
            mse, mae = open_loop_eval.evaluate_single_trajectory(
                policy=policy,
                loader=loader,
                traj_id=traj_id,
                embodiment_tag=embodiment_tag,
                modality_keys=modality_keys,
                steps=evaluation_steps,
                execution_horizon=execution_horizon,
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
                "split": split,
                "checkpoint_step": checkpoint_step_value,
                "trajectory": traj_id,
                "frames": len(error),
                "mae": mae,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(error)),
                "max_absolute_error": float(np.max(np.abs(error))),
            }
        )

        if not skip_trajectory_plots:
            plot_trajectory(
                gt,
                pred,
                labels,
                f"{split_label(split)}: checkpoint {checkpoint_step_value}, trajectory {traj_id}",
                plot_dir / f"trajectory_{traj_id:04d}_joints.png",
                execution_horizon,
            )
            plot_error_heatmap(
                error,
                labels,
                f"{split_label(split)} absolute error: checkpoint {checkpoint_step_value}, trajectory {traj_id}",
                plot_dir / f"trajectory_{traj_id:04d}_error_heatmap.png",
            )

    if not checkpoint_errors:
        raise ValueError(f"The {split} probe contains no episodes")

    combined = np.concatenate(checkpoint_errors, axis=0)
    checkpoint_mse = float(np.mean(combined**2))
    checkpoint_row = {
        "split": split,
        "checkpoint_step": checkpoint_step_value,
        "episodes": len(checkpoint_errors),
        "frames": len(combined),
        "mae": float(np.mean(np.abs(combined))),
        "mse": checkpoint_mse,
        "median_absolute_error": float(np.median(np.abs(combined))),
        "p95_absolute_error": float(np.percentile(np.abs(combined), 95)),
        "rmse": float(np.sqrt(checkpoint_mse)),
        "bias": float(np.mean(combined)),
        "max_absolute_error": float(np.max(np.abs(combined))),
    }

    for index, label in enumerate(labels or []):
        values = combined[:, index]
        mse = float(np.mean(values**2))
        joint_rows.append(
            {
                "split": split,
                "checkpoint_step": checkpoint_step_value,
                "joint": label,
                "mae": float(np.mean(np.abs(values))),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(values)),
                "max_absolute_error": float(np.max(np.abs(values))),
            }
        )

    assert canonical_labels is not None
    return episode_rows, checkpoint_row, joint_rows, canonical_labels


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
        action_keys = (
            modality["action"].modality_keys
            if args.modality_keys is None
            else args.modality_keys
        )
        probe_specs = [("validation", args.dataset_path, args.traj_ids, 0, 0)]
        if args.train_dataset_path is not None:
            probe_specs.append(
                (
                    "train_probe",
                    args.train_dataset_path,
                    args.train_traj_ids,
                    args.train_probe_episodes,
                    args.train_probe_seed,
                )
            )

        for split, dataset_path, selected_ids, episode_count, seed in probe_specs:
            loader = LeRobotEpisodeLoader(
                dataset_path=str(dataset_path), modality_configs=modality
            )
            trajectory_ids = select_trajectory_ids(
                len(loader), selected_ids, episode_count=episode_count, seed=seed
            )
            plot_dir = checkpoint_dir if split == "validation" else checkpoint_dir / split
            plot_dir.mkdir(parents=True, exist_ok=True)
            logging.info(
                "Evaluating %s on %d episode(s) from %s: %s",
                split,
                len(trajectory_ids),
                dataset_path,
                trajectory_ids,
            )

            probe_episode_rows, checkpoint_row, probe_joint_rows, canonical_labels = (
                evaluate_probe(
                    policy=policy,
                    loader=loader,
                    trajectory_ids=trajectory_ids,
                    split=split,
                    checkpoint_step_value=step,
                    plot_dir=plot_dir,
                    embodiment_tag=embodiment_tag,
                    action_keys=action_keys,
                    modality_keys=args.modality_keys,
                    steps=args.steps,
                    execution_horizon=args.execution_horizon,
                    skip_trajectory_plots=args.skip_trajectory_plots,
                    canonical_labels=canonical_labels,
                )
            )
            episode_rows.extend(probe_episode_rows)
            checkpoint_rows.append(checkpoint_row)
            joint_rows.extend(probe_joint_rows)
            del loader

        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    episodes = pd.DataFrame(episode_rows)
    summary = pd.DataFrame(checkpoint_rows).sort_values(["split", "checkpoint_step"])
    joints = pd.DataFrame(joint_rows)
    episodes.to_csv(output_dir / "metrics_per_episode.csv", index=False)
    summary.to_csv(output_dir / "metrics_by_checkpoint.csv", index=False)
    joints.to_csv(output_dir / "metrics_per_joint.csv", index=False)

    plot_checkpoint_progress(summary, output_dir / "checkpoint_error_progress.png")
    checkpoint_summary_csv = output_dir / "checkpoint_metric_summary.csv"
    summary.to_csv(checkpoint_summary_csv, index=False)
    plot_checkpoint_metric_summary(
        summary,
        output_dir / "checkpoint_metric_summary.png",
    )
    validation_summary = summary[summary["split"] == "validation"]
    validation_joints = joints[joints["split"] == "validation"]
    plot_joint_checkpoint_heatmap(
        validation_joints,
        canonical_labels or [],
        output_dir / "joint_error_by_checkpoint.png",
    )
    best_step = int(
        validation_summary.loc[validation_summary["mae"].idxmin(), "checkpoint_step"]
    )
    plot_best_checkpoint_joints(
        validation_joints,
        best_step,
        canonical_labels or [],
        output_dir / "best_checkpoint_joint_mae.png",
    )

    print(summary.to_string(index=False))
    print(f"\nLowest validation MAE: checkpoint {best_step}")
    print(f"Checkpoint summary CSV: {checkpoint_summary_csv}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
