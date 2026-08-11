#!/usr/bin/env python3
"""Evaluate GR00T checkpoints and plot aggregate and per-joint errors."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

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
        "--base-model-path",
        type=Path,
        help=(
            "Optional pretrained model weights to evaluate as logical checkpoint 0. "
            "The processor/statistics are loaded from RUN_DIR/processor so the baseline "
            "uses the same embodiment contract as the finetuned checkpoints."
        ),
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "RUN_DIR/evaluation_exec_hor_<EXECUTION_HORIZON>."
        ),
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--traj-ids", type=int, nargs="*")
    parser.add_argument(
        "--trajectory-plot-episodes",
        type=int,
        default=3,
        help=(
            "Soft target for per-trajectory plots in each split. Plot selection covers every "
            "task first, so the actual count can exceed this value. Use 0 to disable them."
        ),
    )
    parser.add_argument("--trajectory-plot-seed", type=int, default=42)
    parser.add_argument(
        "--train-traj-ids",
        type=int,
        nargs="*",
        help="Specific episode indices from --train-dataset-path to use for the train probe.",
    )
    parser.add_argument(
        "--train-probe-episodes",
        type=int,
        default=None,
        help=(
            "Soft target for train episodes. Selection covers every task first, so the actual "
            "count can exceed this value. Defaults to 3 when --train-dataset-path is provided; "
            "use 0 to evaluate all train episodes."
        ),
    )
    parser.add_argument("--train-probe-seed", type=int, default=42)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument("--steps", type=int, default=0, help="0 evaluates each complete episode")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=42,
        help="Reset the inference RNG to this seed for every evaluated model.",
    )
    parser.add_argument("--modality-keys", nargs="+", default=None)
    parser.add_argument("--skip-trajectory-plots", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Regenerate plots from saved evaluation CSVs without loading checkpoints or "
            "running model inference."
        ),
    )
    args = parser.parse_args()
    if args.train_dataset_path is None and (args.train_traj_ids or args.train_probe_episodes):
        parser.error(
            "--train-traj-ids and --train-probe-episodes require --train-dataset-path"
        )
    return args


def _load_episode_tasks(dataset_path: Path, dataset_size: int) -> dict[int, list[str]]:
    metadata_path = dataset_path / "meta" / "episodes.jsonl"
    if not metadata_path.is_file():
        logging.warning(
            "Task-balanced probe selection unavailable because %s does not exist",
            metadata_path,
        )
        return {}

    episode_tasks = {}
    with metadata_path.open() as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode_id = int(record["episode_index"])
                tasks = [str(task) for task in record.get("tasks", []) if str(task)]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid episode metadata at {metadata_path}:{line_number}"
                ) from exc

            if 0 <= episode_id < dataset_size and tasks:
                episode_tasks[episode_id] = tasks

    return episode_tasks


def select_task_balanced_trajectory_ids(
    dataset_path: Path,
    dataset_size: int,
    candidate_ids: list[int],
    episode_count: int,
    seed: int = 42,
) -> list[int]:
    if episode_count < 0:
        raise ValueError(f"Episode count must be non-negative, got {episode_count}")
    if episode_count == 0 or not candidate_ids:
        return []

    invalid_ids = [traj_id for traj_id in candidate_ids if not 0 <= traj_id < dataset_size]
    if invalid_ids:
        raise IndexError(
            f"Trajectory IDs {invalid_ids} are outside dataset range 0..{dataset_size - 1}"
        )

    rng = np.random.default_rng(seed)
    candidate_set = set(candidate_ids)
    episode_tasks = _load_episode_tasks(dataset_path, dataset_size)

    task_episodes: dict[str, list[int]] = {}
    for episode_id, tasks in episode_tasks.items():
        if episode_id not in candidate_set:
            continue
        for task in tasks:
            task_episodes.setdefault(task, []).append(episode_id)

    # Select one random episode for every task, even when that exceeds the soft target.
    selected = {int(rng.choice(task_episodes[task])) for task in sorted(task_episodes)}

    # When there are fewer tasks than the target, fill from the remaining episodes.
    target_count = min(episode_count, len(candidate_ids))
    remaining = sorted(candidate_set - selected)
    fill_count = min(max(0, target_count - len(selected)), len(remaining))
    if fill_count:
        selected.update(rng.choice(remaining, size=fill_count, replace=False).tolist())

    return sorted(selected)


def select_trajectory_ids(
    dataset_path: Path,
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
        trajectory_ids = select_task_balanced_trajectory_ids(
            dataset_path,
            dataset_size,
            list(range(dataset_size)),
            episode_count,
            seed,
        )
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


@dataclass(frozen=True)
class EvaluationTarget:
    """One physical model/processor pair represented on plots by a training step."""

    step: int
    model_path: Path
    processor_path: Path | None = None

    @property
    def output_name(self) -> str:
        return f"checkpoint-{self.step}"


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


def find_evaluation_targets(
    run_dir: Path,
    selected_steps: list[int] | None,
    base_model_path: Path | None,
) -> list[EvaluationTarget]:
    """Resolve physical checkpoints plus an optional run-compatible step-0 baseline."""

    if re.fullmatch(r"checkpoint-\d+", run_dir.name):
        checkpoints = [run_dir]
    else:
        checkpoints = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    checkpoints.sort(key=checkpoint_step)

    selected = set(selected_steps) if selected_steps else None
    if selected is not None:
        checkpoints = [path for path in checkpoints if checkpoint_step(path) in selected]

    targets = [
        EvaluationTarget(step=checkpoint_step(path), model_path=path) for path in checkpoints
    ]

    include_base = base_model_path is not None and (selected is None or 0 in selected)
    if include_base:
        if any(target.step == 0 for target in targets):
            raise ValueError(
                "Cannot combine --base-model-path with a physical checkpoint-0 directory"
            )
        if not base_model_path.is_dir():
            raise FileNotFoundError(f"Base model directory does not exist: {base_model_path}")
        processor_path = run_dir / "processor"
        required_processor_files = [
            processor_path / "processor_config.json",
            processor_path / "statistics.json",
        ]
        missing_processor_files = [path for path in required_processor_files if not path.is_file()]
        if missing_processor_files:
            missing = ", ".join(str(path) for path in missing_processor_files)
            raise FileNotFoundError(
                "Cannot evaluate the base model with this run's embodiment processor; "
                f"missing: {missing}"
            )
        targets.append(
            EvaluationTarget(
                step=0,
                model_path=base_model_path,
                processor_path=processor_path,
            )
        )

    targets.sort(key=lambda target: target.step)
    if selected is not None:
        found_steps = {target.step for target in targets}
        missing_steps = selected - found_steps
        if missing_steps:
            raise FileNotFoundError(
                f"No model target found for checkpoint step(s): {sorted(missing_steps)}"
            )
    if not targets:
        raise FileNotFoundError(f"No matching model targets found in {run_dir}")
    return targets


def seed_inference(seed: int) -> None:
    """Give every model target the same stochastic flow-sampling sequence."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """Format an elapsed-time or ETA value for compact progress logs."""

    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def default_evaluation_output_dir(run_dir: Path, execution_horizon: int) -> Path:
    """Keep results from different execution horizons in separate directories."""

    return run_dir / f"evaluation_exec_hor_{execution_horizon}"


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
    goal: str | None = None,
) -> None:
    from collections import defaultdict
    from matplotlib.ticker import MaxNLocator, MultipleLocator

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

            # # Red dots reproduce the inference markers from open_loop_eval.py.
            # axis.scatter(
            #     inference_steps,
            #     gt[inference_steps, joint_index],
            #     color="red",
            #     s=16,
            #     zorder=5,
            #     label="inference point",
            # )

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
            axis.set_xlabel("Frame")
            axis.set_xlim(0, max(0, len(gt) - 1))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            axis.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            axis.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(title, fontsize=14, y=0.995)
    header_y = 0.978
    if goal:
        figure.text(
            0.5,
            header_y,
            f"Goal: {goal}",
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
        )
        header_y -= 0.018
    figure.text(
        0.5,
        header_y,
        f"Common y-axis span: {common_y_span:.3f} joint units",
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.94, hspace=0.38, wspace=0.30)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_error_heatmap(
    error: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    goal: str | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 8))
    image = axis.imshow(np.abs(error).T, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Action step")
    axis.set_title(title, pad=30 if goal else None)
    if goal:
        axis.text(
            0.5,
            1.015,
            f"Goal: {goal}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )
    figure.colorbar(image, ax=axis, label="Absolute error (joint units)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_right_trajectory_outputs(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
    title: str,
    plot_dir: Path,
    trajectory_id: int,
    execution_horizon: int,
    goal: str | None,
) -> None:
    indices = right_action_indices(labels)
    if not indices:
        logging.warning("No right_arm/right_hand actions found for trajectory %d", trajectory_id)
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    right_ground_truth = ground_truth[:, indices]
    right_prediction = prediction[:, indices]
    right_labels = [labels[index] for index in indices]
    plot_trajectory(
        right_ground_truth,
        right_prediction,
        right_labels,
        f"{title} — right arm + right hand",
        plot_dir / f"trajectory_{trajectory_id:04d}_joints.png",
        execution_horizon,
        goal,
    )
    plot_error_heatmap(
        right_prediction - right_ground_truth,
        right_labels,
        f"{title} absolute error — right arm + right hand",
        plot_dir / f"trajectory_{trajectory_id:04d}_error_heatmap.png",
        goal,
    )


def summary_splits(summary: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "split" not in summary.columns:
        return [("validation", summary)]
    return [
        (str(split), split_summary.sort_values("checkpoint_step"))
        for split, split_summary in summary.groupby("split", sort=False)
    ]


def split_label(split: str) -> str:
    return split.replace("_", " ").title()


def split_linestyle(split: str) -> str:
    """Use line style, rather than color, to distinguish train from validation."""
    return "--" if split.startswith("train") else "-"


SUMMARY_LINE_WIDTH = 1.0
SUMMARY_MARKER_SIZE = 4.0
SUMMARY_ALPHA = 0.72
MAGNITUDE_METRICS = [
    ("mae", "MAE", "tab:blue"),
    ("rmse", "RMSE", "tab:orange"),
    ("mse", "MSE", "tab:purple"),
    ("median_absolute_error", "Median absolute error", "tab:green"),
    ("p95_absolute_error", "95th-percentile absolute error", "tab:red"),
]


def plot_checkpoint_progress(
    summary: pd.DataFrame,
    path: Path,
    scope_label: str | None = None,
) -> None:
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(10, 5))
    for split, split_summary in summary_splits(summary):
        label = split_label(split)
        linestyle = split_linestyle(split)
        axis.plot(
            split_summary["checkpoint_step"],
            split_summary["mae"],
            marker="o",
            color="tab:blue",
            linestyle=linestyle,
            label=f"{label} MAE",
            linewidth=SUMMARY_LINE_WIDTH,
            markersize=SUMMARY_MARKER_SIZE,
            alpha=SUMMARY_ALPHA,
        )
        axis.plot(
            split_summary["checkpoint_step"],
            split_summary["rmse"],
            marker="o",
            color="tab:orange",
            linestyle=linestyle,
            label=f"{label} RMSE",
            linewidth=SUMMARY_LINE_WIDTH,
            markersize=SUMMARY_MARKER_SIZE,
            alpha=SUMMARY_ALPHA,
        )
    axis.set_xlabel("Checkpoint step")
    axis.set_ylabel("Unnormalized action error")
    if scope_label:
        axis.set_title(scope_label)
    axis.grid(alpha=0.25)
    progress_metrics = [("MAE", "tab:blue"), ("RMSE", "tab:orange")]
    axis.legend(
        handles=[
            handle
            for metric_label, color in progress_metrics
            for handle in (
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    linestyle="-",
                    label=f"{metric_label} — Validation",
                    linewidth=SUMMARY_LINE_WIDTH,
                    markersize=SUMMARY_MARKER_SIZE,
                    alpha=SUMMARY_ALPHA,
                ),
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    linestyle="--",
                    label=f"{metric_label} — Train Probe",
                    linewidth=SUMMARY_LINE_WIDTH,
                    markersize=SUMMARY_MARKER_SIZE,
                    alpha=SUMMARY_ALPHA,
                ),
            )
        ],
        ncol=1,
        handlelength=3.5,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_checkpoint_metric_summary(
    summary: pd.DataFrame,
    path: Path,
    scope_label: str | None = None,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
    )

    validation_magnitude_axis = axes[0, 0]
    train_magnitude_axis = axes[0, 1]
    bias_axis = axes[1, 0]
    maximum_axis = axes[1, 1]

    all_steps = sorted(summary["checkpoint_step"].unique())
    magnitude_axes = {
        "validation": validation_magnitude_axis,
        "train_probe": train_magnitude_axis,
    }
    populated_magnitude_splits = set()
    for split, split_summary in summary_splits(summary):
        steps = split_summary["checkpoint_step"]
        label = split_label(split)
        magnitude_split = "train_probe" if split.startswith("train") else "validation"
        magnitude_axis = magnitude_axes[magnitude_split]
        populated_magnitude_splits.add(magnitude_split)
        split_color = "tab:blue" if split.startswith("train") else "tab:orange"
        for column, metric_label, color in MAGNITUDE_METRICS:
            magnitude_axis.plot(
                steps,
                split_summary[column],
                marker="o",
                color=color,
                linestyle="-",
                label=metric_label,
                linewidth=SUMMARY_LINE_WIDTH,
                markersize=SUMMARY_MARKER_SIZE,
                alpha=SUMMARY_ALPHA,
            )
        bias_axis.plot(
            steps,
            split_summary["bias"],
            marker="o",
            color=split_color,
            linestyle="-",
            label=label,
            linewidth=SUMMARY_LINE_WIDTH,
            markersize=SUMMARY_MARKER_SIZE,
            alpha=SUMMARY_ALPHA,
        )
        maximum_axis.plot(
            steps,
            split_summary["max_absolute_error"],
            marker="o",
            color=split_color,
            linestyle="-",
            label=label,
            linewidth=SUMMARY_LINE_WIDTH,
            markersize=SUMMARY_MARKER_SIZE,
            alpha=SUMMARY_ALPHA,
        )
    for split, axis in magnitude_axes.items():
        axis.set_ylabel("Metric value")
        axis.set_title(f"{split_label(split)} error magnitude")
        if split in populated_magnitude_splits:
            axis.legend(fontsize=8, handlelength=2.5, ncol=1)
        else:
            axis.text(
                0.5,
                0.5,
                f"No {split_label(split).lower()} metrics",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="dimgray",
            )

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

    title = "Aggregate train-probe and validation metrics across checkpoints"
    if scope_label:
        title += f" — {scope_label}"
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_joint_checkpoint_heatmap(
    joints: pd.DataFrame,
    labels: list[str],
    path: Path,
    scope_label: str | None = None,
) -> None:
    table = joints.pivot(index="checkpoint_step", columns="joint", values="mae").reindex(
        columns=labels
    )
    figure, axis = plt.subplots(figsize=(15, max(4, 0.65 * len(table))))
    image = axis.imshow(table.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axis.set_yticks(range(len(table.index)))
    axis.set_yticklabels(table.index)
    axis.set_xlabel("Joint")
    axis.set_ylabel("Checkpoint step")
    title = "Per-joint MAE across checkpoints"
    if scope_label:
        title += f" — {scope_label}"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="MAE (joint units)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_best_checkpoint_joints(
    joints: pd.DataFrame,
    best_step: int,
    labels: list[str],
    path: Path,
    scope_label: str | None = None,
) -> None:
    selected = joints[joints["checkpoint_step"] == best_step].set_index("joint").reindex(labels)
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.bar(range(len(labels)), selected["mae"])
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    axis.set_ylabel("MAE (joint units)")
    title = f"Per-joint MAE at checkpoint {best_step}"
    if scope_label:
        title += f" — {scope_label}"
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


RAW_PREDICTION_COLUMNS = [
    "checkpoint_step",
    "trajectory",
    "frame",
    "joint",
    "ground_truth",
    "prediction",
    "error",
    "absolute_error",
]
RAW_PREDICTION_FILENAMES = {
    "validation": "validation_frame_predictions.csv.gz",
    "train_probe": "train_probe_frame_predictions.csv.gz",
}
RIGHT_ACTION_PREFIXES = ("right_arm[", "right_hand[")


def right_action_indices(labels: list[str]) -> list[int]:
    return [
        index
        for index, label in enumerate(labels)
        if str(label).startswith(RIGHT_ACTION_PREFIXES)
    ]


def right_joint_rows(joints: pd.DataFrame) -> pd.DataFrame:
    return joints[joints["joint"].astype(str).str.startswith(RIGHT_ACTION_PREFIXES)].copy()


def raw_predictions_csv_path(checkpoint_dir: Path, split: str) -> Path:
    try:
        filename = RAW_PREDICTION_FILENAMES[split]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw-prediction split: {split}") from exc
    return checkpoint_dir / filename


def right_summary_from_raw_predictions(
    output_dir: Path,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    split_steps = summary[["split", "checkpoint_step"]].drop_duplicates()
    for split, step_value in split_steps.itertuples(index=False, name=None):
        step = int(step_value)
        raw_path = raw_predictions_csv_path(output_dir / f"checkpoint-{step}", str(split))
        if not raw_path.is_file():
            logging.warning("Cannot calculate right-side metrics; missing %s", raw_path)
            continue

        raw_predictions = pd.read_csv(raw_path, compression="gzip")
        right_predictions = raw_predictions[
            raw_predictions["joint"].astype(str).str.startswith(RIGHT_ACTION_PREFIXES)
        ]
        if right_predictions.empty:
            logging.warning("No right_arm/right_hand actions found in %s", raw_path)
            continue

        error = right_predictions["error"].to_numpy(dtype=float)
        absolute_error = np.abs(error)
        mse = float(np.mean(error**2))
        rows.append(
            {
                "split": split,
                "checkpoint_step": step,
                "episodes": int(right_predictions["trajectory"].nunique()),
                "frames": len(
                    right_predictions[["trajectory", "frame"]].drop_duplicates()
                ),
                "mae": float(np.mean(absolute_error)),
                "mse": mse,
                "median_absolute_error": float(np.median(absolute_error)),
                "p95_absolute_error": float(np.percentile(absolute_error, 95)),
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(error)),
                "max_absolute_error": float(np.max(absolute_error)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["split", "checkpoint_step"])


def raw_action_frame(
    *,
    checkpoint_step_value: int,
    trajectory_id: int,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} does not match prediction shape "
            f"{prediction.shape}"
        )
    if ground_truth.ndim != 2 or ground_truth.shape[1] != len(labels):
        raise ValueError(
            f"Expected [frames, {len(labels)} joints], got {ground_truth.shape}"
        )

    frame_count, joint_count = ground_truth.shape
    error = prediction - ground_truth
    return pd.DataFrame(
        {
            "checkpoint_step": np.full(frame_count * joint_count, checkpoint_step_value),
            "trajectory": np.full(frame_count * joint_count, trajectory_id),
            "frame": np.repeat(np.arange(frame_count), joint_count),
            "joint": np.tile(np.asarray(labels), frame_count),
            "ground_truth": ground_truth.reshape(-1),
            "prediction": prediction.reshape(-1),
            "error": error.reshape(-1),
            "absolute_error": np.abs(error).reshape(-1),
        },
        columns=RAW_PREDICTION_COLUMNS,
    )


def initialize_raw_predictions_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        pd.DataFrame(columns=RAW_PREDICTION_COLUMNS).to_csv(output_file, index=False)


def append_raw_predictions_csv(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "at", encoding="utf-8", newline="") as output_file:
        frame.to_csv(output_file, index=False, header=False)


def evaluate_probe(
    *,
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    trajectory_ids: list[int],
    split: str,
    checkpoint_step_value: int,
    plot_dir: Path,
    right_plot_dir: Path,
    embodiment_tag: EmbodimentTag,
    action_keys: list[str],
    modality_keys: list[str] | None,
    steps: int,
    execution_horizon: int,
    skip_trajectory_plots: bool,
    plot_trajectory_ids: set[int],
    trajectory_goals: dict[int, str],
    raw_predictions_path: Path | None,
    canonical_labels: list[str] | None,
    checkpoint_progress_label: str,
) -> tuple[list[dict], dict, list[dict], list[str]]:
    episode_rows = []
    joint_rows = []
    checkpoint_errors = []
    labels = None
    if raw_predictions_path is not None:
        initialize_raw_predictions_csv(raw_predictions_path)

    probe_started_at = time.perf_counter()
    trajectory_count = len(trajectory_ids)
    for trajectory_index, traj_id in enumerate(trajectory_ids, start=1):
        trajectory_started_at = time.perf_counter()
        trajectory = loader[traj_id]
        evaluation_steps = steps if steps > 0 else len(trajectory)
        evaluation_steps = min(evaluation_steps, len(trajectory))
        inference_count = ceil(evaluation_steps / execution_horizon)
        progress_label = (
            f"{checkpoint_progress_label} | {split} episode "
            f"{trajectory_index}/{trajectory_count} traj={traj_id}"
        )
        goal = trajectory_goals.get(traj_id)
        logging.info(
            "[%s] starting: %d frame(s), %d inference request(s), goal=%r",
            progress_label,
            evaluation_steps,
            inference_count,
            goal,
        )
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
            mse, mae = open_loop_eval.evaluate_single_trajectory(
                policy=policy,
                loader=loader,
                traj_id=traj_id,
                embodiment_tag=embodiment_tag,
                modality_keys=modality_keys,
                steps=evaluation_steps,
                execution_horizon=execution_horizon,
                save_plot_path=None,
                progress_label=progress_label,
            )
        finally:
            open_loop_eval.plot_trajectory_results = original_plotter

        gt = captured["ground_truth"]
        pred = captured["prediction"]
        error = pred - gt
        checkpoint_errors.append(error)
        if raw_predictions_path is not None:
            append_raw_predictions_csv(
                raw_predictions_path,
                raw_action_frame(
                    checkpoint_step_value=checkpoint_step_value,
                    trajectory_id=traj_id,
                    ground_truth=gt,
                    prediction=pred,
                    labels=labels,
                ),
            )
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

        if not skip_trajectory_plots and traj_id in plot_trajectory_ids:
            goal = trajectory_goals.get(traj_id)
            plot_trajectory(
                gt,
                pred,
                labels,
                f"{split_label(split)}: checkpoint {checkpoint_step_value}, trajectory {traj_id}",
                plot_dir / f"trajectory_{traj_id:04d}_joints.png",
                execution_horizon,
                goal,
            )
            plot_error_heatmap(
                error,
                labels,
                f"{split_label(split)} absolute error: checkpoint "
                f"{checkpoint_step_value}, trajectory {traj_id}",
                plot_dir / f"trajectory_{traj_id:04d}_error_heatmap.png",
                goal,
            )
            plot_right_trajectory_outputs(
                gt,
                pred,
                labels,
                f"{split_label(split)}: checkpoint {checkpoint_step_value}, "
                f"trajectory {traj_id}",
                right_plot_dir,
                traj_id,
                execution_horizon,
                goal,
            )

        trajectory_elapsed = time.perf_counter() - trajectory_started_at
        probe_elapsed = time.perf_counter() - probe_started_at
        remaining_trajectories = trajectory_count - trajectory_index
        probe_eta = probe_elapsed / trajectory_index * remaining_trajectories
        logging.info(
            "[%s] complete in %s: MAE=%.6f, MSE=%.6f; split ETA %s",
            progress_label,
            format_duration(trajectory_elapsed),
            mae,
            mse,
            format_duration(probe_eta),
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
    logging.info(
        "[%s | %s] complete in %s: %d episode(s), %d frame(s), MAE=%.6f, RMSE=%.6f",
        checkpoint_progress_label,
        split,
        format_duration(time.perf_counter() - probe_started_at),
        len(checkpoint_errors),
        len(combined),
        checkpoint_row["mae"],
        checkpoint_row["rmse"],
    )

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


def plot_evaluation_summaries(
    output_dir: Path,
    summary: pd.DataFrame,
    joints: pd.DataFrame,
    labels: list[str],
    filename_suffix: str = "",
    scope_label: str | None = None,
) -> int:
    includes_base = bool((summary["checkpoint_step"] == 0).any())
    full_scope_label = scope_label
    if includes_base:
        full_scope_label = " — ".join(
            label for label in (scope_label, "Including base step 0") if label
        )
    plot_checkpoint_progress(
        summary,
        output_dir / f"checkpoint_error_progress{filename_suffix}.png",
        full_scope_label,
    )
    plot_checkpoint_metric_summary(
        summary,
        output_dir / f"checkpoint_metric_summary{filename_suffix}.png",
        full_scope_label,
    )
    validation_summary = summary[summary["split"] == "validation"]
    validation_joints = joints[joints["split"] == "validation"]
    if validation_summary.empty:
        raise ValueError("Saved evaluation metrics contain no validation checkpoints")
    if validation_joints.empty:
        raise ValueError("Saved evaluation metrics contain no validation joint data")

    plot_joint_checkpoint_heatmap(
        validation_joints,
        labels,
        output_dir / f"joint_error_by_checkpoint{filename_suffix}.png",
        full_scope_label,
    )

    finetuned_summary = summary[summary["checkpoint_step"] > 0]
    finetuned_joints = joints[joints["checkpoint_step"] > 0]
    if includes_base and not finetuned_summary.empty:
        finetuned_scope_label = " — ".join(
            label for label in (scope_label, "Finetuned checkpoints only") if label
        )
        finetuned_suffix = f"{filename_suffix}_finetuned_only"
        plot_checkpoint_progress(
            finetuned_summary,
            output_dir / f"checkpoint_error_progress{finetuned_suffix}.png",
            finetuned_scope_label,
        )
        plot_checkpoint_metric_summary(
            finetuned_summary,
            output_dir / f"checkpoint_metric_summary{finetuned_suffix}.png",
            finetuned_scope_label,
        )
        finetuned_validation_joints = finetuned_joints[
            finetuned_joints["split"] == "validation"
        ]
        if not finetuned_validation_joints.empty:
            plot_joint_checkpoint_heatmap(
                finetuned_validation_joints,
                labels,
                output_dir / f"joint_error_by_checkpoint{finetuned_suffix}.png",
                finetuned_scope_label,
            )

    best_step = int(
        validation_summary.loc[validation_summary["mae"].idxmin(), "checkpoint_step"]
    )
    plot_best_checkpoint_joints(
        validation_joints,
        best_step,
        labels,
        output_dir / f"best_checkpoint_joint_mae{filename_suffix}.png",
        scope_label,
    )
    return best_step


def plot_right_evaluation_summaries(
    output_dir: Path,
    summary: pd.DataFrame,
    joints: pd.DataFrame,
    *,
    save_metrics: bool,
) -> int | None:
    right_summary = right_summary_from_raw_predictions(output_dir, summary)
    scoped_joints = right_joint_rows(joints)
    if right_summary.empty or scoped_joints.empty:
        logging.warning(
            "Skipping additional right-arm/hand plots because right-side raw data is unavailable"
        )
        return None

    right_output_dir = output_dir / "right_arm_hand"
    right_output_dir.mkdir(parents=True, exist_ok=True)
    labels = scoped_joints["joint"].drop_duplicates().astype(str).tolist()
    if save_metrics:
        right_summary.to_csv(right_output_dir / "metrics_by_checkpoint.csv", index=False)
        right_summary.to_csv(right_output_dir / "checkpoint_metric_summary.csv", index=False)
        scoped_joints.to_csv(right_output_dir / "metrics_per_joint.csv", index=False)
    return plot_evaluation_summaries(
        right_output_dir,
        right_summary,
        scoped_joints,
        labels,
        scope_label="Right arm + right hand",
    )


def raw_trajectory_arrays(
    raw_predictions: pd.DataFrame,
    trajectory_id: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    missing_columns = set(RAW_PREDICTION_COLUMNS) - set(raw_predictions.columns)
    if missing_columns:
        raise ValueError(
            "Raw prediction CSV is missing columns: " + ", ".join(sorted(missing_columns))
        )

    trajectory = raw_predictions[raw_predictions["trajectory"] == trajectory_id]
    if trajectory.empty:
        raise KeyError(f"Trajectory {trajectory_id} is absent from the raw prediction CSV")

    labels = trajectory["joint"].drop_duplicates().astype(str).tolist()
    frames = sorted(int(frame) for frame in trajectory["frame"].unique())
    indexed = trajectory.set_index(["frame", "joint"])
    if not indexed.index.is_unique:
        raise ValueError(f"Trajectory {trajectory_id} contains duplicate frame/joint rows")

    expected_index = pd.MultiIndex.from_product([frames, labels], names=["frame", "joint"])
    missing_rows = expected_index.difference(indexed.index)
    if len(missing_rows):
        raise ValueError(
            f"Trajectory {trajectory_id} is missing {len(missing_rows)} frame/joint rows"
        )
    ordered = indexed.reindex(expected_index)
    shape = (len(frames), len(labels))
    ground_truth = ordered["ground_truth"].to_numpy().reshape(shape)
    prediction = ordered["prediction"].to_numpy().reshape(shape)
    return ground_truth, prediction, labels


def select_saved_plot_ids(
    *,
    dataset_path: Path | None,
    available_ids: list[int],
    explicit_ids: list[int] | None,
    episode_count: int,
    seed: int,
) -> list[int]:
    if explicit_ids:
        missing_ids = sorted(set(explicit_ids) - set(available_ids))
        if missing_ids:
            raise ValueError(
                f"Requested plot trajectories are absent from saved predictions: {missing_ids}"
            )
        return list(explicit_ids)
    if episode_count == 0 or not available_ids:
        return []
    if dataset_path is None:
        rng = np.random.default_rng(seed)
        count = min(episode_count, len(available_ids))
        return sorted(rng.choice(available_ids, size=count, replace=False).tolist())

    dataset_size = max(available_ids) + 1
    return select_task_balanced_trajectory_ids(
        dataset_path,
        dataset_size,
        available_ids,
        episode_count,
        seed,
    )


def regenerate_trajectory_plots(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_steps: list[int],
) -> None:
    if args.skip_trajectory_plots or args.trajectory_plot_episodes == 0:
        return

    split_specs = [
        ("validation", args.dataset_path, args.traj_ids),
        ("train_probe", args.train_dataset_path, args.train_traj_ids),
    ]
    for step in checkpoint_steps:
        checkpoint_dir = output_dir / f"checkpoint-{step}"
        for split, dataset_path, explicit_ids in split_specs:
            raw_path = raw_predictions_csv_path(checkpoint_dir, split)
            if not raw_path.is_file():
                logging.warning(
                    "Cannot regenerate %s trajectory plots; missing %s",
                    split,
                    raw_path,
                )
                continue

            raw_predictions = pd.read_csv(raw_path, compression="gzip")
            available_ids = sorted(int(value) for value in raw_predictions["trajectory"].unique())
            plot_ids = select_saved_plot_ids(
                dataset_path=dataset_path,
                available_ids=available_ids,
                explicit_ids=explicit_ids,
                episode_count=args.trajectory_plot_episodes,
                seed=args.trajectory_plot_seed,
            )
            episode_tasks = (
                _load_episode_tasks(dataset_path, max(available_ids) + 1)
                if dataset_path is not None and available_ids
                else {}
            )
            trajectory_goals = {
                episode_id: " / ".join(dict.fromkeys(tasks))
                for episode_id, tasks in episode_tasks.items()
            }
            plot_dir = checkpoint_dir if split == "validation" else checkpoint_dir / split
            plot_dir.mkdir(parents=True, exist_ok=True)
            right_checkpoint_dir = output_dir / "right_arm_hand" / f"checkpoint-{step}"
            right_plot_dir = (
                right_checkpoint_dir if split == "validation" else right_checkpoint_dir / split
            )
            logging.info(
                "Regenerating %s trajectory plots for checkpoint %d: %s",
                split,
                step,
                plot_ids,
            )
            for trajectory_id in plot_ids:
                ground_truth, prediction, labels = raw_trajectory_arrays(
                    raw_predictions,
                    trajectory_id,
                )
                error = prediction - ground_truth
                goal = trajectory_goals.get(trajectory_id)
                plot_trajectory(
                    ground_truth,
                    prediction,
                    labels,
                    f"{split_label(split)}: checkpoint {step}, trajectory {trajectory_id}",
                    plot_dir / f"trajectory_{trajectory_id:04d}_joints.png",
                    args.execution_horizon,
                    goal,
                )
                plot_error_heatmap(
                    error,
                    labels,
                    f"{split_label(split)} absolute error: checkpoint "
                    f"{step}, trajectory {trajectory_id}",
                    plot_dir / f"trajectory_{trajectory_id:04d}_error_heatmap.png",
                    goal,
                )
                plot_right_trajectory_outputs(
                    ground_truth,
                    prediction,
                    labels,
                    f"{split_label(split)}: checkpoint {step}, trajectory {trajectory_id}",
                    right_plot_dir,
                    trajectory_id,
                    args.execution_horizon,
                    goal,
                )


def regenerate_plots(args: argparse.Namespace, output_dir: Path) -> None:
    summary_path = output_dir / "metrics_by_checkpoint.csv"
    joints_path = output_dir / "metrics_per_joint.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metrics: {summary_path}")
    if not joints_path.is_file():
        raise FileNotFoundError(f"Missing joint metrics: {joints_path}")

    summary = pd.read_csv(summary_path)
    joints = pd.read_csv(joints_path)
    if args.checkpoint_steps:
        selected_steps = set(args.checkpoint_steps)
        summary = summary[summary["checkpoint_step"].isin(selected_steps)]
        joints = joints[joints["checkpoint_step"].isin(selected_steps)]
        missing_steps = selected_steps - set(int(step) for step in summary["checkpoint_step"])
        if missing_steps:
            raise ValueError(f"Saved metrics do not contain checkpoints: {sorted(missing_steps)}")
    checkpoint_steps = sorted(int(step) for step in summary["checkpoint_step"].unique())
    validation_joints = joints[joints["split"] == "validation"]
    labels = validation_joints["joint"].drop_duplicates().astype(str).tolist()
    best_step = plot_evaluation_summaries(output_dir, summary, joints, labels)
    right_best_step = plot_right_evaluation_summaries(
        output_dir,
        summary,
        joints,
        save_metrics=not bool(args.checkpoint_steps),
    )
    regenerate_trajectory_plots(
        args=args,
        output_dir=output_dir,
        checkpoint_steps=checkpoint_steps,
    )

    print(f"Regenerated plots from saved CSVs in {output_dir}")
    print(f"Lowest validation MAE: checkpoint {best_step}")
    if right_best_step is not None:
        print(f"Lowest right-side validation MAE: checkpoint {right_best_step}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    output_dir = args.output_dir or default_evaluation_output_dir(
        args.run_dir, args.execution_horizon
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        regenerate_plots(args, output_dir)
        return

    targets = find_evaluation_targets(
        args.run_dir,
        args.checkpoint_steps,
        args.base_model_path,
    )
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    episode_rows = []
    checkpoint_rows = []
    joint_rows = []
    canonical_labels = None
    evaluation_started_at = time.perf_counter()
    target_count = len(targets)
    logging.info(
        "Evaluation plan: %d model target(s), checkpoint steps=%s, output=%s",
        target_count,
        [target.step for target in targets],
        output_dir,
    )

    for target_index, target in enumerate(targets, start=1):
        target_started_at = time.perf_counter()
        step = target.step
        checkpoint_progress_label = (
            f"checkpoint {target_index}/{target_count} step={step}"
        )
        checkpoint_dir = output_dir / target.output_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "[%s] loading model=%s processor=%s",
            checkpoint_progress_label,
            target.model_path,
            target.processor_path or target.model_path,
        )

        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(target.model_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            processor_path=(
                str(target.processor_path) if target.processor_path is not None else None
            ),
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        seed_inference(args.inference_seed)
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
                    3 if args.train_probe_episodes is None else args.train_probe_episodes,
                    args.train_probe_seed,
                )
            )

        for split, dataset_path, selected_ids, episode_count, seed in probe_specs:
            loader = LeRobotEpisodeLoader(
                dataset_path=str(dataset_path), modality_configs=modality
            )
            trajectory_ids = select_trajectory_ids(
                dataset_path,
                len(loader),
                selected_ids,
                episode_count=episode_count,
                seed=seed,
            )
            episode_tasks = _load_episode_tasks(dataset_path, len(loader))
            trajectory_goals = {
                episode_id: " / ".join(dict.fromkeys(tasks))
                for episode_id, tasks in episode_tasks.items()
            }
            if args.skip_trajectory_plots or args.trajectory_plot_episodes == 0:
                plot_trajectory_ids = set()
            elif selected_ids:
                # Explicit trajectory selections remain explicit for plotting too.
                plot_trajectory_ids = set(trajectory_ids)
            else:
                plot_trajectory_ids = set(
                    select_task_balanced_trajectory_ids(
                        dataset_path,
                        len(loader),
                        trajectory_ids,
                        args.trajectory_plot_episodes,
                        args.trajectory_plot_seed,
                    )
                )
            plot_dir = checkpoint_dir if split == "validation" else checkpoint_dir / split
            plot_dir.mkdir(parents=True, exist_ok=True)
            right_checkpoint_dir = output_dir / "right_arm_hand" / target.output_name
            right_plot_dir = (
                right_checkpoint_dir if split == "validation" else right_checkpoint_dir / split
            )
            logging.info(
                "[%s | %s] evaluating %d episode(s) from %s: %s",
                checkpoint_progress_label,
                split,
                len(trajectory_ids),
                dataset_path,
                trajectory_ids,
            )
            logging.info(
                "Creating per-trajectory plots for %s episode(s): %s",
                split,
                sorted(plot_trajectory_ids),
            )
            raw_predictions_path = raw_predictions_csv_path(checkpoint_dir, split)

            probe_episode_rows, checkpoint_row, probe_joint_rows, canonical_labels = (
                evaluate_probe(
                    policy=policy,
                    loader=loader,
                    trajectory_ids=trajectory_ids,
                    split=split,
                    checkpoint_step_value=step,
                    plot_dir=plot_dir,
                    right_plot_dir=right_plot_dir,
                    embodiment_tag=embodiment_tag,
                    action_keys=action_keys,
                    modality_keys=args.modality_keys,
                    steps=args.steps,
                    execution_horizon=args.execution_horizon,
                    skip_trajectory_plots=args.skip_trajectory_plots,
                    plot_trajectory_ids=plot_trajectory_ids,
                    trajectory_goals=trajectory_goals,
                    raw_predictions_path=raw_predictions_path,
                    canonical_labels=canonical_labels,
                    checkpoint_progress_label=checkpoint_progress_label,
                )
            )
            logging.info("Saved frame-level %s data to %s", split, raw_predictions_path)
            episode_rows.extend(probe_episode_rows)
            checkpoint_rows.append(checkpoint_row)
            joint_rows.extend(probe_joint_rows)
            del loader

        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target_elapsed = time.perf_counter() - target_started_at
        total_elapsed = time.perf_counter() - evaluation_started_at
        remaining_targets = target_count - target_index
        overall_eta = total_elapsed / target_index * remaining_targets
        logging.info(
            "[%s] finished in %s; overall %d/%d complete, estimated remaining %s",
            checkpoint_progress_label,
            format_duration(target_elapsed),
            target_index,
            target_count,
            format_duration(overall_eta),
        )

    episodes = pd.DataFrame(episode_rows)
    summary = pd.DataFrame(checkpoint_rows).sort_values(["split", "checkpoint_step"])
    joints = pd.DataFrame(joint_rows)
    episodes.to_csv(output_dir / "metrics_per_episode.csv", index=False)
    summary.to_csv(output_dir / "metrics_by_checkpoint.csv", index=False)
    joints.to_csv(output_dir / "metrics_per_joint.csv", index=False)

    checkpoint_summary_csv = output_dir / "checkpoint_metric_summary.csv"
    summary.to_csv(checkpoint_summary_csv, index=False)
    best_step = plot_evaluation_summaries(
        output_dir,
        summary,
        joints,
        canonical_labels or [],
    )
    right_best_step = plot_right_evaluation_summaries(
        output_dir,
        summary,
        joints,
        save_metrics=True,
    )

    print(summary.to_string(index=False))
    print(f"\nLowest validation MAE: checkpoint {best_step}")
    if right_best_step is not None:
        print(f"Lowest right-side validation MAE: checkpoint {right_best_step}")
    print(f"Checkpoint summary CSV: {checkpoint_summary_csv}")
    print(
        "Frame-level validation CSVs: "
        f"{output_dir}/<checkpoint>/validation_frame_predictions.csv.gz"
    )
    print(
        "Frame-level train-probe CSVs: "
        f"{output_dir}/<checkpoint>/train_probe_frame_predictions.csv.gz"
    )
    print(f"Results: {output_dir}")
    logging.info(
        "Evaluation complete: %d model target(s) in %s",
        target_count,
        format_duration(time.perf_counter() - evaluation_started_at),
    )


if __name__ == "__main__":
    main()
