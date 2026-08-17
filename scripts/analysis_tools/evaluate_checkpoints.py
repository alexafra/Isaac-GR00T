#!/usr/bin/env python3
"""Evaluate GR00T checkpoints and plot aggregate and per-joint errors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import gzip
import json
import logging
from math import ceil
from pathlib import Path
import random
import re
import time

import matplotlib


matplotlib.use("Agg")
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import open_loop_eval
from gr00t.policy.gr00t_policy import Gr00tPolicy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch


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
        help=("Output directory. Defaults to RUN_DIR/evaluation_exec_hor_<EXECUTION_HORIZON>."),
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
    parser.add_argument(
        "--run-root-step",
        type=int,
        help=(
            "Evaluate the final weights stored directly in RUN_DIR and label them with this "
            "training step. Useful when checkpoint retention removed the final checkpoint-* "
            "directory but Trainer saved the final model at the run root."
        ),
    )
    parser.add_argument("--steps", type=int, default=0, help="0 evaluates each complete episode")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Number of open-loop observation points sent through the policy together. "
            "Larger values reduce evaluation overhead but can use more GPU memory."
        ),
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=42,
        help="Reset the inference RNG to this seed for every evaluated model.",
    )
    parser.add_argument("--modality-keys", nargs="+", default=None)
    parser.add_argument("--skip-trajectory-plots", action="store_true")
    parser.add_argument(
        "--velocity-analysis",
        action="store_true",
        help=(
            "Opt in to per-trajectory joint-velocity plots and all-frame joint "
            "position/velocity statistics. Position, error, and checkpoint plots are "
            "generated without this flag."
        ),
    )
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
        parser.error("--train-traj-ids and --train-probe-episodes require --train-dataset-path")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
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
    run_root_step: int | None = None,
) -> list[EvaluationTarget]:
    """Resolve physical checkpoints, optional final root weights, and a step-0 baseline."""

    if run_root_step is not None and run_root_step <= 0:
        raise ValueError(f"--run-root-step must be positive, got {run_root_step}")
    if run_root_step is not None and re.fullmatch(r"checkpoint-\d+", run_dir.name):
        raise ValueError("--run-root-step requires RUN_DIR to be a training-run root")

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

    include_run_root = run_root_step is not None and (
        selected is None or run_root_step in selected
    )
    if include_run_root:
        if any(target.step == run_root_step for target in targets):
            raise ValueError(
                f"RUN_DIR and a physical checkpoint both represent step {run_root_step}; "
                "omit --run-root-step for that run"
            )
        required_root_files = [run_dir / "config.json", run_dir / "processor"]
        missing_root_files = [path for path in required_root_files if not path.exists()]
        has_root_weights = any(run_dir.glob("model*.safetensors")) or (
            run_dir / "pytorch_model.bin"
        ).is_file()
        if missing_root_files or not has_root_weights:
            missing = [str(path) for path in missing_root_files]
            if not has_root_weights:
                missing.append(f"{run_dir}/model*.safetensors (or pytorch_model.bin)")
            raise FileNotFoundError(
                "Cannot evaluate final weights from the run root; missing: " + ", ".join(missing)
            )
        targets.append(EvaluationTarget(step=run_root_step, model_path=run_dir))

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


def load_dataset_fps(dataset_path: Path) -> float:
    """Load the sampling rate used to turn action-target differences into velocities."""

    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Cannot calculate action-target velocities; missing dataset metadata: {info_path}"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info["fps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot calculate action-target velocities; {info_path} has no valid fps"
        ) from exc
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"Cannot calculate action-target velocities; {info_path} fps must be positive, "
            f"got {fps}"
        )
    return fps


def joint_labels(
    trajectory: pd.DataFrame,
    column_prefix: str,
    keys: list[str],
) -> list[str]:
    labels = []
    for key in keys:
        column = f"{column_prefix}.{key}"
        if column not in trajectory:
            raise ValueError(f"Trajectory is missing required column {column}")
        width = np.asarray(trajectory.iloc[0][column]).reshape(-1).size
        labels.extend(f"{key}[{index}]" for index in range(width))
    return labels


def action_labels(trajectory: pd.DataFrame, action_keys: list[str]) -> list[str]:
    return joint_labels(trajectory, "action", action_keys)


def save_figure_with_parent(figure, path: Path, **savefig_kwargs) -> None:
    """Create a plot parent at write time and retry once if it disappears concurrently."""

    for attempt in range(2):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(path, **savefig_kwargs)
            return
        except FileNotFoundError:
            if attempt:
                raise
            logging.warning("Plot output directory disappeared; recreating %s", path.parent)


def align_measured_state_to_action_labels(
    measured_state: np.ndarray,
    *,
    trajectory: pd.DataFrame,
    state_keys: list[str],
    action_labels_to_match: list[str],
    frame_count: int,
) -> np.ndarray:
    """Reorder captured observation.state columns to the plotted action-joint schema."""

    measured_state = np.asarray(measured_state)
    if measured_state.ndim != 2:
        raise ValueError(
            f"Expected captured measured state [frames, joints], got {measured_state.shape}"
        )
    state_labels = joint_labels(trajectory, "state", state_keys)
    if measured_state.shape[1] != len(state_labels):
        raise ValueError(
            f"Captured measured state has {measured_state.shape[1]} columns but state schema "
            f"describes {len(state_labels)}: {state_labels}"
        )
    if measured_state.shape[0] < frame_count:
        raise ValueError(
            f"Captured measured state has {measured_state.shape[0]} frames but evaluation "
            f"produced {frame_count} action-target frames"
        )
    if len(set(state_labels)) != len(state_labels):
        raise ValueError(f"State schema contains duplicate joint labels: {state_labels}")
    state_indices = {label: index for index, label in enumerate(state_labels)}
    missing_labels = [label for label in action_labels_to_match if label not in state_indices]
    if missing_labels:
        raise ValueError(
            "Cannot align measured observation.state to action targets; state schema is "
            f"missing: {missing_labels}"
        )
    return measured_state[
        :frame_count,
        [state_indices[label] for label in action_labels_to_match],
    ]


def plot_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    execution_horizon: int,
    goal: str | None = None,
    measured_state: np.ndarray | None = None,
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

    if measured_state is not None:
        measured_state = np.asarray(measured_state)
        if measured_state.shape != gt.shape:
            raise ValueError(
                f"Measured-state shape {measured_state.shape} does not match action-target "
                f"shape {gt.shape}"
            )

    # Find the data range required by each joint across every displayed signal.
    displayed_positions = [gt, pred]
    if measured_state is not None:
        displayed_positions.append(measured_state)
    joint_minimums = np.nanmin(np.stack(displayed_positions), axis=(0, 1))
    joint_maximums = np.nanmax(np.stack(displayed_positions), axis=(0, 1))
    joint_ranges = joint_maximums - joint_minimums

    # Give every subplot the range required by the widest-ranging joint.
    common_y_span = float(np.nanmax(joint_ranges))

    if not np.isfinite(common_y_span) or common_y_span <= 0:
        common_y_span = 1.0

    # Add 10% vertical padding.
    common_y_span *= 1.10

    tick_step = 0.1

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

            if measured_state is not None:
                axis.plot(
                    measured_state[:, joint_index],
                    linewidth=1.2,
                    color="tab:green",
                    label="measured joint state",
                )
            axis.plot(
                gt[:, joint_index],
                linewidth=1.2,
                color="tab:blue",
                label="demonstration action target",
            )
            axis.plot(
                pred[:, joint_index],
                linewidth=1.0,
                alpha=0.85,
                color="tab:orange",
                label="model-predicted action target",
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
            joint_centre = (joint_minimums[joint_index] + joint_maximums[joint_index]) / 2.0

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
        f"Common y-axis span: {common_y_span:.3f} rad; action targets are setpoints, "
        "not measured q",
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.94, hspace=0.38, wspace=0.30)
    save_figure_with_parent(figure, path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def action_target_velocities(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    dataset_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference absolute action targets without inventing a frame-0 velocity."""

    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} does not match prediction shape "
            f"{prediction.shape}"
        )
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected [frames, joints] action targets, got {ground_truth.shape}")
    if not np.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError(f"Dataset FPS must be positive, got {dataset_fps}")

    transition_frames = np.arange(1, ground_truth.shape[0])
    ground_truth_velocity = np.diff(ground_truth, axis=0) * dataset_fps
    prediction_velocity = np.diff(prediction, axis=0) * dataset_fps
    return transition_frames, ground_truth_velocity, prediction_velocity


def plot_action_target_velocities(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    execution_horizon: int,
    dataset_fps: float,
    goal: str | None = None,
    measured_state: np.ndarray | None = None,
) -> None:
    """Plot finite-difference measured positions and action targets without conflating them."""

    from collections import defaultdict

    from matplotlib.ticker import MaxNLocator

    if execution_horizon <= 0:
        raise ValueError(f"Execution horizon must be positive, got {execution_horizon}")
    transition_frames, gt_velocity, pred_velocity = action_target_velocities(
        ground_truth,
        prediction,
        dataset_fps,
    )
    if ground_truth.shape[1] != len(labels):
        raise ValueError(f"Expected {ground_truth.shape[1]} action labels, got {len(labels)}")
    measured_velocity = None
    if measured_state is not None:
        measured_state = np.asarray(measured_state)
        if measured_state.shape != ground_truth.shape:
            raise ValueError(
                f"Measured-state shape {measured_state.shape} does not match action-target "
                f"shape {ground_truth.shape}"
            )
        measured_velocity = np.diff(measured_state, axis=0) * dataset_fps

    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, label in enumerate(labels):
        key = label.split("[")[0]
        groups[key].append((index, label))

    preferred = ["left_arm", "left_hand", "right_arm", "right_hand"]
    ordered_keys = [key for key in preferred if key in groups]
    ordered_keys += [key for key in groups if key not in ordered_keys]
    if not ordered_keys:
        raise ValueError("Cannot plot action-target velocities without action labels")

    n_cols = len(ordered_keys)
    max_rows = max(len(groups[key]) for key in ordered_keys)
    figure, axes = plt.subplots(
        max_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.0 * max_rows),
        sharex=True,
        squeeze=False,
    )

    chunk_transitions = np.arange(
        execution_horizon,
        ground_truth.shape[0],
        execution_horizon,
    )
    for column, key in enumerate(ordered_keys):
        items = groups[key]
        for row in range(max_rows):
            axis = axes[row, column]
            if row >= len(items):
                axis.set_visible(False)
                continue

            joint_index, label = items[row]
            if transition_frames.size:
                gt_joint_velocity = gt_velocity[:, joint_index]
                pred_joint_velocity = pred_velocity[:, joint_index]
                measured_joint_velocity = (
                    measured_velocity[:, joint_index] if measured_velocity is not None else None
                )
                if measured_joint_velocity is not None:
                    axis.plot(
                        transition_frames,
                        measured_joint_velocity,
                        linewidth=1.2,
                        color="tab:green",
                        label="measured joint velocity from state q",
                    )
                axis.plot(
                    transition_frames,
                    gt_joint_velocity,
                    linewidth=1.2,
                    color="tab:blue",
                    label="demonstration action-target velocity",
                )
                axis.plot(
                    transition_frames,
                    pred_joint_velocity,
                    linewidth=1.0,
                    alpha=0.85,
                    color="tab:orange",
                    label="model-predicted action-target velocity",
                )
                velocity_series = [gt_joint_velocity, pred_joint_velocity]
                if measured_joint_velocity is not None:
                    velocity_series.append(measured_joint_velocity)
                finite_values = np.concatenate(
                    [values[np.isfinite(values)] for values in velocity_series]
                )
                max_absolute_velocity = (
                    float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
                )
                if max_absolute_velocity <= 0:
                    max_absolute_velocity = 1.0
                axis.set_ylim(-1.10 * max_absolute_velocity, 1.10 * max_absolute_velocity)
                maximum_parts = []
                if measured_joint_velocity is not None:
                    measured_finite = measured_joint_velocity[np.isfinite(measured_joint_velocity)]
                    measured_max = (
                        float(np.max(np.abs(measured_finite))) if measured_finite.size else np.nan
                    )
                    maximum_parts.append(f"measured={measured_max:.3f}")
                for name, values in (
                    ("demo target", gt_joint_velocity),
                    ("pred target", pred_joint_velocity),
                ):
                    finite = values[np.isfinite(values)]
                    maximum = float(np.max(np.abs(finite))) if finite.size else np.nan
                    maximum_parts.append(f"{name}={maximum:.3f}")
                maximum_text = "max |v| rad/s: " + "; ".join(maximum_parts)
            else:
                maximum_text = "max |v|=n/a (fewer than 2 frames)"
                axis.text(
                    0.5,
                    0.5,
                    "No target-velocity samples\n(fewer than 2 action-target frames)",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="dimgray",
                    fontsize=8,
                )
                axis.set_xlim(0, 1)

            for chunk_transition in chunk_transitions:
                axis.axvline(
                    chunk_transition,
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.30,
                    label=(
                        "cross-chunk transition" if chunk_transition == execution_horizon else None
                    ),
                )

            axis.axhline(0, color="black", linewidth=0.6, alpha=0.35)
            axis.set_title(f"{label}\n{maximum_text}", fontsize=8)
            axis.set_xlabel("Transition frame")
            axis.set_ylabel("Joint / action-target velocity (rad/s)")
            if transition_frames.size:
                axis.set_xlim(1, max(1, ground_truth.shape[0] - 1))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            axis.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            axis.grid(alpha=0.2)

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if legend_handles:
        axes[0, 0].legend(legend_handles, legend_labels, fontsize=7)
    figure.suptitle(
        f"{title} — measured joint and action-target velocities",
        fontsize=14,
    )
    header_y = 0.975
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
        f"v[t] = (position[t] - position[t-1]) x {dataset_fps:g} FPS; measured line "
        "uses state q; target lines are setpoint changes; chunk-boundary transitions "
        "are retained",
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.93, hspace=0.55, wspace=0.35)
    save_figure_with_parent(figure, path, dpi=150, bbox_inches="tight")
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
    save_figure_with_parent(figure, path, dpi=160)
    plt.close(figure)


def plot_right_trajectory_outputs(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
    title: str,
    plot_dir: Path,
    trajectory_id: int,
    execution_horizon: int,
    dataset_fps: float | None,
    goal: str | None,
    measured_state: np.ndarray | None = None,
) -> None:
    indices = right_action_indices(labels)
    if not indices:
        logging.warning("No right_arm/right_hand actions found for trajectory %d", trajectory_id)
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    right_ground_truth = ground_truth[:, indices]
    right_prediction = prediction[:, indices]
    right_measured_state = measured_state[:, indices] if measured_state is not None else None
    right_labels = [labels[index] for index in indices]
    plot_trajectory(
        right_ground_truth,
        right_prediction,
        right_labels,
        f"{title} — right arm + right hand",
        plot_dir / f"trajectory_{trajectory_id:04d}_joints.png",
        execution_horizon,
        goal,
        measured_state=right_measured_state,
    )
    if dataset_fps is not None:
        plot_action_target_velocities(
            right_ground_truth,
            right_prediction,
            right_labels,
            f"{title} — right arm + right hand",
            plot_dir / f"trajectory_{trajectory_id:04d}_joint_velocities.png",
            execution_horizon,
            dataset_fps,
            goal,
            measured_state=right_measured_state,
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
    save_figure_with_parent(figure, path, dpi=180)
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
    save_figure_with_parent(figure, path, dpi=180)
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
    save_figure_with_parent(figure, path, dpi=180)
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
    save_figure_with_parent(figure, path, dpi=180)
    plt.close(figure)


LEGACY_RAW_PREDICTION_COLUMNS = [
    "checkpoint_step",
    "trajectory",
    "frame",
    "joint",
    "ground_truth",
    "prediction",
    "error",
    "absolute_error",
]
RAW_PREDICTION_COLUMNS = [*LEGACY_RAW_PREDICTION_COLUMNS, "measured_state"]
RAW_PREDICTION_FILENAMES = {
    "validation": "validation_frame_predictions.csv.gz",
    "train_probe": "train_probe_frame_predictions.csv.gz",
}
GOAL_METRICS_FILENAME = "metrics_by_goal.csv"
HORIZON_POSITION_METRICS_FILENAME = "metrics_by_horizon_position.csv"
ERROR_SUMMARY_COLUMNS = [
    "split",
    "checkpoint_step",
    "episodes",
    "frames",
    "samples",
    "mae",
    "mse",
    "rmse",
    "median_absolute_error",
    "p95_absolute_error",
    "bias",
    "max_absolute_error",
]
GOAL_METRICS_COLUMNS = [
    "split",
    "checkpoint_step",
    "goal",
    *ERROR_SUMMARY_COLUMNS[2:],
]
HORIZON_POSITION_METRICS_COLUMNS = [
    "split",
    "checkpoint_step",
    "horizon_position",
    *ERROR_SUMMARY_COLUMNS[2:],
]
RIGHT_ACTION_PREFIXES = ("right_arm[", "right_hand[")
JOINT_POSITION_VELOCITY_STATISTICS_FILENAME = "joint_position_velocity_statistics.csv"
JOINT_POSITION_VELOCITY_STATISTICS_COLUMNS = [
    "checkpoint_step",
    "split",
    "execution_horizon",
    "dataset_fps",
    "joint",
    "source",
    "episodes",
    "position_samples",
    "velocity_samples",
    "within_chunk_velocity_samples",
    "chunk_boundary_velocity_samples",
    "position_min_rad",
    "position_max_rad",
    "absolute_position_p95_rad",
    "absolute_position_p99_rad",
    "absolute_position_max_rad",
    "absolute_velocity_p95_rad_s",
    "absolute_velocity_p99_rad_s",
    "absolute_velocity_max_rad_s",
    "within_chunk_absolute_velocity_p95_rad_s",
    "within_chunk_absolute_velocity_p99_rad_s",
    "within_chunk_absolute_velocity_max_rad_s",
    "chunk_boundary_absolute_velocity_p95_rad_s",
    "chunk_boundary_absolute_velocity_p99_rad_s",
    "chunk_boundary_absolute_velocity_max_rad_s",
]


def right_action_indices(labels: list[str]) -> list[int]:
    return [
        index for index, label in enumerate(labels) if str(label).startswith(RIGHT_ACTION_PREFIXES)
    ]


def right_joint_rows(joints: pd.DataFrame) -> pd.DataFrame:
    return joints[joints["joint"].astype(str).str.startswith(RIGHT_ACTION_PREFIXES)].copy()


def raw_predictions_csv_path(checkpoint_dir: Path, split: str) -> Path:
    try:
        filename = RAW_PREDICTION_FILENAMES[split]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw-prediction split: {split}") from exc
    return checkpoint_dir / filename


def latest_checkpoint_steps(checkpoint_steps: list[int], plot_count: int = 2) -> list[int]:
    """Return the numerically latest checkpoint steps to receive plots."""

    if plot_count < 0:
        raise ValueError(f"Plot count must be non-negative, got {plot_count}")
    if plot_count == 0:
        return []
    return sorted({int(step) for step in checkpoint_steps})[-plot_count:]


def _validate_error_rows(raw_predictions: pd.DataFrame) -> None:
    required = {
        "checkpoint_step",
        "trajectory",
        "frame",
        "joint",
        "error",
        "absolute_error",
    }
    missing = required - set(raw_predictions.columns)
    if missing:
        raise ValueError("Raw prediction CSV is missing columns: " + ", ".join(sorted(missing)))
    if raw_predictions.empty:
        raise ValueError("Raw prediction CSV contains no rows")
    errors = raw_predictions["error"].to_numpy(dtype=float)
    absolute_errors = raw_predictions["absolute_error"].to_numpy(dtype=float)
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(absolute_errors)):
        raise ValueError("Raw prediction CSV contains non-finite errors")
    if not np.allclose(absolute_errors, np.abs(errors), rtol=1e-6, atol=1e-8):
        raise ValueError("Raw prediction absolute_error values disagree with abs(error)")


def _summarize_error_groups(
    raw_predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    _validate_error_rows(raw_predictions)
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_values, group in raw_predictions.groupby(grouper, sort=True, dropna=False):
        if len(group_columns) == 1:
            group_values = (group_values,)
        error = group["error"].to_numpy(dtype=float)
        absolute_error = np.abs(error)
        mse = float(np.mean(error**2))
        row = dict(zip(group_columns, group_values, strict=True))
        row.update(
            {
                "episodes": int(group["trajectory"].nunique()),
                "frames": int(len(group[["trajectory", "frame"]].drop_duplicates())),
                "samples": int(len(group)),
                "mae": float(np.mean(absolute_error)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "median_absolute_error": float(np.median(absolute_error)),
                "p95_absolute_error": float(np.percentile(absolute_error, 95)),
                "bias": float(np.mean(error)),
                "max_absolute_error": float(np.max(absolute_error)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def goal_metrics_from_raw_predictions(
    raw_predictions: pd.DataFrame,
    *,
    split: str,
    episode_tasks: dict[int, list[str]],
) -> pd.DataFrame:
    """Summarize each recorded goal without combining or reweighting goals."""

    _validate_error_rows(raw_predictions)
    memberships = []
    for trajectory_value in raw_predictions["trajectory"].drop_duplicates():
        trajectory = int(trajectory_value)
        if float(trajectory_value) != trajectory:
            raise ValueError(f"Trajectory ID must be integral, got {trajectory_value!r}")
        goals = list(dict.fromkeys(str(goal) for goal in episode_tasks.get(trajectory, []) if goal))
        if not goals:
            goals = ["(missing goal metadata)"]
        memberships.extend({"trajectory": trajectory, "goal": goal} for goal in goals)

    expanded = raw_predictions.merge(
        pd.DataFrame(memberships),
        on="trajectory",
        how="left",
        validate="many_to_many",
    )
    expanded["split"] = split
    result = _summarize_error_groups(
        expanded,
        ["split", "checkpoint_step", "goal"],
    )
    return result.reindex(columns=GOAL_METRICS_COLUMNS)


def horizon_position_metrics_from_raw_predictions(
    raw_predictions: pd.DataFrame,
    *,
    split: str,
    execution_horizon: int,
) -> pd.DataFrame:
    """Summarize errors by zero-based action offset within each predicted chunk."""

    if execution_horizon <= 0:
        raise ValueError(f"Execution horizon must be positive, got {execution_horizon}")
    _validate_error_rows(raw_predictions)
    frame_values = pd.to_numeric(raw_predictions["frame"], errors="raise").to_numpy(dtype=float)
    if np.any(frame_values < 0) or not np.all(frame_values == np.floor(frame_values)):
        raise ValueError("Raw prediction frame indices must be non-negative integers")
    positioned = raw_predictions.copy()
    positioned["split"] = split
    positioned["horizon_position"] = frame_values.astype(np.int64) % execution_horizon
    result = _summarize_error_groups(
        positioned,
        ["split", "checkpoint_step", "horizon_position"],
    )
    return result.reindex(columns=HORIZON_POSITION_METRICS_COLUMNS)


def write_extended_evaluation_metrics(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    dataset_paths: dict[str, Path | None],
    execution_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write goal and predicted-horizon-position diagnostic CSVs."""

    required_summary_columns = {"split", "checkpoint_step"}
    missing_summary_columns = required_summary_columns - set(summary.columns)
    if missing_summary_columns:
        raise ValueError(
            "Checkpoint summary is missing columns: "
            + ", ".join(sorted(missing_summary_columns))
    )
    goal_frames = []
    horizon_frames = []
    task_cache: dict[tuple[str, int], dict[int, list[str]]] = {}
    for split_value, step_value in (
        summary[["split", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values(["split", "checkpoint_step"])
        .itertuples(index=False, name=None)
    ):
        split = str(split_value)
        step = int(step_value)
        raw_path = raw_predictions_csv_path(output_dir / f"checkpoint-{step}", split)
        if not raw_path.is_file():
            logging.warning("Cannot calculate extended metrics; missing %s", raw_path)
            continue
        raw_predictions = pd.read_csv(raw_path, compression="gzip")
        _validate_error_rows(raw_predictions)
        raw_steps = set(pd.to_numeric(raw_predictions["checkpoint_step"], errors="raise").astype(int))
        if raw_steps != {step}:
            raise ValueError(
                f"{raw_path} contains checkpoint steps {sorted(raw_steps)}, expected only {step}"
            )

        dataset_path = dataset_paths.get(split)
        if dataset_path is None:
            episode_tasks: dict[int, list[str]] = {}
        else:
            dataset_key = (str(dataset_path.resolve()), int(raw_predictions["trajectory"].max()) + 1)
            episode_tasks = task_cache.get(dataset_key, {})
            if dataset_key not in task_cache:
                episode_tasks = _load_episode_tasks(dataset_path, dataset_key[1])
                task_cache[dataset_key] = episode_tasks

        goal_frames.append(
            goal_metrics_from_raw_predictions(
                raw_predictions,
                split=split,
                episode_tasks=episode_tasks,
            )
        )
        horizon_frames.append(
            horizon_position_metrics_from_raw_predictions(
                raw_predictions,
                split=split,
                execution_horizon=execution_horizon,
            )
        )

    goals = (
        pd.concat(goal_frames, ignore_index=True).sort_values(
            ["split", "checkpoint_step", "goal"]
        )
        if goal_frames
        else pd.DataFrame(columns=GOAL_METRICS_COLUMNS)
    )
    horizon_positions = (
        pd.concat(horizon_frames, ignore_index=True).sort_values(
            ["split", "checkpoint_step", "horizon_position"]
        )
        if horizon_frames
        else pd.DataFrame(columns=HORIZON_POSITION_METRICS_COLUMNS)
    )

    goals.to_csv(output_dir / GOAL_METRICS_FILENAME, index=False)
    horizon_positions.to_csv(output_dir / HORIZON_POSITION_METRICS_FILENAME, index=False)
    logging.info(
        "Saved goal and horizon-position metrics under %s",
        output_dir,
    )
    return goals, horizon_positions


def plot_horizon_position_metrics(
    horizon_metrics: pd.DataFrame,
    checkpoint_steps: list[int],
    path: Path,
) -> None:
    selected = horizon_metrics[
        horizon_metrics["checkpoint_step"].isin(checkpoint_steps)
    ].copy()
    if selected.empty:
        logging.warning(
            "Skipping execution-horizon error plot because no selected metrics are available"
        )
        return
    splits = list(dict.fromkeys(selected["split"].astype(str)))
    figure, axes = plt.subplots(
        len(splits),
        1,
        figsize=(11, 4.8 * len(splits)),
        squeeze=False,
    )
    for axis, split in zip(axes[:, 0], splits, strict=True):
        split_metrics = selected[selected["split"] == split]
        for step, checkpoint in split_metrics.groupby("checkpoint_step", sort=True):
            checkpoint = checkpoint.sort_values("horizon_position")
            axis.plot(
                checkpoint["horizon_position"].to_numpy(dtype=int) + 1,
                checkpoint["mae"],
                marker="o",
                linewidth=SUMMARY_LINE_WIDTH,
                markersize=SUMMARY_MARKER_SIZE,
                label=f"checkpoint {int(step)}",
            )
        positions = sorted(int(value) + 1 for value in split_metrics["horizon_position"].unique())
        axis.set_xticks(positions)
        axis.set_xlabel("Predicted action position within execution chunk (1 = first)")
        axis.set_ylabel("Unnormalized action MAE")
        axis.set_title(f"{split_label(split)} error by predicted horizon position")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    save_figure_with_parent(figure, path, dpi=180)
    plt.close(figure)


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
                "frames": len(right_predictions[["trajectory", "frame"]].drop_duplicates()),
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
    measured_state: np.ndarray | None = None,
) -> pd.DataFrame:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} does not match prediction shape "
            f"{prediction.shape}"
        )
    if ground_truth.ndim != 2 or ground_truth.shape[1] != len(labels):
        raise ValueError(f"Expected [frames, {len(labels)} joints], got {ground_truth.shape}")
    if measured_state is not None:
        measured_state = np.asarray(measured_state)
        if measured_state.shape != ground_truth.shape:
            raise ValueError(
                f"Measured-state shape {measured_state.shape} does not match action-target "
                f"shape {ground_truth.shape}"
            )

    frame_count, joint_count = ground_truth.shape
    error = prediction - ground_truth
    measured_values = (
        measured_state.reshape(-1)
        if measured_state is not None
        else np.full(frame_count * joint_count, np.nan)
    )
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
            "measured_state": measured_values,
        },
        columns=RAW_PREDICTION_COLUMNS,
    )


def ensure_measured_state_in_raw_predictions(
    raw_predictions: pd.DataFrame,
    *,
    dataset_path: Path,
    state_cache: dict[tuple, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fill missing measured q values from exact saved episode/frame/joint parquet rows."""

    missing_columns = set(LEGACY_RAW_PREDICTION_COLUMNS) - set(raw_predictions.columns)
    if missing_columns:
        raise ValueError(
            "Raw prediction CSV is missing columns: " + ", ".join(sorted(missing_columns))
        )
    result = raw_predictions.copy()
    if "measured_state" not in result:
        result["measured_state"] = np.nan
    measured_values = pd.to_numeric(result["measured_state"], errors="coerce")
    if np.all(np.isfinite(measured_values.to_numpy(dtype=float))):
        result["measured_state"] = measured_values
        return result
    if state_cache is None:
        state_cache = {}

    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    modality_path = meta_dir / "modality.json"
    episodes_path = meta_dir / "episodes.jsonl"
    for metadata_path in (info_path, modality_path, episodes_path):
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Cannot load measured observation.state; missing {metadata_path}"
            )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        modality = json.loads(modality_path.read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data_path_pattern = str(info["data_path"])
        chunk_size = int(info["chunks_size"])
        state_schema = modality["state"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot load measured observation.state; invalid dataset metadata under {meta_dir}"
        ) from exc
    if chunk_size <= 0 or not isinstance(state_schema, dict):
        raise ValueError(f"Invalid state/chunk schema under {meta_dir}")

    label_specs: dict[str, tuple[str, int]] = {}
    for label in result["joint"].drop_duplicates().astype(str):
        match = re.fullmatch(r"(.+)\[(\d+)\]", label)
        if match is None:
            raise ValueError(f"Cannot map measured state for invalid joint label {label!r}")
        group_name, local_index_text = match.groups()
        if group_name not in state_schema:
            raise ValueError(
                f"Measured-state schema has no group {group_name!r} required by {label}"
            )
        group_schema = state_schema[group_name]
        try:
            start = int(group_schema["start"])
            end = int(group_schema["end"])
            original_key = str(group_schema.get("original_key", "observation.state"))
            local_index = int(local_index_text)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid measured-state schema for group {group_name!r}") from exc
        if start < 0 or end <= start or not 0 <= local_index < end - start:
            raise ValueError(
                f"Measured-state schema range [{start}, {end}) does not contain {label}"
            )
        label_specs[label] = (original_key, start + local_index)

    key_columns = ["trajectory", "frame", "joint"]
    if result.duplicated(key_columns).any():
        raise ValueError("Raw prediction CSV contains duplicate trajectory/frame/joint rows")

    for trajectory_value, trajectory_rows in result.groupby("trajectory", sort=False):
        trajectory_id = int(trajectory_value)
        if float(trajectory_value) != trajectory_id or not 0 <= trajectory_id < len(episodes):
            raise ValueError(
                f"Saved trajectory ID {trajectory_value!r} is not a valid dataset episode index"
            )
        episode_record = episodes[trajectory_id]
        try:
            episode_index = int(episode_record["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid episode metadata record at index {trajectory_id}") from exc
        parquet_relative_path = data_path_pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        parquet_path = dataset_path / parquet_relative_path
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Cannot load measured observation.state; missing {parquet_path}"
            )
        schema_key = tuple(
            sorted(
                (label, original_key, state_index)
                for label, (original_key, state_index) in label_specs.items()
            )
        )
        cache_key = (str(dataset_path.resolve()), episode_index, schema_key)
        measured_episode = state_cache.get(cache_key)
        if measured_episode is None:
            episode_data = pd.read_parquet(parquet_path)
            if "episode_index" in episode_data:
                parquet_episode_ids = set(
                    pd.to_numeric(episode_data["episode_index"], errors="raise").astype(int)
                )
                if parquet_episode_ids != {episode_index}:
                    raise ValueError(
                        f"{parquet_path} contains episode IDs "
                        f"{sorted(parquet_episode_ids)}, expected only {episode_index}"
                    )
            if "frame_index" not in episode_data:
                raise ValueError(
                    f"Cannot safely align measured state; {parquet_path} has no frame_index column"
                )
            frame_numbers = pd.to_numeric(episode_data["frame_index"], errors="raise").astype(int)
            if frame_numbers.duplicated().any():
                raise ValueError(f"{parquet_path} contains duplicate frame_index values")
            episode_data = episode_data.set_index(frame_numbers)
            measured_episode = pd.DataFrame(index=episode_data.index)
            for label, (original_key, state_index) in label_specs.items():
                if original_key not in episode_data:
                    raise ValueError(
                        f"Measured-state source column {original_key!r} for {label} is "
                        f"absent from {parquet_path}"
                    )
                label_values = []
                for frame, raw_state in episode_data[original_key].items():
                    state_vector = np.asarray(raw_state).reshape(-1)
                    if state_index >= len(state_vector):
                        raise ValueError(
                            f"Measured-state vector {original_key!r} at episode "
                            f"{episode_index} frame {frame} has width {len(state_vector)}, "
                            f"cannot read index {state_index} for {label}"
                        )
                    label_values.append(float(state_vector[state_index]))
                label_array = np.asarray(label_values, dtype=float)
                if not np.all(np.isfinite(label_array)):
                    raise ValueError(
                        f"Measured-state values for episode {episode_index} {label} are non-finite"
                    )
                measured_episode[label] = label_array
            state_cache[cache_key] = measured_episode

        requested_frames = trajectory_rows["frame"].astype(int)
        missing_frames = sorted(set(requested_frames) - set(measured_episode.index))
        if missing_frames:
            raise ValueError(
                f"{parquet_path} is missing saved frame indices {missing_frames} for "
                f"trajectory {trajectory_id}"
            )
        for label, label_rows in trajectory_rows.groupby("joint", sort=False):
            label = str(label)
            label_frames = label_rows["frame"].astype(int).to_numpy()
            loaded_values = measured_episode.loc[label_frames, label].to_numpy(dtype=float)
            existing_values = measured_values.loc[label_rows.index].to_numpy(dtype=float)
            disagreements = np.isfinite(existing_values) & ~np.isclose(
                existing_values,
                loaded_values,
                rtol=1e-6,
                atol=1e-7,
            )
            if np.any(disagreements):
                mismatch_index = int(np.flatnonzero(disagreements)[0])
                raise ValueError(
                    f"Saved measured state disagrees with dataset at trajectory "
                    f"{trajectory_id} frame {label_frames[mismatch_index]} {label}: "
                    f"{existing_values[mismatch_index]} vs {loaded_values[mismatch_index]}"
                )
            result.loc[label_rows.index, "measured_state"] = loaded_values

    if not np.all(np.isfinite(result["measured_state"].to_numpy(dtype=float))):
        raise ValueError("Failed to populate every saved measured-state row")
    return result


def joint_position_velocity_statistics(
    raw_predictions: pd.DataFrame,
    *,
    split: str,
    dataset_fps: float,
    execution_horizon: int,
) -> pd.DataFrame:
    """Summarize measured q and action targets without crossing episode boundaries."""

    missing_columns = set(RAW_PREDICTION_COLUMNS) - set(raw_predictions.columns)
    if missing_columns:
        raise ValueError(
            "Raw prediction CSV is missing columns: " + ", ".join(sorted(missing_columns))
        )
    if not np.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError(f"Dataset FPS must be positive, got {dataset_fps}")
    if execution_horizon <= 0:
        raise ValueError(f"Execution horizon must be positive, got {execution_horizon}")

    rows = []
    for (step_value, joint), joint_frame in raw_predictions.groupby(
        ["checkpoint_step", "joint"],
        sort=False,
    ):
        episode_count = int(joint_frame["trajectory"].nunique())
        for source, value_column in (
            ("measured_state", "measured_state"),
            ("ground_truth_action_target", "ground_truth"),
            ("predicted_action_target", "prediction"),
        ):
            positions = joint_frame[value_column].to_numpy(dtype=float)
            if not np.all(np.isfinite(positions)):
                raise ValueError(
                    f"Checkpoint {step_value} {split} {joint} {source} contains "
                    "non-finite joint positions"
                )

            episode_velocities = []
            within_chunk_velocities = []
            chunk_boundary_velocities = []
            for trajectory_id, episode_frame in joint_frame.groupby("trajectory", sort=False):
                episode_frame = episode_frame.sort_values("frame")
                frame_numbers = episode_frame["frame"].to_numpy(dtype=int)
                if len(np.unique(frame_numbers)) != len(frame_numbers):
                    raise ValueError(
                        f"Checkpoint {step_value} {split} trajectory {trajectory_id} "
                        f"joint {joint} contains duplicate frames"
                    )
                if len(frame_numbers) > 1 and not np.all(np.diff(frame_numbers) == 1):
                    raise ValueError(
                        f"Checkpoint {step_value} {split} trajectory {trajectory_id} "
                        f"joint {joint} contains non-contiguous frames"
                    )
                episode_positions = episode_frame[value_column].to_numpy(dtype=float)
                if len(episode_positions) > 1:
                    velocities = np.diff(episode_positions) * dataset_fps
                    is_chunk_boundary = frame_numbers[1:] % execution_horizon == 0
                    episode_velocities.append(velocities)
                    within_chunk_velocities.append(velocities[~is_chunk_boundary])
                    chunk_boundary_velocities.append(velocities[is_chunk_boundary])

            target_velocities = (
                np.concatenate(episode_velocities) if episode_velocities else np.empty(0)
            )
            within_chunk_target_velocities = (
                np.concatenate(within_chunk_velocities) if within_chunk_velocities else np.empty(0)
            )
            chunk_boundary_target_velocities = (
                np.concatenate(chunk_boundary_velocities)
                if chunk_boundary_velocities
                else np.empty(0)
            )
            absolute_positions = np.abs(positions)
            absolute_velocities = np.abs(target_velocities)
            within_chunk_absolute_velocities = np.abs(within_chunk_target_velocities)
            chunk_boundary_absolute_velocities = np.abs(chunk_boundary_target_velocities)

            def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
                return float(np.percentile(values, percentile)) if values.size else np.nan

            def maximum_or_nan(values: np.ndarray) -> float:
                return float(np.max(values)) if values.size else np.nan

            rows.append(
                {
                    "checkpoint_step": int(step_value),
                    "split": split,
                    "execution_horizon": execution_horizon,
                    "dataset_fps": dataset_fps,
                    "joint": str(joint),
                    "source": source,
                    "episodes": episode_count,
                    "position_samples": len(positions),
                    "velocity_samples": len(target_velocities),
                    "within_chunk_velocity_samples": len(within_chunk_target_velocities),
                    "chunk_boundary_velocity_samples": len(chunk_boundary_target_velocities),
                    "position_min_rad": float(np.min(positions)),
                    "position_max_rad": float(np.max(positions)),
                    "absolute_position_p95_rad": float(np.percentile(absolute_positions, 95)),
                    "absolute_position_p99_rad": float(np.percentile(absolute_positions, 99)),
                    "absolute_position_max_rad": float(np.max(absolute_positions)),
                    "absolute_velocity_p95_rad_s": percentile_or_nan(absolute_velocities, 95),
                    "absolute_velocity_p99_rad_s": percentile_or_nan(absolute_velocities, 99),
                    "absolute_velocity_max_rad_s": maximum_or_nan(absolute_velocities),
                    "within_chunk_absolute_velocity_p95_rad_s": percentile_or_nan(
                        within_chunk_absolute_velocities, 95
                    ),
                    "within_chunk_absolute_velocity_p99_rad_s": percentile_or_nan(
                        within_chunk_absolute_velocities, 99
                    ),
                    "within_chunk_absolute_velocity_max_rad_s": maximum_or_nan(
                        within_chunk_absolute_velocities
                    ),
                    "chunk_boundary_absolute_velocity_p95_rad_s": percentile_or_nan(
                        chunk_boundary_absolute_velocities, 95
                    ),
                    "chunk_boundary_absolute_velocity_p99_rad_s": percentile_or_nan(
                        chunk_boundary_absolute_velocities, 99
                    ),
                    "chunk_boundary_absolute_velocity_max_rad_s": maximum_or_nan(
                        chunk_boundary_absolute_velocities
                    ),
                }
            )

    return pd.DataFrame(rows, columns=JOINT_POSITION_VELOCITY_STATISTICS_COLUMNS)


def write_joint_position_velocity_statistics(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    dataset_paths: dict[str, Path | None],
    execution_horizon: int,
    state_cache: dict[tuple, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build root/right all-frame measured-state and action-target statistics CSVs."""

    statistic_frames = []
    if state_cache is None:
        state_cache = {}
    split_steps = summary[["split", "checkpoint_step"]].drop_duplicates()
    for split_value, step_value in split_steps.itertuples(index=False, name=None):
        split = str(split_value)
        step = int(step_value)
        dataset_path = dataset_paths.get(split)
        if dataset_path is None:
            logging.warning(
                "Cannot calculate %s target position/velocity statistics without its dataset path",
                split,
            )
            continue
        raw_path = raw_predictions_csv_path(output_dir / f"checkpoint-{step}", split)
        if not raw_path.is_file():
            logging.warning(
                "Cannot calculate %s target position/velocity statistics; missing %s",
                split,
                raw_path,
            )
            continue
        raw_predictions = pd.read_csv(raw_path, compression="gzip")
        raw_predictions = ensure_measured_state_in_raw_predictions(
            raw_predictions,
            dataset_path=dataset_path,
            state_cache=state_cache,
        )
        statistic_frames.append(
            joint_position_velocity_statistics(
                raw_predictions,
                split=split,
                dataset_fps=load_dataset_fps(dataset_path),
                execution_horizon=execution_horizon,
            )
        )

    if not statistic_frames:
        logging.warning("No raw predictions were available for target statistics")
        return pd.DataFrame(columns=JOINT_POSITION_VELOCITY_STATISTICS_COLUMNS)

    statistics = pd.concat(statistic_frames, ignore_index=True).sort_values(
        ["split", "checkpoint_step", "joint", "source"]
    )
    statistics_path = output_dir / JOINT_POSITION_VELOCITY_STATISTICS_FILENAME
    statistics.to_csv(statistics_path, index=False)

    right_statistics = statistics[
        statistics["joint"].astype(str).str.startswith(RIGHT_ACTION_PREFIXES)
    ].copy()
    right_output_dir = output_dir / "right_arm_hand"
    right_output_dir.mkdir(parents=True, exist_ok=True)
    right_statistics.to_csv(
        right_output_dir / JOINT_POSITION_VELOCITY_STATISTICS_FILENAME,
        index=False,
    )
    logging.info(
        "Saved all-frame measured-state/action-target position/velocity statistics to %s",
        statistics_path,
    )
    return statistics


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
    dataset_fps: float | None,
    skip_trajectory_plots: bool,
    plot_trajectory_ids: set[int],
    trajectory_goals: dict[int, str],
    raw_predictions_path: Path | None,
    canonical_labels: list[str] | None,
    checkpoint_progress_label: str,
    velocity_analysis: bool = False,
    inference_batch_size: int = 1,
    inference_seed: int | None = None,
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
        if hasattr(loader, "load_episode"):
            trajectory = loader.load_episode(
                traj_id,
                frame_indices=lambda trajectory_length: (
                    open_loop_eval.required_visual_frame_indices(
                        trajectory_length=trajectory_length,
                        steps=steps if steps > 0 else trajectory_length,
                        execution_horizon=execution_horizon,
                        modality_configs=loader.modality_configs,
                    )
                ),
            )
        else:
            # Retain compatibility with simple external/mocked episode loaders.
            trajectory = loader[traj_id]
        evaluation_steps = steps if steps > 0 else len(trajectory)
        evaluation_steps = min(evaluation_steps, len(trajectory))
        inference_count = ceil(evaluation_steps / execution_horizon)
        inference_request_count = ceil(inference_count / inference_batch_size)
        progress_label = (
            f"{checkpoint_progress_label} | {split} episode "
            f"{trajectory_index}/{trajectory_count} traj={traj_id}"
        )
        goal = trajectory_goals.get(traj_id)
        logging.info(
            "[%s] starting: %d frame(s), %d inference point(s) in %d request(s), goal=%r",
            progress_label,
            evaluation_steps,
            inference_count,
            inference_request_count,
            goal,
        )
        labels = action_labels(trajectory, action_keys)
        if canonical_labels is None:
            canonical_labels = labels
        elif labels != canonical_labels:
            raise RuntimeError(
                "Action dimensions changed between datasets, checkpoints, or trajectories"
            )

        captured: dict[str, object] = {}

        def capture_plot(**kwargs) -> None:
            captured["measured_state"] = np.asarray(kwargs["state_joints_across_time"])
            captured["ground_truth"] = np.asarray(kwargs["gt_action_across_time"])
            captured["prediction"] = np.asarray(kwargs["pred_action_across_time"])
            captured["state_keys"] = [str(key) for key in kwargs["state_keys"]]

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
                trajectory=trajectory,
                inference_batch_size=inference_batch_size,
                inference_seed=inference_seed,
            )
        finally:
            open_loop_eval.plot_trajectory_results = original_plotter

        gt = np.asarray(captured["ground_truth"])
        pred = np.asarray(captured["prediction"])
        measured_state = align_measured_state_to_action_labels(
            np.asarray(captured["measured_state"]),
            trajectory=trajectory,
            state_keys=list(captured["state_keys"]),
            action_labels_to_match=labels,
            frame_count=len(gt),
        )
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
                    measured_state=measured_state,
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
                measured_state=measured_state,
            )
            if velocity_analysis:
                if dataset_fps is None:
                    raise ValueError("Velocity analysis requires the dataset FPS")
                plot_action_target_velocities(
                    gt,
                    pred,
                    labels,
                    (
                        f"{split_label(split)}: checkpoint {checkpoint_step_value}, "
                        f"trajectory {traj_id}"
                    ),
                    plot_dir / f"trajectory_{traj_id:04d}_joint_velocities.png",
                    execution_horizon,
                    dataset_fps,
                    goal,
                    measured_state=measured_state,
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
                f"{split_label(split)}: checkpoint {checkpoint_step_value}, trajectory {traj_id}",
                right_plot_dir,
                traj_id,
                execution_horizon,
                dataset_fps if velocity_analysis else None,
                goal,
                measured_state=measured_state,
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
        finetuned_validation_joints = finetuned_joints[finetuned_joints["split"] == "validation"]
        if not finetuned_validation_joints.empty:
            plot_joint_checkpoint_heatmap(
                finetuned_validation_joints,
                labels,
                output_dir / f"joint_error_by_checkpoint{finetuned_suffix}.png",
                finetuned_scope_label,
            )

    best_step = int(validation_summary.loc[validation_summary["mae"].idxmin(), "checkpoint_step"])
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
    plot_checkpoint_steps: set[int] | None = None,
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
    validation_right_summary = right_summary[right_summary["split"] == "validation"]
    best_step = (
        int(
            validation_right_summary.loc[
                validation_right_summary["mae"].idxmin(), "checkpoint_step"
            ]
        )
        if not validation_right_summary.empty
        else None
    )
    if plot_checkpoint_steps is not None:
        right_summary = right_summary[
            right_summary["checkpoint_step"].isin(plot_checkpoint_steps)
        ]
        scoped_joints = scoped_joints[
            scoped_joints["checkpoint_step"].isin(plot_checkpoint_steps)
        ]
    if right_summary.empty or scoped_joints.empty:
        logging.warning("Skipping right-arm/hand plots because no selected checkpoints remain")
        return best_step
    plot_evaluation_summaries(
        right_output_dir,
        right_summary,
        scoped_joints,
        labels,
        scope_label="Right arm + right hand",
    )
    return best_step


def raw_trajectory_arrays(
    raw_predictions: pd.DataFrame,
    trajectory_id: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    missing_columns = set(LEGACY_RAW_PREDICTION_COLUMNS) - set(raw_predictions.columns)
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


def raw_measured_state_array(
    raw_predictions: pd.DataFrame,
    trajectory_id: int,
) -> np.ndarray:
    """Return measured state using the exact frame/joint ordering of a raw trajectory."""

    if "measured_state" not in raw_predictions:
        raise ValueError("Raw prediction data has no measured_state column")
    trajectory = raw_predictions[raw_predictions["trajectory"] == trajectory_id]
    if trajectory.empty:
        raise KeyError(f"Trajectory {trajectory_id} is absent from the raw prediction CSV")
    labels = trajectory["joint"].drop_duplicates().astype(str).tolist()
    frames = sorted(int(frame) for frame in trajectory["frame"].unique())
    indexed = trajectory.set_index(["frame", "joint"])
    if not indexed.index.is_unique:
        raise ValueError(f"Trajectory {trajectory_id} contains duplicate frame/joint rows")
    expected_index = pd.MultiIndex.from_product([frames, labels], names=["frame", "joint"])
    ordered = indexed.reindex(expected_index)
    measured_state = (
        ordered["measured_state"]
        .to_numpy(dtype=float)
        .reshape(
            len(frames),
            len(labels),
        )
    )
    if not np.all(np.isfinite(measured_state)):
        raise ValueError(f"Trajectory {trajectory_id} contains missing measured-state values")
    return measured_state


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
    state_cache: dict[tuple, pd.DataFrame] | None = None,
) -> None:
    if args.skip_trajectory_plots or args.trajectory_plot_episodes == 0:
        return
    if state_cache is None:
        state_cache = {}

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
            if dataset_path is not None:
                raw_predictions = ensure_measured_state_in_raw_predictions(
                    raw_predictions,
                    dataset_path=dataset_path,
                    state_cache=state_cache,
                )
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
            if args.velocity_analysis and dataset_path is None:
                logging.warning(
                    "Cannot regenerate %s target-velocity plots without its dataset path",
                    split,
                )
                dataset_fps = None
            elif args.velocity_analysis:
                dataset_fps = load_dataset_fps(dataset_path)
            else:
                dataset_fps = None
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
                measured_state = (
                    raw_measured_state_array(raw_predictions, trajectory_id)
                    if dataset_path is not None
                    else None
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
                    measured_state=measured_state,
                )
                if dataset_fps is not None:
                    plot_action_target_velocities(
                        ground_truth,
                        prediction,
                        labels,
                        f"{split_label(split)}: checkpoint {step}, trajectory {trajectory_id}",
                        plot_dir / f"trajectory_{trajectory_id:04d}_joint_velocities.png",
                        args.execution_horizon,
                        dataset_fps,
                        goal,
                        measured_state=measured_state,
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
                    dataset_fps,
                    goal,
                    measured_state=measured_state,
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
    state_cache: dict[tuple, pd.DataFrame] = {}
    _, horizon_metrics = write_extended_evaluation_metrics(
        output_dir=output_dir,
        summary=summary,
        dataset_paths={
            "validation": args.dataset_path,
            "train_probe": args.train_dataset_path,
        },
        execution_horizon=args.execution_horizon,
    )
    if args.velocity_analysis:
        write_joint_position_velocity_statistics(
            output_dir=output_dir,
            summary=summary,
            dataset_paths={
                "validation": args.dataset_path,
                "train_probe": args.train_dataset_path,
            },
            execution_horizon=args.execution_horizon,
            state_cache=state_cache,
        )
    if args.checkpoint_steps:
        selected_steps = set(args.checkpoint_steps)
        summary = summary[summary["checkpoint_step"].isin(selected_steps)]
        joints = joints[joints["checkpoint_step"].isin(selected_steps)]
        missing_steps = selected_steps - set(int(step) for step in summary["checkpoint_step"])
        if missing_steps:
            raise ValueError(f"Saved metrics do not contain checkpoints: {sorted(missing_steps)}")
    checkpoint_steps = sorted(int(step) for step in summary["checkpoint_step"].unique())
    plot_checkpoint_steps = latest_checkpoint_steps(checkpoint_steps)
    plot_step_set = set(plot_checkpoint_steps)
    plot_summary = summary[summary["checkpoint_step"].isin(plot_step_set)]
    plot_joints = joints[joints["checkpoint_step"].isin(plot_step_set)]
    validation_summary = summary[summary["split"] == "validation"]
    best_step = int(
        validation_summary.loc[validation_summary["mae"].idxmin(), "checkpoint_step"]
    )
    validation_joints = plot_joints[plot_joints["split"] == "validation"]
    labels = validation_joints["joint"].drop_duplicates().astype(str).tolist()
    plot_evaluation_summaries(output_dir, plot_summary, plot_joints, labels)
    right_best_step = plot_right_evaluation_summaries(
        output_dir,
        summary,
        joints,
        save_metrics=not bool(args.checkpoint_steps),
        plot_checkpoint_steps=plot_step_set,
    )
    plot_horizon_position_metrics(
        horizon_metrics,
        plot_checkpoint_steps,
        output_dir / "error_by_horizon_position.png",
    )
    regenerate_trajectory_plots(
        args=args,
        output_dir=output_dir,
        checkpoint_steps=plot_checkpoint_steps,
        state_cache=state_cache,
    )

    print(f"Regenerated plots from saved CSVs in {output_dir}")
    print(f"Plots limited to latest checkpoint(s): {plot_checkpoint_steps}")
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
        args.run_root_step,
    )
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    episode_rows = []
    checkpoint_rows = []
    joint_rows = []
    canonical_labels = None
    evaluation_started_at = time.perf_counter()
    target_count = len(targets)
    plot_checkpoint_steps = latest_checkpoint_steps([target.step for target in targets])
    plot_step_set = set(plot_checkpoint_steps)
    logging.info(
        "Evaluation plan: %d model target(s), checkpoint steps=%s, plots=%s, output=%s",
        target_count,
        [target.step for target in targets],
        plot_checkpoint_steps,
        output_dir,
    )

    for target_index, target in enumerate(targets, start=1):
        target_started_at = time.perf_counter()
        step = target.step
        checkpoint_progress_label = f"checkpoint {target_index}/{target_count} step={step}"
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
            modality["action"].modality_keys if args.modality_keys is None else args.modality_keys
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
            loader = LeRobotEpisodeLoader(dataset_path=str(dataset_path), modality_configs=modality)
            dataset_fps = load_dataset_fps(dataset_path) if args.velocity_analysis else None
            if dataset_fps is not None and not np.isclose(float(loader.fps), dataset_fps):
                raise ValueError(
                    f"Loader FPS {loader.fps} does not match {dataset_path}/meta/info.json "
                    f"FPS {dataset_fps}"
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
            if (
                step not in plot_step_set
                or args.skip_trajectory_plots
                or args.trajectory_plot_episodes == 0
            ):
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

            probe_episode_rows, checkpoint_row, probe_joint_rows, canonical_labels = evaluate_probe(
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
                dataset_fps=dataset_fps,
                skip_trajectory_plots=args.skip_trajectory_plots,
                plot_trajectory_ids=plot_trajectory_ids,
                trajectory_goals=trajectory_goals,
                raw_predictions_path=raw_predictions_path,
                canonical_labels=canonical_labels,
                checkpoint_progress_label=checkpoint_progress_label,
                velocity_analysis=args.velocity_analysis,
                inference_batch_size=args.inference_batch_size,
                inference_seed=args.inference_seed,
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
    _, horizon_metrics = write_extended_evaluation_metrics(
        output_dir=output_dir,
        summary=summary,
        dataset_paths={
            "validation": args.dataset_path,
            "train_probe": args.train_dataset_path,
        },
        execution_horizon=args.execution_horizon,
    )
    if args.velocity_analysis:
        write_joint_position_velocity_statistics(
            output_dir=output_dir,
            summary=summary,
            dataset_paths={
                "validation": args.dataset_path,
                "train_probe": args.train_dataset_path,
            },
            execution_horizon=args.execution_horizon,
        )

    checkpoint_summary_csv = output_dir / "checkpoint_metric_summary.csv"
    summary.to_csv(checkpoint_summary_csv, index=False)
    plot_summary = summary[summary["checkpoint_step"].isin(plot_step_set)]
    plot_joints = joints[joints["checkpoint_step"].isin(plot_step_set)]
    validation_summary = summary[summary["split"] == "validation"]
    best_step = int(
        validation_summary.loc[validation_summary["mae"].idxmin(), "checkpoint_step"]
    )
    plot_evaluation_summaries(
        output_dir,
        plot_summary,
        plot_joints,
        canonical_labels or [],
    )
    right_best_step = plot_right_evaluation_summaries(
        output_dir,
        summary,
        joints,
        save_metrics=True,
        plot_checkpoint_steps=plot_step_set,
    )
    plot_horizon_position_metrics(
        horizon_metrics,
        plot_checkpoint_steps,
        output_dir / "error_by_horizon_position.png",
    )

    print(summary.to_string(index=False))
    print(f"\nLowest validation MAE: checkpoint {best_step}")
    if right_best_step is not None:
        print(f"Lowest right-side validation MAE: checkpoint {right_best_step}")
    print(f"Plots limited to latest checkpoint(s): {plot_checkpoint_steps}")
    print(f"Checkpoint summary CSV: {checkpoint_summary_csv}")
    print(f"Goal metrics CSV: {output_dir / GOAL_METRICS_FILENAME}")
    print(f"Horizon-position metrics CSV: {output_dir / HORIZON_POSITION_METRICS_FILENAME}")
    print(
        "Frame-level validation CSVs: "
        f"{output_dir}/<checkpoint>/validation_frame_predictions.csv.gz"
    )
    print(
        "Frame-level train-probe CSVs: "
        f"{output_dir}/<checkpoint>/train_probe_frame_predictions.csv.gz"
    )
    if args.velocity_analysis:
        print(
            "All-frame measured-state/action-target position/velocity statistics: "
            f"{output_dir}/{JOINT_POSITION_VELOCITY_STATISTICS_FILENAME}"
        )
    print(f"Results: {output_dir}")
    logging.info(
        "Evaluation complete: %d model target(s) in %s",
        target_count,
        format_duration(time.perf_counter() - evaluation_started_at),
    )


if __name__ == "__main__":
    main()
