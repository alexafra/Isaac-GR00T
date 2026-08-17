#!/usr/bin/env python3
"""Paired episode-bootstrap comparison across completed model evaluations.

Example::

    python -m scripts.analysis_tools.compare_evaluation_bootstrap \
        --evaluation colour=/path/to/colour/evaluation_exec_hor_8 \
        --evaluation depth=/path/to/depth/evaluation_exec_hor_8 \
        --output-dir /path/to/cross_model_comparison

The latest checkpoint present in every input is selected by default. Use
``--checkpoint-step`` to select one explicitly. Episodes are aligned by split and
trajectory before a single set of paired bootstrap draws is shared by every model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("mae", "mse")
REQUIRED_COLUMNS = {"split", "checkpoint_step", "trajectory", "frames"}


@dataclass(frozen=True)
class EvaluationInput:
    label: str
    directory: Path


@dataclass(frozen=True)
class ComparisonResult:
    checkpoint_step: int
    episode_count: int
    model_intervals_path: Path
    pairwise_intervals_path: Path
    metadata_path: Path


def parse_evaluation(value: str) -> EvaluationInput:
    """Parse one ``LABEL=EVALUATION_OUTPUT_DIR`` command-line value."""
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "evaluation must be written as LABEL=EVALUATION_OUTPUT_DIR"
        )
    return EvaluationInput(label.strip(), Path(raw_path).expanduser())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        type=parse_evaluation,
        metavar="LABEL=DIR",
        help="Model label and directory containing metrics_per_episode.csv; repeat per model.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Checkpoint to compare. Defaults to the latest step shared by every model.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Per-episode error column to bootstrap; repeat as needed (default: mae, mse).",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def _validated_integer_column(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{source}: {column} must contain finite integers")
    return numeric.astype(np.int64)


def load_episode_metrics(evaluation: EvaluationInput, split: str) -> pd.DataFrame:
    path = evaluation.directory / "metrics_per_episode.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{evaluation.label}: missing {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    frame = frame.loc[frame["split"].astype(str) == split].copy()
    if frame.empty:
        raise ValueError(f"{path}: split {split!r} has no episode metrics")
    frame["checkpoint_step"] = _validated_integer_column(frame, "checkpoint_step", path)
    frame["trajectory"] = _validated_integer_column(frame, "trajectory", path)
    frame["frames"] = _validated_integer_column(frame, "frames", path)
    if (frame["frames"] <= 0).any():
        raise ValueError(f"{path}: frames must be positive")

    duplicate = frame.duplicated(["checkpoint_step", "trajectory"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["checkpoint_step", "trajectory"]].drop_duplicates()
        raise ValueError(f"{path}: duplicate checkpoint/trajectory rows: {keys.to_dict('records')}")
    return frame


def choose_checkpoint(tables: dict[str, pd.DataFrame], requested_step: int | None = None) -> int:
    common_steps = set.intersection(
        *(set(frame["checkpoint_step"].astype(int)) for frame in tables.values())
    )
    if not common_steps:
        raise ValueError("The evaluations have no checkpoint step in common")
    if requested_step is not None:
        if requested_step not in common_steps:
            availability = {
                label: sorted(set(frame["checkpoint_step"].astype(int)))
                for label, frame in tables.items()
            }
            raise ValueError(
                f"Checkpoint {requested_step} is not present in every evaluation: {availability}"
            )
        return requested_step
    return max(common_steps)


def align_episodes(
    tables: dict[str, pd.DataFrame], checkpoint_step: int, metrics: tuple[str, ...]
) -> tuple[dict[str, pd.DataFrame], list[int], dict[str, list[int]]]:
    selected: dict[str, pd.DataFrame] = {}
    episode_sets: dict[str, set[int]] = {}
    for label, frame in tables.items():
        missing_metrics = set(metrics) - set(frame.columns)
        if missing_metrics:
            raise ValueError(f"{label}: missing metric columns {sorted(missing_metrics)}")
        step_frame = frame.loc[frame["checkpoint_step"] == checkpoint_step].copy()
        step_frame = step_frame.set_index("trajectory", verify_integrity=True).sort_index()
        for metric in metrics:
            values = pd.to_numeric(step_frame[metric], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{label}: metric {metric!r} contains non-finite values")
            step_frame[metric] = values
        selected[label] = step_frame
        episode_sets[label] = set(step_frame.index.astype(int))

    common_episodes = sorted(set.intersection(*episode_sets.values()))
    if not common_episodes:
        raise ValueError(f"Checkpoint {checkpoint_step} has no trajectories shared by every model")

    aligned = {label: frame.loc[common_episodes] for label, frame in selected.items()}
    reference_label = next(iter(aligned))
    reference_frames = aligned[reference_label]["frames"].to_numpy(dtype=np.int64)
    for label, frame in aligned.items():
        frames = frame["frames"].to_numpy(dtype=np.int64)
        if not np.array_equal(frames, reference_frames):
            mismatched = np.asarray(common_episodes)[frames != reference_frames].tolist()
            raise ValueError(
                f"{label} and {reference_label} evaluated different frame counts for "
                f"trajectories {mismatched}"
            )

    dropped = {label: sorted(episode_sets[label] - set(common_episodes)) for label in selected}
    return aligned, common_episodes, dropped


def _confidence_interval(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return float(lower), float(upper)


def compare_evaluations(
    evaluations: list[EvaluationInput],
    output_dir: Path,
    *,
    split: str = "validation",
    checkpoint_step: int | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> ComparisonResult:
    if len(evaluations) < 2:
        raise ValueError("At least two --evaluation inputs are required")
    labels = [evaluation.label for evaluation in evaluations]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Evaluation labels must be unique, got {labels}")
    if not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("Metrics must be a non-empty list without duplicates")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    tables = {
        evaluation.label: load_episode_metrics(evaluation, split) for evaluation in evaluations
    }
    selected_step = choose_checkpoint(tables, checkpoint_step)
    aligned, trajectories, dropped = align_episodes(tables, selected_step, metrics)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(trajectories),
        size=(resamples, len(trajectories)),
        dtype=np.int32,
    )
    model_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for metric in metrics:
        values = {
            label: frame[metric].to_numpy(dtype=np.float64) for label, frame in aligned.items()
        }
        for label, model_values in values.items():
            bootstrap_means = model_values[sample_indices].mean(axis=1)
            ci_lower, ci_upper = _confidence_interval(bootstrap_means, confidence_level)
            model_rows.append(
                {
                    "split": split,
                    "checkpoint_step": selected_step,
                    "model": label,
                    "metric": metric,
                    "episodes": len(trajectories),
                    "episode_mean": float(model_values.mean()),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "confidence_level": confidence_level,
                    "resamples": resamples,
                    "seed": seed,
                }
            )

        for model_a, model_b in itertools.combinations(labels, 2):
            difference = values[model_a] - values[model_b]
            bootstrap_differences = difference[sample_indices].mean(axis=1)
            ci_lower, ci_upper = _confidence_interval(bootstrap_differences, confidence_level)
            model_a_wins = int(np.count_nonzero(difference < 0.0))
            model_b_wins = int(np.count_nonzero(difference > 0.0))
            ties = int(np.count_nonzero(difference == 0.0))
            pairwise_rows.append(
                {
                    "split": split,
                    "checkpoint_step": selected_step,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "episodes": len(trajectories),
                    "mean_difference_a_minus_b": float(difference.mean()),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "confidence_level": confidence_level,
                    "ci_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
                    "model_a_episode_wins": model_a_wins,
                    "model_b_episode_wins": model_b_wins,
                    "episode_ties": ties,
                    "resamples": resamples,
                    "seed": seed,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cross_model_bootstrap_intervals.csv"
    pairwise_path = output_dir / "cross_model_pairwise_bootstrap.csv"
    metadata_path = output_dir / "cross_model_bootstrap_metadata.json"
    pd.DataFrame(model_rows).to_csv(model_path, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    metadata = {
        "split": split,
        "checkpoint_step": selected_step,
        "metrics": list(metrics),
        "aggregation": "unweighted mean of per-episode metrics",
        "bootstrap_unit": "episode",
        "paired": True,
        "episodes": len(trajectories),
        "trajectories": trajectories,
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "evaluations": [
            {
                "label": evaluation.label,
                "directory": str(evaluation.directory.resolve()),
                "dropped_non_common_trajectories": dropped[evaluation.label],
            }
            for evaluation in evaluations
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return ComparisonResult(
        checkpoint_step=selected_step,
        episode_count=len(trajectories),
        model_intervals_path=model_path,
        pairwise_intervals_path=pairwise_path,
        metadata_path=metadata_path,
    )


def main() -> int:
    args = parse_args()
    try:
        result = compare_evaluations(
            args.evaluation,
            args.output_dir,
            split=args.split,
            checkpoint_step=args.checkpoint_step,
            metrics=tuple(args.metrics or DEFAULT_METRICS),
            resamples=args.resamples,
            seed=args.seed,
            confidence_level=args.confidence_level,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Compared {len(args.evaluation)} models at {args.split} checkpoint "
        f"{result.checkpoint_step} across {result.episode_count} shared episodes."
    )
    print(f"Model intervals:    {result.model_intervals_path}")
    print(f"Pairwise intervals: {result.pairwise_intervals_path}")
    print(f"Metadata:           {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
