#!/usr/bin/env python3
"""Queue and collate normalized action metrics for the known Inspire runs.

This command is deliberately conservative.  It has an explicit run allowlist,
waits for the active three-model pipeline to finish successfully, validates all
saved frame-level evidence before changing derived artifacts, and processes one
model at a time with CUDA hidden.  It never evaluates a checkpoint.

The per-run work is delegated to ``normalized_action_metrics`` and to the
evaluator's ``--plots-only`` path.  The latter removes only its narrowly
allowlisted stale raw-error PNGs and recreates them using normalized errors.
Frame-prediction archives are hashed before and after every run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXECUTION_HORIZON = 8
DEFAULT_MODELS_ROOT = Path("/home/alex/Development/Models/inspire")
DEFAULT_DATASETS_ROOT = Path("/home/alex/Development/Datasets/lerobot2")
DEFAULT_OUTPUT_DIR = DEFAULT_MODELS_ROOT / "inspire_normalized_action_metrics_exec_hor_8"
DEFAULT_PIPELINE_UNIT = "groot-inspire-452-full-20260915.service"
DEFAULT_PIPELINE_LOG = Path(
    "/home/alex/Development/logs/groot/training/groot-inspire-all-tasks-452-20260915.log"
)
DEFAULT_PIPELINE_STATUS = Path(
    "/home/alex/Development/logs/groot/training/"
    "multi_finetune_evaluation_25k_all_tasks_452eps_20260915_evaluation_status.tsv"
)
PIPELINE_SUCCESS_MARKER = "All training and evaluation experiments finished successfully."
PREDICTION_FILENAMES = {
    "validation": "validation_frame_predictions.csv.gz",
    "train_probe": "train_probe_frame_predictions.csv.gz",
}
REQUIRED_PREDICTION_COLUMNS = {
    "checkpoint_step",
    "trajectory",
    "frame",
    "joint",
    "ground_truth",
    "prediction",
    "error",
    "absolute_error",
    "measured_state",
}
COLLATED_SCOPES = ("overall", "arms", "hands")
COLLATED_METRICS = ("mae", "mse", "raw_mae", "raw_mse")
CANONICAL_FORMULA_VERSION = "processor_training_coordinates_v1"
DIAGNOSTIC_FORMULA_VERSION = "unclipped_q01_q99_minus1_plus1_v1"
CANONICAL_METRIC_SOURCE_COLUMNS = {"mae": "mae", "mse": "mse"}
DIAGNOSTIC_METRIC_SOURCE_COLUMNS = {
    "mae": "diagnostic_unclipped_q01_q99_mae",
    "mse": "diagnostic_unclipped_q01_q99_mse",
}
ALL_MODEL_PLOT_WITH_CHECKPOINT_0 = (
    "inspire_processor_contract_all_7_models_validation_action_metrics_with_checkpoint_0.png"
)
ALL_MODEL_PLOT_WITHOUT_CHECKPOINT_0 = (
    "inspire_processor_contract_all_7_models_validation_action_metrics_without_checkpoint_0.png"
)
LEGACY_RGB_PLOT_WITH_CHECKPOINT_0 = (
    "inspire_processor_contract_legacy_red_cup_rgb_82eps_validation_action_metrics_"
    "with_checkpoint_0.png"
)
LEGACY_RGB_PLOT_WITHOUT_CHECKPOINT_0 = (
    "inspire_processor_contract_legacy_red_cup_rgb_82eps_validation_action_metrics_"
    "without_checkpoint_0.png"
)
OBSOLETE_GLOBAL_PLOT = "inspire_validation_normalized_action_metrics.png"
OBSOLETE_AMBIGUOUS_PLOTS = (
    OBSOLETE_GLOBAL_PLOT,
    "inspire_all_7_models_validation_normalized_action_metrics_with_checkpoint_0.png",
    "inspire_all_7_models_validation_normalized_action_metrics_without_checkpoint_0.png",
    "inspire_legacy_red_cup_rgb_82eps_validation_normalized_action_metrics_with_checkpoint_0.png",
    "inspire_legacy_red_cup_rgb_82eps_validation_normalized_action_metrics_"
    "without_checkpoint_0.png",
)
BEST_FLAG_FOR_METRIC = {
    "mae": "is_best_mae",
    "mse": "is_best_mse",
    "raw_mae": "is_best_raw_mae",
    "raw_mse": "is_best_raw_mse",
}
BEST_MODEL_COLUMNS = (
    "model",
    "checkpoint",
    "mse_total",
    "mse_hands",
    "mse_arms",
    "mae_total",
    "mae_hands",
    "mae_arms",
)


@dataclass(frozen=True)
class InspireRunSpec:
    """One explicitly approved model/evaluation and its current dataset location."""

    key: str
    model_relative: Path
    dataset_relative: Path
    checkpoint_steps: tuple[int, ...]
    has_train_probe: bool = True
    pipeline_model: bool = False


@dataclass(frozen=True)
class ComparisonPlotSpec:
    key: str
    filename: str
    run_keys: tuple[str, ...]
    include_checkpoint_zero: bool
    title_subject: str
    required_checkpoint_zero_run_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricFamilySpec:
    key: str
    formula_version: str
    source_columns: Mapping[str, str]
    plot_title: str
    axis_label: str
    filename_token: str
    floored_samples_column: str


FIVE_CHECKPOINTS = (5000, 10000, 15000, 20000, 25000)
SIX_RGB_CHECKPOINTS = (0, *FIVE_CHECKPOINTS)

# This is intentionally not filesystem discovery.  New or similarly named runs
# must be reviewed and added explicitly before the batch command will touch them.
INSPIRE_RUN_ALLOWLIST: tuple[InspireRunSpec, ...] = (
    InspireRunSpec(
        "all_tasks_rgb",
        Path("inspire_c_rgb_patch_tuned_bf16_batch_32_acc_1_25k_all_tasks_452eps_20260915"),
        Path("inspire/all_tasks_452eps_20260915"),
        SIX_RGB_CHECKPOINTS,
        pipeline_model=True,
    ),
    InspireRunSpec(
        "all_tasks_normals",
        Path(
            "inspire_c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_"
            "bf16_batch_32_acc_1_25k_all_tasks_452eps_20260915"
        ),
        Path("inspire/all_tasks_452eps_20260915"),
        FIVE_CHECKPOINTS,
        pipeline_model=True,
    ),
    InspireRunSpec(
        "all_tasks_depth",
        Path(
            "inspire_c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_"
            "bf16_batch_32_acc_1_25k_all_tasks_452eps_20260915"
        ),
        Path("inspire/all_tasks_452eps_20260915"),
        FIVE_CHECKPOINTS,
        pipeline_model=True,
    ),
    InspireRunSpec(
        "red_cup_rgb",
        Path(
            "inspire_c_rgb_patch_tuned_bf16_batch_32_acc_1_25k_"
            "inspire_pick_place_red_cup_08_13_20260913"
        ),
        Path("inspire/pick_place_red_cup_08_13"),
        SIX_RGB_CHECKPOINTS,
    ),
    InspireRunSpec(
        "red_cup_normals",
        Path(
            "inspire_c_normals_6ch_early_fusion_patch_tuned_"
            "normals_init_rgb_mean_bf16_batch_32_acc_1_25k_"
            "inspire_pick_place_red_cup_08_13_20260913"
        ),
        Path("inspire/pick_place_red_cup_08_13"),
        FIVE_CHECKPOINTS,
    ),
    InspireRunSpec(
        "red_cup_depth",
        Path(
            "inspire_c_d1_4ch_early_fusion_patch_tuned_"
            "depth_init_rgb_mean_bf16_batch_32_acc_1_25k_"
            "inspire_pick_place_red_cup_08_13_20260913"
        ),
        Path("inspire/pick_place_red_cup_08_13"),
        FIVE_CHECKPOINTS,
    ),
    InspireRunSpec(
        "legacy_red_cup_rgb_82eps",
        Path(
            "inspire_e2_ftp_colour_only_rgb_redcup_pick_put_82eps_"
            "bf16_b32a1_h32_20k_20260912T150904Z"
        ),
        Path("inspire/inspire_e2_ftp_colour_only_test_20260912"),
        (0, 15000, 20000),
        has_train_probe=False,
    ),
)

ALL_MODEL_COMPARISON_RUN_KEYS = tuple(spec.key for spec in INSPIRE_RUN_ALLOWLIST)
ALL_MODEL_COMPARISON_LABELS = {
    "all_tasks_rgb": "All-tasks RGB",
    "all_tasks_normals": "All-tasks surface normals",
    "all_tasks_depth": "All-tasks RGB-D",
    "red_cup_rgb": "Red-cup RGB",
    "red_cup_normals": "Red-cup surface normals",
    "red_cup_depth": "Red-cup RGB-D",
    "legacy_red_cup_rgb_82eps": "Legacy red-cup RGB (82 eps)",
}
LEGACY_RGB_RUN_KEY = "legacy_red_cup_rgb_82eps"
RGB_CHECKPOINT_ZERO_RUN_KEYS = ("all_tasks_rgb", "red_cup_rgb", LEGACY_RGB_RUN_KEY)
COMPARISON_PLOT_SPECS = (
    ComparisonPlotSpec(
        key="all_7_models_with_checkpoint_0",
        filename=ALL_MODEL_PLOT_WITH_CHECKPOINT_0,
        run_keys=ALL_MODEL_COMPARISON_RUN_KEYS,
        include_checkpoint_zero=True,
        title_subject="All seven Inspire models",
        required_checkpoint_zero_run_keys=RGB_CHECKPOINT_ZERO_RUN_KEYS,
    ),
    ComparisonPlotSpec(
        key="all_7_models_without_checkpoint_0",
        filename=ALL_MODEL_PLOT_WITHOUT_CHECKPOINT_0,
        run_keys=ALL_MODEL_COMPARISON_RUN_KEYS,
        include_checkpoint_zero=False,
        title_subject="All seven Inspire models",
    ),
    ComparisonPlotSpec(
        key="legacy_red_cup_rgb_82eps_with_checkpoint_0",
        filename=LEGACY_RGB_PLOT_WITH_CHECKPOINT_0,
        run_keys=(LEGACY_RGB_RUN_KEY,),
        include_checkpoint_zero=True,
        title_subject="Legacy red-cup RGB model (82 episodes)",
        required_checkpoint_zero_run_keys=(LEGACY_RGB_RUN_KEY,),
    ),
    ComparisonPlotSpec(
        key="legacy_red_cup_rgb_82eps_without_checkpoint_0",
        filename=LEGACY_RGB_PLOT_WITHOUT_CHECKPOINT_0,
        run_keys=(LEGACY_RGB_RUN_KEY,),
        include_checkpoint_zero=False,
        title_subject="Legacy red-cup RGB model (82 episodes)",
    ),
)
CANONICAL_METRIC_FAMILY = MetricFamilySpec(
    key="processor_contract",
    formula_version=CANONICAL_FORMULA_VERSION,
    source_columns=CANONICAL_METRIC_SOURCE_COLUMNS,
    plot_title=(
        "processor-contract validation error (prediction and target each clipped to [-1, +1])"
    ),
    axis_label="Processor-contract",
    filename_token="processor_contract",
    floored_samples_column="floored_samples",
)
DIAGNOSTIC_METRIC_FAMILY = MetricFamilySpec(
    key="q01_q99_diagnostic",
    formula_version=DIAGNOSTIC_FORMULA_VERSION,
    source_columns=DIAGNOSTIC_METRIC_SOURCE_COLUMNS,
    plot_title="q01-q99 diagnostic validation error (unclipped)",
    axis_label="q01-q99 diagnostic",
    filename_token="q01_q99_diagnostic",
    floored_samples_column="diagnostic_unclipped_q01_q99_floored_samples",
)
DIAGNOSTIC_COMPARISON_PLOT_SPECS = tuple(
    ComparisonPlotSpec(
        key=spec.key,
        filename=spec.filename.replace(
            CANONICAL_METRIC_FAMILY.filename_token,
            DIAGNOSTIC_METRIC_FAMILY.filename_token,
        ),
        run_keys=spec.run_keys,
        include_checkpoint_zero=spec.include_checkpoint_zero,
        title_subject=spec.title_subject,
        required_checkpoint_zero_run_keys=spec.required_checkpoint_zero_run_keys,
    )
    for spec in COMPARISON_PLOT_SPECS
)


@dataclass(frozen=True)
class ResolvedRun:
    spec: InspireRunSpec
    model_dir: Path
    dataset_dir: Path
    evaluation_dir: Path
    statistics_path: Path


@dataclass(frozen=True)
class EvidenceSnapshot:
    raw_sha256: Mapping[Path, str]
    png_paths: frozenset[Path]
    protected_plot_paths: frozenset[Path]


def _resolved_child(root: Path, relative: Path) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Allowlisted path escapes root {root}: {relative}") from exc
    return candidate


def resolve_runs(
    *,
    models_root: Path = DEFAULT_MODELS_ROOT,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    specs: Sequence[InspireRunSpec] = INSPIRE_RUN_ALLOWLIST,
) -> list[ResolvedRun]:
    resolved = []
    for spec in specs:
        model_dir = _resolved_child(models_root, spec.model_relative)
        dataset_dir = _resolved_child(datasets_root, spec.dataset_relative)
        statistics_candidates = (
            model_dir / "experiment_cfg" / "dataset_statistics.json",
            model_dir / "processor" / "statistics.json",
        )
        statistics_path = next(
            (candidate for candidate in statistics_candidates if candidate.is_file()),
            statistics_candidates[0],
        )
        resolved.append(
            ResolvedRun(
                spec=spec,
                model_dir=model_dir,
                dataset_dir=dataset_dir,
                evaluation_dir=model_dir / f"evaluation_exec_hor_{EXECUTION_HORIZON}",
                statistics_path=statistics_path,
            )
        )
    return resolved


def prediction_paths(run: ResolvedRun) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {split: [] for split in PREDICTION_FILENAMES}
    for split, filename in PREDICTION_FILENAMES.items():
        if split == "train_probe" and not run.spec.has_train_probe:
            continue
        result[split] = [
            run.evaluation_dir / f"checkpoint-{step}" / filename
            for step in run.spec.checkpoint_steps
        ]
    return result


def _prediction_steps_on_disk(evaluation_dir: Path, filename: str) -> set[int]:
    steps = set()
    for path in evaluation_dir.glob(f"checkpoint-*/{filename}"):
        try:
            steps.add(int(path.parent.name.removeprefix("checkpoint-")))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid checkpoint directory beside prediction file: {path}"
            ) from exc
    return steps


def _validate_prediction_archive(path: Path, expected_step: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty saved prediction evidence: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = REQUIRED_PREDICTION_COLUMNS - set(header)
            if missing:
                raise RuntimeError(
                    f"Prediction evidence {path} is missing columns: {sorted(missing)}"
                )
            first = next(reader, None)
            if first is None:
                raise RuntimeError(f"Prediction evidence contains no data rows: {path}")
            try:
                first_step = int(first["checkpoint_step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid checkpoint_step in prediction evidence: {path}"
                ) from exc
            if first_step != expected_step:
                raise RuntimeError(
                    f"Prediction evidence step mismatch for {path}: "
                    f"expected={expected_step}, actual={first_step}"
                )
            # Consume the stream so a truncated archive or gzip CRC failure is
            # found during the all-run preflight, before any plots are changed.
            for _ in handle:
                pass
    except (OSError, EOFError, UnicodeError) as exc:
        raise RuntimeError(f"Unreadable gzip prediction evidence: {path}") from exc


def _summary_steps(evaluation_dir: Path, split: str) -> set[int]:
    path = evaluation_dir / "checkpoint_metric_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint summary: {path}")
    summary = pd.read_csv(path)
    required = {"split", "checkpoint_step", "mae", "mse"}
    missing = required - set(summary.columns)
    if missing:
        raise RuntimeError(f"Checkpoint summary {path} is missing columns: {sorted(missing)}")
    rows = summary.loc[summary["split"].astype(str) == split, "checkpoint_step"]
    return {int(step) for step in pd.to_numeric(rows, errors="raise")}


def validate_run_evidence(run: ResolvedRun) -> None:
    """Fail on missing *or unexpected* checkpoint evidence for one allowlisted run."""

    if not run.model_dir.is_dir():
        raise FileNotFoundError(f"Missing allowlisted model directory: {run.model_dir}")
    if not run.evaluation_dir.is_dir():
        raise FileNotFoundError(f"Missing evaluation directory: {run.evaluation_dir}")
    if not run.statistics_path.is_file():
        raise FileNotFoundError(f"Missing training-only statistics: {run.statistics_path}")
    for split in ("validation", "train"):
        info = run.dataset_dir / split / "meta" / "info.json"
        if not info.is_file():
            raise FileNotFoundError(f"Missing remapped dataset metadata: {info}")

    expected = set(run.spec.checkpoint_steps)
    paths = prediction_paths(run)
    for split, filename in PREDICTION_FILENAMES.items():
        expected_for_split = (
            expected if split == "validation" or run.spec.has_train_probe else set()
        )
        steps_for_split = (
            run.spec.checkpoint_steps if split == "validation" or run.spec.has_train_probe else ()
        )
        actual = _prediction_steps_on_disk(run.evaluation_dir, filename)
        if actual != expected_for_split:
            raise RuntimeError(
                f"{run.spec.key} {split} checkpoint evidence differs from allowlist: "
                f"expected={sorted(expected_for_split)}, actual={sorted(actual)}"
            )
        for step, path in zip(steps_for_split, paths[split], strict=True):
            _validate_prediction_archive(path, step)

    if _summary_steps(run.evaluation_dir, "validation") != expected:
        raise RuntimeError(f"{run.spec.key} validation checkpoint summary is incomplete")
    expected_train = expected if run.spec.has_train_probe else set()
    if _summary_steps(run.evaluation_dir, "train_probe") != expected_train:
        raise RuntimeError(f"{run.spec.key} train-probe checkpoint summary is incomplete")
    for step in expected - {0}:
        if not (run.model_dir / f"checkpoint-{step}").is_dir():
            raise FileNotFoundError(f"Missing model checkpoint directory: checkpoint-{step}")


def _parse_systemctl_properties(output: str) -> dict[str, str]:
    properties = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key] = value
    return properties


def read_user_unit_status(
    unit: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    result = runner(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus,MainPID",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Could not inspect pipeline unit {unit}: {detail}")
    properties = _parse_systemctl_properties(result.stdout)
    if "ActiveState" not in properties:
        raise RuntimeError(f"Pipeline unit {unit} returned no ActiveState")
    return properties


def wait_for_pipeline_unit(
    unit: str,
    *,
    wait: bool,
    poll_seconds: float,
    timeout_seconds: float,
    status_reader: Callable[[str], Mapping[str, str]] = read_user_unit_status,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Mapping[str, str]:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("poll_seconds and timeout_seconds must be positive")
    started = monotonic()
    while True:
        status = status_reader(unit)
        active_state = status.get("ActiveState", "")
        if active_state == "inactive":
            if status.get("Result") != "success" or status.get("ExecMainStatus") != "0":
                raise RuntimeError(f"Pipeline unit did not finish successfully: {dict(status)}")
            return status
        if active_state == "failed":
            raise RuntimeError(f"Pipeline unit failed: {dict(status)}")
        if active_state not in {"active", "activating", "deactivating", "reloading"}:
            raise RuntimeError(f"Unexpected pipeline unit state: {dict(status)}")
        if not wait:
            raise RuntimeError(f"Pipeline unit is still {active_state}; refusing to postprocess")
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(f"Timed out after {elapsed:.1f}s waiting for pipeline unit {unit}")
        sleeper(min(poll_seconds, timeout_seconds - elapsed))


def _tail_contains(path: Path, marker: str, *, tail_bytes: int = 4 * 1024 * 1024) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes))
        tail = handle.read()
    return marker.encode("utf-8") in tail


def validate_pipeline_status_tsv(path: Path, pipeline_runs: Sequence[ResolvedRun]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pipeline status TSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {
        (stage, str(run.model_dir.resolve()))
        for run in pipeline_runs
        for stage in ("training", "evaluation")
    }
    actual = {(str(row.get("stage")), str(Path(row.get("model", "")).resolve())) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Pipeline status rows differ from the six expected entries: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    failures = [
        row for row in rows if row.get("status") != "PASS" or str(row.get("exit_code")) != "0"
    ]
    if failures:
        raise RuntimeError(f"Pipeline status contains non-PASS rows: {failures}")


def validate_pipeline_completion_evidence(
    *,
    pipeline_log: Path,
    pipeline_status: Path,
    runs: Sequence[ResolvedRun],
) -> None:
    pipeline_runs = [run for run in runs if run.spec.pipeline_model]
    if len(pipeline_runs) != 3:
        raise RuntimeError(f"Expected exactly three pipeline models, found {len(pipeline_runs)}")
    if not _tail_contains(pipeline_log, PIPELINE_SUCCESS_MARKER):
        raise RuntimeError(f"Pipeline log lacks final success marker: {pipeline_log}")
    validate_pipeline_status_tsv(pipeline_status, pipeline_runs)
    depth_runs = [run for run in pipeline_runs if run.spec.key == "all_tasks_depth"]
    if len(depth_runs) != 1:
        raise RuntimeError("Allowlist must contain exactly one current all-tasks depth run")
    validate_run_evidence(depth_runs[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_evidence(run: ResolvedRun) -> EvidenceSnapshot:
    raw_paths = [path for paths in prediction_paths(run).values() for path in paths]
    png_paths = frozenset(path.resolve() for path in run.evaluation_dir.rglob("*.png"))
    protected = frozenset(
        path
        for path in png_paths
        if path.name.endswith("_joints.png") or path.name.endswith("_joint_velocities.png")
    )
    return EvidenceSnapshot(
        raw_sha256={path.resolve(): _sha256(path) for path in raw_paths},
        png_paths=png_paths,
        protected_plot_paths=protected,
    )


def is_allowlisted_stale_plot(path: Path, evaluation_dir: Path) -> bool:
    path = path.resolve()
    evaluation_dir = evaluation_dir.resolve()
    try:
        relative = path.relative_to(evaluation_dir)
    except ValueError:
        return False
    if fnmatch.fnmatch(path.name, "trajectory_*_error_heatmap.png"):
        return True
    if relative.parent not in {Path("."), Path("right_arm_hand")}:
        return False
    return any(
        fnmatch.fnmatch(path.name, pattern)
        for pattern in (
            "checkpoint_error_progress*.png",
            "checkpoint_metric_summary*.png",
            "joint_error_by_checkpoint*.png",
            "best_checkpoint_joint_mae*.png",
            "error_by_horizon_position.png",
        )
    )


def verify_preserved_evidence(run: ResolvedRun, before: EvidenceSnapshot) -> None:
    after_raw = {path: _sha256(path) for path in before.raw_sha256}
    changed = [path for path, digest in before.raw_sha256.items() if after_raw[path] != digest]
    if changed:
        raise RuntimeError(f"Frame-prediction evidence changed unexpectedly: {changed}")
    missing_protected = [path for path in before.protected_plot_paths if not path.is_file()]
    if missing_protected:
        raise RuntimeError(f"Joint/velocity plots were removed unexpectedly: {missing_protected}")
    after_png = frozenset(path.resolve() for path in run.evaluation_dir.rglob("*.png"))
    removed = before.png_paths - after_png
    unexpected = [
        path for path in removed if not is_allowlisted_stale_plot(path, run.evaluation_dir)
    ]
    if unexpected:
        raise RuntimeError(f"Non-error plots were removed unexpectedly: {unexpected}")


def _cpu_only_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    return environment


def postprocess_run(
    run: ResolvedRun,
    *,
    python_executable: Path,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> pd.DataFrame:
    """Normalize and replot exactly one run, then return its small scope table."""

    before = snapshot_evidence(run)
    environment = _cpu_only_environment()
    normalized_command = [
        str(python_executable),
        "-m",
        "scripts.analysis_tools.normalized_action_metrics",
        "--run-dir",
        str(run.model_dir),
        "--evaluation-dir",
        str(run.evaluation_dir),
        "--statistics-path",
        str(run.statistics_path),
        "--embodiment-tag",
        "new_embodiment",
        "--execution-horizon",
        str(EXECUTION_HORIZON),
    ]
    runner(normalized_command, check=True, cwd=repo_root, env=environment)

    plot_command = [
        str(python_executable),
        "-m",
        "scripts.analysis_tools.evaluate_checkpoints",
        "--run-dir",
        str(run.model_dir),
        "--dataset-path",
        str(run.dataset_dir / "validation"),
        "--output-dir",
        str(run.evaluation_dir),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--execution-horizon",
        str(EXECUTION_HORIZON),
        "--plots-only",
    ]
    if run.spec.has_train_probe:
        plot_command.extend(["--train-dataset-path", str(run.dataset_dir / "train")])
    runner(plot_command, check=True, cwd=repo_root, env=environment)
    verify_preserved_evidence(run, before)

    scope_path = run.evaluation_dir / "normalized_metrics_by_scope.csv"
    if not scope_path.is_file():
        raise FileNotFoundError(f"Normalized scope table was not produced: {scope_path}")
    return pd.read_csv(scope_path)


def prepare_long_collation(
    run_tables: Sequence[tuple[ResolvedRun, pd.DataFrame]],
    *,
    metric_family: MetricFamilySpec = CANONICAL_METRIC_FAMILY,
) -> pd.DataFrame:
    frames = []
    for run_order, (run, table) in enumerate(run_tables):
        required = {
            "split",
            "checkpoint_step",
            "scope",
            "episodes",
            "frames",
            "dimensions",
            "samples",
            metric_family.floored_samples_column,
            "raw_mae",
            "raw_mse",
            *metric_family.source_columns.values(),
        }
        missing = required - set(table.columns)
        if missing:
            raise RuntimeError(f"{run.spec.key} scope table is missing columns: {sorted(missing)}")
        selected = table[
            (table["split"].astype(str) == "validation")
            & table["scope"].astype(str).isin(COLLATED_SCOPES)
        ].copy()
        for output_column, source_column in metric_family.source_columns.items():
            selected[output_column] = pd.to_numeric(selected[source_column], errors="raise")
        selected["floored_samples"] = pd.to_numeric(
            selected[metric_family.floored_samples_column], errors="raise"
        ).astype(int)
        actual_steps = {int(step) for step in selected["checkpoint_step"]}
        if actual_steps != set(run.spec.checkpoint_steps):
            raise RuntimeError(
                f"{run.spec.key} normalized scope steps differ from allowlist: "
                f"{sorted(actual_steps)}"
            )
        counts = selected.groupby("checkpoint_step")["scope"].nunique()
        if len(counts) != len(run.spec.checkpoint_steps) or not (counts == 3).all():
            raise RuntimeError(f"{run.spec.key} lacks overall/arms/hands rows per checkpoint")
        selected.insert(0, "run_order", run_order)
        selected.insert(1, "run_key", run.spec.key)
        selected.insert(2, "metric_family", metric_family.key)
        selected["model_name"] = run.model_dir.name
        selected["model_dir"] = str(run.model_dir)
        selected["dataset_dir"] = str(run.dataset_dir)
        frames.append(selected)

    result = pd.concat(frames, ignore_index=True)
    group_columns = ["run_key", "scope"]
    for metric, flag in BEST_FLAG_FOR_METRIC.items():
        values = pd.to_numeric(result[metric], errors="raise")
        minima = values.groupby([result[column] for column in group_columns]).transform("min")
        result[flag] = np.isclose(values, minima, rtol=0.0, atol=1e-15)
    scope_order = {scope: index for index, scope in enumerate(COLLATED_SCOPES)}
    result["_scope_order"] = result["scope"].map(scope_order)
    result = result.sort_values(["run_order", "checkpoint_step", "_scope_order"]).drop(
        columns="_scope_order"
    )
    preferred = [
        "run_order",
        "run_key",
        "metric_family",
        "model_name",
        "model_dir",
        "dataset_dir",
        "split",
        "checkpoint_step",
        "scope",
        "episodes",
        "frames",
        "dimensions",
        "samples",
        "floored_samples",
        "mae",
        "mse",
        "raw_mae",
        "raw_mse",
        *BEST_FLAG_FOR_METRIC.values(),
    ]
    return result.reindex(columns=preferred).reset_index(drop=True)


def prepare_wide_collation(long: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "run_order",
        "run_key",
        "metric_family",
        "model_name",
        "model_dir",
        "dataset_dir",
        "split",
        "checkpoint_step",
    ]
    values = [
        "episodes",
        "frames",
        "dimensions",
        "samples",
        "floored_samples",
        *COLLATED_METRICS,
        *BEST_FLAG_FOR_METRIC.values(),
    ]
    wide = long.pivot(index=metadata, columns="scope", values=values)
    wide.columns = [f"{scope}_{metric}" for metric, scope in wide.columns]
    wide = wide.reset_index().sort_values(["run_order", "checkpoint_step"])
    ordered = metadata + [
        f"{scope}_{metric}"
        for scope in COLLATED_SCOPES
        for metric in values
        if f"{scope}_{metric}" in wide.columns
    ]
    wide = wide.reindex(columns=ordered)
    for column in wide.columns:
        if column.endswith(tuple(BEST_FLAG_FOR_METRIC.values())):
            wide[column] = wide[column].astype(bool)
    return wide.reset_index(drop=True)


def prepare_best_model_summary(
    long: pd.DataFrame, *, selection_metric: str = "mse"
) -> pd.DataFrame:
    """Return one row per model at its best overall normalized-error checkpoint.

    Exact metric ties are resolved deterministically in favor of the lower checkpoint
    step.  Every reported scope and metric is then read from that same checkpoint.
    """

    if selection_metric not in {"mae", "mse"}:
        raise ValueError("selection_metric must be 'mae' or 'mse'")
    required = {"run_order", "run_key", "checkpoint_step", "scope", "mae", "mse"}
    missing = required - set(long.columns)
    if missing:
        raise RuntimeError(f"Long collation is missing best-model columns: {sorted(missing)}")

    rows = []
    for (run_order, run_key), run_frame in long.groupby(
        ["run_order", "run_key"], sort=False, dropna=False
    ):
        relevant = run_frame[run_frame["scope"].isin(COLLATED_SCOPES)].copy()
        relevant["checkpoint_step"] = pd.to_numeric(
            relevant["checkpoint_step"], errors="raise"
        ).astype(int)
        relevant["mae"] = pd.to_numeric(relevant["mae"], errors="raise")
        relevant["mse"] = pd.to_numeric(relevant["mse"], errors="raise")
        if not np.isfinite(relevant[["mae", "mse"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{run_key} contains non-finite normalized metrics")
        duplicate_scopes = relevant.duplicated(["checkpoint_step", "scope"], keep=False)
        if duplicate_scopes.any():
            raise RuntimeError(
                f"{run_key} has duplicate scope rows at a checkpoint; selection is ambiguous"
            )

        overall = relevant[relevant["scope"] == "overall"].sort_values(
            [selection_metric, "checkpoint_step"], kind="stable"
        )
        if overall.empty:
            raise RuntimeError(f"{run_key} has no overall validation metric")
        selected_step = int(overall.iloc[0]["checkpoint_step"])
        selected = relevant[relevant["checkpoint_step"] == selected_step].set_index("scope")
        missing_scopes = set(COLLATED_SCOPES) - set(selected.index)
        if missing_scopes:
            raise RuntimeError(
                f"{run_key} checkpoint {selected_step} lacks scopes: {sorted(missing_scopes)}"
            )
        rows.append(
            {
                "run_order": int(run_order),
                "model": str(run_key),
                "checkpoint": selected_step,
                "mse_total": float(selected.at["overall", "mse"]),
                "mse_hands": float(selected.at["hands", "mse"]),
                "mse_arms": float(selected.at["arms", "mse"]),
                "mae_total": float(selected.at["overall", "mae"]),
                "mae_hands": float(selected.at["hands", "mae"]),
                "mae_arms": float(selected.at["arms", "mae"]),
            }
        )

    result = pd.DataFrame(rows).sort_values("run_order").drop(columns="run_order")
    return result.reindex(columns=BEST_MODEL_COLUMNS).reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_comparison_frame(
    long: pd.DataFrame,
    *,
    run_keys: Sequence[str],
    include_checkpoint_zero: bool,
    required_checkpoint_zero_run_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """Select an explicit run subset and optionally remove checkpoint zero."""

    required = {"run_key", "checkpoint_step", "scope", "mae", "mse"}
    missing = required - set(long.columns)
    if missing:
        raise RuntimeError(f"Long collation is missing plot columns: {sorted(missing)}")
    run_keys = tuple(run_keys)
    if not run_keys or len(set(run_keys)) != len(run_keys):
        raise ValueError("run_keys must be a nonempty sequence of unique run keys")
    unknown = set(run_keys) - set(ALL_MODEL_COMPARISON_RUN_KEYS)
    if unknown:
        raise ValueError(f"Comparison includes non-allowlisted run keys: {sorted(unknown)}")
    required_zero = set(required_checkpoint_zero_run_keys)
    if not include_checkpoint_zero and required_zero:
        raise ValueError("Cannot require checkpoint zero when it is excluded")
    if not required_zero <= set(run_keys):
        raise ValueError("Required checkpoint-zero keys must be selected run keys")
    frame = long[
        long["run_key"].astype(str).isin(run_keys) & long["scope"].astype(str).isin(COLLATED_SCOPES)
    ].copy()
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="raise").astype(int)
    zero_runs = set(frame.loc[frame["checkpoint_step"] == 0, "run_key"].astype(str))
    missing_zero = required_zero - zero_runs
    if missing_zero:
        raise RuntimeError(
            f"Comparison lacks required checkpoint-0 rows for: {sorted(missing_zero)}"
        )
    if not include_checkpoint_zero:
        frame = frame[frame["checkpoint_step"] != 0].copy()
    actual_runs = set(frame["run_key"].astype(str))
    if actual_runs != set(run_keys):
        raise RuntimeError(
            "Comparison data differs from its selected run keys: "
            f"expected={sorted(run_keys)}, actual={sorted(actual_runs)}"
        )
    if frame.duplicated(["run_key", "checkpoint_step", "scope"]).any():
        raise RuntimeError("Comparison has duplicate run/checkpoint/scope rows")
    for metric in ("mae", "mse"):
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        values = frame[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"Comparison {metric} values must be finite and positive")
    scope_counts = frame.groupby(["run_key", "checkpoint_step"])["scope"].nunique()
    if scope_counts.empty or not (scope_counts == len(COLLATED_SCOPES)).all():
        raise RuntimeError("Comparison requires overall/arms/hands at every selected checkpoint")
    return frame.reset_index(drop=True)


def _shared_log_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Shared logarithmic limits require finite positive values")
    low = float(np.log10(values.min()))
    high = float(np.log10(values.max()))
    span = max(high - low, 0.2)
    padding = max(span * 0.06, 0.04)
    return 10 ** (low - padding), 10 ** (high + padding)


def line_values_with_checkpoint_gaps(
    checkpoint_steps: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    maximum_contiguous_gap: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so a line never implies checkpoints across a retention gap."""

    steps = np.asarray(checkpoint_steps, dtype=float)
    metric_values = np.asarray(values, dtype=float)
    if steps.ndim != 1 or metric_values.ndim != 1 or len(steps) != len(metric_values):
        raise ValueError("checkpoint_steps and values must be equally sized one-dimensional data")
    if maximum_contiguous_gap <= 0:
        raise ValueError("maximum_contiguous_gap must be positive")
    if len(steps) < 2:
        return steps, metric_values
    expanded_steps: list[float] = [float(steps[0])]
    expanded_values: list[float] = [float(metric_values[0])]
    for index in range(1, len(steps)):
        if steps[index] - steps[index - 1] > maximum_contiguous_gap:
            expanded_steps.append(float("nan"))
            expanded_values.append(float("nan"))
        expanded_steps.append(float(steps[index]))
        expanded_values.append(float(metric_values[index]))
    return np.asarray(expanded_steps), np.asarray(expanded_values)


def build_comparison_figure(
    long: pd.DataFrame,
    *,
    spec: ComparisonPlotSpec,
    metric_family: MetricFamilySpec = CANONICAL_METRIC_FAMILY,
) -> tuple[object, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plotted = prepare_comparison_frame(
        long,
        run_keys=spec.run_keys,
        include_checkpoint_zero=spec.include_checkpoint_zero,
        required_checkpoint_zero_run_keys=spec.required_checkpoint_zero_run_keys,
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey="row")
    for column, scope in enumerate(COLLATED_SCOPES):
        scoped = plotted[plotted["scope"] == scope]
        for run_key in spec.run_keys:
            run_frame = scoped[scoped["run_key"] == run_key]
            run_frame = run_frame.sort_values("checkpoint_step")
            checkpoint_steps = run_frame["checkpoint_step"].to_numpy(dtype=float)
            label = ALL_MODEL_COMPARISON_LABELS[run_key]
            for row_index, metric in ((0, "mae"), (1, "mse")):
                plot_steps, plot_values = line_values_with_checkpoint_gaps(
                    checkpoint_steps, run_frame[metric].to_numpy(dtype=float)
                )
                axes[row_index, column].plot(
                    plot_steps / 1000.0,
                    plot_values,
                    marker="o",
                    label=label,
                )
        axes[0, column].set_title(f"{scope}: {metric_family.axis_label} MAE")
        axes[1, column].set_title(f"{scope}: {metric_family.axis_label} MSE")
        for row in range(2):
            axes[row, column].set_xlabel("Checkpoint step (thousands)")
            axes[row, column].set_yscale("log")
            axes[row, column].grid(alpha=0.25)
    for row_index, metric in ((0, "mae"), (1, "mse")):
        limits = _shared_log_limits(plotted[metric].to_numpy(dtype=float))
        for column in range(3):
            axes[row_index, column].set_ylim(limits)
    axes[0, 0].set_ylabel(f"{metric_family.axis_label} MAE")
    axes[1, 0].set_ylabel(f"{metric_family.axis_label} MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(spec.run_keys)),
        fontsize=8.5,
    )
    checkpoint_scope = (
        "including available checkpoint 0 baselines"
        if spec.include_checkpoint_zero
        else "excluding checkpoint 0 baselines"
    )
    figure.suptitle(f"{spec.title_subject}: {metric_family.plot_title} ({checkpoint_scope})")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    return figure, axes


def write_comparison_plot(
    long: pd.DataFrame,
    path: Path,
    *,
    spec: ComparisonPlotSpec,
    metric_family: MetricFamilySpec = CANONICAL_METRIC_FAMILY,
) -> None:
    figure, _ = build_comparison_figure(long, spec=spec, metric_family=metric_family)
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=160)
        plt.close(figure)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(figure)


def _validate_png(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 8:
        raise RuntimeError(f"Comparison plot was not created: {path}")
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Comparison plot is not a valid PNG: {path}")


def write_comparison_plots(
    long: pd.DataFrame,
    output_dir: Path,
    *,
    metric_family: MetricFamilySpec = CANONICAL_METRIC_FAMILY,
    plot_specs: Sequence[ComparisonPlotSpec] = COMPARISON_PLOT_SPECS,
) -> dict[str, Path]:
    """Write one metric family's explicit comparison plot set."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {spec.key: output_dir / spec.filename for spec in plot_specs}
    for spec in plot_specs:
        write_comparison_plot(long, paths[spec.key], spec=spec, metric_family=metric_family)
        _validate_png(paths[spec.key])
    return paths


def write_metric_family_comparison_plots(
    canonical_long: pd.DataFrame,
    diagnostic_long: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Write canonical and diagnostic sets before removing any ambiguous predecessor."""

    paths = {
        CANONICAL_METRIC_FAMILY.key: write_comparison_plots(
            canonical_long,
            output_dir,
            metric_family=CANONICAL_METRIC_FAMILY,
            plot_specs=COMPARISON_PLOT_SPECS,
        ),
        DIAGNOSTIC_METRIC_FAMILY.key: write_comparison_plots(
            diagnostic_long,
            output_dir,
            metric_family=DIAGNOSTIC_METRIC_FAMILY,
            plot_specs=DIAGNOSTIC_COMPARISON_PLOT_SPECS,
        ),
    }
    for family_paths in paths.values():
        for path in family_paths.values():
            _validate_png(path)
    for filename in OBSOLETE_AMBIGUOUS_PLOTS:
        obsolete = output_dir / filename
        if obsolete.is_file():
            obsolete.unlink()
    return paths


def comparison_plot_provenance(
    paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, object]:
    return {
        "metric_families": {
            family.key: {
                "formula_version": family.formula_version,
                "plots": {
                    spec.key: {
                        "path": str(paths[family.key][spec.key]),
                        "run_keys": list(spec.run_keys),
                        "include_checkpoint_0": spec.include_checkpoint_zero,
                        "required_checkpoint_zero_run_keys": list(
                            spec.required_checkpoint_zero_run_keys
                        ),
                    }
                    for spec in plot_specs
                },
            }
            for family, plot_specs in (
                (CANONICAL_METRIC_FAMILY, COMPARISON_PLOT_SPECS),
                (DIAGNOSTIC_METRIC_FAMILY, DIAGNOSTIC_COMPARISON_PLOT_SPECS),
            )
        },
        "scopes": list(COLLATED_SCOPES),
        "axes": "logarithmic; shared y limits across all three scope panels within each metric row",
        "mae_and_mse_limits": "independent row-wise limits because MAE and MSE differ",
        "between_plot_limits": (
            "with-0 and without-0 figures autoscale independently so the latter shows the "
            "fine-grained trained-checkpoint band"
        ),
        "checkpoint_gap_rendering": (
            "lines break across adjacent saved checkpoints more than 5000 steps apart; "
            "markers remain at every observed checkpoint"
        ),
        "checkpoint_markers": "uniform circles for every observed checkpoint",
        "obsolete_ambiguous_plots_removed_after_replacements": list(OBSOLETE_AMBIGUOUS_PLOTS),
    }


def update_existing_comparison_artifacts(output_dir: Path) -> dict[str, dict[str, Path]]:
    """Regenerate only comparison plots/provenance from an existing long CSV."""

    output_dir = output_dir.expanduser().resolve()
    canonical_long_path = output_dir / "inspire_validation_action_metrics_long.csv"
    diagnostic_long_path = (
        output_dir / "inspire_q01_q99_diagnostic_validation_action_metrics_long.csv"
    )
    provenance_path = output_dir / "inspire_validation_action_metrics_provenance.json"
    if not canonical_long_path.is_file():
        raise FileNotFoundError(f"Missing canonical long comparison table: {canonical_long_path}")
    if not diagnostic_long_path.is_file():
        raise FileNotFoundError(
            f"Missing q01-q99 diagnostic long comparison table: {diagnostic_long_path}"
        )
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing comparison provenance: {provenance_path}")
    canonical_long = pd.read_csv(canonical_long_path)
    diagnostic_long = pd.read_csv(diagnostic_long_path)
    paths = write_metric_family_comparison_plots(canonical_long, diagnostic_long, output_dir)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise RuntimeError(f"Comparison provenance must be a JSON object: {provenance_path}")
    provenance["comparison_plots_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["comparison_plots"] = comparison_plot_provenance(paths)
    _atomic_json(provenance, provenance_path)
    return paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pipeline-unit", default=DEFAULT_PIPELINE_UNIT)
    parser.add_argument("--pipeline-log", type=Path, default=DEFAULT_PIPELINE_LOG)
    parser.add_argument("--pipeline-status", type=Path, default=DEFAULT_PIPELINE_STATUS)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument(
        "--skip-pipeline-gate",
        action="store_true",
        help="Testing/recovery only: skip unit/log/TSV gates; run evidence is still validated.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the completion gates and all evidence without writing anything.",
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    runs = resolve_runs(models_root=args.models_root, datasets_root=args.datasets_root)
    if not args.skip_pipeline_gate:
        wait_for_pipeline_unit(
            args.pipeline_unit,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        validate_pipeline_completion_evidence(
            pipeline_log=args.pipeline_log,
            pipeline_status=args.pipeline_status,
            runs=runs,
        )
    for run in runs:
        validate_run_evidence(run)
    print(f"Validated {len(runs)} allowlisted Inspire runs; no checkpoint inference will run.")
    if args.check_only:
        return

    run_tables = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] normalizing and replotting {run.spec.key}: {run.model_dir}")
        table = postprocess_run(
            run,
            python_executable=args.python_executable,
            repo_root=repo_root,
        )
        run_tables.append((run, table))

    canonical_long = prepare_long_collation(run_tables, metric_family=CANONICAL_METRIC_FAMILY)
    diagnostic_long = prepare_long_collation(run_tables, metric_family=DIAGNOSTIC_METRIC_FAMILY)
    canonical_wide = prepare_wide_collation(canonical_long)
    diagnostic_wide = prepare_wide_collation(diagnostic_long)
    best_models_mse = prepare_best_model_summary(canonical_long, selection_metric="mse")
    best_models_mae = prepare_best_model_summary(canonical_long, selection_metric="mae")
    diagnostic_best_mse = prepare_best_model_summary(diagnostic_long, selection_metric="mse")
    diagnostic_best_mae = prepare_best_model_summary(diagnostic_long, selection_metric="mae")
    best_tables = (
        best_models_mse,
        best_models_mae,
        diagnostic_best_mse,
        diagnostic_best_mae,
    )
    if any(len(table) != len(runs) for table in best_tables):
        raise RuntimeError(
            f"Expected one row per run in all four best-model summaries: "
            f"expected={len(runs)}, actual={[len(table) for table in best_tables]}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    long_path = output_dir / "inspire_validation_action_metrics_long.csv"
    wide_path = output_dir / "inspire_validation_action_metrics_wide.csv"
    canonical_long_path = (
        output_dir / "inspire_processor_contract_validation_action_metrics_long.csv"
    )
    canonical_wide_path = (
        output_dir / "inspire_processor_contract_validation_action_metrics_wide.csv"
    )
    diagnostic_long_path = (
        output_dir / "inspire_q01_q99_diagnostic_validation_action_metrics_long.csv"
    )
    diagnostic_wide_path = (
        output_dir / "inspire_q01_q99_diagnostic_validation_action_metrics_wide.csv"
    )
    best_models_mse_path = output_dir / "inspire_best_models_by_normalized_mse.csv"
    best_models_mae_path = output_dir / "inspire_best_models_by_normalized_mae.csv"
    diagnostic_best_mse_path = output_dir / "inspire_q01_q99_diagnostic_best_models_by_mse.csv"
    diagnostic_best_mae_path = output_dir / "inspire_q01_q99_diagnostic_best_models_by_mae.csv"
    provenance_path = output_dir / "inspire_validation_action_metrics_provenance.json"
    _atomic_csv(canonical_long, long_path)
    _atomic_csv(canonical_wide, wide_path)
    _atomic_csv(canonical_long, canonical_long_path)
    _atomic_csv(canonical_wide, canonical_wide_path)
    _atomic_csv(diagnostic_long, diagnostic_long_path)
    _atomic_csv(diagnostic_wide, diagnostic_wide_path)
    _atomic_csv(best_models_mse, best_models_mse_path)
    _atomic_csv(best_models_mae, best_models_mae_path)
    _atomic_csv(diagnostic_best_mse, diagnostic_best_mse_path)
    _atomic_csv(diagnostic_best_mae, diagnostic_best_mae_path)
    comparison_plot_paths = write_metric_family_comparison_plots(
        canonical_long, diagnostic_long, output_dir
    )
    _atomic_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_horizon": EXECUTION_HORIZON,
            "validation_only": True,
            "canonical_metric_family": {
                "key": CANONICAL_METRIC_FAMILY.key,
                "formula_version": CANONICAL_METRIC_FAMILY.formula_version,
                "arms": (
                    "relative action in raw min/max training coordinates; normalized prediction "
                    "and target each clipped to [-1,+1]"
                ),
                "hands": (
                    "absolute action in q01/q99 training coordinates; normalized prediction and "
                    "target each clipped to [-1,+1]"
                ),
                "error_range_after_operand_clipping": "[-2,+2]",
                "long": str(canonical_long_path),
                "wide": str(canonical_wide_path),
                "compatibility_long": str(long_path),
                "compatibility_wide": str(wide_path),
            },
            "diagnostic_metric_family": {
                "key": DIAGNOSTIC_METRIC_FAMILY.key,
                "formula_version": DIAGNOSTIC_METRIC_FAMILY.formula_version,
                "definition": "all dimensions q01/q99 mapped to [-1,+1], errors unclipped",
                "long": str(diagnostic_long_path),
                "wide": str(diagnostic_wide_path),
                "best_mse": str(diagnostic_best_mse_path),
                "best_mae": str(diagnostic_best_mae_path),
            },
            "normalized_headline_metrics": ["mae", "mse"],
            "physical_unit_counterparts": ["raw_mae", "raw_mse"],
            "best_flags": BEST_FLAG_FOR_METRIC,
            "best_model_summaries": {
                "normalized_mse": {
                    "path": str(best_models_mse_path),
                    "selection_metric": "validation overall processor-contract mse",
                },
                "normalized_mae": {
                    "path": str(best_models_mae_path),
                    "selection_metric": "validation overall processor-contract mae",
                },
                "tie_break": "lowest checkpoint step",
                "all_metrics_from_selected_checkpoint": True,
                "columns": list(BEST_MODEL_COLUMNS),
            },
            "comparison_plots": comparison_plot_provenance(comparison_plot_paths),
            "processing": "sequential, CPU-only, plots-only; no checkpoint inference",
            "runs": [
                {
                    "run_key": run.spec.key,
                    "model_dir": str(run.model_dir),
                    "dataset_dir": str(run.dataset_dir),
                    "evaluation_dir": str(run.evaluation_dir),
                    "statistics_path": str(run.statistics_path),
                    "checkpoint_steps": list(run.spec.checkpoint_steps),
                    "has_train_probe": run.spec.has_train_probe,
                }
                for run in runs
            ],
        },
        provenance_path,
    )
    print(f"canonical long: {canonical_long_path}")
    print(f"canonical wide: {canonical_wide_path}")
    print(f"q01-q99 diagnostic long: {diagnostic_long_path}")
    print(f"q01-q99 diagnostic wide: {diagnostic_wide_path}")
    print(f"best models by normalized MSE: {best_models_mse_path}")
    print(f"best models by normalized MAE: {best_models_mae_path}")
    for family_key, family_paths in comparison_plot_paths.items():
        for plot_key, path in family_paths.items():
            print(f"comparison plot {family_key}/{plot_key}: {path}")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()
