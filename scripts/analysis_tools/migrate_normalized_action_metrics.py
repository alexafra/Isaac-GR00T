#!/usr/bin/env python3
"""Move in-place normalized metrics aside and restore legacy raw-unit tables.

This utility has a deliberately two-phase contract:

``prepare``
    Reads immutable ``*_frame_predictions.csv.gz`` archives, copies every
    normalized result currently in the source evaluation directory into a
    path-disjoint destination, reconstructs the four pre-normalization raw-unit
    aggregate tables, and records hashes/mtimes.  It never changes the source.

``commit``
    Re-verifies the prepared snapshot, atomically installs the reconstructed
    legacy aggregate tables, removes only the seven explicitly known files
    added by ``normalized_action_metrics``, and proves that every source PNG and
    raw prediction archive retained both its hash and mtime.

No model is loaded and no plots or prediction archives are opened for writing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scripts.analysis_tools.normalized_action_metrics import PREDICTION_FILENAMES


ADDED_NORMALIZED_FILES = (
    "normalized_metrics_by_scope.csv",
    "normalized_metrics_per_episode.csv",
    "normalized_metrics_per_joint.csv",
    "unclipped_q01_q99_metrics_by_scope.csv",
    "unclipped_q01_q99_metrics_per_episode.csv",
    "unclipped_q01_q99_metrics_per_joint.csv",
    "normalized_metrics_provenance.json",
)
OVERWRITTEN_AGGREGATE_FILES = (
    "metrics_by_checkpoint.csv",
    "checkpoint_metric_summary.csv",
    "metrics_per_episode.csv",
    "metrics_per_joint.csv",
)
SOURCE_NORMALIZED_RESULT_FILES = (*ADDED_NORMALIZED_FILES, *OVERWRITTEN_AGGREGATE_FILES)

CHECKPOINT_COLUMNS = (
    "split",
    "checkpoint_step",
    "episodes",
    "frames",
    "mae",
    "mse",
    "median_absolute_error",
    "p95_absolute_error",
    "rmse",
    "bias",
    "max_absolute_error",
)
EPISODE_COLUMNS = (
    "split",
    "checkpoint_step",
    "trajectory",
    "frames",
    "mae",
    "mse",
    "rmse",
    "bias",
    "max_absolute_error",
)
JOINT_COLUMNS = (
    "split",
    "checkpoint_step",
    "joint",
    "mae",
    "mse",
    "rmse",
    "bias",
    "max_absolute_error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _records(paths: Iterable[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    return [_record(path, relative_to=relative_to) for path in sorted(paths)]


def _assert_record(root: Path, record: dict[str, Any], *, require_mtime: bool) -> Path:
    path = root / record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Prepared source file is now missing: {path}")
    actual_hash = _sha256(path)
    if actual_hash != record["sha256"] or path.stat().st_size != record["size"]:
        raise RuntimeError(f"Prepared source file content changed: {path}")
    if require_mtime and path.stat().st_mtime_ns != record["mtime_ns"]:
        raise RuntimeError(f"Prepared source file mtime changed: {path}")
    return path


def _assert_disjoint(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError(
            "Migration output must be path-disjoint from the source evaluation directory; "
            f"source={source}, output={destination}"
        )


def _prediction_paths(evaluation_dir: Path) -> list[tuple[str, int, Path]]:
    results: list[tuple[str, int, Path]] = []
    checkpoint_dirs = []
    for path in evaluation_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError as exc:
            raise RuntimeError(f"Invalid checkpoint directory: {path}") from exc
        checkpoint_dirs.append((step, path))
    for step, checkpoint_dir in sorted(checkpoint_dirs):
        for split, filename in PREDICTION_FILENAMES.items():
            path = checkpoint_dir / filename
            if path.is_file():
                results.append((split, step, path))
    if not results:
        raise FileNotFoundError(f"No saved frame predictions under {evaluation_dir}")
    return results


def _raw_metrics(error: np.ndarray, *, include_distribution: bool) -> dict[str, float]:
    # The prediction evidence was serialized from float32 action arrays.  The
    # decimal representation round-trips those values exactly, so retaining the
    # original dtype reproduces the evaluator's NumPy reductions rather than
    # silently promoting them to float64.
    values = np.asarray(error, dtype=np.float32)
    absolute = np.abs(values)
    mse = float(np.mean(values**2))
    result = {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(values)),
        "max_absolute_error": float(np.max(absolute)),
    }
    if include_distribution:
        result.update(
            {
                "median_absolute_error": float(np.median(absolute)),
                "p95_absolute_error": float(np.percentile(absolute, 95)),
            }
        )
    return result


def _load_prediction_matrix(
    path: Path, *, expected_step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "checkpoint_step",
            "trajectory",
            "frame",
            "joint",
            "error",
            "absolute_error",
        ],
        dtype={
            "checkpoint_step": np.int64,
            "trajectory": np.int64,
            "frame": np.int64,
            "joint": str,
            "error": np.float32,
            "absolute_error": np.float32,
        },
    )
    if frame.empty:
        raise ValueError(f"Prediction archive contains no rows: {path}")
    if set(frame["checkpoint_step"].unique()) != {expected_step}:
        raise ValueError(f"Prediction archive checkpoint mismatch: {path}")
    error = frame["error"].to_numpy(dtype=np.float32)
    absolute = frame["absolute_error"].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(error)) or not np.array_equal(absolute, np.abs(error)):
        raise ValueError(f"Prediction archive has invalid error evidence: {path}")

    first_trajectory = int(frame.iloc[0]["trajectory"])
    first_frame = int(frame.iloc[0]["frame"])
    first_block = frame[(frame["trajectory"] == first_trajectory) & (frame["frame"] == first_frame)]
    labels = first_block["joint"].astype(str).tolist()
    width = len(labels)
    if width == 0 or len(frame) % width:
        raise ValueError(f"Prediction archive is not a rectangular action table: {path}")
    row_count = len(frame)
    block_count = row_count // width
    joint_values = frame["joint"].astype(str).to_numpy()
    if not np.array_equal(joint_values, np.tile(np.asarray(labels), block_count)):
        raise ValueError(f"Prediction archive joint order changes between frames: {path}")

    trajectories = frame["trajectory"].to_numpy(dtype=np.int64).reshape(block_count, width)
    frames = frame["frame"].to_numpy(dtype=np.int64).reshape(block_count, width)
    if not np.all(trajectories == trajectories[:, :1]) or not np.all(frames == frames[:, :1]):
        raise ValueError(f"Prediction archive interleaves trajectory/frame rows: {path}")
    block_trajectories = trajectories[:, 0]
    block_frames = frames[:, 0]
    pairs = np.column_stack((block_trajectories, block_frames))
    if len(np.unique(pairs, axis=0)) != block_count:
        raise ValueError(f"Prediction archive contains duplicate trajectory/frame blocks: {path}")
    return (
        error.reshape(block_count, width),
        block_trajectories,
        block_frames,
        labels,
    )


def reconstruct_legacy_raw_tables(evaluation_dir: Path) -> dict[str, pd.DataFrame]:
    """Rebuild the evaluator's pre-normalization aggregate tables from raw evidence."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    checkpoint_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    canonical_labels: list[str] | None = None
    for split, step, path in _prediction_paths(evaluation_dir):
        errors, trajectories, _frames, labels = _load_prediction_matrix(path, expected_step=step)
        if canonical_labels is None:
            canonical_labels = labels
        elif labels != canonical_labels:
            raise ValueError(f"Action labels differ across prediction archives: {path}")

        trajectory_ids = list(dict.fromkeys(int(value) for value in trajectories))
        for trajectory in trajectory_ids:
            selected = errors[trajectories == trajectory]
            metrics = _raw_metrics(selected, include_distribution=False)
            episode_rows.append(
                {
                    "split": split,
                    "checkpoint_step": step,
                    "trajectory": trajectory,
                    "frames": len(selected),
                    **metrics,
                }
            )

        checkpoint_rows.append(
            {
                "split": split,
                "checkpoint_step": step,
                "episodes": len(trajectory_ids),
                "frames": len(errors),
                **_raw_metrics(errors, include_distribution=True),
            }
        )
        for index, label in enumerate(labels):
            joint_rows.append(
                {
                    "split": split,
                    "checkpoint_step": step,
                    "joint": label,
                    **_raw_metrics(errors[:, index], include_distribution=False),
                }
            )

    checkpoints = pd.DataFrame(checkpoint_rows).sort_values(
        ["split", "checkpoint_step"], ignore_index=True
    )
    episodes = pd.DataFrame(episode_rows)
    joints = pd.DataFrame(joint_rows)
    checkpoints = checkpoints.reindex(columns=CHECKPOINT_COLUMNS)
    episodes = episodes.reindex(columns=EPISODE_COLUMNS)
    joints = joints.reindex(columns=JOINT_COLUMNS)
    return {
        "metrics_by_checkpoint.csv": checkpoints,
        "checkpoint_metric_summary.csv": checkpoints.copy(),
        "metrics_per_episode.csv": episodes,
        "metrics_per_joint.csv": joints,
    }


def _validate_raw_projection(evaluation_dir: Path, reconstructed: dict[str, pd.DataFrame]) -> None:
    """Check reconstructed raw values against the raw columns saved in new tables."""

    column_sets = {
        "metrics_by_checkpoint.csv": CHECKPOINT_COLUMNS,
        "checkpoint_metric_summary.csv": CHECKPOINT_COLUMNS,
        "metrics_per_episode.csv": EPISODE_COLUMNS,
        "metrics_per_joint.csv": JOINT_COLUMNS,
    }
    key_columns = {
        "metrics_by_checkpoint.csv": ["split", "checkpoint_step"],
        "checkpoint_metric_summary.csv": ["split", "checkpoint_step"],
        "metrics_per_episode.csv": ["split", "checkpoint_step", "trajectory"],
        "metrics_per_joint.csv": ["split", "checkpoint_step", "joint"],
    }
    metric_columns = (
        "mae",
        "mse",
        "rmse",
        "median_absolute_error",
        "p95_absolute_error",
        "bias",
        "max_absolute_error",
    )
    for filename, expected in reconstructed.items():
        current = pd.read_csv(evaluation_dir / filename)
        keys = key_columns[filename]
        raw_mapping = {
            column: f"raw_{column}" for column in metric_columns if column in column_sets[filename]
        }
        required = {*keys, *raw_mapping.values()}
        missing = required - set(current.columns)
        if missing:
            raise ValueError(
                f"Current normalized table {filename} lacks raw restoration evidence: "
                f"{sorted(missing)}"
            )
        projected = current[[*keys, *raw_mapping.values()]].rename(
            columns={value: key for key, value in raw_mapping.items()}
        )
        merged = expected.merge(projected, on=keys, suffixes=("_reconstructed", "_saved"))
        if len(merged) != len(expected) or len(projected) != len(expected):
            raise ValueError(f"Raw restoration rows do not align for {filename}")
        for metric in raw_mapping:
            if not np.allclose(
                merged[f"{metric}_reconstructed"],
                merged[f"{metric}_saved"],
                rtol=2e-6,
                atol=2e-8,
            ):
                difference = np.max(
                    np.abs(merged[f"{metric}_reconstructed"] - merged[f"{metric}_saved"])
                )
                raise ValueError(
                    f"Reconstructed raw {metric} disagrees with saved evidence in "
                    f"{filename}; max_abs_difference={difference}"
                )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_migration(*, evaluation_dir: Path, output_dir: Path) -> Path:
    """Prepare a migration without changing a byte or mtime in the source tree."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _assert_disjoint(evaluation_dir, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Migration output already exists: {output_dir}")
    missing = [
        name for name in SOURCE_NORMALIZED_RESULT_FILES if not (evaluation_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Source is missing expected normalized result files: {missing}")

    protected_paths = [
        path
        for path in evaluation_dir.rglob("*")
        if path.is_file()
        and (path.suffix.lower() == ".png" or path.name in PREDICTION_FILENAMES.values())
    ]
    protected_before = _records(protected_paths, relative_to=evaluation_dir)
    normalized_before = _records(
        (evaluation_dir / name for name in SOURCE_NORMALIZED_RESULT_FILES),
        relative_to=evaluation_dir,
    )
    reconstructed = reconstruct_legacy_raw_tables(evaluation_dir)
    _validate_raw_projection(evaluation_dir, reconstructed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        new_metrics = staging / "new_metrics"
        legacy = staging / "legacy_raw_restoration"
        new_metrics.mkdir()
        legacy.mkdir()
        for name in SOURCE_NORMALIZED_RESULT_FILES:
            shutil.copy2(evaluation_dir / name, new_metrics / name)
        for name, frame in reconstructed.items():
            frame.to_csv(legacy / name, index=False)

        protected_after = _records(protected_paths, relative_to=evaluation_dir)
        if protected_after != protected_before:
            raise RuntimeError("Preparing migration changed protected PNG/prediction evidence")
        current_after = _records(
            (evaluation_dir / name for name in SOURCE_NORMALIZED_RESULT_FILES),
            relative_to=evaluation_dir,
        )
        if current_after != normalized_before:
            raise RuntimeError("Preparing migration changed current source result files")

        manifest = {
            "schema_version": 1,
            "status": "prepared",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_evaluation_dir": str(evaluation_dir),
            "output_dir": str(output_dir),
            "source_normalized_results": normalized_before,
            "protected_source_evidence": protected_before,
            "new_metrics_copies": _records(new_metrics.iterdir(), relative_to=staging),
            "legacy_raw_restoration": _records(legacy.iterdir(), relative_to=staging),
            "files_removed_from_source_on_commit": list(ADDED_NORMALIZED_FILES),
            "files_replaced_in_source_on_commit": list(OVERWRITTEN_AGGREGATE_FILES),
            "restoration_contract": {
                "source": "immutable frame-prediction error rows",
                "numeric_dtype": "float32, matching the serialized action arrays",
                "schemas": {
                    "checkpoint": list(CHECKPOINT_COLUMNS),
                    "episode": list(EPISODE_COLUMNS),
                    "joint": list(JOINT_COLUMNS),
                },
                "semantic_restoration": "verified",
                "byte_exact_original_restoration": (
                    "unverifiable because no pre-overwrite byte hashes or backup files exist; "
                    "the values, row order, schemas, and pandas CSV serialization are rebuilt "
                    "from the exact float32 evidence and original evaluator semantics"
                ),
            },
        }
        _write_json(staging / "migration_manifest.json", manifest)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "migration_manifest.json"


def _verify_prepared_tree(prepared_dir: Path, manifest: dict[str, Any]) -> None:
    for section in ("new_metrics_copies", "legacy_raw_restoration"):
        for record in manifest[section]:
            _assert_record(prepared_dir, record, require_mtime=False)


def _install_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.restore-{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def commit_migration(*, evaluation_dir: Path, prepared_dir: Path) -> Path:
    """Commit one already prepared and fully re-verified source restoration."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    prepared_dir = prepared_dir.expanduser().resolve()
    manifest_path = prepared_dir / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        raise RuntimeError(f"Migration is not in prepared state: {manifest_path}")
    if Path(manifest["source_evaluation_dir"]).resolve() != evaluation_dir:
        raise RuntimeError("Prepared migration belongs to a different source evaluation directory")
    _assert_disjoint(evaluation_dir, prepared_dir)
    _verify_prepared_tree(prepared_dir, manifest)
    for record in manifest["protected_source_evidence"]:
        _assert_record(evaluation_dir, record, require_mtime=True)
    for record in manifest["source_normalized_results"]:
        _assert_record(evaluation_dir, record, require_mtime=True)

    # Validate candidates semantically one final time before the first source write.
    reconstructed = reconstruct_legacy_raw_tables(evaluation_dir)
    for name, expected in reconstructed.items():
        candidate = pd.read_csv(prepared_dir / "legacy_raw_restoration" / name)
        pd.testing.assert_frame_equal(candidate, expected, check_dtype=False, rtol=2e-6, atol=2e-8)

    try:
        for name in OVERWRITTEN_AGGREGATE_FILES:
            _install_atomic(
                prepared_dir / "legacy_raw_restoration" / name,
                evaluation_dir / name,
            )
        for name in ADDED_NORMALIZED_FILES:
            (evaluation_dir / name).unlink()

        for name, expected in reconstructed.items():
            actual = pd.read_csv(evaluation_dir / name)
            if list(actual.columns) != list(expected.columns):
                raise RuntimeError(f"Restored legacy schema mismatch: {evaluation_dir / name}")
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=2e-6, atol=2e-8)
        for record in manifest["protected_source_evidence"]:
            _assert_record(evaluation_dir, record, require_mtime=True)
        remaining = [name for name in ADDED_NORMALIZED_FILES if (evaluation_dir / name).exists()]
        if remaining:
            raise RuntimeError(f"New normalized source artifacts remain after commit: {remaining}")
    except BaseException:
        # Restore the exact prepared in-place normalized state if any commit
        # step fails.  Protected evidence is never part of this rollback.
        for name in SOURCE_NORMALIZED_RESULT_FILES:
            _install_atomic(prepared_dir / "new_metrics" / name, evaluation_dir / name)
        raise

    manifest["status"] = "committed"
    manifest["committed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["restored_source_aggregates"] = _records(
        (evaluation_dir / name for name in OVERWRITTEN_AGGREGATE_FILES),
        relative_to=evaluation_dir,
    )
    manifest["protected_source_evidence_after_commit"] = _records(
        (evaluation_dir / record["path"] for record in manifest["protected_source_evidence"]),
        relative_to=evaluation_dir,
    )
    _write_json(manifest_path, manifest)
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Prepare and audit; do not mutate source")
    prepare.add_argument("--evaluation-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    commit = subparsers.add_parser("commit", help="Commit an audited prepared migration")
    commit.add_argument("--evaluation-dir", type=Path, required=True)
    commit.add_argument("--prepared-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_migration(
            evaluation_dir=args.evaluation_dir,
            output_dir=args.output_dir,
        )
        print(f"Prepared migration without changing source: {manifest}")
    else:
        manifest = commit_migration(
            evaluation_dir=args.evaluation_dir,
            prepared_dir=args.prepared_dir,
        )
        print(f"Committed migration: {manifest}")


if __name__ == "__main__":
    main()
