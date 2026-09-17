import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.analysis_tools.migrate_normalized_action_metrics import (
    ADDED_NORMALIZED_FILES,
    CHECKPOINT_COLUMNS,
    EPISODE_COLUMNS,
    JOINT_COLUMNS,
    OVERWRITTEN_AGGREGATE_FILES,
    commit_migration,
    prepare_migration,
)
from scripts.analysis_tools.normalized_action_metrics import (
    ActionErrorNormalizer,
    write_metric_tables,
)


def _statistics():
    return {
        "new_embodiment": {
            "action": {
                "left_hand": {"q01": [0.0], "q99": [1.0]},
                "right_hand": {"q01": [0.0], "q99": [2.0]},
            },
            "relative_action": {
                "left_arm": {
                    "min": [[-2.0], [-4.0]],
                    "max": [[2.0], [4.0]],
                    "q01": [[-1.0], [-2.0]],
                    "q99": [[1.0], [2.0]],
                },
                "right_arm": {
                    "min": [[-1.0], [-2.0]],
                    "max": [[1.0], [2.0]],
                    "q01": [[-0.5], [-1.0]],
                    "q99": [[0.5], [1.0]],
                },
            },
        }
    }


def _prediction_frame(step: int, *, trajectory_offset: int = 0) -> pd.DataFrame:
    labels = ["left_arm[0]", "right_arm[0]", "left_hand[0]", "right_hand[0]"]
    # Explicit float32 evidence exercises the legacy evaluator's dtype contract.
    errors = np.asarray(
        [
            [0.1, -0.2, 0.3, -0.4],
            [0.2, -0.1, 0.4, -0.3],
            [-0.5, 0.25, -0.125, 0.75],
            [-0.25, 0.5, -0.75, 0.125],
        ],
        dtype=np.float32,
    )
    trajectories = np.repeat([trajectory_offset, trajectory_offset + 1], 2 * len(labels))
    frames = np.tile(np.repeat([0, 1], len(labels)), 2)
    error = errors.reshape(-1)
    return pd.DataFrame(
        {
            "checkpoint_step": np.repeat(step, len(error)),
            "trajectory": trajectories,
            "frame": frames,
            "joint": np.tile(labels, 4),
            "ground_truth": np.zeros(len(error), dtype=np.float32),
            "prediction": error,
            "error": error,
            "absolute_error": np.abs(error),
            "measured_state": np.zeros(len(error), dtype=np.float32),
        }
    )


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int, int]]:
    return {
        path: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in paths
    }


def _normalized_source(tmp_path: Path) -> tuple[Path, list[Path]]:
    evaluation_dir = tmp_path / "model" / "evaluation_exec_hor_2"
    checkpoint_dir = evaluation_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)
    validation = checkpoint_dir / "validation_frame_predictions.csv.gz"
    train = checkpoint_dir / "train_probe_frame_predictions.csv.gz"
    _prediction_frame(100).to_csv(validation, index=False, compression="gzip")
    _prediction_frame(100, trajectory_offset=20).to_csv(train, index=False, compression="gzip")
    plot = evaluation_dir / "checkpoint_error_progress.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\nprotected plot")
    write_metric_tables(
        evaluation_dir=evaluation_dir,
        normalizer=ActionErrorNormalizer(_statistics(), execution_horizon=2),
    )
    return evaluation_dir, [validation, train, plot]


def test_prepare_is_read_only_and_commit_restores_legacy_source(tmp_path):
    evaluation_dir, protected = _normalized_source(tmp_path)
    protected_before = _snapshot(protected)
    normalized_before = {
        name: (evaluation_dir / name).read_bytes()
        for name in (*ADDED_NORMALIZED_FILES, *OVERWRITTEN_AGGREGATE_FILES)
    }
    output_dir = tmp_path / "separate_metrics_root" / "simple_rgb"

    manifest_path = prepare_migration(
        evaluation_dir=evaluation_dir,
        output_dir=output_dir,
    )

    assert _snapshot(protected) == protected_before
    assert all(
        (evaluation_dir / name).read_bytes() == contents
        for name, contents in normalized_before.items()
    )
    assert all(
        (output_dir / "new_metrics" / name).read_bytes() == contents
        for name, contents in normalized_before.items()
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "prepared"
    assert manifest["restoration_contract"]["semantic_restoration"] == "verified"

    commit_migration(evaluation_dir=evaluation_dir, prepared_dir=output_dir)

    assert _snapshot(protected) == protected_before
    assert not any((evaluation_dir / name).exists() for name in ADDED_NORMALIZED_FILES)
    assert list(pd.read_csv(evaluation_dir / "metrics_by_checkpoint.csv").columns) == list(
        CHECKPOINT_COLUMNS
    )
    assert list(pd.read_csv(evaluation_dir / "checkpoint_metric_summary.csv").columns) == list(
        CHECKPOINT_COLUMNS
    )
    assert list(pd.read_csv(evaluation_dir / "metrics_per_episode.csv").columns) == list(
        EPISODE_COLUMNS
    )
    assert list(pd.read_csv(evaluation_dir / "metrics_per_joint.csv").columns) == list(
        JOINT_COLUMNS
    )
    summary = pd.read_csv(evaluation_dir / "metrics_by_checkpoint.csv")
    assert summary["split"].tolist() == ["train_probe", "validation"]
    assert summary["episodes"].tolist() == [2, 2]
    assert summary["frames"].tolist() == [4, 4]
    expected = _prediction_frame(100)["error"].to_numpy(dtype=np.float32)
    assert summary.loc[summary["split"] == "validation", "mae"].iloc[0] == pytest.approx(
        float(np.mean(np.abs(expected)))
    )
    assert (output_dir / "migration_manifest.json").is_file()
    assert json.loads((output_dir / "migration_manifest.json").read_text())["status"] == "committed"


def test_commit_fails_closed_if_source_changed_after_prepare(tmp_path):
    evaluation_dir, protected = _normalized_source(tmp_path)
    output_dir = tmp_path / "separate_metrics_root" / "run"
    prepare_migration(evaluation_dir=evaluation_dir, output_dir=output_dir)
    aggregate_before = {
        name: (evaluation_dir / name).read_bytes() for name in OVERWRITTEN_AGGREGATE_FILES
    }
    changed = evaluation_dir / ADDED_NORMALIZED_FILES[0]
    changed.write_bytes(changed.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="content changed"):
        commit_migration(evaluation_dir=evaluation_dir, prepared_dir=output_dir)

    assert all(
        (evaluation_dir / name).read_bytes() == contents
        for name, contents in aggregate_before.items()
    )
    assert all(path.is_file() for path in protected)
    assert json.loads((output_dir / "migration_manifest.json").read_text())["status"] == "prepared"
