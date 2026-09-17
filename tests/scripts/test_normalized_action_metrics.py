import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scripts.analysis_tools.normalized_action_metrics import (
    DIAGNOSTIC_PREFIX,
    ActionErrorNormalizer,
    load_saved_predictions,
    summarize_by_scope,
    summarize_per_joint,
    write_metric_tables,
)


def _statistics():
    return {
        "new_embodiment": {
            "action": {
                "left_hand": {"q01": [0.0, 0.25], "q99": [1.0, 0.75]},
                "right_hand": {"q01": [0.0], "q99": [2.0]},
            },
            "relative_action": {
                "left_arm": {
                    "min": [[-2.0, -4.0], [-4.0, -8.0]],
                    "max": [[2.0, 4.0], [4.0, 8.0]],
                    "q01": [[-1.0, -2.0], [-2.0, -4.0]],
                    "q99": [[1.0, 2.0], [2.0, 4.0]],
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


def _raw_frame(error=None):
    labels = [
        "left_arm[0]",
        "left_arm[1]",
        "right_arm[0]",
        "left_hand[0]",
        "left_hand[1]",
        "right_hand[0]",
    ]
    if error is None:
        error = np.array(
            [
                [1.0, 2.0, 0.5, 0.5, 0.25, 1.0],
                [2.0, 4.0, 1.0, 0.5, 0.25, 1.0],
            ]
        )
    ground_truth = np.zeros_like(error)
    prediction = ground_truth + error
    return pd.DataFrame(
        {
            "checkpoint_step": np.repeat(100, error.size),
            "trajectory": np.repeat(3, error.size),
            "frame": np.repeat(np.arange(len(error)), len(labels)),
            "joint": np.tile(labels, len(error)),
            "ground_truth": ground_truth.reshape(-1),
            "prediction": prediction.reshape(-1),
            "error": error.reshape(-1),
            "absolute_error": np.abs(error).reshape(-1),
            "measured_state": np.zeros(error.size),
        }
    )


def test_canonical_normalization_matches_processor_bounds_and_preserves_old_diagnostic():
    normalizer = ActionErrorNormalizer(_statistics(), execution_horizon=2)

    normalized = normalizer.normalize_predictions(_raw_frame())
    matrix = normalized["normalized_error"].to_numpy().reshape(2, 6)

    # Arm min/max ranges are twice as wide as q01/q99, while hands still use
    # q01/q99. Canonical processor-coordinate errors therefore differ by body part.
    np.testing.assert_allclose(matrix[:, :3], 0.5)
    np.testing.assert_allclose(matrix[:, 3:], [[1.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(normalized[f"{DIAGNOSTIC_PREFIX}_error"], 1.0)
    assert set(normalized.loc[normalized["body_part"] == "arms", "normalization_source"]) == {
        "relative_action.min/max"
    }
    assert set(normalized.loc[normalized["body_part"] == "hands", "normalization_source"]) == {
        "action.q01/q99"
    }

    larger = _raw_frame()
    larger.loc[0, "prediction"] = 5.0
    larger.loc[0, "error"] = 5.0
    larger.loc[0, "absolute_error"] = 5.0
    normalized_larger = normalizer.normalize_predictions(larger)
    assert normalized_larger.loc[0, "normalized_error"] == 1.0
    assert normalized_larger.loc[0, f"{DIAGNOSTIC_PREFIX}_error"] == 5.0


def test_canonical_clips_ground_truth_and_prediction_before_calculating_error():
    raw = _raw_frame()
    row = raw.index[raw["joint"] == "left_hand[0]"][0]
    raw.loc[row, "ground_truth"] = 10.0
    raw.loc[row, "prediction"] = 11.0
    raw.loc[row, "error"] = 1.0
    raw.loc[row, "absolute_error"] = 1.0

    normalized = ActionErrorNormalizer(_statistics(), execution_horizon=2).normalize_predictions(
        raw
    )

    # Both operands saturate to +1, so the canonical error is zero. The old
    # unclipped q01/q99 diagnostic deliberately preserves the out-of-range error.
    assert normalized.loc[row, "normalized_ground_truth"] == 1.0
    assert normalized.loc[row, "normalized_prediction"] == 1.0
    assert normalized.loc[row, "normalized_error"] == 0.0
    assert normalized.loc[row, f"{DIAGNOSTIC_PREFIX}_error"] == 2.0


def test_relative_arm_anchor_is_measured_state_at_each_chunk_start():
    statistics = _statistics()
    raw = pd.DataFrame(
        {
            "checkpoint_step": [100] * 4,
            "trajectory": [3] * 4,
            "frame": [0, 1, 2, 3],
            "joint": ["left_arm[0]"] * 4,
            "ground_truth": [10.0, 11.0, 20.0, 22.0],
            "prediction": [11.0, 12.0, 21.0, 23.0],
            "error": [1.0] * 4,
            "absolute_error": [1.0] * 4,
            # The non-boundary values must never be used as chunk anchors.
            "measured_state": [10.0, 999.0, 20.0, 999.0],
        }
    )

    normalized = ActionErrorNormalizer(statistics, execution_horizon=2).normalize_predictions(raw)

    np.testing.assert_allclose(normalized["normalization_anchor"], [10.0, 10.0, 20.0, 20.0])
    # frame 3 GT/pred relative values are +2/+3 with [-4,+4] bounds.
    assert normalized.loc[3, "normalized_ground_truth"] == pytest.approx(0.5)
    assert normalized.loc[3, "normalized_prediction"] == pytest.approx(0.75)
    assert normalized.loc[3, "normalized_error"] == pytest.approx(0.25)


def test_explicit_unclipped_diagnostic_does_not_require_arm_anchor_evidence():
    raw = _raw_frame().drop(columns="measured_state")
    normalizer = ActionErrorNormalizer(_statistics(), execution_horizon=2)

    diagnostic = normalizer.normalize_unclipped_q01_q99_predictions(raw)

    np.testing.assert_allclose(diagnostic[f"{DIAGNOSTIC_PREFIX}_error"], 1.0)
    assert "normalized_error" not in diagnostic
    with pytest.raises(ValueError, match="measured_state"):
        normalizer.normalize_predictions(raw)


def test_scope_metrics_are_equal_per_dimension_and_retain_raw_metrics():
    raw = _raw_frame()
    raw["split"] = "validation"
    normalized = ActionErrorNormalizer(_statistics(), execution_horizon=2).normalize_predictions(
        raw
    )

    scopes = summarize_by_scope(normalized).set_index("scope")

    assert scopes.loc["overall", "dimensions"] == 6
    assert scopes.loc["arms", "dimensions"] == 3
    assert scopes.loc["hands", "dimensions"] == 3
    assert scopes.loc["left", "dimensions"] == 4
    assert scopes.loc["right", "dimensions"] == 2
    assert scopes.loc["overall", "mae"] == pytest.approx(7 / 12)
    assert scopes.loc["overall", "mse"] == pytest.approx(11 / 24)
    assert scopes.loc["overall", f"{DIAGNOSTIC_PREFIX}_mae"] == pytest.approx(1.0)
    assert scopes.loc["overall", f"{DIAGNOSTIC_PREFIX}_mse"] == pytest.approx(1.0)
    assert scopes.loc["overall", "raw_mae"] == pytest.approx(np.mean(np.abs(raw["error"])))
    assert scopes.loc["overall", "raw_mse"] == pytest.approx(np.mean(raw["error"] ** 2))

    joints = summarize_per_joint(normalized).set_index("joint")
    assert joints.loc["left_arm[0]", "normalization_range_min"] == pytest.approx(4.0)
    assert joints.loc["left_arm[0]", "normalization_range_max"] == pytest.approx(8.0)
    assert joints.loc["left_arm[0]", f"{DIAGNOSTIC_PREFIX}_range_min"] == pytest.approx(2.0)
    assert joints.loc["left_hand[0]", "normalization_range_min"] == pytest.approx(1.0)


def test_range_floor_is_recorded_but_negative_or_missing_statistics_fail_closed():
    statistics = _statistics()
    statistics["new_embodiment"]["action"]["left_hand"]["q99"][0] = 0.0
    normalizer = ActionErrorNormalizer(statistics, execution_horizon=2, range_floor=0.01)

    normalized = normalizer.normalize_predictions(_raw_frame())
    rows = normalized[normalized["joint"] == "left_hand[0]"]
    assert rows["normalization_range_floored"].all()
    assert (rows["normalization_effective_range"] == 0.01).all()
    assert (rows["normalized_ground_truth"] == 0.0).all()
    assert (rows["normalized_prediction"] == 0.0).all()

    statistics["new_embodiment"]["action"]["left_hand"]["q99"][0] = -0.1
    with pytest.raises(ValueError, match="upper bounds are below lower bounds"):
        ActionErrorNormalizer(statistics, execution_horizon=2).normalize_predictions(_raw_frame())

    del statistics["new_embodiment"]["relative_action"]["right_arm"]
    with pytest.raises(KeyError, match="relative_action.right_arm"):
        ActionErrorNormalizer(statistics, execution_horizon=2).normalize_predictions(_raw_frame())


def test_unequal_joint_sample_counts_fail_instead_of_silently_changing_weights():
    raw = _raw_frame()
    raw["split"] = "validation"
    normalized = ActionErrorNormalizer(_statistics(), execution_horizon=2).normalize_predictions(
        raw
    )

    with pytest.raises(ValueError, match="same sample count"):
        summarize_by_scope(normalized.iloc[:-1])


def test_standalone_writer_preserves_frame_predictions_byte_for_byte(tmp_path):
    run_dir = tmp_path / "run"
    evaluation_dir = run_dir / "evaluation_exec_hor_2"
    checkpoint_dir = evaluation_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)
    statistics_path = run_dir / "experiment_cfg" / "dataset_statistics.json"
    statistics_path.parent.mkdir(parents=True)
    statistics_path.write_text(json.dumps(_statistics()))

    raw = _raw_frame()
    raw_path = checkpoint_dir / "validation_frame_predictions.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    normalizer = ActionErrorNormalizer.from_file(statistics_path, execution_horizon=2)
    outputs = write_metric_tables(evaluation_dir=evaluation_dir, normalizer=normalizer)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before
    assert all(path.is_file() for path in outputs.values())
    loaded, paths = load_saved_predictions(evaluation_dir)
    assert paths == [raw_path]
    assert set(loaded["split"]) == {"validation"}
    scopes = pd.read_csv(outputs["scopes"])
    assert set(scopes["scope"]) == {
        "overall",
        "arms",
        "hands",
        "left",
        "right",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    provenance = json.loads(outputs["provenance"].read_text())
    assert provenance["input_prediction_files_rewritten"] is False
    assert provenance["clip_normalized_operands"] is True
    assert provenance["clip_normalized_error"] is False
    assert provenance["normalized_operand_range"] == [-1.0, 1.0]
    assert provenance["normalized_error_range"] == [-2.0, 2.0]
    assert "clip_normalized_errors" not in provenance
    assert provenance["formula_version"] == "processor_training_coordinates_v1"
    assert provenance["diagnostic_formula_version"] == "unclipped_q01_q99_minus1_plus1_v1"
    assert outputs["diagnostic_scopes"].is_file()

    canonical = pd.read_csv(outputs["metrics_by_checkpoint"])
    assert canonical.columns[:6].tolist() == [
        "split",
        "checkpoint_step",
        "episodes",
        "frames",
        "mae",
        "mse",
    ]
    assert "model_name" not in canonical
    assert "raw_mae" in canonical
    assert "raw_mse" in canonical


def test_isolated_output_preserves_every_source_file_hash_and_mtime(tmp_path):
    run_dir = tmp_path / "model"
    evaluation_dir = run_dir / "evaluation_exec_hor_2"
    checkpoint_dir = evaluation_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)
    statistics_path = run_dir / "experiment_cfg" / "dataset_statistics.json"
    statistics_path.parent.mkdir(parents=True)
    statistics_path.write_text(json.dumps(_statistics()))

    raw_path = checkpoint_dir / "validation_frame_predictions.csv.gz"
    _raw_frame().to_csv(raw_path, index=False, compression="gzip")
    # These names deliberately collide with the writer's in-place names.  An
    # isolated run must neither replace nor remove any of them (or any plot).
    for name, contents in {
        "metrics_by_checkpoint.csv": b"legacy checkpoint table\n",
        "checkpoint_metric_summary.csv": b"legacy summary table\n",
        "metrics_per_episode.csv": b"legacy episode table\n",
        "metrics_per_joint.csv": b"legacy joint table\n",
        "normalized_metrics_by_scope.csv": b"older derived table\n",
        "checkpoint_error_progress.png": b"\x89PNG\r\n\x1a\nlegacy",
    }.items():
        (evaluation_dir / name).write_bytes(contents)

    def snapshot(root):
        return {
            path.relative_to(root): (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = snapshot(evaluation_dir)
    isolated_dir = tmp_path / "dex3_comparison" / "simple_rgb"
    normalizer = ActionErrorNormalizer.from_file(statistics_path, execution_horizon=2)
    outputs = write_metric_tables(
        evaluation_dir=evaluation_dir,
        output_dir=isolated_dir,
        normalizer=normalizer,
        model_dir=run_dir,
    )

    assert snapshot(evaluation_dir) == before
    assert all(path.is_relative_to(isolated_dir) for path in outputs.values())
    assert all(path.is_file() for path in outputs.values())
    assert not list(isolated_dir.rglob("*.png"))
    provenance = json.loads(outputs["provenance"].read_text())
    assert provenance["source_evaluation_dir"] == str(evaluation_dir.resolve())
    assert provenance["output_dir"] == str(isolated_dir.resolve())
    assert provenance["isolated_output"] is True
    assert provenance["source_aggregate_csvs_rewritten"] is False
    assert provenance["source_pngs_rewritten"] is False
    assert set(provenance["metric_families"]) == {
        "processor_contract",
        "q01_q99_diagnostic",
        "raw_joint_units",
    }

    combined = pd.read_csv(outputs["scopes"])
    assert {"mae", "mse", "raw_mae", "raw_mse"}.issubset(combined.columns)
    assert {
        f"{DIAGNOSTIC_PREFIX}_mae",
        f"{DIAGNOSTIC_PREFIX}_mse",
    }.issubset(combined.columns)


@pytest.mark.parametrize("relative_output", [".", "child", ".."])
def test_explicit_output_rejects_non_disjoint_source_paths(tmp_path, relative_output):
    evaluation_dir = tmp_path / "evaluation_exec_hor_2"
    checkpoint_dir = evaluation_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)
    _raw_frame().to_csv(
        checkpoint_dir / "validation_frame_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    output_dir = (evaluation_dir / relative_output).resolve()

    with pytest.raises(ValueError, match="path-disjoint"):
        write_metric_tables(
            evaluation_dir=evaluation_dir,
            output_dir=output_dir,
            normalizer=ActionErrorNormalizer(_statistics(), execution_horizon=2),
        )


def test_dex3_28_dimensions_are_discovered_from_statistics_without_inspire_hardcoding():
    width = 7
    statistics = {
        "new_embodiment": {
            "relative_action": {
                group: {
                    "min": [[-1.0] * width, [-2.0] * width],
                    "max": [[1.0] * width, [2.0] * width],
                    "q01": [[-1.0] * width, [-2.0] * width],
                    "q99": [[1.0] * width, [2.0] * width],
                }
                for group in ("left_arm", "right_arm")
            },
            "action": {
                group: {"q01": [-1.0] * width, "q99": [1.0] * width}
                for group in ("left_hand", "right_hand")
            },
        }
    }
    labels = [
        f"{group}[{index}]"
        for group in ("left_arm", "right_arm", "left_hand", "right_hand")
        for index in range(width)
    ]
    # Half of every selected interval at both horizon positions -> normalized +1.
    errors = np.array(
        [
            [1.0] * 28,
            [2.0] * 14 + [1.0] * 14,
        ]
    )
    raw = pd.DataFrame(
        {
            "checkpoint_step": np.repeat(200, errors.size),
            "trajectory": np.repeat(0, errors.size),
            "frame": np.repeat([0, 1], 28),
            "joint": np.tile(labels, 2),
            "ground_truth": np.zeros(errors.size),
            "prediction": errors.reshape(-1),
            "error": errors.reshape(-1),
            "absolute_error": errors.reshape(-1),
            "measured_state": np.zeros(errors.size),
            "split": np.repeat("validation", errors.size),
        }
    )

    normalized = ActionErrorNormalizer(statistics, execution_horizon=2).normalize_predictions(raw)
    scopes = summarize_by_scope(normalized).set_index("scope")

    assert normalized["joint"].nunique() == 28
    np.testing.assert_allclose(normalized["normalized_error"], 1.0)
    assert scopes.loc["overall", "dimensions"] == 28
    assert scopes.loc["arms", "dimensions"] == 14
    assert scopes.loc["hands", "dimensions"] == 14
    assert scopes.loc["left", "dimensions"] == 14
    assert scopes.loc["right", "dimensions"] == 14
