from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.analysis_tools.collate_dex3_normalized_action_metrics import (
    DEX3_MODEL_SPECS,
    METRIC_FAMILIES,
    OUTPUT_COLUMNS,
    PROCESSOR_CONTRACT,
    Q01_Q99_DIAGNOSTIC,
    build_checkpoint_figure,
    generate_report,
    load_metric_families,
    snapshot_source_tree,
)


def _metric_value(
    *, model_index: int, checkpoint: int, scope: str, metric: str
) -> float:
    model_offset = 0.01 * model_index
    scope_offset = {"overall": 0.0, "hands": 0.02, "arms": 0.04}[scope]
    if checkpoint == 0:
        checkpoint_value = 0.9 if metric == "mse" else 0.8
    elif checkpoint == 100:
        checkpoint_value = 0.2 if metric == "mse" else 0.3
    elif checkpoint == 200:
        # MSE ties checkpoint 100, while MAE improves at checkpoint 200.
        checkpoint_value = 0.2 if metric == "mse" else 0.1
    else:  # pragma: no cover - fixture misuse guard
        raise AssertionError(checkpoint)
    return checkpoint_value + model_offset + scope_offset


def _write_isolated_inputs(input_root: Path) -> None:
    for model_index, model in enumerate(DEX3_MODEL_SPECS):
        model_dir = input_root / model.directory
        model_dir.mkdir(parents=True)
        checkpoints = [0, 100, 200] if model.key.endswith("_rgb") else [100, 200]
        canonical_rows = []
        diagnostic_rows = []
        for checkpoint in checkpoints:
            for scope in ("overall", "arms", "hands"):
                mae = _metric_value(
                    model_index=model_index,
                    checkpoint=checkpoint,
                    scope=scope,
                    metric="mae",
                )
                mse = _metric_value(
                    model_index=model_index,
                    checkpoint=checkpoint,
                    scope=scope,
                    metric="mse",
                )
                canonical_rows.append(
                    {
                        "split": "validation",
                        "checkpoint_step": checkpoint,
                        "scope": scope,
                        "mae": mae,
                        "mse": mse,
                    }
                )
                diagnostic_rows.append(
                    {
                        "split": "validation",
                        "checkpoint_step": checkpoint,
                        "scope": scope,
                        "diagnostic_unclipped_q01_q99_mae": mae * 10,
                        "diagnostic_unclipped_q01_q99_mse": mse * 10,
                    }
                )
        # Train rows prove that collation is validation-only.
        canonical_rows.append(
            {
                "split": "train_probe",
                "checkpoint_step": 999,
                "scope": "overall",
                "mae": 0.001,
                "mse": 0.001,
            }
        )
        pd.DataFrame(canonical_rows).to_csv(
            model_dir / PROCESSOR_CONTRACT.source_filename, index=False
        )
        pd.DataFrame(diagnostic_rows).to_csv(
            model_dir / Q01_Q99_DIAGNOSTIC.source_filename, index=False
        )
        (model_dir / "normalized_metrics_provenance.json").write_text(
            json.dumps(
                {
                    "formula_version": PROCESSOR_CONTRACT.formula_version,
                    "diagnostic_formula_version": Q01_Q99_DIAGNOSTIC.formula_version,
                    # This path must never be followed by the collator.
                    "input_prediction_files": ["/must/not/be/read/source-model.csv.gz"],
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture()
def isolated_inputs(tmp_path: Path) -> Path:
    input_root = tmp_path / "isolated" / "models"
    _write_isolated_inputs(input_root)
    return input_root


def test_report_has_fixed_schema_order_best_selection_and_preserves_sources(
    isolated_inputs: Path, tmp_path: Path
) -> None:
    before = snapshot_source_tree(isolated_inputs)
    output_root = tmp_path / "isolated" / "reports"

    assert generate_report(isolated_inputs, output_root) == output_root.resolve()

    after = snapshot_source_tree(isolated_inputs)
    assert after == before
    expected_order = [model.key for model in DEX3_MODEL_SPECS]
    for family in METRIC_FAMILIES:
        family_dir = output_root / family.key
        all_checkpoints = pd.read_csv(family_dir / "all_checkpoints.csv")
        best_mse = pd.read_csv(family_dir / "best_by_mse.csv")
        best_mae = pd.read_csv(family_dir / "best_by_mae.csv")
        for table in (all_checkpoints, best_mse, best_mae):
            assert tuple(table.columns) == OUTPUT_COLUMNS
        assert list(all_checkpoints["model"].drop_duplicates()) == expected_order
        assert list(best_mse["model"]) == expected_order
        assert list(best_mae["model"]) == expected_order
        assert set(best_mse["checkpoint"]) == {100}
        assert set(best_mae["checkpoint"]) == {200}

        # The hands/arms values must come from the same checkpoint selected by
        # the overall metric, not from independently best scope checkpoints.
        source_100 = all_checkpoints[
            (all_checkpoints["model"] == "simple_tasks_rgb")
            & (all_checkpoints["checkpoint"] == 100)
        ].iloc[0]
        selected_100 = best_mse[best_mse["model"] == "simple_tasks_rgb"].iloc[0]
        pd.testing.assert_series_equal(selected_100, source_100, check_names=False)

        assert (family_dir / "comparison_with_checkpoint_0.png").is_file()
        assert (family_dir / "comparison_without_checkpoint_0.png").is_file()
        for model in DEX3_MODEL_SPECS:
            assert (family_dir / "per_model" / f"{model.key}_checkpoints.png").is_file()

    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_tree"]["hashes_and_mtimes_unchanged"] is True
    assert provenance["source_tree"]["digest_before"] == provenance["source_tree"][
        "digest_after"
    ]
    assert provenance["output_columns"] == list(OUTPUT_COLUMNS)


def test_figures_use_circles_log_scale_and_shared_row_limits(
    isolated_inputs: Path,
) -> None:
    from matplotlib import pyplot as plt

    frame = load_metric_families(isolated_inputs)[PROCESSOR_CONTRACT.key]
    model_keys = tuple(model.key for model in DEX3_MODEL_SPECS)
    figure, axes = build_checkpoint_figure(
        frame,
        family=PROCESSOR_CONTRACT,
        model_keys=model_keys,
        include_checkpoint_zero=False,
        subject="All six Dex3 models",
    )
    try:
        assert "processor-contract" in figure._suptitle.get_text()
        for axis in axes.flat:
            assert axis.get_lines()
            assert {line.get_marker() for line in axis.get_lines()} == {"o"}
            assert all((line.get_xdata() > 0).all() for line in axis.get_lines())
        for row in range(2):
            assert {axes[row, column].get_yscale() for column in range(3)} == {"log"}
            first_limits = axes[row, 0].get_ylim()
            for column in range(1, 3):
                assert axes[row, column].get_ylim() == pytest.approx(first_limits)
    finally:
        plt.close(figure)


def test_rejects_output_inside_or_above_source_tree(
    isolated_inputs: Path,
) -> None:
    before = snapshot_source_tree(isolated_inputs)
    with pytest.raises(ValueError, match="path-disjoint"):
        generate_report(isolated_inputs, isolated_inputs / "reports")
    with pytest.raises(ValueError, match="path-disjoint"):
        generate_report(isolated_inputs, isolated_inputs.parent)
    assert snapshot_source_tree(isolated_inputs) == before
