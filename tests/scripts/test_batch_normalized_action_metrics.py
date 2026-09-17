from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
from scripts.analysis_tools.batch_normalized_action_metrics import (
    ALL_MODEL_COMPARISON_RUN_KEYS,
    ALL_MODEL_PLOT_WITH_CHECKPOINT_0,
    ALL_MODEL_PLOT_WITHOUT_CHECKPOINT_0,
    BEST_FLAG_FOR_METRIC,
    CANONICAL_METRIC_FAMILY,
    COLLATED_SCOPES,
    COMPARISON_PLOT_SPECS,
    DIAGNOSTIC_COMPARISON_PLOT_SPECS,
    DIAGNOSTIC_METRIC_FAMILY,
    EXECUTION_HORIZON,
    INSPIRE_RUN_ALLOWLIST,
    LEGACY_RGB_PLOT_WITH_CHECKPOINT_0,
    LEGACY_RGB_PLOT_WITHOUT_CHECKPOINT_0,
    OBSOLETE_AMBIGUOUS_PLOTS,
    PIPELINE_SUCCESS_MARKER,
    REQUIRED_PREDICTION_COLUMNS,
    EvidenceSnapshot,
    InspireRunSpec,
    ResolvedRun,
    build_comparison_figure,
    is_allowlisted_stale_plot,
    line_values_with_checkpoint_gaps,
    postprocess_run,
    prepare_best_model_summary,
    prepare_comparison_frame,
    prepare_long_collation,
    prepare_wide_collation,
    update_existing_comparison_artifacts,
    validate_pipeline_completion_evidence,
    validate_run_evidence,
    verify_preserved_evidence,
    wait_for_pipeline_unit,
    write_metric_family_comparison_plots,
)


def _run(tmp_path: Path, *, steps: tuple[int, ...] = (5000,), train: bool = True) -> ResolvedRun:
    model_dir = tmp_path / "Models" / "model"
    dataset_dir = tmp_path / "Datasets" / "inspire" / "dataset"
    evaluation_dir = model_dir / f"evaluation_exec_hor_{EXECUTION_HORIZON}"
    statistics_path = model_dir / "experiment_cfg" / "dataset_statistics.json"
    statistics_path.parent.mkdir(parents=True)
    statistics_path.write_text("{}\n", encoding="utf-8")
    for split in ("train", "validation"):
        metadata = dataset_dir / split / "meta" / "info.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("{}\n", encoding="utf-8")
    for step in steps:
        checkpoint_dir = evaluation_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True)
        if step:
            (model_dir / f"checkpoint-{step}").mkdir(exist_ok=True)
        filenames = ["validation_frame_predictions.csv.gz"]
        if train:
            filenames.append("train_probe_frame_predictions.csv.gz")
        for filename in filenames:
            with gzip.open(checkpoint_dir / filename, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_PREDICTION_COLUMNS))
                writer.writeheader()
                row = {column: 0 for column in REQUIRED_PREDICTION_COLUMNS}
                row["checkpoint_step"] = step
                writer.writerow(row)
    summary_rows = [
        {"split": split, "checkpoint_step": step, "mae": 0.1, "mse": 0.01}
        for step in steps
        for split in (("validation", "train_probe") if train else ("validation",))
    ]
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(evaluation_dir / "checkpoint_metric_summary.csv", index=False)
    return ResolvedRun(
        spec=InspireRunSpec(
            "test",
            Path("model"),
            Path("inspire/dataset"),
            steps,
            has_train_probe=train,
        ),
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        evaluation_dir=evaluation_dir,
        statistics_path=statistics_path,
    )


def _scope_table(steps: tuple[int, ...], values: dict[tuple[int, str], float]) -> pd.DataFrame:
    rows = []
    for step in steps:
        for scope in COLLATED_SCOPES:
            mse = values[(step, scope)]
            rows.append(
                {
                    "split": "validation",
                    "checkpoint_step": step,
                    "scope": scope,
                    "episodes": 2,
                    "frames": 10,
                    "dimensions": 26 if scope == "overall" else 13,
                    "samples": 260 if scope == "overall" else 130,
                    "floored_samples": 0,
                    "mae": mse + 0.1,
                    "mse": mse,
                    "raw_mae": mse + 0.2,
                    "raw_mse": mse + 0.3,
                    "diagnostic_unclipped_q01_q99_mae": mse + 1.1,
                    "diagnostic_unclipped_q01_q99_mse": mse + 1.0,
                    "diagnostic_unclipped_q01_q99_floored_samples": 3,
                }
            )
    return pd.DataFrame(rows)


def _comparison_long(tmp_path: Path) -> pd.DataFrame:
    run_steps = {
        "all_tasks_rgb": (0, 5000),
        "all_tasks_normals": (5000, 10000),
        "all_tasks_depth": (5000, 10000),
        "red_cup_rgb": (0, 5000),
        "red_cup_normals": (5000, 10000),
        "red_cup_depth": (5000, 10000),
        "legacy_red_cup_rgb_82eps": (0, 15000, 20000),
    }
    run_tables = []
    for run_index, (key, steps) in enumerate(run_steps.items(), start=1):
        base = _run(tmp_path / key, steps=steps)
        run = ResolvedRun(
            spec=InspireRunSpec(
                key,
                base.spec.model_relative,
                base.spec.dataset_relative,
                steps,
                has_train_probe=base.spec.has_train_probe,
            ),
            model_dir=base.model_dir,
            dataset_dir=base.dataset_dir,
            evaluation_dir=base.evaluation_dir,
            statistics_path=base.statistics_path,
        )
        values = {
            (step, scope): (
                10.0 if step == 0 else 0.01 * run_index + 0.001 * scope_index + step / 1_000_000
            )
            for step in steps
            for scope_index, scope in enumerate(COLLATED_SCOPES)
        }
        run_tables.append((run, _scope_table(steps, values)))
    return prepare_long_collation(run_tables)


def test_allowlist_is_exactly_seven_with_current_dataset_remaps() -> None:
    assert [spec.key for spec in INSPIRE_RUN_ALLOWLIST] == [
        "all_tasks_rgb",
        "all_tasks_normals",
        "all_tasks_depth",
        "red_cup_rgb",
        "red_cup_normals",
        "red_cup_depth",
        "legacy_red_cup_rgb_82eps",
    ]
    assert {spec.dataset_relative for spec in INSPIRE_RUN_ALLOWLIST} == {
        Path("inspire/all_tasks_452eps_20260915"),
        Path("inspire/pick_place_red_cup_08_13"),
        Path("inspire/inspire_e2_ftp_colour_only_test_20260912"),
    }
    assert sum(spec.pipeline_model for spec in INSPIRE_RUN_ALLOWLIST) == 3
    legacy = next(spec for spec in INSPIRE_RUN_ALLOWLIST if spec.key == "legacy_red_cup_rgb_82eps")
    assert legacy.checkpoint_steps == (0, 15000, 20000)


def test_wait_for_pipeline_requires_inactive_success() -> None:
    statuses = iter(
        [
            {"ActiveState": "active", "Result": "success", "ExecMainStatus": "0"},
            {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"},
        ]
    )
    ticks = iter([0.0, 1.0])
    slept = []

    result = wait_for_pipeline_unit(
        "pipeline.service",
        wait=True,
        poll_seconds=3,
        timeout_seconds=10,
        status_reader=lambda _unit: next(statuses),
        sleeper=slept.append,
        monotonic=lambda: next(ticks),
    )

    assert result["ActiveState"] == "inactive"
    assert slept == [3]


@pytest.mark.parametrize(
    "status",
    [
        {"ActiveState": "inactive", "Result": "exit-code", "ExecMainStatus": "1"},
        {"ActiveState": "failed", "Result": "exit-code", "ExecMainStatus": "1"},
    ],
)
def test_wait_for_pipeline_rejects_failure(status: dict[str, str]) -> None:
    with pytest.raises(RuntimeError):
        wait_for_pipeline_unit(
            "pipeline.service",
            wait=True,
            poll_seconds=1,
            timeout_seconds=10,
            status_reader=lambda _unit: status,
        )


def test_run_evidence_requires_exact_prediction_steps(tmp_path: Path) -> None:
    run = _run(tmp_path, steps=(5000, 10000))
    validate_run_evidence(run)

    extra = run.evaluation_dir / "checkpoint-15000" / "validation_frame_predictions.csv.gz"
    extra.parent.mkdir()
    with gzip.open(extra, "wt", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(sorted(REQUIRED_PREDICTION_COLUMNS))
    with pytest.raises(RuntimeError, match="differs from allowlist"):
        validate_run_evidence(run)


def test_run_evidence_accepts_validation_only_legacy_run(tmp_path: Path) -> None:
    validate_run_evidence(_run(tmp_path, steps=(15000, 20000), train=False))


def test_completion_gate_requires_log_six_pass_rows_and_depth_evidence(tmp_path: Path) -> None:
    runs = []
    for key in ("all_tasks_rgb", "all_tasks_normals", "all_tasks_depth"):
        run = _run(tmp_path / key)
        run = ResolvedRun(
            spec=InspireRunSpec(
                key,
                run.spec.model_relative,
                run.spec.dataset_relative,
                run.spec.checkpoint_steps,
                pipeline_model=True,
            ),
            model_dir=run.model_dir,
            dataset_dir=run.dataset_dir,
            evaluation_dir=run.evaluation_dir,
            statistics_path=run.statistics_path,
        )
        runs.append(run)
    log = tmp_path / "pipeline.log"
    log.write_text(f"anything\n{PIPELINE_SUCCESS_MARKER}\n", encoding="utf-8")
    status = tmp_path / "status.tsv"
    with status.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stage", "model", "status", "exit_code"], delimiter="\t"
        )
        writer.writeheader()
        for run in runs:
            for stage in ("training", "evaluation"):
                writer.writerow(
                    {"stage": stage, "model": run.model_dir, "status": "PASS", "exit_code": 0}
                )

    validate_pipeline_completion_evidence(pipeline_log=log, pipeline_status=status, runs=runs)

    rows = list(csv.DictReader(status.open(encoding="utf-8"), delimiter="\t"))
    rows[-1]["status"] = "FAIL"
    with status.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError):
        validate_pipeline_completion_evidence(pipeline_log=log, pipeline_status=status, runs=runs)


def test_stale_plot_allowlist_is_narrow_and_protects_joint_velocity_plots(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    assert is_allowlisted_stale_plot(evaluation / "checkpoint_error_progress.png", evaluation)
    assert is_allowlisted_stale_plot(
        evaluation / "checkpoint-5000" / "trajectory_0_error_heatmap.png", evaluation
    )
    assert not is_allowlisted_stale_plot(
        evaluation / "checkpoint-5000" / "trajectory_0_joints.png", evaluation
    )
    assert not is_allowlisted_stale_plot(
        evaluation / "checkpoint-5000" / "trajectory_0_joint_velocities.png", evaluation
    )
    assert not is_allowlisted_stale_plot(
        evaluation / "arbitrary" / "checkpoint_metric_summary.png", evaluation
    )


def test_verify_preserved_evidence_rejects_raw_archive_change(tmp_path: Path) -> None:
    run = _run(tmp_path)
    raw = run.evaluation_dir / "checkpoint-5000" / "validation_frame_predictions.csv.gz"
    before = EvidenceSnapshot(
        raw_sha256={raw.resolve(): "not-the-real-digest"},
        png_paths=frozenset(),
        protected_plot_paths=frozenset(),
    )
    with pytest.raises(RuntimeError, match="evidence changed"):
        verify_preserved_evidence(run, before)


def test_long_and_wide_collation_mark_per_run_scope_bests(tmp_path: Path) -> None:
    run = _run(tmp_path, steps=(5000, 10000))
    values = {
        (5000, "overall"): 0.3,
        (10000, "overall"): 0.2,
        (5000, "arms"): 0.1,
        (10000, "arms"): 0.4,
        (5000, "hands"): 0.5,
        (10000, "hands"): 0.2,
    }
    scope_table = _scope_table(run.spec.checkpoint_steps, values)
    long = prepare_long_collation([(run, scope_table)])
    assert len(long) == 6
    for scope, expected_step in {"overall": 10000, "arms": 5000, "hands": 10000}.items():
        row = long[(long["scope"] == scope) & long["is_best_mse"]].iloc[0]
        assert row["checkpoint_step"] == expected_step
        assert all(long[long["scope"] == scope][list(BEST_FLAG_FOR_METRIC.values())].sum() == 1)

    wide = prepare_wide_collation(long)
    assert len(wide) == 2
    assert bool(wide.loc[wide["checkpoint_step"] == 10000, "overall_is_best_mse"].iloc[0])
    assert {"overall_mse", "arms_raw_mae", "hands_is_best_raw_mse"} <= set(wide.columns)

    diagnostic = prepare_long_collation(
        [(run, scope_table)], metric_family=DIAGNOSTIC_METRIC_FAMILY
    )
    assert set(long["metric_family"]) == {CANONICAL_METRIC_FAMILY.key}
    assert set(diagnostic["metric_family"]) == {DIAGNOSTIC_METRIC_FAMILY.key}
    canonical_row = long[(long["checkpoint_step"] == 5000) & (long["scope"] == "overall")].iloc[0]
    diagnostic_row = diagnostic[
        (diagnostic["checkpoint_step"] == 5000) & (diagnostic["scope"] == "overall")
    ].iloc[0]
    assert canonical_row["mse"] == pytest.approx(0.3)
    assert canonical_row["floored_samples"] == 0
    assert diagnostic_row["mse"] == pytest.approx(1.3)
    assert diagnostic_row["mae"] == pytest.approx(1.4)
    assert diagnostic_row["floored_samples"] == 3


def test_best_model_summary_breaks_mse_tie_by_lower_step_and_keeps_one_checkpoint(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, steps=(5000, 10000))
    values = {
        (5000, "overall"): 0.2,
        (10000, "overall"): 0.2,
        (5000, "arms"): 0.4,
        (10000, "arms"): 0.1,
        (5000, "hands"): 0.5,
        (10000, "hands"): 0.05,
    }
    long = prepare_long_collation([(run, _scope_table(run.spec.checkpoint_steps, values))])

    best = prepare_best_model_summary(long)

    assert list(best.columns) == [
        "model",
        "checkpoint",
        "mse_total",
        "mse_hands",
        "mse_arms",
        "mae_total",
        "mae_hands",
        "mae_arms",
    ]
    assert len(best) == 1
    assert best.loc[0, "model"] == "test"
    assert best.loc[0, "checkpoint"] == 5000
    assert best.loc[0, "mse_total"] == pytest.approx(0.2)
    assert best.loc[0, "mse_hands"] == pytest.approx(0.5)
    assert best.loc[0, "mse_arms"] == pytest.approx(0.4)
    assert best.loc[0, "mae_total"] == pytest.approx(0.3)
    assert best.loc[0, "mae_hands"] == pytest.approx(0.6)
    assert best.loc[0, "mae_arms"] == pytest.approx(0.5)

    long.loc[(long["scope"] == "overall") & (long["checkpoint_step"] == 5000), "mae"] = 0.4
    long.loc[(long["scope"] == "overall") & (long["checkpoint_step"] == 10000), "mae"] = 0.1
    best_mae = prepare_best_model_summary(long, selection_metric="mae")
    assert list(best_mae.columns) == list(best.columns)
    assert best_mae.loc[0, "checkpoint"] == 10000
    assert best_mae.loc[0, "mse_total"] == pytest.approx(0.2)
    assert best_mae.loc[0, "mse_hands"] == pytest.approx(0.05)
    assert best_mae.loc[0, "mse_arms"] == pytest.approx(0.1)
    assert best_mae.loc[0, "mae_total"] == pytest.approx(0.1)
    assert best_mae.loc[0, "mae_hands"] == pytest.approx(0.15)
    assert best_mae.loc[0, "mae_arms"] == pytest.approx(0.2)


def test_comparison_figures_use_selected_runs_and_shared_row_limits(tmp_path: Path) -> None:
    from matplotlib import pyplot as plt

    long = _comparison_long(tmp_path)
    with_zero_spec = next(
        spec for spec in COMPARISON_PLOT_SPECS if spec.key == "all_7_models_with_checkpoint_0"
    )
    without_zero_spec = next(
        spec for spec in COMPARISON_PLOT_SPECS if spec.key == "all_7_models_without_checkpoint_0"
    )
    with_zero = prepare_comparison_frame(
        long,
        run_keys=with_zero_spec.run_keys,
        include_checkpoint_zero=True,
        required_checkpoint_zero_run_keys=with_zero_spec.required_checkpoint_zero_run_keys,
    )
    without_zero = prepare_comparison_frame(
        long,
        run_keys=without_zero_spec.run_keys,
        include_checkpoint_zero=False,
    )
    assert set(with_zero["run_key"]) == set(ALL_MODEL_COMPARISON_RUN_KEYS)
    assert set(with_zero.loc[with_zero["checkpoint_step"] == 0, "run_key"]) == {
        "all_tasks_rgb",
        "red_cup_rgb",
        "legacy_red_cup_rgb_82eps",
    }
    assert set(without_zero["run_key"]) == set(ALL_MODEL_COMPARISON_RUN_KEYS)
    assert 0 not in set(without_zero["checkpoint_step"])

    with_figure, with_axes = build_comparison_figure(long, spec=with_zero_spec)
    without_figure, without_axes = build_comparison_figure(long, spec=without_zero_spec)
    diagnostic_long = long.copy()
    diagnostic_long["metric_family"] = DIAGNOSTIC_METRIC_FAMILY.key
    diagnostic_long[["mae", "mse"]] *= 2
    diagnostic_with_zero_spec = next(
        spec
        for spec in DIAGNOSTIC_COMPARISON_PLOT_SPECS
        if spec.key == "all_7_models_with_checkpoint_0"
    )
    diagnostic_figure, diagnostic_axes = build_comparison_figure(
        diagnostic_long,
        spec=diagnostic_with_zero_spec,
        metric_family=DIAGNOSTIC_METRIC_FAMILY,
    )
    try:
        for axes in (with_axes, without_axes):
            assert axes[0, 0].get_ylim() == pytest.approx(axes[0, 1].get_ylim())
            assert axes[0, 0].get_ylim() == pytest.approx(axes[0, 2].get_ylim())
            assert axes[1, 0].get_ylim() == pytest.approx(axes[1, 1].get_ylim())
            assert axes[1, 0].get_ylim() == pytest.approx(axes[1, 2].get_ylim())
            assert all(line.get_marker() == "o" for axis in axes.flat for line in axis.lines)
            assert all(not axis.collections for axis in axes.flat)
        for row in range(2):
            with_limits = with_axes[row, 0].get_ylim()
            without_limits = without_axes[row, 0].get_ylim()
            assert without_limits[1] / without_limits[0] < with_limits[1] / with_limits[0]
        legacy_line = with_axes[0, 0].get_lines()[-1]
        assert legacy_line.get_xdata()[0] == 0
        assert pd.isna(legacy_line.get_xdata()[1])
        assert list(legacy_line.get_xdata()[2:]) == [15, 20]
        assert "prediction and target each clipped to [-1, +1]" in (
            with_figure._suptitle.get_text()
        )
        assert diagnostic_axes[0, 0].get_title() == ("overall: q01-q99 diagnostic MAE")
        assert "q01-q99 diagnostic validation error (unclipped)" in (
            diagnostic_figure._suptitle.get_text()
        )
    finally:
        plt.close(with_figure)
        plt.close(without_figure)
        plt.close(diagnostic_figure)


def test_line_values_break_only_checkpoint_gaps_over_5000() -> None:
    steps, values = line_values_with_checkpoint_gaps([0, 15000, 20000], [5.0, 0.12, 0.11])
    assert steps[0] == 0
    assert values[0] == 5.0
    assert pd.isna(steps[1]) and pd.isna(values[1])
    assert list(steps[2:]) == [15000, 20000]
    assert list(values[2:]) == [0.12, 0.11]


def test_comparison_artifact_update_is_narrow_and_records_eight_explicit_plots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    long_path = output_dir / "inspire_validation_action_metrics_long.csv"
    canonical_long = _comparison_long(tmp_path / "data")
    canonical_long.to_csv(long_path, index=False)
    long_before = long_path.read_bytes()
    diagnostic_long_path = (
        output_dir / "inspire_q01_q99_diagnostic_validation_action_metrics_long.csv"
    )
    diagnostic_long = canonical_long.copy()
    diagnostic_long["metric_family"] = DIAGNOSTIC_METRIC_FAMILY.key
    diagnostic_long["mae"] *= 2
    diagnostic_long["mse"] *= 2
    diagnostic_long.to_csv(diagnostic_long_path, index=False)
    diagnostic_long_before = diagnostic_long_path.read_bytes()
    provenance_path = output_dir / "inspire_validation_action_metrics_provenance.json"
    provenance_path.write_text('{"existing": true}\n', encoding="utf-8")
    obsolete_paths = [output_dir / filename for filename in OBSOLETE_AMBIGUOUS_PLOTS]
    for obsolete in obsolete_paths:
        obsolete.write_bytes(b"old ambiguous plot")
    unrelated_plot = output_dir / "keep_me.png"
    unrelated_plot.write_bytes(b"unrelated")
    raw_archive = output_dir / "validation_frame_predictions.csv.gz"
    raw_archive.write_bytes(b"raw prediction sentinel")

    paths = update_existing_comparison_artifacts(output_dir)

    assert set(paths) == {
        CANONICAL_METRIC_FAMILY.key,
        DIAGNOSTIC_METRIC_FAMILY.key,
    }
    assert set(paths[CANONICAL_METRIC_FAMILY.key]) == {spec.key for spec in COMPARISON_PLOT_SPECS}
    assert set(paths[DIAGNOSTIC_METRIC_FAMILY.key]) == {
        spec.key for spec in DIAGNOSTIC_COMPARISON_PLOT_SPECS
    }
    assert {path.name for path in paths[CANONICAL_METRIC_FAMILY.key].values()} == {
        ALL_MODEL_PLOT_WITH_CHECKPOINT_0,
        ALL_MODEL_PLOT_WITHOUT_CHECKPOINT_0,
        LEGACY_RGB_PLOT_WITH_CHECKPOINT_0,
        LEGACY_RGB_PLOT_WITHOUT_CHECKPOINT_0,
    }
    assert all(
        "q01_q99_diagnostic" in path.name for path in paths[DIAGNOSTIC_METRIC_FAMILY.key].values()
    )
    flattened_paths = [path for family in paths.values() for path in family.values()]
    assert len(flattened_paths) == 8
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in flattened_paths)
    assert all(not obsolete.exists() for obsolete in obsolete_paths)
    assert unrelated_plot.read_bytes() == b"unrelated"
    assert raw_archive.read_bytes() == b"raw prediction sentinel"
    assert long_path.read_bytes() == long_before
    assert diagnostic_long_path.read_bytes() == diagnostic_long_before
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["existing"] is True
    provenance_families = provenance["comparison_plots"]["metric_families"]
    assert set(provenance_families) == set(paths)
    for family_key, family_paths in paths.items():
        assert set(provenance_families[family_key]["plots"]) == set(family_paths)
    assert "autoscale independently" in provenance["comparison_plots"]["between_plot_limits"]
    assert provenance["comparison_plots"]["checkpoint_markers"] == (
        "uniform circles for every observed checkpoint"
    )


def test_comparison_plot_set_keeps_obsolete_plot_until_required_zero_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    obsolete_paths = [output_dir / filename for filename in OBSOLETE_AMBIGUOUS_PLOTS]
    for obsolete in obsolete_paths:
        obsolete.write_bytes(b"old ambiguous plot")
    canonical_long = _comparison_long(tmp_path / "data")
    canonical_long = canonical_long[
        ~(
            (canonical_long["run_key"] == "legacy_red_cup_rgb_82eps")
            & (canonical_long["checkpoint_step"] == 0)
        )
    ]
    diagnostic_long = canonical_long.copy()
    diagnostic_long["metric_family"] = DIAGNOSTIC_METRIC_FAMILY.key

    with pytest.raises(RuntimeError, match="lacks required checkpoint-0"):
        write_metric_family_comparison_plots(canonical_long, diagnostic_long, output_dir)

    assert all(obsolete.read_bytes() == b"old ambiguous plot" for obsolete in obsolete_paths)


def test_postprocess_is_cpu_only_and_uses_remapped_dataset_for_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    table = _scope_table(
        run.spec.checkpoint_steps, {(5000, scope): 0.1 for scope in COLLATED_SCOPES}
    )
    commands: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        if command[2] == "scripts.analysis_tools.normalized_action_metrics":
            table.to_csv(run.evaluation_dir / "normalized_metrics_by_scope.csv", index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    result = postprocess_run(
        run,
        python_executable=Path("python"),
        repo_root=tmp_path,
        runner=fake_runner,
    )

    assert len(result) == 3
    assert commands[0][2] == "scripts.analysis_tools.normalized_action_metrics"
    assert commands[1][2] == "scripts.analysis_tools.evaluate_checkpoints"
    assert "--plots-only" in commands[1]
    assert commands[1][commands[1].index("--dataset-path") + 1] == str(
        run.dataset_dir / "validation"
    )
    assert commands[1][commands[1].index("--train-dataset-path") + 1] == str(
        run.dataset_dir / "train"
    )
