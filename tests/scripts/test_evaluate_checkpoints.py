from argparse import Namespace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts.analysis_tools import evaluate_checkpoints
from scripts.analysis_tools.evaluate_checkpoints import (
    RAW_PREDICTION_COLUMNS,
    action_target_velocities,
    append_raw_predictions_csv,
    default_evaluation_output_dir,
    ensure_measured_state_in_raw_predictions,
    evaluate_probe,
    find_evaluation_targets,
    format_duration,
    initialize_raw_predictions_csv,
    joint_position_velocity_statistics,
    load_dataset_fps,
    plot_action_target_velocities,
    plot_checkpoint_metric_summary,
    plot_checkpoint_progress,
    plot_error_heatmap,
    plot_evaluation_summaries,
    plot_trajectory,
    raw_action_frame,
    raw_predictions_csv_path,
    raw_trajectory_arrays,
    regenerate_plots,
    save_figure_with_parent,
    seed_inference,
    select_task_balanced_trajectory_ids,
    select_trajectory_ids,
)
import torch


def _write_episode_metadata(dataset_path, episode_tasks, fps=30):
    metadata_dir = dataset_path / "meta"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "info.json").write_text(json.dumps({"fps": fps}))
    with (metadata_dir / "episodes.jsonl").open("w") as metadata_file:
        for episode_id, tasks in enumerate(episode_tasks):
            metadata_file.write(json.dumps({"episode_index": episode_id, "tasks": tasks}) + "\n")


def _write_state_parquet_dataset(dataset_path, episode_states, state_groups, fps=30):
    info = {
        "fps": fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info))
    modality_state = {
        group: {
            "start": start,
            "end": end,
            "original_key": "observation.state",
        }
        for group, (start, end) in state_groups.items()
    }
    (dataset_path / "meta" / "modality.json").write_text(json.dumps({"state": modality_state}))
    for episode_index, states in enumerate(episode_states):
        states = np.asarray(states)
        parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "episode_index": np.full(len(states), episode_index),
                "frame_index": np.arange(len(states)),
                "observation.state": [row for row in states],
            }
        ).to_parquet(parquet_path, index=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_velocity_analysis_cli_is_opt_in(tmp_path, monkeypatch, enabled):
    argv = [
        "evaluate_checkpoints",
        "--run-dir",
        str(tmp_path / "run"),
        "--dataset-path",
        str(tmp_path / "validation"),
    ]
    if enabled:
        argv.append("--velocity-analysis")
    monkeypatch.setattr(sys, "argv", argv)

    assert evaluate_checkpoints.parse_args().velocity_analysis is enabled


def test_inference_batch_size_cli_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_checkpoints",
            "--run-dir",
            str(tmp_path / "run"),
            "--dataset-path",
            str(tmp_path / "validation"),
            "--inference-batch-size",
            "8",
        ],
    )

    assert evaluate_checkpoints.parse_args().inference_batch_size == 8


def test_plot_selection_exceeds_soft_target_to_cover_every_task(tmp_path):
    episode_tasks = [["a"], ["a"], ["b"], ["b"], ["c"], ["d"]]
    _write_episode_metadata(tmp_path, episode_tasks)

    selected = select_task_balanced_trajectory_ids(
        tmp_path,
        len(episode_tasks),
        list(range(len(episode_tasks))),
        episode_count=3,
        seed=7,
    )

    covered_tasks = {task for episode_id in selected for task in episode_tasks[episode_id]}
    assert len(selected) == 4
    assert covered_tasks == {"a", "b", "c", "d"}


def test_plot_selection_fills_to_soft_target_when_there_are_fewer_tasks(tmp_path):
    episode_tasks = [["a"], ["a"], ["b"], ["b"], ["b"]]
    _write_episode_metadata(tmp_path, episode_tasks)

    selected = select_task_balanced_trajectory_ids(
        tmp_path,
        len(episode_tasks),
        list(range(len(episode_tasks))),
        episode_count=3,
        seed=7,
    )

    covered_tasks = {task for episode_id in selected for task in episode_tasks[episode_id]}
    assert len(selected) == 3
    assert covered_tasks == {"a", "b"}


def test_explicit_trajectory_ids_override_task_balancing(tmp_path):
    selected = select_trajectory_ids(tmp_path, 5, [1, 4], episode_count=3, seed=7)

    assert selected == [1, 4]


def test_default_validation_selection_still_evaluates_every_episode(tmp_path):
    selected = select_trajectory_ids(tmp_path, 5, None, episode_count=0, seed=7)

    assert selected == [0, 1, 2, 3, 4]


def _write_run_processor(run_dir):
    processor_dir = run_dir / "processor"
    processor_dir.mkdir(parents=True)
    (processor_dir / "processor_config.json").write_text("{}")
    (processor_dir / "statistics.json").write_text("{}")
    return processor_dir


def test_evaluation_targets_include_run_compatible_base_as_step_zero(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    processor_dir = _write_run_processor(run_dir)
    (run_dir / "checkpoint-2000").mkdir()
    (run_dir / "checkpoint-4000").mkdir()
    base_model = tmp_path / "base"
    base_model.mkdir()

    targets = find_evaluation_targets(run_dir, None, base_model)

    assert [target.step for target in targets] == [0, 2000, 4000]
    assert targets[0].model_path == base_model
    assert targets[0].processor_path == processor_dir
    assert targets[1].processor_path is None


def test_evaluation_target_selection_can_choose_only_base_or_only_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_processor(run_dir)
    checkpoint = run_dir / "checkpoint-2000"
    checkpoint.mkdir()
    base_model = tmp_path / "base"
    base_model.mkdir()

    base_only = find_evaluation_targets(run_dir, [0], base_model)
    checkpoint_only = find_evaluation_targets(run_dir, [2000], base_model)

    assert [(target.step, target.model_path) for target in base_only] == [(0, base_model)]
    assert [(target.step, target.model_path) for target in checkpoint_only] == [(2000, checkpoint)]


def test_evaluation_targets_can_include_final_weights_from_run_root(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_processor(run_dir)
    (run_dir / "config.json").write_text("{}")
    (run_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    checkpoint = run_dir / "checkpoint-25000"
    checkpoint.mkdir()

    targets = find_evaluation_targets(run_dir, [25000, 30000], None, 30000)

    assert [(target.step, target.model_path) for target in targets] == [
        (25000, checkpoint),
        (30000, run_dir),
    ]


def test_run_root_step_rejects_duplicate_physical_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint-30000").mkdir()

    with pytest.raises(ValueError, match="both represent step 30000"):
        find_evaluation_targets(run_dir, [30000], None, 30000)


def test_base_evaluation_requires_run_processor_statistics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base_model = tmp_path / "base"
    base_model.mkdir()

    with pytest.raises(FileNotFoundError, match="embodiment processor"):
        find_evaluation_targets(run_dir, None, base_model)


def test_inference_seed_restarts_torch_noise_for_each_model():
    seed_inference(123)
    first = torch.randn(8)
    seed_inference(123)
    second = torch.randn(8)

    torch.testing.assert_close(first, second)


def test_progress_duration_format_is_compact():
    assert format_duration(4.6) == "5s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3723) == "1h 02m 03s"


def test_default_evaluation_output_records_execution_horizon(tmp_path):
    assert default_evaluation_output_dir(tmp_path, 8) == tmp_path / "evaluation_exec_hor_8"
    assert default_evaluation_output_dir(tmp_path, 16) == tmp_path / "evaluation_exec_hor_16"


def test_action_target_velocity_uses_dataset_fps_and_transition_frames():
    ground_truth = np.array([[0.0, 1.0], [0.1, 0.8], [0.4, 1.0]])
    prediction = np.array([[0.0, 1.0], [0.2, 0.9], [0.3, 1.2]])

    frames, ground_truth_velocity, prediction_velocity = action_target_velocities(
        ground_truth,
        prediction,
        dataset_fps=10,
    )

    np.testing.assert_array_equal(frames, [1, 2])
    np.testing.assert_allclose(ground_truth_velocity, [[1.0, -2.0], [3.0, 2.0]])
    np.testing.assert_allclose(prediction_velocity, [[2.0, -1.0], [1.0, 3.0]])


def test_action_target_velocity_does_not_fabricate_frame_zero_sample():
    frames, ground_truth_velocity, prediction_velocity = action_target_velocities(
        np.array([[0.25, -0.5]]),
        np.array([[0.3, -0.4]]),
        dataset_fps=30,
    )

    assert frames.shape == (0,)
    assert ground_truth_velocity.shape == (0, 2)
    assert prediction_velocity.shape == (0, 2)


def test_load_dataset_fps_requires_positive_metadata_value(tmp_path):
    _write_episode_metadata(tmp_path, [[]], fps=15)
    assert load_dataset_fps(tmp_path) == 15.0

    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 0}))
    with pytest.raises(ValueError, match="fps must be positive"):
        load_dataset_fps(tmp_path)


def test_figure_save_recreates_parent_if_it_disappears_during_first_write(tmp_path):
    class DisappearingParentFigure:
        def __init__(self):
            self.calls = 0

        def savefig(self, path, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                path.parent.rmdir()
                raise FileNotFoundError(path)
            assert path.parent.is_dir()

    figure = DisappearingParentFigure()
    output_path = tmp_path / "checkpoint-16000" / "train_probe" / "velocity.png"

    save_figure_with_parent(figure, output_path, dpi=150)

    assert figure.calls == 2
    assert output_path.parent.is_dir()


@pytest.mark.parametrize("velocity_analysis", [False, True])
def test_evaluate_probe_passes_checkpoint_split_and_episode_progress(
    tmp_path, monkeypatch, caplog, velocity_analysis
):
    loaded_trajectory_ids = []

    class FakeLoader:
        def __getitem__(self, trajectory_id):
            loaded_trajectory_ids.append(trajectory_id)
            value = np.asarray([float(trajectory_id)], dtype=np.float32)
            return pd.DataFrame(
                {
                    "state.right_arm": [value, value, value],
                    "action.right_arm": [value, value, value],
                }
            )

    progress_labels = []
    inference_batch_sizes = []

    def fake_evaluate_single_trajectory(**kwargs):
        progress_labels.append(kwargs["progress_label"])
        inference_batch_sizes.append(kwargs["inference_batch_size"])
        assert kwargs["trajectory"] is not None
        trajectory_id = kwargs["traj_id"]
        ground_truth = np.full((3, 1), trajectory_id, dtype=np.float32)
        prediction = ground_truth + 0.25
        evaluate_checkpoints.open_loop_eval.plot_trajectory_results(
            state_joints_across_time=ground_truth,
            gt_action_across_time=ground_truth,
            pred_action_across_time=prediction,
            traj_id=trajectory_id,
            state_keys=["right_arm"],
            action_keys=["right_arm"],
            execution_horizon=kwargs["execution_horizon"],
            save_plot_path=None,
        )
        return 0.0625, 0.25

    monkeypatch.setattr(
        evaluate_checkpoints.open_loop_eval,
        "evaluate_single_trajectory",
        fake_evaluate_single_trajectory,
    )
    velocity_plot_calls = []
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_trajectory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_error_heatmap",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_action_target_velocities",
        lambda *_args, **_kwargs: velocity_plot_calls.append(
            (_args[4], _args[6], _kwargs["measured_state"].copy())
        ),
    )
    caplog.set_level("INFO")

    raw_predictions_path = tmp_path / "validation_frame_predictions.csv.gz"
    evaluate_probe(
        policy=SimpleNamespace(),
        loader=FakeLoader(),
        trajectory_ids=[4, 9],
        split="validation",
        checkpoint_step_value=2000,
        plot_dir=tmp_path / "plots",
        right_plot_dir=tmp_path / "right_plots",
        embodiment_tag=evaluate_checkpoints.EmbodimentTag.NEW_EMBODIMENT,
        action_keys=["right_arm"],
        modality_keys=["right_arm"],
        steps=0,
        execution_horizon=8,
        dataset_fps=30,
        skip_trajectory_plots=False,
        plot_trajectory_ids={4},
        trajectory_goals={4: "pick", 9: "place"},
        raw_predictions_path=raw_predictions_path,
        canonical_labels=None,
        checkpoint_progress_label="checkpoint 2/11 step=2000",
        velocity_analysis=velocity_analysis,
        inference_batch_size=3,
    )

    assert loaded_trajectory_ids == [4, 9]
    assert inference_batch_sizes == [3, 3]
    assert progress_labels == [
        "checkpoint 2/11 step=2000 | validation episode 1/2 traj=4",
        "checkpoint 2/11 step=2000 | validation episode 2/2 traj=9",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("validation episode 1/2 traj=4" in message for message in messages)
    assert any("validation episode 2/2 traj=9" in message for message in messages)
    expected_velocity_plots = (
        [
            (tmp_path / "plots" / "trajectory_0004_joint_velocities.png", 30),
            (tmp_path / "right_plots" / "trajectory_0004_joint_velocities.png", 30),
        ]
        if velocity_analysis
        else []
    )
    assert [(path, fps) for path, fps, _state in velocity_plot_calls] == expected_velocity_plots
    for _path, _fps, measured_state in velocity_plot_calls:
        np.testing.assert_allclose(measured_state, np.full((3, 1), 4))
    saved_raw = pd.read_csv(raw_predictions_path, compression="gzip")
    assert set(saved_raw["trajectory"]) == {4, 9}
    for trajectory_id in (4, 9):
        np.testing.assert_allclose(
            saved_raw.loc[
                saved_raw["trajectory"] == trajectory_id,
                "measured_state",
            ],
            trajectory_id,
        )


def test_joint_trajectory_plot_labels_visible_frame_axes(tmp_path, monkeypatch):
    saved_figures = []

    def capture_figure(figure, *_args, **_kwargs):
        saved_figures.append(figure)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure)
    gt = np.zeros((32, 4), dtype=np.float32)
    pred = np.ones((32, 4), dtype=np.float32)
    measured_state = np.full((32, 4), 0.5, dtype=np.float32)
    labels = ["left_arm[0]", "left_arm[1]", "right_arm[0]", "right_arm[1]"]

    plot_trajectory(
        gt,
        pred,
        labels,
        "test trajectory",
        tmp_path / "trajectory_joints.png",
        execution_horizon=16,
        goal="move the red cup",
        measured_state=measured_state,
    )

    plot_error_heatmap(
        pred - gt,
        labels,
        "test trajectory errors",
        tmp_path / "trajectory_error_heatmap.png",
        goal="move the red cup",
    )

    assert len(saved_figures) == 2
    frame_axes = [axis for axis in saved_figures[0].axes if axis.get_xlabel() == "Frame"]
    assert len(frame_axes) == 4
    assert all(any(label.get_visible() for label in axis.get_xticklabels()) for axis in frame_axes)
    position_line_labels = {line.get_label() for line in frame_axes[0].lines}
    assert "measured joint state" in position_line_labels
    assert "demonstration action target" in position_line_labels
    assert "model-predicted action target" in position_line_labels
    assert any(text.get_text() == "Goal: move the red cup" for text in saved_figures[0].texts)
    assert any(
        text.get_text() == "Goal: move the red cup"
        for axis in saved_figures[1].axes
        for text in axis.texts
    )


def test_target_velocity_plot_marks_cross_chunk_transition_and_joint_maxima(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, *_args, **_kwargs: saved_figures.append(figure),
    )
    ground_truth = np.array([[0.0], [0.1], [0.3], [0.2]])
    prediction = np.array([[0.0], [0.2], [0.1], [0.4]])
    measured_state = np.array([[0.0], [0.05], [0.15], [0.1]])

    plot_action_target_velocities(
        ground_truth,
        prediction,
        ["right_hand[0]"],
        "test trajectory",
        tmp_path / "trajectory_joint_velocities.png",
        execution_horizon=2,
        dataset_fps=10,
        goal="release the cup",
        measured_state=measured_state,
    )

    assert len(saved_figures) == 1
    figure = saved_figures[0]
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    np.testing.assert_array_equal(
        lines["demonstration action-target velocity"].get_xdata(),
        [1, 2, 3],
    )
    np.testing.assert_allclose(
        lines["demonstration action-target velocity"].get_ydata(),
        [1.0, 2.0, -1.0],
    )
    np.testing.assert_array_equal(lines["cross-chunk transition"].get_xdata(), [2, 2])
    np.testing.assert_allclose(
        lines["measured joint velocity from state q"].get_ydata(),
        [0.5, 1.0, -0.5],
    )
    assert "measured=1.000" in axis.get_title()
    assert "demo target=2.000" in axis.get_title()
    assert "pred target=3.000" in axis.get_title()
    assert axis.get_xlabel() == "Transition frame"
    assert axis.get_ylabel() == "Joint / action-target velocity (rad/s)"
    assert "measured joint and action-target velocities" in figure._suptitle.get_text()
    assert any(
        "chunk-boundary transitions are retained" in text.get_text() for text in figure.texts
    )


def test_target_velocity_plot_explains_single_frame_has_no_sample(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, *_args, **_kwargs: saved_figures.append(figure),
    )

    plot_action_target_velocities(
        np.array([[0.25]]),
        np.array([[0.3]]),
        ["right_arm[0]"],
        "single frame",
        tmp_path / "trajectory_joint_velocities.png",
        execution_horizon=8,
        dataset_fps=30,
    )

    axis = saved_figures[0].axes[0]
    assert "fewer than 2 frames" in axis.get_title()
    assert any("No target-velocity samples" in text.get_text() for text in axis.texts)
    assert not any(
        line.get_label()
        in {
            "demonstration action-target velocity",
            "model-predicted action-target velocity",
        }
        for line in axis.lines
    )


def test_checkpoint_summary_splits_magnitude_metrics_with_matching_colours(tmp_path, monkeypatch):
    saved_figures = []

    def capture_figure(figure, *_args, **_kwargs):
        saved_figures.append(figure)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure)
    rows = []
    for split, offset in (("train_probe", 0.0), ("validation", 0.1)):
        for step in (100, 200):
            rows.append(
                {
                    "split": split,
                    "checkpoint_step": step,
                    "mae": 0.2 + offset,
                    "mse": 0.04 + offset,
                    "median_absolute_error": 0.1 + offset,
                    "p95_absolute_error": 0.4 + offset,
                    "rmse": 0.3 + offset,
                    "bias": 0.01 + offset,
                    "max_absolute_error": 0.5 + offset,
                }
            )

    plot_checkpoint_metric_summary(pd.DataFrame(rows), tmp_path / "summary.png")

    axes_by_title = {axis.get_title(): axis for axis in saved_figures[0].axes}
    for title in ("Aggregate prediction bias", "Worst observed error"):
        split_lines = {line.get_label(): line for line in axes_by_title[title].lines}
        assert split_lines["Train Probe"].get_color() == "tab:blue"
        assert split_lines["Validation"].get_color() == "tab:orange"
        assert split_lines["Train Probe"].get_linestyle() == "-"
        assert split_lines["Validation"].get_linestyle() == "-"

    expected_metric_colours = {
        "MAE": "tab:blue",
        "RMSE": "tab:orange",
        "MSE": "tab:purple",
        "Median absolute error": "tab:green",
        "95th-percentile absolute error": "tab:red",
    }
    for title in ("Validation error magnitude", "Train Probe error magnitude"):
        magnitude_lines = {line.get_label(): line for line in axes_by_title[title].lines}
        assert set(magnitude_lines) == set(expected_metric_colours)
        for metric, colour in expected_metric_colours.items():
            line = magnitude_lines[metric]
            assert line.get_color() == colour
            assert line.get_linewidth() == 1.0
            assert line.get_markersize() == 4.0
            assert line.get_alpha() == 0.72
        legend = axes_by_title[title].get_legend()
        assert legend._ncols == 1
        assert [text.get_text() for text in legend.get_texts()] == list(expected_metric_colours)


def test_checkpoint_progress_legend_explains_color_and_line_style(tmp_path, monkeypatch):
    saved_figures = []

    def capture_figure(figure, *_args, **_kwargs):
        saved_figures.append(figure)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure)
    summary = pd.DataFrame(
        [
            {
                "split": split,
                "checkpoint_step": step,
                "mae": mae,
                "rmse": rmse,
            }
            for split, mae, rmse in (
                ("train_probe", 0.2, 0.3),
                ("validation", 0.25, 0.35),
            )
            for step in (100, 200)
        ]
    )

    plot_checkpoint_progress(summary, tmp_path / "progress.png")

    legend = saved_figures[0].axes[0].get_legend()
    assert legend._ncols == 1
    assert [text.get_text() for text in legend.get_texts()] == [
        "MAE — Validation",
        "MAE — Train Probe",
        "RMSE — Validation",
        "RMSE — Train Probe",
    ]


def test_step_zero_summaries_also_create_finetuned_only_views(tmp_path, monkeypatch):
    summary = pd.DataFrame(
        [
            {
                "split": split,
                "checkpoint_step": step,
                "mae": 0.2,
            }
            for split in ("train_probe", "validation")
            for step in (0, 2000)
        ]
    )
    joints = pd.DataFrame(
        [
            {
                "split": split,
                "checkpoint_step": step,
                "joint": "right_arm[0]",
                "mae": 0.2,
            }
            for split in ("train_probe", "validation")
            for step in (0, 2000)
        ]
    )
    saved_paths = []
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_checkpoint_progress",
        lambda _summary, path, _scope: saved_paths.append(path.name),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_checkpoint_metric_summary",
        lambda _summary, path, _scope: saved_paths.append(path.name),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_joint_checkpoint_heatmap",
        lambda _joints, _labels, path, _scope: saved_paths.append(path.name),
    )
    monkeypatch.setattr(evaluate_checkpoints, "plot_best_checkpoint_joints", lambda *_args: None)

    plot_evaluation_summaries(tmp_path, summary, joints, ["right_arm[0]"])

    assert saved_paths == [
        "checkpoint_error_progress.png",
        "checkpoint_metric_summary.png",
        "joint_error_by_checkpoint.png",
        "checkpoint_error_progress_finetuned_only.png",
        "checkpoint_metric_summary_finetuned_only.png",
        "joint_error_by_checkpoint_finetuned_only.png",
    ]


def test_raw_validation_predictions_round_trip_as_compressed_csv(tmp_path):
    output_path = tmp_path / "validation_frame_predictions.csv.gz"
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    prediction = np.array([[1.25, 1.5], [2.0, 5.0]], dtype=np.float32)
    measured_state = np.array([[0.9, 1.9], [2.9, 3.9]], dtype=np.float32)
    frame = raw_action_frame(
        checkpoint_step_value=200,
        trajectory_id=7,
        ground_truth=ground_truth,
        prediction=prediction,
        labels=["left_arm[0]", "right_arm[0]"],
        measured_state=measured_state,
    )

    initialize_raw_predictions_csv(output_path)
    append_raw_predictions_csv(output_path, frame)
    saved = pd.read_csv(output_path, compression="gzip")

    assert list(saved.columns) == RAW_PREDICTION_COLUMNS
    assert len(saved) == 4
    assert set(saved["checkpoint_step"]) == {200}
    assert set(saved["trajectory"]) == {7}
    assert saved["joint"].tolist() == [
        "left_arm[0]",
        "right_arm[0]",
        "left_arm[0]",
        "right_arm[0]",
    ]
    np.testing.assert_allclose(saved["error"], [0.25, -0.5, -1.0, 1.0])
    np.testing.assert_allclose(saved["absolute_error"], [0.25, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(saved["measured_state"], measured_state.reshape(-1))

    saved_ground_truth, saved_prediction, saved_labels = raw_trajectory_arrays(saved, 7)
    assert saved_labels == ["left_arm[0]", "right_arm[0]"]
    np.testing.assert_allclose(saved_ground_truth, ground_truth)
    np.testing.assert_allclose(saved_prediction, prediction)


def test_legacy_raw_predictions_load_exact_saved_state_frames_from_parquet(tmp_path, monkeypatch):
    _write_episode_metadata(tmp_path, [["task"]])
    states = np.array(
        [
            [0.0, 10.0],
            [0.1, 10.1],
            [0.2, 10.2],
            [0.3, 10.3],
        ]
    )
    _write_state_parquet_dataset(
        tmp_path,
        [states],
        {"left_arm": (0, 1), "right_arm": (1, 2)},
    )
    legacy = raw_action_frame(
        checkpoint_step_value=10,
        trajectory_id=0,
        ground_truth=np.zeros((2, 2)),
        prediction=np.ones((2, 2)),
        labels=["left_arm[0]", "right_arm[0]"],
    ).drop(columns="measured_state")
    legacy["frame"] = [1, 1, 3, 3]
    parquet_reads = []
    original_read_parquet = pd.read_parquet

    def counted_read_parquet(*args, **kwargs):
        parquet_reads.append(args[0])
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)
    state_cache = {}

    populated = ensure_measured_state_in_raw_predictions(
        legacy,
        dataset_path=tmp_path,
        state_cache=state_cache,
    )
    populated_again = ensure_measured_state_in_raw_predictions(
        legacy,
        dataset_path=tmp_path,
        state_cache=state_cache,
    )

    np.testing.assert_allclose(populated["measured_state"], [0.1, 10.1, 0.3, 10.3])
    np.testing.assert_allclose(populated_again["measured_state"], populated["measured_state"])
    assert len(parquet_reads) == 1


def test_joint_statistics_never_difference_across_trajectories_and_split_chunk_speeds():
    labels = ["right_hand[0]"]
    trajectory_zero = raw_action_frame(
        checkpoint_step_value=200,
        trajectory_id=0,
        ground_truth=np.array([[-0.2], [0.0], [0.3]]),
        prediction=np.array([[-0.1], [0.1], [-0.1]]),
        labels=labels,
        measured_state=np.array([[-0.3], [-0.1], [0.2]]),
    )
    trajectory_one = raw_action_frame(
        checkpoint_step_value=200,
        trajectory_id=1,
        ground_truth=np.array([[10.0], [10.2]]),
        prediction=np.array([[20.0], [19.5]]),
        labels=labels,
        measured_state=np.array([[5.0], [5.1]]),
    )

    statistics = joint_position_velocity_statistics(
        pd.concat([trajectory_zero, trajectory_one], ignore_index=True),
        split="validation",
        dataset_fps=10,
        execution_horizon=2,
    ).set_index("source")

    ground_truth = statistics.loc["ground_truth_action_target"]
    assert ground_truth["execution_horizon"] == 2
    assert ground_truth["dataset_fps"] == 10
    assert ground_truth["episodes"] == 2
    assert ground_truth["position_samples"] == 5
    assert ground_truth["velocity_samples"] == 3
    assert ground_truth["within_chunk_velocity_samples"] == 2
    assert ground_truth["chunk_boundary_velocity_samples"] == 1
    assert ground_truth["position_min_rad"] == pytest.approx(-0.2)
    assert ground_truth["position_max_rad"] == pytest.approx(10.2)
    assert ground_truth["absolute_position_p95_rad"] == pytest.approx(
        np.percentile([0.2, 0.0, 0.3, 10.0, 10.2], 95)
    )
    assert ground_truth["absolute_position_p99_rad"] == pytest.approx(
        np.percentile([0.2, 0.0, 0.3, 10.0, 10.2], 99)
    )
    assert ground_truth["absolute_position_max_rad"] == pytest.approx(10.2)
    assert ground_truth["absolute_velocity_p95_rad_s"] == pytest.approx(
        np.percentile([2.0, 3.0, 2.0], 95)
    )
    assert ground_truth["absolute_velocity_p99_rad_s"] == pytest.approx(
        np.percentile([2.0, 3.0, 2.0], 99)
    )
    assert ground_truth["absolute_velocity_max_rad_s"] == pytest.approx(3.0)
    assert ground_truth["within_chunk_absolute_velocity_max_rad_s"] == pytest.approx(2.0)
    assert ground_truth["chunk_boundary_absolute_velocity_max_rad_s"] == pytest.approx(3.0)

    prediction = statistics.loc["predicted_action_target"]
    assert prediction["velocity_samples"] == 3
    assert prediction["absolute_velocity_max_rad_s"] == pytest.approx(5.0)
    assert prediction["within_chunk_absolute_velocity_max_rad_s"] == pytest.approx(5.0)
    assert prediction["chunk_boundary_absolute_velocity_max_rad_s"] == pytest.approx(2.0)

    measured = statistics.loc["measured_state"]
    assert measured["velocity_samples"] == 3
    assert measured["absolute_velocity_max_rad_s"] == pytest.approx(3.0)
    assert measured["position_min_rad"] == pytest.approx(-0.3)
    assert measured["position_max_rad"] == pytest.approx(5.1)


def test_plots_only_regenerates_validation_and_train_probe_trajectories(tmp_path, monkeypatch):
    output_dir = tmp_path / "evaluation"
    checkpoint_dir = output_dir / "checkpoint-10"
    validation_dataset = tmp_path / "validation"
    train_dataset = tmp_path / "train"
    _write_episode_metadata(validation_dataset, [["validate the red cup"]])
    _write_episode_metadata(train_dataset, [["train with the blue cup"]])
    state_groups = {"left_arm": (0, 1), "right_arm": (1, 2)}
    _write_state_parquet_dataset(
        validation_dataset,
        [np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]])],
        state_groups,
    )
    _write_state_parquet_dataset(
        train_dataset,
        [np.array([[0.0, 0.0], [-0.1, -0.2], [-0.2, -0.4]])],
        state_groups,
    )

    summary = pd.DataFrame(
        [
            {"split": "validation", "checkpoint_step": 10, "mae": 0.2},
            {"split": "train_probe", "checkpoint_step": 10, "mae": 0.1},
        ]
    )
    joints = pd.DataFrame(
        [
            {
                "split": split,
                "checkpoint_step": 10,
                "joint": joint,
                "mae": mae,
            }
            for split, mae in (("validation", 0.2), ("train_probe", 0.1))
            for joint in ("left_arm[0]", "right_arm[0]")
        ]
    )
    output_dir.mkdir(parents=True)
    summary.to_csv(output_dir / "metrics_by_checkpoint.csv", index=False)
    joints.to_csv(output_dir / "metrics_per_joint.csv", index=False)

    for split in ("validation", "train_probe"):
        raw_path = raw_predictions_csv_path(checkpoint_dir, split)
        initialize_raw_predictions_csv(raw_path)
        append_raw_predictions_csv(
            raw_path,
            raw_action_frame(
                checkpoint_step_value=10,
                trajectory_id=0,
                ground_truth=np.zeros((3, 2)),
                prediction=np.ones((3, 2)),
                labels=["left_arm[0]", "right_arm[0]"],
            ),
        )

    summary_plot_calls = []
    trajectory_plot_calls = []
    velocity_plot_calls = []
    heatmap_calls = []
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_checkpoint_progress",
        lambda *_args: summary_plot_calls.append("progress"),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_checkpoint_metric_summary",
        lambda *_args: summary_plot_calls.append("summary"),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_joint_checkpoint_heatmap",
        lambda *_args: summary_plot_calls.append("heatmap"),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_best_checkpoint_joints",
        lambda *_args: summary_plot_calls.append("best"),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_trajectory",
        lambda *_args, **_kwargs: trajectory_plot_calls.append(
            (_args[4], _args[6], _kwargs["measured_state"].copy())
        ),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_error_heatmap",
        lambda *_args, **_kwargs: heatmap_calls.append((_args[3], _args[4])),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_action_target_velocities",
        lambda *_args, **_kwargs: velocity_plot_calls.append(
            (_args[4], _args[6], _kwargs["measured_state"].copy())
        ),
    )
    args = Namespace(
        checkpoint_steps=None,
        dataset_path=validation_dataset,
        execution_horizon=16,
        skip_trajectory_plots=False,
        train_dataset_path=train_dataset,
        train_traj_ids=None,
        traj_ids=None,
        trajectory_plot_episodes=1,
        trajectory_plot_seed=42,
        velocity_analysis=True,
    )

    regenerate_plots(args, output_dir)

    assert summary_plot_calls == [
        "progress",
        "summary",
        "heatmap",
        "best",
        "progress",
        "summary",
        "heatmap",
        "best",
    ]
    assert [(path, goal) for path, goal, _state in trajectory_plot_calls] == [
        (checkpoint_dir / "trajectory_0000_joints.png", "validate the red cup"),
        (
            output_dir / "right_arm_hand" / "checkpoint-10" / "trajectory_0000_joints.png",
            "validate the red cup",
        ),
        (
            checkpoint_dir / "train_probe" / "trajectory_0000_joints.png",
            "train with the blue cup",
        ),
        (
            output_dir
            / "right_arm_hand"
            / "checkpoint-10"
            / "train_probe"
            / "trajectory_0000_joints.png",
            "train with the blue cup",
        ),
    ]
    assert len(heatmap_calls) == 4
    assert [(path, fps) for path, fps, _state in velocity_plot_calls] == [
        (checkpoint_dir / "trajectory_0000_joint_velocities.png", 30.0),
        (
            output_dir
            / "right_arm_hand"
            / "checkpoint-10"
            / "trajectory_0000_joint_velocities.png",
            30.0,
        ),
        (
            checkpoint_dir / "train_probe" / "trajectory_0000_joint_velocities.png",
            30.0,
        ),
        (
            output_dir
            / "right_arm_hand"
            / "checkpoint-10"
            / "train_probe"
            / "trajectory_0000_joint_velocities.png",
            30.0,
        ),
    ]
    assert (output_dir / "right_arm_hand" / "metrics_by_checkpoint.csv").is_file()
    assert (output_dir / "right_arm_hand" / "metrics_per_joint.csv").is_file()
    statistics = pd.read_csv(output_dir / "joint_position_velocity_statistics.csv")
    assert set(statistics["split"]) == {"validation", "train_probe"}
    assert set(statistics["source"]) == {
        "measured_state",
        "ground_truth_action_target",
        "predicted_action_target",
    }
    assert set(statistics["execution_horizon"]) == {16}
    assert set(statistics["dataset_fps"]) == {30.0}
    assert len(statistics) == 12
    right_statistics = pd.read_csv(
        output_dir / "right_arm_hand" / "joint_position_velocity_statistics.csv"
    )
    assert set(right_statistics["joint"]) == {"right_arm[0]"}
    assert len(right_statistics) == 6


def test_plots_only_skips_velocity_statistics_without_opt_in(tmp_path, monkeypatch):
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    pd.DataFrame([{"split": "validation", "checkpoint_step": 10, "mae": 0.2}]).to_csv(
        output_dir / "metrics_by_checkpoint.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "split": "validation",
                "checkpoint_step": 10,
                "joint": "right_arm[0]",
                "mae": 0.2,
            }
        ]
    ).to_csv(output_dir / "metrics_per_joint.csv", index=False)

    statistics_calls = []
    trajectory_calls = []
    monkeypatch.setattr(evaluate_checkpoints, "plot_evaluation_summaries", lambda *_args: 10)
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_right_evaluation_summaries",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "write_joint_position_velocity_statistics",
        lambda **kwargs: statistics_calls.append(kwargs),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "regenerate_trajectory_plots",
        lambda **kwargs: trajectory_calls.append(kwargs),
    )
    args = Namespace(
        checkpoint_steps=None,
        dataset_path=tmp_path / "validation",
        execution_horizon=16,
        train_dataset_path=None,
        velocity_analysis=False,
    )

    regenerate_plots(args, output_dir)

    assert statistics_calls == []
    assert len(trajectory_calls) == 1


def test_main_plots_only_skips_checkpoint_loading(tmp_path, monkeypatch):
    calls = []
    args = Namespace(
        output_dir=tmp_path / "evaluation",
        plots_only=True,
        run_dir=tmp_path / "run",
    )
    monkeypatch.setattr(evaluate_checkpoints, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate_checkpoints,
        "regenerate_plots",
        lambda parsed_args, output_dir: calls.append((parsed_args, output_dir)),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "find_checkpoints",
        lambda *_args: (_ for _ in ()).throw(AssertionError("checkpoint loading was reached")),
    )

    evaluate_checkpoints.main()

    assert calls == [(args, args.output_dir)]


def test_latest_checkpoint_steps_selects_only_two_numerically_latest():
    assert evaluate_checkpoints.latest_checkpoint_steps([20_000, 0, 4_000, 16_000]) == [
        16_000,
        20_000,
    ]
    assert evaluate_checkpoints.latest_checkpoint_steps([8_000]) == [8_000]


def test_goal_and_horizon_position_metrics_use_every_scalar_error():
    episode_zero = raw_action_frame(
        checkpoint_step_value=10,
        trajectory_id=0,
        ground_truth=np.zeros((4, 1)),
        prediction=np.array([[1.0], [2.0], [3.0], [4.0]]),
        labels=["right_arm[0]"],
    )
    episode_one = raw_action_frame(
        checkpoint_step_value=10,
        trajectory_id=1,
        ground_truth=np.zeros((4, 1)),
        prediction=np.zeros((4, 1)),
        labels=["right_arm[0]"],
    )
    raw = pd.concat([episode_zero, episode_one], ignore_index=True)

    goals = evaluate_checkpoints.goal_metrics_from_raw_predictions(
        raw,
        split="validation",
        episode_tasks={0: ["pick", "shared"], 1: ["place", "shared"]},
    ).set_index("goal")
    assert goals.loc["pick", "mae"] == pytest.approx(2.5)
    assert goals.loc["place", "mae"] == pytest.approx(0.0)
    assert goals.loc["shared", "mae"] == pytest.approx(1.25)
    assert goals.loc["shared", "episodes"] == 2
    assert goals.loc["shared", "frames"] == 8

    positions = evaluate_checkpoints.horizon_position_metrics_from_raw_predictions(
        raw,
        split="validation",
        execution_horizon=2,
    ).set_index("horizon_position")
    assert positions.loc[0, "mae"] == pytest.approx(1.0)
    assert positions.loc[0, "mse"] == pytest.approx(2.5)
    assert positions.loc[1, "mae"] == pytest.approx(1.5)
    assert positions.loc[1, "mse"] == pytest.approx(5.0)


def test_horizon_position_plot_excludes_older_checkpoints(tmp_path, monkeypatch):
    saved_figures = []
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda figure, *_args, **_kwargs: saved_figures.append(figure),
    )
    metrics = pd.DataFrame(
        [
            {
                "split": "validation",
                "checkpoint_step": step,
                "horizon_position": position,
                "mae": step / 1000 + position,
            }
            for step in (10, 20, 30)
            for position in (0, 1)
        ]
    )

    evaluate_checkpoints.plot_horizon_position_metrics(
        metrics,
        [20, 30],
        tmp_path / "horizon.png",
    )

    line_labels = {line.get_label() for line in saved_figures[0].axes[0].lines}
    assert line_labels == {"checkpoint 20", "checkpoint 30"}


def test_plots_only_scopes_every_checkpoint_plot_to_latest_two(tmp_path, monkeypatch):
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    summary = pd.DataFrame(
        [
            {"split": "validation", "checkpoint_step": step, "mae": step / 1000}
            for step in (10, 20, 30)
        ]
    )
    joints = pd.DataFrame(
        [
            {
                "split": "validation",
                "checkpoint_step": step,
                "joint": "right_arm[0]",
                "mae": step / 1000,
            }
            for step in (10, 20, 30)
        ]
    )
    summary.to_csv(output_dir / "metrics_by_checkpoint.csv", index=False)
    joints.to_csv(output_dir / "metrics_per_joint.csv", index=False)
    plot_calls = {}
    monkeypatch.setattr(
        evaluate_checkpoints,
        "write_extended_evaluation_metrics",
        lambda **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_evaluation_summaries",
        lambda _output, scoped_summary, _joints, _labels: plot_calls.update(
            aggregate=sorted(scoped_summary["checkpoint_step"].unique())
        ),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_right_evaluation_summaries",
        lambda *_args, **kwargs: plot_calls.update(
            right=sorted(kwargs["plot_checkpoint_steps"])
        ),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_horizon_position_metrics",
        lambda _metrics, steps, _path: plot_calls.update(horizon=steps),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "regenerate_trajectory_plots",
        lambda **kwargs: plot_calls.update(trajectories=kwargs["checkpoint_steps"]),
    )
    args = Namespace(
        checkpoint_steps=None,
        dataset_path=tmp_path / "validation",
        execution_horizon=8,
        train_dataset_path=None,
        velocity_analysis=False,
    )

    regenerate_plots(args, output_dir)

    assert plot_calls == {
        "aggregate": [20, 30],
        "right": [20, 30],
        "horizon": [20, 30],
        "trajectories": [20, 30],
    }
