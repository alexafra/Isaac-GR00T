import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.analysis_tools import evaluate_checkpoints
from scripts.analysis_tools.evaluate_checkpoints import (
    RAW_PREDICTION_COLUMNS,
    append_raw_predictions_csv,
    default_evaluation_output_dir,
    evaluate_probe,
    find_evaluation_targets,
    format_duration,
    initialize_raw_predictions_csv,
    plot_checkpoint_metric_summary,
    plot_checkpoint_progress,
    plot_error_heatmap,
    plot_evaluation_summaries,
    plot_trajectory,
    raw_action_frame,
    raw_predictions_csv_path,
    raw_trajectory_arrays,
    regenerate_plots,
    seed_inference,
    select_task_balanced_trajectory_ids,
    select_trajectory_ids,
)


def _write_episode_metadata(dataset_path, episode_tasks):
    metadata_dir = dataset_path / "meta"
    metadata_dir.mkdir(parents=True)
    with (metadata_dir / "episodes.jsonl").open("w") as metadata_file:
        for episode_id, tasks in enumerate(episode_tasks):
            metadata_file.write(
                json.dumps({"episode_index": episode_id, "tasks": tasks}) + "\n"
            )


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


def test_evaluate_probe_passes_checkpoint_split_and_episode_progress(tmp_path, monkeypatch, caplog):
    class FakeLoader:
        def __getitem__(self, trajectory_id):
            value = np.asarray([float(trajectory_id)], dtype=np.float32)
            return pd.DataFrame({"action.right_arm": [value, value, value]})

    progress_labels = []

    def fake_evaluate_single_trajectory(**kwargs):
        progress_labels.append(kwargs["progress_label"])
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
    caplog.set_level("INFO")

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
        skip_trajectory_plots=True,
        plot_trajectory_ids=set(),
        trajectory_goals={4: "pick", 9: "place"},
        raw_predictions_path=None,
        canonical_labels=None,
        checkpoint_progress_label="checkpoint 2/11 step=2000",
    )

    assert progress_labels == [
        "checkpoint 2/11 step=2000 | validation episode 1/2 traj=4",
        "checkpoint 2/11 step=2000 | validation episode 2/2 traj=9",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("validation episode 1/2 traj=4" in message for message in messages)
    assert any("validation episode 2/2 traj=9" in message for message in messages)


def test_joint_trajectory_plot_labels_visible_frame_axes(tmp_path, monkeypatch):
    saved_figures = []

    def capture_figure(figure, *_args, **_kwargs):
        saved_figures.append(figure)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure)
    gt = np.zeros((32, 4), dtype=np.float32)
    pred = np.ones((32, 4), dtype=np.float32)
    labels = ["left_arm[0]", "left_arm[1]", "right_arm[0]", "right_arm[1]"]

    plot_trajectory(
        gt,
        pred,
        labels,
        "test trajectory",
        tmp_path / "trajectory_joints.png",
        execution_horizon=16,
        goal="move the red cup",
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
    assert any(text.get_text() == "Goal: move the red cup" for text in saved_figures[0].texts)
    assert any(
        text.get_text() == "Goal: move the red cup"
        for axis in saved_figures[1].axes
        for text in axis.texts
    )


def test_checkpoint_summary_splits_magnitude_metrics_with_matching_colours(
    tmp_path, monkeypatch
):
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
        assert [text.get_text() for text in legend.get_texts()] == list(
            expected_metric_colours
        )


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
    frame = raw_action_frame(
        checkpoint_step_value=200,
        trajectory_id=7,
        ground_truth=ground_truth,
        prediction=prediction,
        labels=["left_arm[0]", "right_arm[0]"],
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

    saved_ground_truth, saved_prediction, saved_labels = raw_trajectory_arrays(saved, 7)
    assert saved_labels == ["left_arm[0]", "right_arm[0]"]
    np.testing.assert_allclose(saved_ground_truth, ground_truth)
    np.testing.assert_allclose(saved_prediction, prediction)


def test_plots_only_regenerates_validation_and_train_probe_trajectories(tmp_path, monkeypatch):
    output_dir = tmp_path / "evaluation"
    checkpoint_dir = output_dir / "checkpoint-10"
    validation_dataset = tmp_path / "validation"
    train_dataset = tmp_path / "train"
    _write_episode_metadata(validation_dataset, [["validate the red cup"]])
    _write_episode_metadata(train_dataset, [["train with the blue cup"]])

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
        lambda *_args: trajectory_plot_calls.append((_args[4], _args[6])),
    )
    monkeypatch.setattr(
        evaluate_checkpoints,
        "plot_error_heatmap",
        lambda *_args: heatmap_calls.append((_args[3], _args[4])),
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
    assert trajectory_plot_calls == [
        (checkpoint_dir / "trajectory_0000_joints.png", "validate the red cup"),
        (
            output_dir
            / "right_arm_hand"
            / "checkpoint-10"
            / "trajectory_0000_joints.png",
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
    assert (output_dir / "right_arm_hand" / "metrics_by_checkpoint.csv").is_file()
    assert (output_dir / "right_arm_hand" / "metrics_per_joint.csv").is_file()


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
