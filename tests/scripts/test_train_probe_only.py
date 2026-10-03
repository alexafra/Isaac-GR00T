"""CPU tests for isolated training diagnostics; no checkpoint inference is run."""
from argparse import Namespace
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

if os.environ.get("EVALUATOR_TEST_MODULE"):
    spec = importlib.util.spec_from_file_location("staged_evaluate_checkpoints", os.environ["EVALUATOR_TEST_MODULE"])
    evaluate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluate
    spec.loader.exec_module(evaluate)
else:
    from scripts.analysis_tools import evaluate_checkpoints as evaluate


def args_for(tmp_path, **overrides):
    result = dict(run_dir=tmp_path / "run", output_dir=tmp_path / "run/probe_only",
                  train_dataset_path=tmp_path / "train", dataset_path=tmp_path / "validation",
                  execution_horizon=8, plots_only=False, train_probe_only=True,
                  traj_ids=None, train_traj_ids=None, train_probe_episodes=5, train_probe_seed=42)
    result.update(overrides)
    return Namespace(**result)


def test_probe_only_plan_never_contains_validation(tmp_path):
    args = args_for(tmp_path)
    assert evaluate.evaluation_probe_specs(args) == [("train_probe", args.train_dataset_path, None, 5, 42)]
    args.train_probe_only = False
    assert [row[0] for row in evaluate.evaluation_probe_specs(args)] == ["validation", "train_probe"]


@pytest.mark.parametrize("omission", ["--train-dataset-path", "--output-dir"])
def test_probe_only_requires_train_and_explicit_output(tmp_path, monkeypatch, omission):
    argv = ["evaluate", "--run-dir", str(tmp_path / "run"), "--train-probe-only"]
    for flag, value in (("--train-dataset-path", tmp_path / "train"), ("--output-dir", tmp_path / "probe")):
        if flag != omission:
            argv.extend([flag, str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        evaluate.parse_args()


def test_validation_required_only_for_default_mode(tmp_path, monkeypatch):
    argv = ["evaluate", "--run-dir", str(tmp_path / "run")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        evaluate.parse_args()
    monkeypatch.setattr(sys, "argv", argv + ["--train-probe-only", "--train-dataset-path", str(tmp_path / "train"),
                                            "--output-dir", str(tmp_path / "new_probe")])
    assert evaluate.parse_args().dataset_path is None


@pytest.mark.parametrize("relative", ["evaluation_exec_hor_8", "evaluation_exec_hor_8/nested",
                                     "test_evaluation_exec_hor_8", "normalized_action_metrics_exec_hor_8",
                                     "test_normalized_action_metrics_exec_hor_8", "."])
def test_canonical_output_trees_protected(tmp_path, relative):
    args = args_for(tmp_path, output_dir=tmp_path / "run" / relative)
    with pytest.raises(ValueError, match="path-disjoint"):
        evaluate.validate_train_probe_output(args)


def test_populated_output_and_symlink_to_canonical_refused(tmp_path):
    args = args_for(tmp_path)
    args.output_dir.mkdir(parents=True)
    sentinel = args.output_dir / "do_not_change.csv"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match="populated"):
        evaluate.validate_train_probe_output(args)
    assert sentinel.read_text() == "original"
    target = args.run_dir / "test_evaluation_exec_hor_8"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    args.output_dir = alias
    with pytest.raises(ValueError, match="path-disjoint"):
        evaluate.validate_train_probe_output(args)


def test_train_only_plots_never_select_best_checkpoint(tmp_path, monkeypatch):
    summary = pd.DataFrame([dict(split="train_probe", checkpoint_step=5000, mae=.1)])
    joints = pd.DataFrame([dict(split="train_probe", checkpoint_step=5000, joint="left_arm[0]", mae=.1)])
    for name in ("plot_checkpoint_progress", "plot_checkpoint_metric_summary"):
        monkeypatch.setattr(evaluate, name, lambda *a, **k: None)
    monkeypatch.setattr(evaluate, "plot_best_checkpoint_joints", lambda *a, **k: pytest.fail("No training-based selection"))
    assert evaluate.plot_evaluation_summaries(tmp_path, summary, joints, ["left_arm[0]"]) is None


def test_main_training_only_writes_new_output_without_opening_validation(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    validation_metrics = args.run_dir / "evaluation_exec_hor_8/metrics_by_checkpoint.csv"
    test_metrics = args.run_dir / "test_evaluation_exec_hor_8/metrics_by_checkpoint.csv"
    for path in (validation_metrics, test_metrics):
        path.parent.mkdir(parents=True)
        path.write_text("unchanged held-out metrics\n")
    argv = ["evaluate", "--run-dir", str(args.run_dir), "--train-probe-only", "--train-dataset-path",
            str(args.train_dataset_path), "--output-dir", str(args.output_dir), "--execution-horizon", "8",
            "--skip-trajectory-plots", "--checkpoint-steps", "5000"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluate, "default_statistics_path", lambda run: tmp_path / "unused_stats.json")
    monkeypatch.setattr(evaluate.ActionErrorNormalizer, "from_file", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "find_evaluation_targets", lambda *a: [SimpleNamespace(
        step=5000, output_name="checkpoint-5000", model_path=tmp_path / "unused_model", processor_path=None)])
    policy = SimpleNamespace(model=SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=4)),
                             get_modality_config=lambda: {"action": SimpleNamespace(modality_keys=["left_arm"])})
    monkeypatch.setattr(evaluate, "Gr00tPolicy", lambda **kwargs: policy)
    monkeypatch.setattr(evaluate, "seed_inference", lambda seed: None)
    monkeypatch.setattr(evaluate.torch.cuda, "is_available", lambda: False)
    opened = []
    class Loader:
        def __init__(self, dataset_path, modality_configs):
            opened.append(Path(dataset_path))
            assert Path(dataset_path) == args.train_dataset_path
        def __len__(self):
            return 1
    monkeypatch.setattr(evaluate, "LeRobotEpisodeLoader", Loader)
    monkeypatch.setattr(evaluate, "select_trajectory_ids", lambda *a, **k: [0])
    monkeypatch.setattr(evaluate, "_load_episode_tasks", lambda *a: {0: ["goal"]})
    def fake_probe(**kwargs):
        assert kwargs["split"] == "train_probe"
        row = dict(split="train_probe", checkpoint_step=5000, mae=.1)
        return [row], row, [dict(row, joint="left_arm[0]")], ["left_arm[0]"]
    monkeypatch.setattr(evaluate, "evaluate_probe", fake_probe)
    monkeypatch.setattr(evaluate, "write_extended_evaluation_metrics", lambda **k: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(evaluate, "plot_evaluation_summaries", lambda *a, **k: None)
    monkeypatch.setattr(evaluate, "plot_right_evaluation_summaries", lambda *a, **k: None)
    monkeypatch.setattr(evaluate, "plot_horizon_position_metrics", lambda *a, **k: None)
    def normalized(**kwargs):
        assert kwargs["evaluation_dir"] == args.output_dir
        return {"scopes": args.output_dir / "normalized_metrics_by_scope.csv"}
    monkeypatch.setattr(evaluate, "write_normalized_metric_tables", normalized)
    evaluate.main()
    assert opened == [args.train_dataset_path]
    assert set(pd.read_csv(args.output_dir / "metrics_by_checkpoint.csv")["split"]) == {"train_probe"}
    for path in (validation_metrics, test_metrics):
        assert path.read_text() == "unchanged held-out metrics\n"
