import json

import numpy as np
import pandas as pd
import pytest
from scripts.analysis_tools.compare_evaluation_bootstrap import EvaluationInput, compare_evaluations


def _write_metrics(directory, rows):
    directory.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(directory / "metrics_per_episode.csv", index=False)


def _rows(step, mae, *, frames=(10, 20, 30), trajectories=(0, 1, 2)):
    return [
        {
            "split": "validation",
            "checkpoint_step": step,
            "trajectory": trajectory,
            "frames": frame_count,
            "mae": value,
            "mse": value**2,
        }
        for trajectory, frame_count, value in zip(trajectories, frames, mae, strict=True)
    ]


def test_compares_latest_common_checkpoint_with_paired_episode_bootstrap(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    c_dir = tmp_path / "c"
    _write_metrics(
        a_dir,
        _rows(100, (5.0, 5.0, 5.0))
        + _rows(200, (1.0, 2.0, 3.0))
        + _rows(200, (9.0,), frames=(40,), trajectories=(3,)),
    )
    _write_metrics(
        b_dir,
        _rows(100, (6.0, 6.0, 6.0)) + _rows(200, (2.0, 2.0, 4.0)) + _rows(300, (1.0, 1.0, 1.0)),
    )
    _write_metrics(
        c_dir,
        _rows(100, (7.0, 7.0, 7.0)) + _rows(200, (0.5, 2.5, 3.5)),
    )
    evaluations = [
        EvaluationInput("a", a_dir),
        EvaluationInput("b", b_dir),
        EvaluationInput("c", c_dir),
    ]

    result = compare_evaluations(
        evaluations,
        tmp_path / "comparison",
        resamples=2_000,
        seed=7,
    )

    assert result.checkpoint_step == 200
    assert result.episode_count == 3
    model_intervals = pd.read_csv(result.model_intervals_path)
    a_mae = model_intervals.query("model == 'a' and metric == 'mae'").iloc[0]
    assert a_mae["episode_mean"] == pytest.approx(2.0)
    assert a_mae["ci_lower"] <= a_mae["episode_mean"] <= a_mae["ci_upper"]

    pairwise = pd.read_csv(result.pairwise_intervals_path)
    a_vs_b = pairwise.query("model_a == 'a' and model_b == 'b' and metric == 'mae'").iloc[0]
    assert a_vs_b["mean_difference_a_minus_b"] == pytest.approx(-2.0 / 3.0)
    assert a_vs_b["model_a_episode_wins"] == 2
    assert a_vs_b["model_b_episode_wins"] == 0
    assert a_vs_b["episode_ties"] == 1

    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["trajectories"] == [0, 1, 2]
    assert metadata["evaluations"][0]["dropped_non_common_trajectories"] == [3]

    repeated = compare_evaluations(
        evaluations,
        tmp_path / "comparison_repeated",
        resamples=2_000,
        seed=7,
    )
    pd.testing.assert_frame_equal(
        pairwise,
        pd.read_csv(repeated.pairwise_intervals_path),
    )


def test_explicit_checkpoint_must_exist_in_every_evaluation(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _write_metrics(a_dir, _rows(100, (1.0, 2.0, 3.0)) + _rows(200, (1.0, 2.0, 3.0)))
    _write_metrics(b_dir, _rows(100, (1.0, 2.0, 3.0)))

    with pytest.raises(ValueError, match="Checkpoint 200 is not present in every evaluation"):
        compare_evaluations(
            [EvaluationInput("a", a_dir), EvaluationInput("b", b_dir)],
            tmp_path / "comparison",
            checkpoint_step=200,
            resamples=10,
        )


def test_rejects_different_frame_counts_for_paired_episode(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _write_metrics(a_dir, _rows(200, (1.0, 2.0, 3.0)))
    _write_metrics(b_dir, _rows(200, (1.0, 2.0, 3.0), frames=(10, 21, 30)))

    with pytest.raises(ValueError, match=r"different frame counts.*\[1\]"):
        compare_evaluations(
            [EvaluationInput("a", a_dir), EvaluationInput("b", b_dir)],
            tmp_path / "comparison",
            resamples=10,
        )


def test_pairwise_interval_is_computed_from_paired_differences(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    _write_metrics(a_dir, _rows(200, (1.0, 100.0, 1.0)))
    _write_metrics(b_dir, _rows(200, (2.0, 101.0, 2.0)))

    result = compare_evaluations(
        [EvaluationInput("a", a_dir), EvaluationInput("b", b_dir)],
        tmp_path / "comparison",
        metrics=("mae",),
        resamples=100,
        seed=9,
    )
    row = pd.read_csv(result.pairwise_intervals_path).iloc[0]

    assert row["mean_difference_a_minus_b"] == pytest.approx(-1.0)
    assert np.asarray([row["ci_lower"], row["ci_upper"]]) == pytest.approx([-1.0, -1.0])
    assert bool(row["ci_excludes_zero"])
