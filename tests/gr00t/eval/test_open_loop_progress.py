import logging
from types import SimpleNamespace

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import open_loop_eval
import numpy as np
import pandas as pd


class _FakeLoader:
    def __init__(self) -> None:
        self.modality_configs = {
            "state": SimpleNamespace(modality_keys=["right_arm"]),
            "action": SimpleNamespace(modality_keys=["right_arm"], delta_indices=list(range(8))),
            "language": SimpleNamespace(modality_keys=["task"]),
        }
        values = [np.zeros(1, dtype=np.float32) for _ in range(17)]
        self.trajectory = pd.DataFrame({"state.right_arm": values, "action.right_arm": values})

    def __getitem__(self, _trajectory_id: int) -> pd.DataFrame:
        return self.trajectory


class _FakePolicy:
    def get_action(self, _observation):
        return {"right_arm": np.zeros((1, 8, 1), dtype=np.float32)}, {}


def test_open_loop_progress_identifies_context_and_reaches_100_percent(monkeypatch, caplog):
    monkeypatch.setattr(
        open_loop_eval.PolicyHorizonSpec,
        "from_modality_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        open_loop_eval,
        "extract_step_data",
        lambda *_args, **_kwargs: SimpleNamespace(
            states={"right_arm": np.zeros((1, 1), dtype=np.float32)},
            images={},
            text="move",
        ),
    )
    monkeypatch.setattr(
        open_loop_eval,
        "parse_observation_gr00t",
        lambda _observation, _config: {
            "state": {"right_arm": np.zeros((1, 1, 1), dtype=np.float32)},
            "language": {"task": [["move"]]},
        },
    )
    monkeypatch.setattr(open_loop_eval, "plot_trajectory_results", lambda **_kwargs: None)
    caplog.set_level(logging.INFO)

    open_loop_eval.evaluate_single_trajectory(
        policy=_FakePolicy(),
        loader=_FakeLoader(),
        traj_id=4,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        steps=17,
        execution_horizon=8,
        progress_label="checkpoint 2/11 step=2000 | validation episode 3/42 traj=4",
    )

    progress_messages = [
        record.getMessage() for record in caplog.records if "inference" in record.getMessage()
    ]
    assert len(progress_messages) == 3
    assert "checkpoint 2/11 step=2000" in progress_messages[0]
    assert "inference 1/3 complete" in progress_messages[0]
    assert "inference 3/3 complete" in progress_messages[-1]
    assert "frames 17-17/17 (100.0%)" in progress_messages[-1]


def test_open_loop_batches_inference_points_without_reordering(monkeypatch):
    loader = _FakeLoader()
    policy_batch_sizes = []
    policy_noise_seeds = []

    class BatchedPolicy:
        def get_action(self, observation, options=None):
            batch_size = observation["state"]["right_arm"].shape[0]
            policy_batch_sizes.append(batch_size)
            policy_noise_seeds.extend(options["noise_seeds"])
            return {"right_arm": np.zeros((batch_size, 8, 1), dtype=np.float32)}, {}

    monkeypatch.setattr(
        open_loop_eval.PolicyHorizonSpec,
        "from_modality_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        open_loop_eval,
        "extract_step_data",
        lambda *_args, **_kwargs: SimpleNamespace(
            states={"right_arm": np.zeros((1, 1), dtype=np.float32)},
            images={},
            text="move",
        ),
    )
    monkeypatch.setattr(
        open_loop_eval,
        "parse_observation_gr00t",
        lambda _observation, _config: {
            "state": {"right_arm": np.zeros((1, 1, 1), dtype=np.float32)},
            "language": {"task": [["move"]]},
        },
    )
    captured = {}
    monkeypatch.setattr(
        open_loop_eval,
        "plot_trajectory_results",
        lambda **kwargs: captured.update(kwargs),
    )

    mse, mae = open_loop_eval.evaluate_single_trajectory(
        policy=BatchedPolicy(),
        loader=loader,
        traj_id=4,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        steps=17,
        execution_horizon=8,
        inference_batch_size=2,
        inference_seed=42,
        trajectory=loader.trajectory,
    )

    assert policy_batch_sizes == [2, 1]
    assert policy_noise_seeds == [
        open_loop_eval.inference_noise_seed(42, 4, step) for step in (0, 8, 16)
    ]
    assert captured["pred_action_across_time"].shape == (17, 1)
    assert mse == 0
    assert mae == 0


def test_required_visual_frames_follow_open_loop_stride_and_temporal_offsets():
    modality_configs = {
        "video": SimpleNamespace(delta_indices=[0, 1]),
        "mask": SimpleNamespace(delta_indices=[0]),
    }

    indices = open_loop_eval.required_visual_frame_indices(
        trajectory_length=18,
        steps=17,
        execution_horizon=8,
        modality_configs=modality_configs,
    )

    np.testing.assert_array_equal(indices, [0, 1, 8, 9, 16, 17])
