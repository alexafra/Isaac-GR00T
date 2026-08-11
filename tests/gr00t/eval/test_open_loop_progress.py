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
        lambda observation, _config: observation,
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
