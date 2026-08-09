# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from gr00t.data.types import ModalityConfig
import gr00t.eval.run_gr00t_server as server_module
from gr00t.eval.run_gr00t_server import (
    ServerConfig,
    _load_checkpoint_action_output_contract,
    _load_deployment_dataset_contract,
    _load_json_modality_configs,
    _verify_checkpoint_dataset_path,
    main,
)
import pytest


def test_dataset_layout_json_raises_actionable(tmp_path):
    # A dataset's meta/modality.json (start/end layout), not ModalityConfig fields.
    p = tmp_path / "modality.json"
    p.write_text(json.dumps({"state": {"single_arm": {"start": 0, "end": 5}}}))

    with pytest.raises(ValueError) as exc:
        _load_json_modality_configs(p)

    msg = str(exc.value)
    assert "ModalityConfig" in msg
    assert ".py" in msg


def test_valid_modality_config_json_loads(tmp_path):
    payload = {"action": {"delta_indices": [0, 1], "modality_keys": ["x"]}}
    p = tmp_path / "mc.json"
    p.write_text(json.dumps(payload))

    configs = _load_json_modality_configs(p)

    assert set(configs) == set(payload)
    assert isinstance(configs["action"], ModalityConfig)
    assert configs["action"].delta_indices == payload["action"]["delta_indices"]
    assert configs["action"].modality_keys == payload["action"]["modality_keys"]


def test_load_deployment_dataset_contract(tmp_path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "TestBot",
                "fps": 30,
                "features": {
                    "observation.state": {"names": [["joint_a", "joint_b"]]},
                    "action": {"names": [["joint_a", "joint_b"]]},
                    "observation.images.ego_view": {"shape": [480, 640, 3]},
                },
            }
        )
    )

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["robot_type"] == "TestBot"
    assert contract["fps"] == 30.0
    assert contract["observation_state_names"] == ["joint_a", "joint_b"]
    assert contract["action_names"] == ["joint_a", "joint_b"]
    assert contract["ego_view_shape"] == [480, 640, 3]
    assert len(contract["sha256"]) == 64


def test_deployment_dataset_must_match_checkpoint_training_path(tmp_path):
    model = tmp_path / "checkpoint"
    experiment = model / "experiment_cfg"
    experiment.mkdir(parents=True)
    trained_dataset = tmp_path / "trained"
    wrong_dataset = tmp_path / "wrong"
    (experiment / "config.yaml").write_text(
        """
data:
  datasets:
    - embodiment_tag: new_embodiment
      dataset_paths:
        - %s
""".strip()
        % trained_dataset
    )

    _verify_checkpoint_dataset_path(model, "new_embodiment", trained_dataset)
    with pytest.raises(ValueError, match="single training dataset"):
        _verify_checkpoint_dataset_path(model, "new_embodiment", wrong_dataset)


def _write_processor_config(tmp_path, *, use_relative_action, representations):
    model = tmp_path / "checkpoint"
    model.mkdir()
    action_keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
    (model / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "use_relative_action": use_relative_action,
                    "modality_configs": {
                        "new_embodiment": {
                            "action": {
                                "modality_keys": action_keys,
                                "action_configs": [
                                    {"rep": representation} for representation in representations
                                ],
                            }
                        }
                    },
                }
            }
        )
    )
    return model


def test_checkpoint_action_output_contract_describes_relative_decode(tmp_path):
    model = _write_processor_config(
        tmp_path,
        use_relative_action=True,
        representations=["RELATIVE", "RELATIVE", "ABSOLUTE", "ABSOLUTE"],
    )

    contract = _load_checkpoint_action_output_contract(model, "new_embodiment")

    assert contract == {
        "semantics": "absolute_joint_position",
        "use_relative_action": True,
        "relative_keys_decoded_to_absolute": ["left_arm", "right_arm"],
    }


def test_checkpoint_action_output_contract_rejects_undecoded_relative_actions(tmp_path):
    model = _write_processor_config(
        tmp_path,
        use_relative_action=False,
        representations=["RELATIVE", "RELATIVE", "ABSOLUTE", "ABSOLUTE"],
    )

    contract = _load_checkpoint_action_output_contract(model, "new_embodiment")

    assert contract == {
        "semantics": "checkpoint_native",
        "use_relative_action": False,
        "relative_keys_decoded_to_absolute": [],
    }


def test_checkpoint_action_output_contract_recognizes_native_absolute_actions(tmp_path):
    model = _write_processor_config(
        tmp_path,
        use_relative_action=False,
        representations=["ABSOLUTE"] * 4,
    )

    assert _load_checkpoint_action_output_contract(model, "new_embodiment")["semantics"] == (
        "absolute_joint_position"
    )


def test_checkpoint_action_output_contract_requires_boolean_processor_flag(tmp_path):
    model = _write_processor_config(
        tmp_path,
        use_relative_action="false",
        representations=["RELATIVE", "RELATIVE", "ABSOLUTE", "ABSOLUTE"],
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        _load_checkpoint_action_output_contract(model, "new_embodiment")


def test_hugging_face_model_id_does_not_require_local_deployment_metadata(monkeypatch):
    captured = {}

    class DummyServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self):
            pass

    monkeypatch.setattr(server_module, "Gr00tPolicy", lambda **_kwargs: object())
    monkeypatch.setattr(server_module, "PolicyServer", DummyServer)
    monkeypatch.setattr(
        server_module,
        "_load_checkpoint_action_output_contract",
        lambda *_args: pytest.fail("deployment-only processor metadata was loaded"),
    )

    main(ServerConfig(model_path="nvidia/GR00T-N1.7-3B", device="cpu"))

    assert "action_output_contract" not in captured["policy_metadata"]
