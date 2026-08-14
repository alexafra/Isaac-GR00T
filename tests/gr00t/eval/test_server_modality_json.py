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
import gr00t.policy.gr00t_policy as policy_module
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
                    "observation.images.ego_view": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
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
    assert contract["video_shapes"] == {"ego_view": [480, 640, 3]}
    assert "depth_encoding" not in contract
    assert len(contract["sha256"]) == 64


def _write_deployment_dataset(tmp_path, *, reverse_video_order=False, depth_encoding=None):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    image_features = [
        (
            "observation.images.ego_view",
            {"dtype": "video", "shape": [480, 640, 3]},
        ),
        (
            "observation.images.depth_gray_view",
            {"dtype": "video", "shape": [480, 640, 3]},
        ),
    ]
    if reverse_video_order:
        image_features.reverse()
    info = {
        "robot_type": "TestBot",
        "fps": 30,
        "features": {
            "observation.state": {"names": [["joint_a", "joint_b"]]},
            "action": {"names": [["joint_a", "joint_b"]]},
            **dict(image_features),
        },
        "depth_encoding": depth_encoding
        or {
            "source_key": "depth_0",
            "feature_key": "observation.images.depth_gray_view",
            "encoding": "linear_grayscale_replicated_rgb",
            "default_scale_m_per_unit": 0.001,
            "near_m": 0.25,
            "far_m": 1.0,
            "invalid_value": 0,
            "valid_value_range": [1, 255],
        },
        # Existing datasets may only have the old raw sidecar. It is provenance,
        # not a prerequisite for serving the already-trained depth view.
        "raw_depth_encoding": {"scale_m_per_unit": 0.001},
    }
    (meta / "info.json").write_text(json.dumps(info))
    return dataset


def test_load_deployment_dataset_contract_includes_normalized_depth(tmp_path):
    dataset = _write_deployment_dataset(tmp_path)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["video_shapes"] == {
        "depth_gray_view": [480, 640, 3],
        "ego_view": [480, 640, 3],
    }
    assert contract["depth_encoding"] == {
        "source_key": "depth_0",
        "feature_key": "observation.images.depth_gray_view",
        "encoding": "linear_grayscale_replicated_rgb",
        "near_m": 0.25,
        "far_m": 1.0,
        "invalid_value": 0,
        "valid_value_range": [1, 255],
    }


def test_deployment_contract_hash_is_independent_of_feature_insertion_order(tmp_path):
    first = _write_deployment_dataset(tmp_path / "first")
    second = _write_deployment_dataset(tmp_path / "second", reverse_video_order=True)

    assert (
        _load_deployment_dataset_contract(first)["sha256"]
        == (_load_deployment_dataset_contract(second)["sha256"])
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_key", "observation.images.wrong", "feature_key"),
        ("near_m", 2.0, "near_m < far_m"),
        ("valid_value_range", [255, 1], "valid_value_range"),
    ],
)
def test_deployment_contract_rejects_malformed_depth_encoding(tmp_path, field, value, message):
    encoding = {
        "source_key": "depth_0",
        "feature_key": "observation.images.depth_gray_view",
        "encoding": "linear_grayscale_replicated_rgb",
        "near_m": 0.25,
        "far_m": 1.0,
        "invalid_value": 0,
        "valid_value_range": [1, 255],
    }
    encoding[field] = value
    dataset = _write_deployment_dataset(tmp_path, depth_encoding=encoding)

    with pytest.raises(ValueError, match=message):
        _load_deployment_dataset_contract(dataset)


def _surface_normals_encoding():
    return {
        "source_key": "depth_0",
        "aligned_to": "color_0",
        "feature_key": "observation.images.surface_normals_view",
        "encoding": "camera_xyz_uint8",
        "encoding_version": 1,
        "depth_scale_source": "episode.info.depth.scale_m_per_unit",
        "default_scale_m_per_unit": 0.001,
        "intrinsics": {
            "model": "pinhole",
            "width": 640,
            "height": 480,
            "fx": 605.421508789062,
            "fy": 605.590515136719,
            "cx": 321.856811523438,
            "cy": 242.249740600586,
        },
        "axis_order": ["x", "y", "z"],
        "coordinate_frame": "camera_optical_x_right_y_down_z_forward",
        "orientation": "camera_facing_dot_normal_point_lte_zero",
        "method": "central_difference_3d",
        "neighbor_offset_pixels": 1,
        "max_neighbor_depth_delta_m": 0.05,
        "invalid_value": [0, 0, 0],
        "valid_component_range": [1, 255],
    }


def _write_surface_normals_dataset(tmp_path, *, encoding=None, include_metadata=True):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    info = {
        "robot_type": "TestBot",
        "fps": 30,
        "features": {
            "observation.state": {"names": [["joint_a", "joint_b"]]},
            "action": {"names": [["joint_a", "joint_b"]]},
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.surface_normals_view": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
        },
    }
    if include_metadata:
        info["surface_normals_encoding"] = encoding or _surface_normals_encoding()
    (meta / "info.json").write_text(json.dumps(info))
    return dataset


def test_load_deployment_dataset_contract_includes_surface_normals(tmp_path):
    dataset = _write_surface_normals_dataset(tmp_path)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["video_shapes"]["surface_normals_view"] == [480, 640, 3]
    assert contract["surface_normals_encoding"] == _surface_normals_encoding()
    assert len(contract["sha256"]) == 64


def test_surface_normals_view_requires_versioned_encoding_metadata(tmp_path):
    dataset = _write_surface_normals_dataset(tmp_path, include_metadata=False)

    with pytest.raises(ValueError, match="requires a surface_normals_encoding object"):
        _load_deployment_dataset_contract(dataset)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(encoding_version=2), "encoding_version"),
        (
            lambda value: value["intrinsics"].update(width=848),
            "intrinsics resolution",
        ),
        (
            lambda value: value.update(feature_key="observation.images.wrong"),
            "feature_key",
        ),
        (
            lambda value: value.update(max_neighbor_depth_delta_m=0.0),
            "max_neighbor_depth_delta_m",
        ),
    ],
)
def test_deployment_contract_rejects_malformed_surface_normals_encoding(tmp_path, mutate, message):
    encoding = _surface_normals_encoding()
    mutate(encoding)
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    with pytest.raises(ValueError, match=message):
        _load_deployment_dataset_contract(dataset)


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
    vision_contract = {"version": 1, "mode": "separate_views"}  # earlyfusion

    class DummyServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self):
            pass

    monkeypatch.setattr(server_module, "Gr00tPolicy", lambda **_kwargs: type("Policy", (), {"get_vision_input_contract": lambda _self: vision_contract})())  # fmt: skip  # earlyfusion
    monkeypatch.setattr(server_module, "PolicyServer", DummyServer)
    monkeypatch.setattr(
        server_module,
        "_load_checkpoint_action_output_contract",
        lambda *_args: pytest.fail("deployment-only processor metadata was loaded"),
    )

    main(ServerConfig(model_path="nvidia/GR00T-N1.7-3B", device="cpu"))

    assert "action_output_contract" not in captured["policy_metadata"]
    assert captured["policy_metadata"]["vision_input_contract"] == vision_contract  # earlyfusion
    assert captured["policy_metadata"]["rtc"] == {
        "protocol_version": 1,
        "physical_action_tail": True,
        "backend": "pytorch",
    }


def test_sim_policy_wrapper_does_not_advertise_physical_tail_rtc(monkeypatch):
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
    monkeypatch.setattr(policy_module, "Gr00tSimPolicyWrapper", lambda policy: policy)
    monkeypatch.setattr(server_module, "PolicyServer", DummyServer)

    main(
        ServerConfig(
            model_path="nvidia/GR00T-N1.7-3B",
            device="cpu",
            use_sim_policy_wrapper=True,
        )
    )

    assert "rtc" not in captured["policy_metadata"]
