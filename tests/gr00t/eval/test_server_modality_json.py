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

from gr00t.data.depth_colormap import fixed_turbo_depth_colormap_contract
from gr00t.data.types import ModalityConfig
import gr00t.eval.run_gr00t_server as server_module
from gr00t.eval.run_gr00t_server import (
    ServerConfig,
    _load_checkpoint_action_output_contract,
    _load_deployment_dataset_contract,
    _load_deployment_task_contract,
    _load_json_modality_configs,
    _validate_checkpoint_dataset_vision_contract,
    _verify_checkpoint_dataset_path,
    main,
)
import gr00t.policy.gr00t_policy as policy_module
import pytest


def _late_fusion_vision_contract(
    auxiliary_key, stage="pre_vision_language_adapter"
):
    input_dim, output_dim = {
        "pre_vision_language_adapter": (2048, 1024),
        "post_vision_language_adapter": (4096, 2048),
    }[stage]
    parameters_per_adapter = input_dim * output_dim + output_dim
    return {
        "version": 2,
        "mode": "post_vision_late_fusion",
        "input_channels": 3,
        "channel_layout": ["ego_view:0", "ego_view:1", "ego_view:2"],
        "patch_embed_init": "original_rgb",
        "patch_embed_trainable": False,
        "wire_video_keys": ["ego_view", auxiliary_key],
        "post_vision_fusion": {
            "version": 1,
            "adapter_type": "concat_linear",
            "adapter_bias": True,
            "adapter_bias_init": "zeros",
            "adapter_count": 4,
            "deepstack_layers": [5, 11, 17],
            "adapter_init": "rgb50_geo50",
            "rgb_init_scale": 0.5,
            "geometry_init_scale": 0.5,
            "stage": stage,
            "adapter_input_dim": input_dim,
            "adapter_output_dim": output_dim,
            "parameters_per_adapter": parameters_per_adapter,
            "total_adapter_parameters": 4 * parameters_per_adapter,
            "source_keys": ["ego_view", auxiliary_key],
        },
    }


def _write_tasks(dataset, instructions):
    meta = dataset / "meta"
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["total_tasks"] = len(instructions)
    info_path.write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": instruction}) + "\n"
            for index, instruction in enumerate(instructions)
        )
    )


def test_load_deployment_task_contract_has_exact_ordered_allowlist_and_hash(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    instructions = ["pick up the red cup.", "put down the red cup."]
    _write_tasks(dataset, instructions)

    contract = _load_deployment_task_contract(dataset)

    assert contract == {
        "schema_version": 1,
        "instructions": instructions,
        "sha256": "1dc8e65b8210cd017ad48081d860208805ad4638785496059eef743fd1d633e7",
    }


@pytest.mark.parametrize(
    ("records", "total_tasks", "message"),
    [
        ([{"task_index": 1, "task": "pick"}], 1, "contiguous from zero"),
        ([{"task_index": 0, "task": "pick"}], 2, "record count"),
        (
            [
                {"task_index": 0, "task": "same"},
                {"task_index": 1, "task": "same"},
            ],
            2,
            "duplicate instruction",
        ),
        ([{"task_index": 0, "task": "bad\ntext"}], 1, "printable"),
    ],
)
def test_deployment_task_contract_rejects_ambiguous_or_malformed_tasks(
    tmp_path, records, total_tasks, message
):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_tasks": total_tasks}))
    (meta / "tasks.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match=message):
        _load_deployment_task_contract(dataset)


def _write_two_joint_modality(dataset):
    (dataset / "meta" / "modality.json").write_text(
        json.dumps(
            {
                "state": {"joints": {"start": 0, "end": 2}},
                "action": {"joints": {"start": 0, "end": 2}},
            }
        )
    )


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
                    "observation.state": {
                        "shape": [2],
                        "names": [["joint_a", "joint_b"]],
                    },
                    "action": {"shape": [2], "names": [["joint_a", "joint_b"]]},
                    "observation.images.ego_view": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                },
            }
        )
    )
    _write_two_joint_modality(dataset)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["robot_type"] == "TestBot"
    assert contract["fps"] == 30.0
    assert contract["observation_state_names"] == ["joint_a", "joint_b"]
    assert contract["action_names"] == ["joint_a", "joint_b"]
    assert contract["observation_state_shape"] == [2]
    assert contract["action_shape"] == [2]
    assert contract["state_layout"] == {"joints": {"start": 0, "end": 2, "dim": 2}}
    assert contract["action_layout"] == {"joints": {"start": 0, "end": 2, "dim": 2}}
    assert contract["ego_view_shape"] == [480, 640, 3]
    assert contract["video_shapes"] == {"ego_view": [480, 640, 3]}
    assert "depth_encoding" not in contract
    assert len(contract["sha256"]) == 64


@pytest.mark.parametrize(
    ("state_layout", "message"),
    [
        ({"first": {"start": 0, "end": 1}}, "ends at 1, expected 2"),
        (
            {
                "first": {"start": 0, "end": 2},
                "overlap": {"start": 1, "end": 2},
            },
            "exactly partition",
        ),
        ({"too_long": {"start": 0, "end": 3}}, "0 <= start < end <= 2"),
    ],
)
def test_deployment_contract_rejects_invalid_state_layout(tmp_path, state_layout, message):
    dataset = _write_deployment_dataset(tmp_path)
    modality_path = dataset / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())
    modality["state"] = state_layout
    modality_path.write_text(json.dumps(modality))

    with pytest.raises(ValueError, match=message):
        _load_deployment_dataset_contract(dataset)


def test_deployment_contract_rejects_name_dimension_mismatch(tmp_path):
    dataset = _write_deployment_dataset(tmp_path)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [3]
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="one non-empty name per dimension"):
        _load_deployment_dataset_contract(dataset)


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
            "observation.state": {
                "shape": [2],
                "names": [["joint_a", "joint_b"]],
            },
            "action": {"shape": [2], "names": [["joint_a", "joint_b"]]},
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
    _write_two_joint_modality(dataset)
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


def _surface_normals_encoding_v2():
    encoding = _surface_normals_encoding()
    encoding["encoding_version"] = 2
    encoding["depth_valid_range_m"] = {
        "near_m": 0.25,
        "far_m": 1.0,
        "inclusive": True,
        "required_samples": ["center", "left", "right", "up", "down"],
    }
    return encoding


def _camera_calibration_identity():
    return {
        "source": "converter.profile.d435i-254322071415",
        "schema": "realsense_rgbd_calibration.v1",
        "camera": {
            "model": "Intel RealSense D435I",
            "serial": "254322071415",
            "product_id": "0B3A",
            "firmware": "5.15.1.55",
        },
        "fingerprint": f"sha256:{'a' * 64}",
    }


def _write_surface_normals_dataset(tmp_path, *, encoding=None, include_metadata=True):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    info = {
        "robot_type": "TestBot",
        "fps": 30,
        "features": {
            "observation.state": {
                "shape": [2],
                "names": [["joint_a", "joint_b"]],
            },
            "action": {"shape": [2], "names": [["joint_a", "joint_b"]]},
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
    _write_two_joint_modality(dataset)
    return dataset


def test_load_deployment_dataset_contract_includes_surface_normals(tmp_path):
    dataset = _write_surface_normals_dataset(tmp_path)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["video_shapes"]["surface_normals_view"] == [480, 640, 3]
    assert contract["surface_normals_encoding"] == _surface_normals_encoding()
    assert len(contract["sha256"]) == 64


def test_load_deployment_dataset_contract_accepts_masked_surface_normals_v2(tmp_path):
    dataset = _write_surface_normals_dataset(
        tmp_path,
        encoding=_surface_normals_encoding_v2(),
    )

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["surface_normals_encoding"] == _surface_normals_encoding_v2()


@pytest.mark.parametrize(
    ("auxiliary_key", "encoding_key"),
    [
        ("depth_gray_view", "depth_encoding"),
        ("surface_normals_view", "surface_normals_encoding"),
    ],
)
@pytest.mark.parametrize(
    "stage", ["pre_vision_language_adapter", "post_vision_language_adapter"]
)
def test_late_fusion_checkpoint_contract_binds_aligned_deployment_views(
    auxiliary_key, encoding_key, stage
):
    vision_contract = _late_fusion_vision_contract(auxiliary_key, stage)
    dataset_contract = {
        "video_shapes": {
            "ego_view": [480, 640, 3],
            auxiliary_key: [480, 640, 3],
        },
        encoding_key: {"encoding": "test"},
    }

    _validate_checkpoint_dataset_vision_contract(vision_contract, dataset_contract)


def test_late_fusion_checkpoint_contract_rejects_unaligned_deployment_views():
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    dataset_contract = {
        "video_shapes": {
            "ego_view": [480, 640, 3],
            "depth_gray_view": [240, 320, 3],
        },
        "depth_encoding": {"encoding": "test"},
    }

    with pytest.raises(ValueError, match="identical HWC shapes"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, dataset_contract)


def test_late_fusion_checkpoint_contract_requires_auxiliary_encoding_metadata():
    vision_contract = _late_fusion_vision_contract("surface_normals_view")
    dataset_contract = {
        "video_shapes": {
            "ego_view": [480, 640, 3],
            "surface_normals_view": [480, 640, 3],
        }
    }

    with pytest.raises(ValueError, match="surface_normals_encoding metadata"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, dataset_contract)


def test_fixed_turbo_checkpoint_requires_exact_gray_depth_dataset_contract():
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["depth_colormap"] = fixed_turbo_depth_colormap_contract()
    dataset_contract = {
        "video_shapes": {
            "ego_view": [480, 640, 3],
            "depth_gray_view": [480, 640, 3],
        },
        "depth_encoding": {
            "encoding": "linear_grayscale_replicated_rgb",
            "invalid_value": 0,
            "valid_value_range": [1, 255],
        },
    }

    _validate_checkpoint_dataset_vision_contract(vision_contract, dataset_contract)

    for field, bad_value in (
        ("encoding", "frame_normalized_gray"),
        ("invalid_value", 255),
        ("valid_value_range", [0, 255]),
    ):
        malformed = dict(dataset_contract)
        malformed["depth_encoding"] = dict(dataset_contract["depth_encoding"])
        malformed["depth_encoding"][field] = bad_value
        with pytest.raises(ValueError, match=f"mismatch for {field}"):
            _validate_checkpoint_dataset_vision_contract(vision_contract, malformed)

    with pytest.raises(ValueError, match="requires deployment depth_encoding metadata"):
        _validate_checkpoint_dataset_vision_contract(
            vision_contract,
            {"video_shapes": dataset_contract["video_shapes"]},
        )


def test_late_fusion_checkpoint_contract_rejects_legacy_fusion_stage_name():
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["post_vision_fusion"]["stage"] = "pre_merger"

    with pytest.raises(ValueError, match="must declare stage"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, {})


def test_late_fusion_checkpoint_contract_rejects_stage_dimension_mismatch():
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["post_vision_fusion"]["stage"] = (
        "post_vision_language_adapter"
    )

    with pytest.raises(ValueError, match="architecture mismatch for adapter_input_dim"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, {})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_bias", False),
        ("adapter_bias_init", "ones"),
        ("deepstack_layers", [4, 10, 16]),
        ("parameters_per_adapter", 0),
        ("total_adapter_parameters", 0),
    ],
)
def test_late_fusion_checkpoint_contract_rejects_architecture_metadata_mismatch(
    field, value
):
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["post_vision_fusion"][field] = value

    with pytest.raises(ValueError, match=f"architecture mismatch for {field}"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, {})


def test_late_fusion_checkpoint_contract_requires_frozen_patch_embedding():
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["patch_embed_trainable"] = True

    with pytest.raises(ValueError, match="patch_embed_trainable=false"):
        _validate_checkpoint_dataset_vision_contract(vision_contract, {})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_init", "rgb90_geo10", "adapter_init 'rgb50_geo50'"),
        ("rgb_init_scale", 0.9, "RGB=0.5 and geometry=0.5"),
        ("geometry_init_scale", 0.1, "RGB=0.5 and geometry=0.5"),
    ],
)
def test_late_fusion_checkpoint_contract_rejects_another_initialization(
    field, value, message
):
    vision_contract = _late_fusion_vision_contract("depth_gray_view")
    vision_contract["post_vision_fusion"][field] = value

    with pytest.raises(ValueError, match=message):
        _validate_checkpoint_dataset_vision_contract(vision_contract, {})


def test_existing_vision_modes_keep_existing_dataset_validation_behavior():
    _validate_checkpoint_dataset_vision_contract(
        {"version": 1, "mode": "early_channel_fusion"},
        {},
    )


def test_surface_normals_contract_preserves_camera_calibration_identity(tmp_path):
    encoding = _surface_normals_encoding()
    encoding["camera_calibration"] = _camera_calibration_identity()
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["surface_normals_encoding"]["camera_calibration"] == (
        _camera_calibration_identity()
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(source=""), "source"),
        (lambda value: value.update(schema="wrong"), "schema"),
        (lambda value: value["camera"].pop("firmware"), "camera"),
        (lambda value: value.update(fingerprint="sha256:ABC"), "fingerprint"),
    ],
)
def test_surface_normals_contract_rejects_bad_camera_calibration(tmp_path, mutate, message):
    encoding = _surface_normals_encoding()
    calibration = _camera_calibration_identity()
    mutate(calibration)
    encoding["camera_calibration"] = calibration
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    with pytest.raises(ValueError, match=message):
        _load_deployment_dataset_contract(dataset)


def test_surface_normals_view_requires_versioned_encoding_metadata(tmp_path):
    dataset = _write_surface_normals_dataset(tmp_path, include_metadata=False)

    with pytest.raises(ValueError, match="requires a surface_normals_encoding object"):
        _load_deployment_dataset_contract(dataset)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(encoding_version=3), "encoding_version"),
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("depth_valid_range_m"), "depth_valid_range_m"),
        (
            lambda value: value["depth_valid_range_m"].update(inclusive=False),
            "inclusive",
        ),
        (
            lambda value: value["depth_valid_range_m"].update(near_m=1.0),
            "near_m < far_m",
        ),
        (
            lambda value: value["depth_valid_range_m"].update(extra=True),
            "exactly",
        ),
        (
            lambda value: value["depth_valid_range_m"].update(required_samples=["center"]),
            "required_samples must be exactly",
        ),
        (
            lambda value: value.update(range_semantics="center_only"),
            "unsupported fields",
        ),
    ],
)
def test_surface_normals_v2_rejects_malformed_depth_valid_range(tmp_path, mutate, message):
    encoding = _surface_normals_encoding_v2()
    mutate(encoding)
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    with pytest.raises(ValueError, match=message):
        _load_deployment_dataset_contract(dataset)


@pytest.mark.parametrize(
    "range_value",
    [
        None,
        {
            "near_m": 0.25,
            "far_m": 1.0,
            "inclusive": True,
            "required_samples": ["center", "left", "right", "up", "down"],
        },
    ],
)
def test_surface_normals_v1_rejects_v2_range_field(tmp_path, range_value):
    encoding = _surface_normals_encoding()
    encoding["depth_valid_range_m"] = range_value
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    with pytest.raises(ValueError, match="version 1.*depth_valid_range_m"):
        _load_deployment_dataset_contract(dataset)


def test_surface_normals_v2_range_must_match_canonical_depth_contract(tmp_path):
    encoding = _surface_normals_encoding_v2()
    encoding["depth_valid_range_m"]["far_m"] = 0.9
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)

    with pytest.raises(ValueError, match="must exactly match.*canonical depth encoding"):
        _load_deployment_dataset_contract(dataset)


def test_surface_normals_v2_range_must_match_declared_depth_view(tmp_path):
    encoding = _surface_normals_encoding_v2()
    encoding["depth_valid_range_m"]["far_m"] = 0.9
    dataset = _write_surface_normals_dataset(tmp_path, encoding=encoding)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.depth_gray_view"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
    }
    info["depth_encoding"] = {
        "source_key": "depth_0",
        "feature_key": "observation.images.depth_gray_view",
        "encoding": "linear_grayscale_replicated_rgb",
        "default_scale_m_per_unit": 0.001,
        "near_m": 0.25,
        "far_m": 1.0,
        "invalid_value": 0,
        "valid_value_range": [1, 255],
    }
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="must exactly match.*dataset depth_encoding"):
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


def test_checkpoint_bound_task_contract_is_advertised_in_policy_metadata(tmp_path, monkeypatch):
    captured = {}
    model = tmp_path / "checkpoint"
    dataset = tmp_path / "dataset"
    model.mkdir()
    dataset.mkdir()
    expected_task_contract = {
        "schema_version": 1,
        "instructions": ["pick up the red cup.", "put down the red cup."],
        "sha256": "1dc8e65b8210cd017ad48081d860208805ad4638785496059eef743fd1d633e7",
    }

    class DummyPolicy:
        @staticmethod
        def get_vision_input_contract():
            return {"version": 1}

    class DummyServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self):
            pass

    monkeypatch.setattr(server_module, "Gr00tPolicy", lambda **_kwargs: DummyPolicy())
    monkeypatch.setattr(server_module, "PolicyServer", DummyServer)
    monkeypatch.setattr(server_module, "_verify_checkpoint_dataset_path", lambda *_args: None)
    monkeypatch.setattr(
        server_module,
        "_load_checkpoint_action_output_contract",
        lambda *_args: {"semantics": "absolute_joint_position"},
    )
    monkeypatch.setattr(
        server_module,
        "_load_deployment_dataset_contract",
        lambda *_args: {"robot_type": "test"},
    )
    monkeypatch.setattr(
        server_module,
        "_load_deployment_task_contract",
        lambda *_args: expected_task_contract,
    )

    main(
        ServerConfig(
            model_path=str(model),
            deployment_dataset_path=str(dataset),
            device="cpu",
        )
    )

    assert captured["policy_metadata"]["task_contract"] == expected_task_contract


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
