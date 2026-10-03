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

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

from gr00t.data.depth_colormap import fixed_turbo_depth_colormap_contract
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.eval.live_model_aliases import DEFAULT_BLIND_STUDY
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro
import yaml


DEFAULT_MODEL_SERVER_PORT = 5555
IMAGE_FEATURE_PREFIX = "observation.images."
SURFACE_NORMALS_VIEW = "surface_normals_view"
SURFACE_NORMALS_ENCODING = "camera_xyz_uint8"
LEGACY_SURFACE_NORMALS_ENCODING_VERSION = 1
SURFACE_NORMALS_ENCODING_VERSION = 2
SUPPORTED_SURFACE_NORMALS_ENCODING_VERSIONS = (
    LEGACY_SURFACE_NORMALS_ENCODING_VERSION,
    SURFACE_NORMALS_ENCODING_VERSION,
)
SURFACE_NORMALS_DEPTH_RANGE_REQUIRED_SAMPLES = (
    "center",
    "left",
    "right",
    "up",
    "down",
)
DEFAULT_DEPTH_NEAR_M = 0.25
DEFAULT_DEPTH_FAR_M = 1.0
TASK_CONTRACT_SCHEMA_VERSION = 1
REALSENSE_RGBD_CALIBRATION_SCHEMA = "realsense_rgbd_calibration.v1"
POST_VISION_LATE_FUSION_MODE = "post_vision_late_fusion"
POST_VISION_LATE_FUSION_STAGE_SPECS = {
    "pre_vision_language_adapter": {"input_dim": 2048, "output_dim": 1024},
    "post_vision_language_adapter": {"input_dim": 4096, "output_dim": 2048},
}


def _task_contract_sha256(instructions: list[str]) -> str:
    """Hash the versioned, ordered instruction allowlist sent to clients."""

    canonical = json.dumps(
        {
            "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
            "instructions": instructions,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_deployment_task_contract(dataset_path: Path) -> dict:
    """Load an exact, ordered task allowlist from a checkpoint-bound dataset."""

    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    info_path = dataset_path / "meta" / "info.json"
    try:
        with open(info_path) as file:
            info = json.load(file)
        expected_count = info["total_tasks"]
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise ValueError("Dataset info.json total_tasks must be a positive integer")

        records: list[dict] = []
        with open(tasks_path) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Dataset tasks.jsonl line {line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Dataset tasks.jsonl line {line_number} must contain an object"
                    )
                records.append(record)

        if len(records) != expected_count:
            raise ValueError(
                "Dataset tasks.jsonl record count does not match info.json total_tasks: "
                f"got {len(records)}, expected {expected_count}"
            )

        instructions: list[str] = []
        for expected_index, record in enumerate(records):
            task_index = record.get("task_index")
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or task_index != expected_index
            ):
                raise ValueError(
                    "Dataset tasks.jsonl task_index values must be contiguous from zero; "
                    f"record {expected_index} has {task_index!r}"
                )
            instruction = record.get("task")
            if (
                not isinstance(instruction, str)
                or not instruction
                or len(instruction) > 256
                or any(ord(character) < 32 for character in instruction)
            ):
                raise ValueError(
                    f"Dataset tasks.jsonl task {expected_index} must be 1..256 printable characters"
                )
            instructions.append(instruction)
        if len(set(instructions)) != len(instructions):
            raise ValueError("Dataset tasks.jsonl contains duplicate instruction text")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot build a deployment task contract from {tasks_path}: {exc}"
        ) from exc

    return {
        "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "instructions": instructions,
        "sha256": _task_contract_sha256(instructions),
    }


def _load_end_effector_contract(info: dict) -> dict | None:
    """Validate optional end-effector provenance carried by converted datasets."""

    raw = info.get("end_effector")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Dataset end_effector must be an object")

    schema_version = raw.get("schema_version")
    hand_dof = raw.get("hand_dof")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("Dataset end_effector.schema_version must be integer 1")
    if isinstance(hand_dof, bool) or not isinstance(hand_dof, int) or hand_dof <= 0:
        raise ValueError("Dataset end_effector.hand_dof must be a positive integer")

    for field in ("type", "protocol", "value_unit", "zero_semantics"):
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Dataset end_effector.{field} must be a non-empty string")
    one_semantics = raw.get("one_semantics")
    if one_semantics is not None and (not isinstance(one_semantics, str) or not one_semantics):
        raise ValueError("Dataset end_effector.one_semantics must be null or a non-empty string")
    if raw.get("canonical_order") != "left_then_right":
        raise ValueError("Dataset end_effector.canonical_order must be 'left_then_right'")

    value_range = raw.get("value_range")
    if (
        not isinstance(value_range, list)
        or len(value_range) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in value_range
        )
        or value_range[0] > value_range[1]
    ):
        raise ValueError("Dataset end_effector.value_range must contain two ordered finite numbers")

    normalized = dict(raw)
    normalized["value_range"] = [float(value) for value in value_range]
    for side in ("left", "right"):
        field = f"{side}_joint_names"
        names = raw.get(field)
        if (
            not isinstance(names, list)
            or len(names) != hand_dof
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"Dataset end_effector.{field} must contain {hand_dof} unique non-empty names"
            )
        normalized[field] = list(names)
    return normalized


def _validate_end_effector_layout(
    end_effector: dict,
    layouts: dict[str, dict],
    feature_names: dict[str, list[str]],
) -> None:
    """Bind end-effector provenance to the hand slices and feature names."""

    hand_dof = end_effector["hand_dof"]
    for section, names in feature_names.items():
        layout = layouts[section]
        for side in ("left", "right"):
            group_name = f"{side}_hand"
            group = layout.get(group_name)
            if not isinstance(group, dict) or group.get("dim") != hand_dof:
                raise ValueError(
                    f"Dataset modality.json.{section}.{group_name} must have dimension {hand_dof}"
                )
            expected_names = end_effector[f"{side}_joint_names"]
            actual_names = names[group["start"] : group["end"]]
            if actual_names != expected_names:
                raise ValueError(
                    f"Dataset {section} names for {group_name} do not match end_effector provenance"
                )
        if layout["left_hand"]["start"] >= layout["right_hand"]["start"]:
            raise ValueError(
                f"Dataset modality.json.{section} must place left_hand before right_hand"
            )


def _load_vector_feature(feature: dict, feature_key: str) -> tuple[list[int], list[str]]:
    """Validate and normalize a one-dimensional state/action feature contract."""

    if not isinstance(feature, dict):
        raise ValueError(f"Deployment feature {feature_key!r} must be an object")
    shape = feature.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or not isinstance(shape[0], int)
        or isinstance(shape[0], bool)
        or shape[0] <= 0
    ):
        raise ValueError(
            f"Deployment feature {feature_key!r} must have a positive one-dimensional shape"
        )

    names = feature.get("names")
    if isinstance(names, dict):
        if set(names) != {"motors"}:
            raise ValueError(
                f"Deployment feature {feature_key!r} names object must contain only 'motors'"
            )
        names = names["motors"]
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if (
        not isinstance(names, list)
        or len(names) != shape[0]
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError(
            f"Deployment feature {feature_key!r} must provide one non-empty name per dimension"
        )
    return list(shape), list(names)


def _load_modality_layout(meta_dir: Path, feature_dims: dict[str, int]) -> dict[str, dict]:
    """Load state/action slices and prove that each exactly partitions its vector."""

    modality_path = meta_dir / "modality.json"
    with open(modality_path) as file:
        modality = json.load(file)
    if not isinstance(modality, dict):
        raise ValueError("Dataset modality.json must contain a JSON object")

    normalized: dict[str, dict] = {}
    for section, feature_dim in feature_dims.items():
        groups = modality.get(section)
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"Dataset modality.json.{section} must be a non-empty object")

        ordered_groups: list[tuple[int, str, int]] = []
        for group_name, raw_slice in groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError(f"Dataset modality.json.{section} has an empty group name")
            if not isinstance(raw_slice, dict):
                raise ValueError(f"Dataset modality.json.{section}.{group_name} must be an object")
            start = raw_slice.get("start")
            end = raw_slice.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > feature_dim
            ):
                raise ValueError(
                    f"Dataset modality.json.{section}.{group_name} must satisfy "
                    f"0 <= start < end <= {feature_dim}"
                )
            ordered_groups.append((start, group_name, end))

        ordered_groups.sort()
        cursor = 0
        section_contract: dict[str, dict[str, int]] = {}
        for start, group_name, end in ordered_groups:
            if start != cursor:
                raise ValueError(
                    f"Dataset modality.json.{section} must exactly partition [0, {feature_dim}); "
                    f"expected the next slice at {cursor}, got {group_name!r} at {start}"
                )
            section_contract[group_name] = {
                "start": start,
                "end": end,
                "dim": end - start,
            }
            cursor = end
        if cursor != feature_dim:
            raise ValueError(
                f"Dataset modality.json.{section} ends at {cursor}, expected {feature_dim}"
            )
        normalized[section] = section_contract
    return normalized


def _load_video_shapes(features: dict) -> dict[str, list[int]]:
    """Return a stable view-name -> HWC mapping for LeRobot image features."""

    video_shapes: dict[str, list[int]] = {}
    for feature_key in sorted(features):
        if not feature_key.startswith(IMAGE_FEATURE_PREFIX):
            continue
        feature = features[feature_key]
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"Deployment image feature {feature_key!r} must have dtype 'video'")
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape
            )
        ):
            raise ValueError(
                f"Deployment image feature {feature_key!r} must have a positive integer HWC shape"
            )
        if shape[-1] != 3:
            raise ValueError(
                f"Deployment image feature {feature_key!r} must have three channels, got {shape}"
            )
        view_name = feature_key.removeprefix(IMAGE_FEATURE_PREFIX)
        if not view_name:
            raise ValueError("Deployment image feature has an empty view name")
        video_shapes[view_name] = list(shape)
    if "ego_view" not in video_shapes:
        raise ValueError("Deployment dataset has no observation.images.ego_view video feature")
    return video_shapes


def _load_depth_encoding(info: dict, video_shapes: dict[str, list[int]]) -> dict | None:
    """Normalize the model-visible depth encoding without binding sensor-native units."""

    has_depth_view = "depth_gray_view" in video_shapes
    raw_encoding = info.get("depth_encoding")
    if not has_depth_view:
        if raw_encoding is not None:
            raise ValueError(
                "Dataset declares depth_encoding but has no observation.images.depth_gray_view"
            )
        return None
    if not isinstance(raw_encoding, dict):
        raise ValueError("Dataset depth_gray_view requires a depth_encoding object")

    required = {
        "source_key": str,
        "feature_key": str,
        "encoding": str,
        "near_m": (int, float),
        "far_m": (int, float),
        "invalid_value": int,
        "valid_value_range": list,
    }
    for field, expected_type in required.items():
        value = raw_encoding.get(field)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ValueError(f"Dataset depth_encoding.{field} is missing or malformed")

    if raw_encoding["feature_key"] != f"{IMAGE_FEATURE_PREFIX}depth_gray_view":
        raise ValueError("Dataset depth_encoding.feature_key does not identify depth_gray_view")
    near_m = float(raw_encoding["near_m"])
    far_m = float(raw_encoding["far_m"])
    if not math.isfinite(near_m) or not math.isfinite(far_m) or far_m <= near_m:
        raise ValueError("Dataset depth_encoding requires finite near_m < far_m")
    valid_range = raw_encoding["valid_value_range"]
    if (
        len(valid_range) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in valid_range)
        or valid_range[0] > valid_range[1]
    ):
        raise ValueError("Dataset depth_encoding.valid_value_range must be two ordered integers")

    return {
        "source_key": raw_encoding["source_key"],
        "feature_key": raw_encoding["feature_key"],
        "encoding": raw_encoding["encoding"],
        "near_m": near_m,
        "far_m": far_m,
        "invalid_value": raw_encoding["invalid_value"],
        "valid_value_range": list(valid_range),
    }


def _load_surface_normals_encoding(info: dict, video_shapes: dict[str, list[int]]) -> dict | None:
    """Validate the versioned transform used for the model-visible normals view."""

    has_normals_view = SURFACE_NORMALS_VIEW in video_shapes
    raw_encoding = info.get("surface_normals_encoding")
    if not has_normals_view:
        if raw_encoding is not None:
            raise ValueError(
                "Dataset declares surface_normals_encoding but has no "
                f"{IMAGE_FEATURE_PREFIX}{SURFACE_NORMALS_VIEW}"
            )
        return None
    if not isinstance(raw_encoding, dict):
        raise ValueError(
            f"Dataset {SURFACE_NORMALS_VIEW} requires a surface_normals_encoding object"
        )

    expected_strings = {
        "source_key": "depth_0",
        "aligned_to": "color_0",
        "feature_key": f"{IMAGE_FEATURE_PREFIX}{SURFACE_NORMALS_VIEW}",
        "encoding": SURFACE_NORMALS_ENCODING,
        "depth_scale_source": "episode.info.depth.scale_m_per_unit",
        "coordinate_frame": "camera_optical_x_right_y_down_z_forward",
        "orientation": "camera_facing_dot_normal_point_lte_zero",
        "method": "central_difference_3d",
    }
    for field, expected in expected_strings.items():
        if raw_encoding.get(field) != expected:
            raise ValueError(f"Dataset surface_normals_encoding.{field} must be {expected!r}")

    encoding_version = raw_encoding.get("encoding_version")
    if (
        isinstance(encoding_version, bool)
        or not isinstance(encoding_version, int)
        or encoding_version not in SUPPORTED_SURFACE_NORMALS_ENCODING_VERSIONS
    ):
        raise ValueError(
            "Dataset surface_normals_encoding.encoding_version must be one of the supported "
            f"versions {list(SUPPORTED_SURFACE_NORMALS_ENCODING_VERSIONS)!r}"
        )

    depth_valid_range = raw_encoding.get("depth_valid_range_m")
    normalized_depth_valid_range = None
    if encoding_version == LEGACY_SURFACE_NORMALS_ENCODING_VERSION:
        if "depth_valid_range_m" in raw_encoding:
            raise ValueError(
                "Dataset legacy surface_normals_encoding version 1 must not declare "
                "depth_valid_range_m"
            )
    else:
        if not isinstance(depth_valid_range, dict) or set(depth_valid_range) != {
            "near_m",
            "far_m",
            "inclusive",
            "required_samples",
        }:
            raise ValueError(
                "Dataset surface_normals_encoding.depth_valid_range_m must contain exactly "
                "near_m, far_m, inclusive, and required_samples"
            )
        if depth_valid_range["inclusive"] is not True:
            raise ValueError(
                "Dataset surface_normals_encoding.depth_valid_range_m.inclusive must be true"
            )
        if depth_valid_range["required_samples"] != list(
            SURFACE_NORMALS_DEPTH_RANGE_REQUIRED_SAMPLES
        ):
            raise ValueError(
                "Dataset surface_normals_encoding.depth_valid_range_m.required_samples "
                "must be exactly "
                f"{list(SURFACE_NORMALS_DEPTH_RANGE_REQUIRED_SAMPLES)!r}"
            )
        near_m = depth_valid_range["near_m"]
        far_m = depth_valid_range["far_m"]
        if (
            isinstance(near_m, bool)
            or isinstance(far_m, bool)
            or not isinstance(near_m, (int, float))
            or not isinstance(far_m, (int, float))
            or not math.isfinite(near_m)
            or not math.isfinite(far_m)
            or near_m < 0.0
            or far_m <= near_m
        ):
            raise ValueError(
                "Dataset surface_normals_encoding.depth_valid_range_m requires finite "
                "0 <= near_m < far_m"
            )
        normalized_depth_valid_range = {
            "near_m": float(near_m),
            "far_m": float(far_m),
            "inclusive": True,
            "required_samples": list(SURFACE_NORMALS_DEPTH_RANGE_REQUIRED_SAMPLES),
        }

    expected_top_level_fields = set(expected_strings) | {
        "encoding_version",
        "default_scale_m_per_unit",
        "intrinsics",
        "axis_order",
        "neighbor_offset_pixels",
        "max_neighbor_depth_delta_m",
        "invalid_value",
        "valid_component_range",
        "camera_calibration",
    }
    if encoding_version == SURFACE_NORMALS_ENCODING_VERSION:
        expected_top_level_fields.add("depth_valid_range_m")
    unknown_fields = set(raw_encoding) - expected_top_level_fields
    if unknown_fields:
        raise ValueError(
            "Dataset surface_normals_encoding declares unsupported fields: "
            f"{sorted(unknown_fields)!r}"
        )

    default_scale = raw_encoding.get("default_scale_m_per_unit")
    if (
        isinstance(default_scale, bool)
        or not isinstance(default_scale, (int, float))
        or not math.isfinite(default_scale)
        or default_scale <= 0
    ):
        raise ValueError(
            "Dataset surface_normals_encoding.default_scale_m_per_unit must be positive and finite"
        )

    intrinsics = raw_encoding.get("intrinsics")
    expected_intrinsics_fields = {"model", "width", "height", "fx", "fy", "cx", "cy"}
    if (
        not isinstance(intrinsics, dict)
        or set(intrinsics) != expected_intrinsics_fields
        or intrinsics.get("model") != "pinhole"
    ):
        raise ValueError(
            "Dataset surface_normals_encoding.intrinsics must contain exactly the pinhole "
            f"fields {sorted(expected_intrinsics_fields)!r}"
        )
    normalized_intrinsics = {"model": "pinhole"}
    for field in ("width", "height"):
        value = intrinsics.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Dataset surface_normals_encoding.intrinsics.{field} must be a positive integer"
            )
        normalized_intrinsics[field] = value
    for field in ("fx", "fy", "cx", "cy"):
        value = intrinsics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Dataset surface_normals_encoding.intrinsics.{field} must be finite")
        if field in ("fx", "fy") and value <= 0:
            raise ValueError(
                f"Dataset surface_normals_encoding.intrinsics.{field} must be positive"
            )
        normalized_intrinsics[field] = float(value)

    view_height, view_width, _ = video_shapes[SURFACE_NORMALS_VIEW]
    if (normalized_intrinsics["height"], normalized_intrinsics["width"]) != (
        view_height,
        view_width,
    ):
        raise ValueError(
            "Dataset surface_normals_encoding intrinsics resolution does not match "
            f"{SURFACE_NORMALS_VIEW} shape"
        )

    if raw_encoding.get("axis_order") != ["x", "y", "z"]:
        raise ValueError("Dataset surface_normals_encoding.axis_order must be ['x', 'y', 'z']")
    neighbor_offset = raw_encoding.get("neighbor_offset_pixels")
    if isinstance(neighbor_offset, bool) or neighbor_offset != 1:
        raise ValueError("Dataset surface_normals_encoding.neighbor_offset_pixels must be 1")
    max_depth_delta = raw_encoding.get("max_neighbor_depth_delta_m")
    if (
        isinstance(max_depth_delta, bool)
        or not isinstance(max_depth_delta, (int, float))
        or not math.isfinite(max_depth_delta)
        or max_depth_delta <= 0
    ):
        raise ValueError(
            "Dataset surface_normals_encoding.max_neighbor_depth_delta_m must be positive and finite"
        )
    if raw_encoding.get("invalid_value") != [0, 0, 0]:
        raise ValueError("Dataset surface_normals_encoding.invalid_value must be [0, 0, 0]")
    if raw_encoding.get("valid_component_range") != [1, 255]:
        raise ValueError("Dataset surface_normals_encoding.valid_component_range must be [1, 255]")

    normalized_encoding = {
        **expected_strings,
        "encoding_version": encoding_version,
        "default_scale_m_per_unit": float(default_scale),
        "intrinsics": normalized_intrinsics,
        "axis_order": ["x", "y", "z"],
        "neighbor_offset_pixels": neighbor_offset,
        "max_neighbor_depth_delta_m": float(max_depth_delta),
        "invalid_value": [0, 0, 0],
        "valid_component_range": [1, 255],
    }
    if normalized_depth_valid_range is not None:
        normalized_encoding["depth_valid_range_m"] = normalized_depth_valid_range
    camera_calibration = raw_encoding.get("camera_calibration")
    if camera_calibration is not None:
        if not isinstance(camera_calibration, dict) or set(camera_calibration) != {
            "source",
            "schema",
            "camera",
            "fingerprint",
        }:
            raise ValueError(
                "Dataset surface_normals_encoding.camera_calibration must contain exactly "
                "source, schema, camera, and fingerprint"
            )
        source = camera_calibration["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "Dataset surface_normals_encoding.camera_calibration.source must be a "
                "non-empty string"
            )
        if camera_calibration["schema"] != REALSENSE_RGBD_CALIBRATION_SCHEMA:
            raise ValueError(
                "Dataset surface_normals_encoding.camera_calibration.schema must be "
                f"{REALSENSE_RGBD_CALIBRATION_SCHEMA!r}"
            )
        camera = camera_calibration["camera"]
        camera_fields = ("model", "serial", "product_id", "firmware")
        if not isinstance(camera, dict) or set(camera) != set(camera_fields):
            raise ValueError(
                "Dataset surface_normals_encoding.camera_calibration.camera must contain "
                "exactly model, serial, product_id, and firmware"
            )
        normalized_camera = {}
        for field in camera_fields:
            value = camera[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Dataset surface_normals_encoding.camera_calibration.camera."
                    f"{field} must be a non-empty string"
                )
            normalized_camera[field] = value
        fingerprint = camera_calibration["fingerprint"]
        digest = fingerprint.removeprefix("sha256:") if isinstance(fingerprint, str) else ""
        if (
            not isinstance(fingerprint, str)
            or not fingerprint.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "Dataset surface_normals_encoding.camera_calibration.fingerprint must be "
                "sha256:<64 lowercase hex digits>"
            )
        normalized_encoding["camera_calibration"] = {
            "source": source,
            "schema": REALSENSE_RGBD_CALIBRATION_SCHEMA,
            "camera": normalized_camera,
            "fingerprint": fingerprint,
        }
    return normalized_encoding


def _load_deployment_dataset_contract(dataset_path: Path) -> dict:
    """Load the hardware-facing parts of a LeRobot dataset contract."""

    info_path = dataset_path / "meta" / "info.json"
    try:
        with open(info_path) as file:
            info = json.load(file)
        features = info["features"]
        state_shape, state_names = _load_vector_feature(
            features["observation.state"], "observation.state"
        )
        action_shape, action_names = _load_vector_feature(features["action"], "action")
        modality_layout = _load_modality_layout(
            info_path.parent,
            {"state": state_shape[0], "action": action_shape[0]},
        )
        video_shapes = _load_video_shapes(features)
        end_effector = _load_end_effector_contract(info)
        if end_effector is not None:
            _validate_end_effector_layout(
                end_effector,
                modality_layout,
                {
                    "state": state_names,
                    "action": action_names,
                },
            )
        depth_encoding = _load_depth_encoding(info, video_shapes)
        surface_normals_encoding = _load_surface_normals_encoding(info, video_shapes)
        if (
            surface_normals_encoding is not None
            and surface_normals_encoding["encoding_version"] == SURFACE_NORMALS_ENCODING_VERSION
        ):
            expected_near_m = (
                depth_encoding["near_m"] if depth_encoding is not None else DEFAULT_DEPTH_NEAR_M
            )
            expected_far_m = (
                depth_encoding["far_m"] if depth_encoding is not None else DEFAULT_DEPTH_FAR_M
            )
            normal_range = surface_normals_encoding["depth_valid_range_m"]
            if normal_range["near_m"] != expected_near_m or normal_range["far_m"] != expected_far_m:
                source = (
                    "dataset depth_encoding"
                    if depth_encoding is not None
                    else "canonical depth encoding"
                )
                raise ValueError(
                    "Dataset surface-normal v2 depth_valid_range_m must exactly match "
                    f"the {source} near/far bounds "
                    f"[{expected_near_m}, {expected_far_m}]"
                )
        contract = {
            "robot_type": str(info["robot_type"]),
            "fps": float(info["fps"]),
            "observation_state_shape": state_shape,
            "action_shape": action_shape,
            "observation_state_names": list(state_names),
            "action_names": list(action_names),
            "state_layout": modality_layout["state"],
            "action_layout": modality_layout["action"],
            # Retained for clients deployed before the multi-view contract.
            "ego_view_shape": list(video_shapes["ego_view"]),
            "video_shapes": video_shapes,
        }
        if end_effector is not None:
            contract["end_effector"] = end_effector
        if depth_encoding is not None:
            contract["depth_encoding"] = depth_encoding
        if surface_normals_encoding is not None:
            contract["surface_normals_encoding"] = surface_normals_encoding
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot build a deployment contract from {info_path}: {exc}") from exc
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def _validate_checkpoint_dataset_vision_contract(
    vision_contract: dict, dataset_contract: dict
) -> None:
    """Bind a post-vision checkpoint's ordered sources to deployment data.

    Existing RGB, separate-view, and early-channel-fusion deployments retain
    their current validation behavior.  A late-fusion checkpoint is stricter:
    it has two semantically ordered, spatially aligned three-channel sources,
    and deployment must provide the encoding metadata for the auxiliary view.
    """

    depth_colormap = vision_contract.get("depth_colormap")
    if depth_colormap is not None:
        expected_colormap = fixed_turbo_depth_colormap_contract()
        if depth_colormap != expected_colormap:
            raise ValueError(
                "Checkpoint depth-colormap contract does not match fixed_turbo_v1: "
                f"got {depth_colormap!r}, expected {expected_colormap!r}"
            )
        depth_encoding = dataset_contract.get("depth_encoding")
        if not isinstance(depth_encoding, dict):
            raise ValueError(
                "fixed_turbo_v1 requires deployment depth_encoding metadata"
            )
        expected_encoding = {
            "encoding": "linear_grayscale_replicated_rgb",
            "invalid_value": 0,
            "valid_value_range": [1, 255],
        }
        for field, expected in expected_encoding.items():
            if depth_encoding.get(field) != expected:
                raise ValueError(
                    "fixed_turbo_v1 deployment depth encoding mismatch for "
                    f"{field}: got {depth_encoding.get(field)!r}, expected {expected!r}"
                )

    if vision_contract.get("mode") != POST_VISION_LATE_FUSION_MODE:
        return

    if vision_contract.get("version") != 2:
        raise ValueError("Post-vision late-fusion vision contract must use schema version 2")
    if vision_contract.get("patch_embed_trainable") is not False:
        raise ValueError(
            "Post-vision late-fusion contract must declare patch_embed_trainable=false"
        )
    fusion = vision_contract.get("post_vision_fusion")
    if not isinstance(fusion, dict):
        raise ValueError("Post-vision late-fusion contract is missing fusion metadata")
    stage = fusion.get("stage")
    if stage not in POST_VISION_LATE_FUSION_STAGE_SPECS:
        raise ValueError(
            "Post-vision late-fusion contract must declare stage "
            "'pre_vision_language_adapter' or 'post_vision_language_adapter'"
        )
    stage_spec = POST_VISION_LATE_FUSION_STAGE_SPECS[stage]
    parameters_per_adapter = (
        stage_spec["input_dim"] * stage_spec["output_dim"]
        + stage_spec["output_dim"]
    )
    expected_architecture = {
        "version": 1,
        "adapter_type": "concat_linear",
        "adapter_bias": True,
        "adapter_bias_init": "zeros",
        "adapter_count": 4,
        "deepstack_layers": [5, 11, 17],
        "adapter_input_dim": stage_spec["input_dim"],
        "adapter_output_dim": stage_spec["output_dim"],
        "parameters_per_adapter": parameters_per_adapter,
        "total_adapter_parameters": 4 * parameters_per_adapter,
    }
    for field, expected in expected_architecture.items():
        if fusion.get(field) != expected:
            raise ValueError(
                "Post-vision late-fusion contract architecture mismatch for "
                f"{field}: got {fusion.get(field)!r}, expected {expected!r}"
            )
    if fusion.get("adapter_init") != "rgb50_geo50":
        raise ValueError(
            "Post-vision late-fusion contract must declare adapter_init 'rgb50_geo50'"
        )
    if fusion.get("rgb_init_scale") != 0.5 or fusion.get("geometry_init_scale") != 0.5:
        raise ValueError(
            "Post-vision late-fusion contract must declare exact initialization scales "
            "RGB=0.5 and geometry=0.5"
        )
    source_keys = fusion.get("source_keys")
    if source_keys != vision_contract.get("wire_video_keys"):
        raise ValueError("Post-vision late-fusion source keys do not match the wire video keys")
    if (
        not isinstance(source_keys, list)
        or len(source_keys) != 2
        or source_keys[0] != "ego_view"
        or source_keys[1] not in {"depth_gray_view", SURFACE_NORMALS_VIEW}
    ):
        raise ValueError("Post-vision late fusion requires ordered sources ['ego_view', auxiliary]")

    video_shapes = dataset_contract.get("video_shapes")
    if not isinstance(video_shapes, dict):
        raise ValueError("Deployment dataset contract is missing video_shapes")
    try:
        primary_shape = video_shapes[source_keys[0]]
        auxiliary_shape = video_shapes[source_keys[1]]
    except KeyError as exc:
        raise ValueError(
            f"Deployment dataset is missing late-fusion video source {exc.args[0]!r}"
        ) from exc
    if primary_shape != auxiliary_shape:
        raise ValueError(
            "Post-vision late-fusion RGB and auxiliary views must have identical HWC shapes: "
            f"RGB={primary_shape!r}, auxiliary={auxiliary_shape!r}"
        )
    if not isinstance(primary_shape, list) or len(primary_shape) != 3 or primary_shape[-1] != 3:
        raise ValueError("Post-vision late-fusion sources must be aligned three-channel HWC videos")

    auxiliary_key = source_keys[1]
    if auxiliary_key == "depth_gray_view" and "depth_encoding" not in dataset_contract:
        raise ValueError(
            "Post-vision depth late fusion requires deployment depth_encoding metadata"
        )
    if auxiliary_key == SURFACE_NORMALS_VIEW and "surface_normals_encoding" not in dataset_contract:
        raise ValueError(
            "Post-vision normals late fusion requires deployment surface_normals_encoding metadata"
        )


def _verify_checkpoint_dataset_path(
    model_path: Path, embodiment_tag: str, deployment_dataset_path: Path
) -> None:
    """Bind an asserted deployment dataset to the checkpoint's saved training config."""

    config_path = model_path / "experiment_cfg" / "config.yaml"
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
        datasets = config["data"]["datasets"]
        recorded = [
            Path(path).resolve()
            for dataset in datasets
            if dataset.get("embodiment_tag") == embodiment_tag
            for path in dataset["dataset_paths"]
        ]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Cannot verify deployment data against checkpoint config {config_path}: {exc}"
        ) from exc
    requested = deployment_dataset_path.resolve()
    if recorded != [requested]:
        raise ValueError(
            "Deployment dataset must be the single training dataset recorded for "
            f"embodiment '{embodiment_tag}'. Checkpoint records {recorded}; got {requested}."
        )


def _load_checkpoint_action_output_contract(model_path: Path, embodiment_tag: str) -> dict:
    """Describe how ``Gr00tPolicy.get_action`` decodes checkpoint outputs."""

    candidates = [
        model_path / "processor_config.json",
        model_path / "processor" / "processor_config.json",
    ]
    config_path = next((path for path in candidates if path.is_file()), candidates[0])
    try:
        with open(config_path) as file:
            processor = json.load(file)["processor_kwargs"]
        use_relative_action = processor["use_relative_action"]
        if not isinstance(use_relative_action, bool):
            raise TypeError(
                "processor_kwargs.use_relative_action must be a boolean, got "
                f"{use_relative_action!r}"
            )
        action = processor["modality_configs"][embodiment_tag]["action"]
        keys = action["modality_keys"]
        configs = action["action_configs"]
        relative_keys = [
            key for key, config in zip(keys, configs, strict=True) if config["rep"] == "RELATIVE"
        ]
        representations = {config["rep"] for config in configs}
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine action output semantics from {config_path}: {exc}"
        ) from exc
    supported_representations = representations <= {"RELATIVE", "ABSOLUTE"}
    absolute_output = supported_representations and (use_relative_action or not relative_keys)
    return {
        "semantics": "absolute_joint_position" if absolute_output else "checkpoint_native",
        "use_relative_action": use_relative_action,
        "relative_keys_decoded_to_absolute": relative_keys if use_relative_action else [],
    }


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


def _model_identity(model_path: str) -> dict[str, str | None]:
    """Identify the loaded model for recording provenance, without reading weights."""

    path = Path(model_path)
    checkpoint = (
        path.name
        if path.name.startswith("checkpoint-") and path.name[len("checkpoint-") :].isdigit()
        else None
    )
    model_dir = path.parent if checkpoint is not None else path
    return {
        "model_name": model_dir.name or model_dir.resolve().name,
        "checkpoint": checkpoint,
        "model_path": model_path,
    }


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    model: str | None = None
    """Visible live-folder alias: rgb, separate, or fusion."""

    blind: str | None = None
    """Independent hidden mode: A or B; defaults to RGB versus surface-normals separate views."""

    blind_study: str = DEFAULT_BLIND_STUDY
    """Reuse this study name across sessions; never reshuffled on restart."""

    live_model_root: str = "/home/alex/Development/Models/inspire_live_test"
    """Folder containing live_models.json and all aliased checkpoints."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    deployment_dataset_path: str | None = None
    """Dataset root whose meta/info.json identifies the physical deployment contract"""


def main(config: ServerConfig):
    import uuid
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    policy_metadata = {
        "embodiment_tag": config.embodiment_tag.value,
        "embodiment_name": config.embodiment_tag.name,
        "server_instance_id": uuid.uuid4().hex,
    }
    if config.model_path is not None:
        policy_metadata["model_identity"] = getattr(config, "_public_identity", None) or _model_identity(config.model_path)
    if config.model_path is not None and not config.use_sim_policy_wrapper:
        policy_metadata["vision_input_contract"] = policy.get_vision_input_contract()  # earlyfusion
        # Explicit capability handshake for clients that need model-side RTC
        # inpainting. ReplayPolicy and older servers deliberately omit it so a
        # real-robot client can fail before arming rather than silently falling
        # back to independent asynchronous chunks.
        policy_metadata["rtc"] = {
            "protocol_version": 1,
            "physical_action_tail": True,
            "backend": "pytorch",
            "pure_inference_guidance": (
                getattr(
                    getattr(getattr(policy, "model", None), "action_head", None),
                    "supports_rtc_pure",
                    False,
                ) is True
            ),
        }
    if config.deployment_dataset_path is not None:
        deployment_dataset_path = Path(config.deployment_dataset_path)
        if config.model_path is not None:
            _verify_checkpoint_dataset_path(
                Path(config.model_path), config.embodiment_tag.value, deployment_dataset_path
            )
            policy_metadata["action_output_contract"] = _load_checkpoint_action_output_contract(
                Path(config.model_path), config.embodiment_tag.value
            )
        elif Path(config.dataset_path).resolve() != deployment_dataset_path.resolve():
            raise ValueError("Replay policy --deployment-dataset-path must equal --dataset-path")
        dataset_contract = _load_deployment_dataset_contract(deployment_dataset_path)
        if "vision_input_contract" in policy_metadata:
            _validate_checkpoint_dataset_vision_contract(
                policy_metadata["vision_input_contract"], dataset_contract
            )
        policy_metadata["dataset_contract"] = dataset_contract
        policy_metadata["task_contract"] = _load_deployment_task_contract(deployment_dataset_path)

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        policy_metadata=policy_metadata,
    ) as server:
        announcer = getattr(config, "_ready_announce", None)
        if announcer is not None:
            announcer(f"Blind model {config.blind.upper()} ready on {config.host}:{config.port} (study {config.blind_study})")
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


def launch(config: ServerConfig):
    """Resolve aliases before entering the unchanged full-path server workflow."""
    from gr00t.eval.live_model_aliases import DEFAULT_DATASET, private_server_output, resolve_live_model
    if config.model is not None or config.blind is not None:
        if config.model_path is not None or config.dataset_path is not None:
            raise ValueError("--model/--blind cannot be combined with --model-path/--dataset-path")
        try:
            selection = resolve_live_model(model=config.model, blind=config.blind,
                                           study=config.blind_study, root=Path(config.live_model_root))
        except Exception:
            if config.blind is not None:
                raise SystemExit("Blind selection failed; mapping was not reshuffled. Inspect the private study directory.") from None
            raise
        config.model_path = str(selection.checkpoint)
        config._public_identity = selection.identity
        if config.deployment_dataset_path is None:
            config.deployment_dataset_path = DEFAULT_DATASET
        if selection.private_directory is not None:
            print(f"Loading blind model {config.blind.upper()} (study {config.blind_study}); identity logs kept private.", flush=True)
            with private_server_output(selection.private_directory) as announce:
                config._ready_announce = announce
                try:
                    return main(config)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    announce("Blind server failed; detailed diagnostics are in the private study directory.")
                    raise SystemExit(1) from None
    return main(config)


if __name__ == "__main__":
    launch(tyro.cli(ServerConfig))
