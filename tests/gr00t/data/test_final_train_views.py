"""Regression coverage for final train/validation views and calibration provenance."""

import copy

import pytest

from gr00t.data.virtual_dataset import VirtualDatasetError, build_virtual_dataset
from tests.gr00t.data.test_virtual_dataset import (
    GOALS,
    _read_json,
    _read_jsonl,
    _split_source_with_provenance,
    _write_dataset,
    _write_json,
)


def test_final_trainval_keeps_provenance_test_disjoint_and_train_then_val_order(tmp_path):
    first = _split_source_with_provenance(
        tmp_path / "first", {"train": "a" * 64, "validation": "b" * 64, "test": "c" * 64}
    )
    second = _split_source_with_provenance(
        tmp_path / "second",
        {"train": "d" * 64, "validation": "e" * 64, "test": "f" * 64},
        identity_offset=10,
    )
    combined = build_virtual_dataset([first, second], tmp_path / "combined", require_splits=True)
    final = build_virtual_dataset([combined / "train", combined / "validation"], tmp_path / "trainval")
    training = _read_jsonl(final / "meta/virtual_episode_sources.jsonl")
    held_out = _read_jsonl(combined / "test/meta/virtual_episode_sources.jsonl")
    assert [record["data_json_sha256"] for record in training] == [x * 64 for x in "adbe"]
    assert {record["data_json_sha256"] for record in held_out} == {x * 64 for x in "cf"}
    assert _read_json(final / "meta/info.json")["total_episodes"] == 4
    assert _read_json(final / "meta/info.json")["total_frames"] == 140
    assert _read_json(final / "meta/info.json")["total_tasks"] == 1
    assert not (final / "meta/relative_stats.json").exists()


def _calibrated_sources(tmp_path):
    sources = [tmp_path / "first", tmp_path / "second"]
    full_calibration = {
        "schema": "realsense_rgbd_calibration.v1",
        "camera": {"serial": "camera-1"},
        "color": {"fx": 609.3858642578125, "fy": 609.4705200195312},
        "depth": {"fx": 397.5912170410156},
        "depth_to_color": {"translation_m": [0.0148, 0.00088, 0.00073]},
        "fingerprint": "sha256:" + "a" * 64,
    }
    for index, root in enumerate(sources):
        _write_dataset(root, [(index, 35, GOALS[0])])
        info = _read_json(root / "meta/info.json")
        info["camera_calibration"] = copy.deepcopy(full_calibration)
        info["surface_normals_encoding"] = {
            "depth_valid_range_m": {"near_m": 0.25, "far_m": 1.0},
            "camera_calibration": {
                "schema": full_calibration["schema"],
                "camera": full_calibration["camera"],
                "fingerprint": full_calibration["fingerprint"],
                "source": "converter.profile.camera-1" if index == 0 else "episode.info.depth.calibration",
            },
        }
        _write_json(root / "meta/info.json", info)
    return sources


def test_calibration_source_label_only_is_not_encoding_mismatch(tmp_path):
    sources = _calibrated_sources(tmp_path)
    output = build_virtual_dataset(sources, tmp_path / "combined")
    assert _read_json(output / "meta/info.json")["total_episodes"] == 2
    # Preserve the original metadata; ignore the label only for compatibility.
    assert _read_json(output / "meta/info.json")["surface_normals_encoding"]["camera_calibration"]["source"] == "converter.profile.camera-1"


@pytest.mark.parametrize("change", ["intrinsics", "extrinsics", "fingerprint", "range", "camera"])
def test_calibration_actual_contract_changes_still_rejected(tmp_path, change):
    sources = _calibrated_sources(tmp_path)
    info = _read_json(sources[1] / "meta/info.json")
    if change == "intrinsics":
        info["camera_calibration"]["color"]["fx"] += 1
    elif change == "extrinsics":
        info["camera_calibration"]["depth_to_color"]["translation_m"][0] += 0.01
    elif change == "fingerprint":
        info["surface_normals_encoding"]["camera_calibration"]["fingerprint"] = "sha256:" + "b" * 64
    elif change == "range":
        info["surface_normals_encoding"]["depth_valid_range_m"]["far_m"] = 1.5
    else:
        info["camera_calibration"]["camera"]["serial"] = "camera-2"
    _write_json(sources[1] / "meta/info.json", info)
    with pytest.raises(VirtualDatasetError, match="Incompatible"):
        build_virtual_dataset(sources, tmp_path / "combined")
