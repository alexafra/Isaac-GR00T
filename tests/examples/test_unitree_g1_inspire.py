import importlib.util
import json
from pathlib import Path
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionRepresentation
from gr00t.eval.run_gr00t_server import _load_deployment_dataset_contract
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "examples" / "UnitreeG1" / "g1_inspire_headonly_config.py"
LAYOUT_PATH = REPO_ROOT / "examples" / "UnitreeG1" / "modality_inspire.json"
GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
CONFIG_RECIPES = [
    ("g1_inspire_headonly_config", ["ego_view"], ["ego_view:0", "ego_view:1", "ego_view:2"]),
    (
        "g1_inspire_head_3_channel_gray_depth_config",
        ["ego_view", "depth_gray_view"],
        ["ego_view:0", "ego_view:1", "ego_view:2"],
    ),
    (
        "g1_inspire_head_depth_gray_only_config",
        ["depth_gray_view"],
        ["depth_gray_view:0", "depth_gray_view:1", "depth_gray_view:2"],
    ),
    (
        "g1_inspire_head_4_channel_gray_depth_fusion_config",
        ["ego_view", "depth_gray_view"],
        ["ego_view:0", "ego_view:1", "ego_view:2", "depth_gray_view:0"],
    ),
    (
        "g1_inspire_head_3_channel_surface_normals_config",
        ["ego_view", "surface_normals_view"],
        ["ego_view:0", "ego_view:1", "ego_view:2"],
    ),
    (
        "g1_inspire_head_surface_normals_only_config",
        ["surface_normals_view"],
        ["surface_normals_view:0", "surface_normals_view:1", "surface_normals_view:2"],
    ),
    (
        "g1_inspire_head_6_channel_surface_normals_fusion_config",
        ["ego_view", "surface_normals_view"],
        [
            "ego_view:0",
            "ego_view:1",
            "ego_view:2",
            "surface_normals_view:0",
            "surface_normals_view:1",
            "surface_normals_view:2",
        ],
    ),
]
EXPECTED_LAYOUT = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}
INSPIRE_JOINT_NAMES = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
    "kRightHandPinky",
    "kRightHandRing",
    "kRightHandMiddle",
    "kRightHandIndex",
    "kRightHandThumbBend",
    "kRightHandThumbRotation",
]
INSPIRE_END_EFFECTOR = {
    "schema_version": 1,
    "type": "inspire",
    "protocol": "dfx",
    "hand_dof": 6,
    "value_unit": "normalized_open_fraction",
    "value_range": [0.0, 1.0],
    "zero_semantics": "fully_closed",
    "one_semantics": "fully_open",
    "left_joint_names": INSPIRE_JOINT_NAMES[14:20],
    "right_joint_names": INSPIRE_JOINT_NAMES[20:26],
    "canonical_order": "left_then_right",
}


def _load_inspire_config(config_name="g1_inspire_headonly_config"):
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    previous = MODALITY_CONFIGS.pop(tag, None)
    sys.path.insert(0, str(CONFIG_PATH.parent))
    try:
        config_path = CONFIG_PATH.with_name(f"{config_name}.py")
        spec = importlib.util.spec_from_file_location(f"unitree_{config_name}", config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, config_name)
        assert MODALITY_CONFIGS[tag] is config
        return config
    finally:
        sys.path.remove(str(CONFIG_PATH.parent))
        MODALITY_CONFIGS.pop(tag, None)
        if previous is not None:
            MODALITY_CONFIGS[tag] = previous


def _write_inspire_dataset(tmp_path, *, end_effector=None):
    dataset = tmp_path / "inspire_dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "Unitree_G1_Inspire_HeadOnly",
                "fps": 30,
                "end_effector": INSPIRE_END_EFFECTOR if end_effector is None else end_effector,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [26],
                        "names": [INSPIRE_JOINT_NAMES],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [26],
                        "names": [INSPIRE_JOINT_NAMES],
                    },
                    "observation.images.ego_view": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                },
            }
        )
    )
    (meta / "modality.json").write_text(LAYOUT_PATH.read_text())
    return dataset


def test_inspire_layout_is_native_26d_and_matches_state_action_order():
    with open(LAYOUT_PATH) as file:
        layout = json.load(file)

    assert layout["state"] == EXPECTED_LAYOUT
    assert layout["action"] == EXPECTED_LAYOUT
    assert list(layout["state"]) == GROUPS
    assert layout["state"]["left_hand"]["end"] - layout["state"]["left_hand"]["start"] == 6
    assert layout["state"]["right_hand"]["end"] - layout["state"]["right_hand"]["start"] == 6
    assert max(group["end"] for group in layout["state"].values()) == 26


def test_inspire_layout_preserves_head_camera_geometry_keys():
    with open(LAYOUT_PATH) as file:
        layout = json.load(file)

    assert layout["video"] == {
        "ego_view": {"original_key": "observation.images.ego_view"},
        "depth_gray_view": {"original_key": "observation.images.depth_gray_view"},
        "surface_normals_view": {"original_key": "observation.images.surface_normals_view"},
    }


def test_all_inspire_visual_recipes_preserve_g1_action_contract():
    for config_name, video_keys, channel_layout in CONFIG_RECIPES:
        config = _load_inspire_config(config_name)

        assert config["video"].modality_keys == video_keys
        assert config["video"].vision_channel_layout == channel_layout
        assert config["state"].modality_keys == GROUPS
        assert config["action"].modality_keys == GROUPS
        assert config["action"].delta_indices == list(range(32))
        assert [item.rep for item in config["action"].action_configs] == [
            ActionRepresentation.RELATIVE,
            ActionRepresentation.RELATIVE,
            ActionRepresentation.ABSOLUTE,
            ActionRepresentation.ABSOLUTE,
        ]


def test_server_contract_preserves_native_inspire_shape_names_and_group_dims(tmp_path):
    dataset = _write_inspire_dataset(tmp_path)

    contract = _load_deployment_dataset_contract(dataset)

    assert contract["robot_type"] == "Unitree_G1_Inspire_HeadOnly"
    assert contract["observation_state_shape"] == [26]
    assert contract["action_shape"] == [26]
    assert contract["observation_state_names"] == INSPIRE_JOINT_NAMES
    assert contract["action_names"] == INSPIRE_JOINT_NAMES
    assert contract["end_effector"] == INSPIRE_END_EFFECTOR
    assert {name: item["dim"] for name, item in contract["state_layout"].items()} == {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
    }
    assert contract["action_layout"] == contract["state_layout"]


def test_server_contract_rejects_end_effector_hand_name_mismatch(tmp_path):
    end_effector = json.loads(json.dumps(INSPIRE_END_EFFECTOR))
    end_effector["left_joint_names"][0] = "wrong"
    dataset = _write_inspire_dataset(tmp_path, end_effector=end_effector)

    with pytest.raises(ValueError, match="state names for left_hand"):
        _load_deployment_dataset_contract(dataset)


def test_server_contract_rejects_malformed_end_effector_value_range(tmp_path):
    end_effector = json.loads(json.dumps(INSPIRE_END_EFFECTOR))
    end_effector["value_range"] = [float("nan"), 1.0]
    dataset = _write_inspire_dataset(tmp_path, end_effector=end_effector)

    with pytest.raises(ValueError, match="value_range"):
        _load_deployment_dataset_contract(dataset)


def test_server_contract_hash_distinguishes_dfx_from_ftp(tmp_path):
    dataset = _write_inspire_dataset(tmp_path)
    dfx_contract = _load_deployment_dataset_contract(dataset)

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["end_effector"]["protocol"] = "ftp"
    info_path.write_text(json.dumps(info))
    ftp_contract = _load_deployment_dataset_contract(dataset)

    assert dfx_contract["end_effector"]["protocol"] == "dfx"
    assert ftp_contract["end_effector"]["protocol"] == "ftp"
    assert dfx_contract["sha256"] != ftp_contract["sha256"]
