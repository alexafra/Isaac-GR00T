import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_unitree_raw_episode_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_unitree_raw_episode_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
validate_episode_contract = MODULE.validate_episode_contract


def _provenance(protocol="dfx"):
    return {
        "schema_version": 1,
        "type": "inspire",
        "protocol": protocol,
        "hand_dof": 6,
        "value_unit": "normalized_open_fraction",
        "value_range": [0.0, 1.0],
        "zero_semantics": "fully_closed",
        "one_semantics": "fully_open",
        "left_joint_names": [
            "kLeftHandPinky",
            "kLeftHandRing",
            "kLeftHandMiddle",
            "kLeftHandIndex",
            "kLeftHandThumbBend",
            "kLeftHandThumbRotation",
        ],
        "right_joint_names": [
            "kRightHandPinky",
            "kRightHandRing",
            "kRightHandMiddle",
            "kRightHandIndex",
            "kRightHandThumbBend",
            "kRightHandThumbRotation",
        ],
        "canonical_order": "left_then_right",
    }


def _write_episode(tmp_path, hand_dof, provenance=None, *, action_hand_dof=None):
    episode = tmp_path / "episode_0001"
    episode.mkdir()
    action_hand_dof = hand_dof if action_hand_dof is None else action_hand_dof

    def section(selected_hand_dof):
        return {
            "left_arm": {"qpos": [0.0] * 7},
            "right_arm": {"qpos": [0.0] * 7},
            "left_ee": {"qpos": [0.0] * selected_hand_dof},
            "right_ee": {"qpos": [0.0] * selected_hand_dof},
        }

    info = {}
    if provenance is not None:
        info["end_effector"] = provenance
    payload = {
        "info": info,
        "data": [{"states": section(hand_dof), "actions": section(action_hand_dof)}],
    }
    (episode / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode


def _replace_qpos_value(episode, section, key, value):
    data_path = episode / "data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["data"][0][section][key]["qpos"][0] = value
    data_path.write_text(json.dumps(payload), encoding="utf-8")


def test_accepts_legacy_dex3_without_provenance(tmp_path):
    episode = _write_episode(tmp_path, 7)

    summary = validate_episode_contract([episode], "dex3")

    assert summary["vector_dim"] == 28
    assert summary["legacy_episodes_without_provenance"] == 1


def test_accepts_native_inspire_dfx_with_provenance(tmp_path):
    episode = _write_episode(tmp_path, 6, _provenance())

    summary = validate_episode_contract([episode], "inspire-dfx")

    assert summary["vector_dim"] == 26
    assert summary["provenance"]["protocol"] == "dfx"


def test_accepts_native_inspire_ftp_only_with_ftp_provenance(tmp_path):
    episode = _write_episode(tmp_path, 6, _provenance("ftp"))

    summary = validate_episode_contract([episode], "inspire-ftp")

    assert summary["vector_dim"] == 26
    assert summary["provenance"]["protocol"] == "ftp"


def test_inspire_requires_provenance(tmp_path):
    episode = _write_episode(tmp_path, 6)

    with pytest.raises(ValueError, match="info.end_effector is required"):
        validate_episode_contract([episode], "inspire-dfx")


def test_rejects_inspire_protocol_mismatch(tmp_path):
    episode = _write_episode(tmp_path, 6, _provenance("ftp"))

    with pytest.raises(ValueError, match="protocol='ftp'.*inspire-dfx"):
        validate_episode_contract([episode], "inspire-dfx")


def test_rejects_selected_hand_dimension_mismatch(tmp_path):
    episode = _write_episode(tmp_path, 6, _provenance(), action_hand_dof=7)

    with pytest.raises(ValueError, match="actions component lengths.*inspire-dfx"):
        validate_episode_contract([episode], "inspire-dfx")


def test_rejects_explicit_dex3_inspire_mismatch(tmp_path):
    episode = _write_episode(tmp_path, 6, _provenance())

    with pytest.raises(ValueError, match="type='inspire'.*--end-effector dex3"):
        validate_episode_contract([episode], "dex3")


def test_rejects_inspire_semantics_mismatch(tmp_path):
    provenance = _provenance()
    provenance["value_unit"] = "radian"
    episode = _write_episode(tmp_path, 6, provenance)

    with pytest.raises(ValueError, match="value_unit='radian'.*native Inspire"):
        validate_episode_contract([episode], "inspire-dfx")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version must be integer 1"),
        ("schema_version", 1.0, "schema_version must be integer 1"),
        ("hand_dof", 6.0, "hand_dof=6.0"),
        ("value_range", [False, True], "value_range must contain two ordered finite numbers"),
        ("value_range", [0.0, math.inf], "value_range must contain two ordered finite numbers"),
    ],
)
def test_rejects_provenance_values_with_ambiguous_json_types(tmp_path, field, value, message):
    provenance = _provenance()
    provenance[field] = value
    episode = _write_episode(tmp_path, 6, provenance)

    with pytest.raises(ValueError, match=message):
        validate_episode_contract([episode], "inspire-dfx")


def test_rejects_incomplete_explicit_dex3_provenance_but_keeps_legacy_compatible(tmp_path):
    episode = _write_episode(
        tmp_path,
        7,
        {
            "schema_version": 1,
            "type": "dex3",
            "protocol": "dex3",
            "hand_dof": 7,
        },
    )

    with pytest.raises(ValueError, match="value_unit must be a non-empty string"):
        validate_episode_contract([episode], "dex3")


@pytest.mark.parametrize("value", ["0.5", True], ids=["string", "bool"])
def test_rejects_non_numeric_qpos_values(tmp_path, value):
    episode = _write_episode(tmp_path, 7)
    _replace_qpos_value(episode, "actions", "left_arm", value)

    with pytest.raises(ValueError, match=r"actions\.left_arm\.qpos\[0\].*JSON number"):
        validate_episode_contract([episode], "dex3")


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_rejects_non_finite_qpos_values(tmp_path, value):
    episode = _write_episode(tmp_path, 7)
    _replace_qpos_value(episode, "states", "right_arm", value)

    with pytest.raises(ValueError, match=r"states\.right_arm\.qpos\[0\].*must be finite"):
        validate_episode_contract([episode], "dex3")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("states", "left_ee", -0.001),
        ("actions", "right_ee", 1.001),
    ],
)
def test_rejects_out_of_range_inspire_hand_values(tmp_path, section, key, value):
    episode = _write_episode(tmp_path, 6, _provenance())
    _replace_qpos_value(episode, section, key, value)

    with pytest.raises(ValueError, match=r"must be within \[0\.0, 1\.0\] for Inspire"):
        validate_episode_contract([episode], "inspire-dfx")
