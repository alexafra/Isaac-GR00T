#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the state/action and end-effector contract of raw Unitree episodes.

This preflight is intentionally dependency-free so conversion wrappers can fail
before creating or replacing a LeRobot dataset.  Older Dex3 recordings did not
store explicit end-effector provenance, so that one legacy case remains valid.
New Inspire recordings must carry the provenance written by xr_teleoperate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EndEffectorContract:
    selector: str
    raw_type: str
    raw_protocol: str | None
    hand_dof: int
    vector_dim: int
    require_provenance: bool


END_EFFECTOR_CONTRACTS = {
    "dex3": EndEffectorContract(
        selector="dex3",
        raw_type="dex3",
        raw_protocol=None,
        hand_dof=7,
        vector_dim=28,
        require_provenance=False,
    ),
    "inspire-dfx": EndEffectorContract(
        selector="inspire-dfx",
        raw_type="inspire",
        raw_protocol="dfx",
        hand_dof=6,
        vector_dim=26,
        require_provenance=True,
    ),
    "inspire-ftp": EndEffectorContract(
        selector="inspire-ftp",
        raw_type="inspire",
        raw_protocol="ftp",
        hand_dof=6,
        vector_dim=26,
        require_provenance=True,
    ),
}

INSPIRE_LEFT_JOINT_NAMES = (
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
)
INSPIRE_RIGHT_JOINT_NAMES = (
    "kRightHandPinky",
    "kRightHandRing",
    "kRightHandMiddle",
    "kRightHandIndex",
    "kRightHandThumbBend",
    "kRightHandThumbRotation",
)


def _qpos_length(
    frame: Mapping[str, Any],
    section: str,
    key: str,
    context: str,
    *,
    require_unit_interval: bool,
) -> int:
    section_value = frame.get(section)
    if not isinstance(section_value, Mapping):
        raise ValueError(f"{context}: missing or invalid {section}")
    component = section_value.get(key)
    if not isinstance(component, Mapping):
        raise ValueError(f"{context}: missing or invalid {section}.{key}")
    qpos = component.get("qpos")
    if not isinstance(qpos, Sequence) or isinstance(qpos, (str, bytes)):
        raise ValueError(f"{context}: missing or invalid {section}.{key}.qpos")

    for value_index, value in enumerate(qpos):
        value_context = f"{context}: {section}.{key}.qpos[{value_index}]"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{value_context} must be a JSON number (not bool), got {value!r}")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{value_context} must be finite, got {value!r}")
        if require_unit_interval and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{value_context} must be within [0.0, 1.0] for Inspire, got {value!r}"
            )
    return len(qpos)


def _validate_provenance(
    end_effector: Any,
    contract: EndEffectorContract,
    episode: Path,
) -> dict[str, Any] | None:
    if end_effector is None:
        if contract.require_provenance:
            raise ValueError(
                f"{episode}: info.end_effector is required for {contract.selector}; "
                "record with the Inspire-aware teleop or add verified provenance"
            )
        return None
    if not isinstance(end_effector, Mapping):
        raise ValueError(f"{episode}: info.end_effector must be an object")

    schema_version = end_effector.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(
            f"{episode}: info.end_effector.schema_version must be integer 1, got {schema_version!r}"
        )

    actual_type = end_effector.get("type")
    if actual_type != contract.raw_type:
        raise ValueError(
            f"{episode}: info.end_effector.type={actual_type!r} does not match "
            f"--end-effector {contract.selector} (expected {contract.raw_type!r})"
        )

    actual_hand_dof = end_effector.get("hand_dof")
    if (
        isinstance(actual_hand_dof, bool)
        or not isinstance(actual_hand_dof, int)
        or actual_hand_dof != contract.hand_dof
    ):
        raise ValueError(
            f"{episode}: info.end_effector.hand_dof={actual_hand_dof!r} does not "
            f"match {contract.selector} ({contract.hand_dof})"
        )

    for field in ("type", "protocol", "value_unit", "zero_semantics"):
        value = end_effector.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{episode}: info.end_effector.{field} must be a non-empty string")
    one_semantics = end_effector.get("one_semantics")
    if one_semantics is not None and (not isinstance(one_semantics, str) or not one_semantics):
        raise ValueError(
            f"{episode}: info.end_effector.one_semantics must be null or a non-empty string"
        )

    value_range = end_effector.get("value_range")
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
        raise ValueError(
            f"{episode}: info.end_effector.value_range must contain two ordered finite numbers"
        )

    if contract.raw_protocol is not None:
        actual_protocol = end_effector.get("protocol")
        if actual_protocol != contract.raw_protocol:
            raise ValueError(
                f"{episode}: info.end_effector.protocol={actual_protocol!r} does not "
                f"match --end-effector {contract.selector} "
                f"(expected {contract.raw_protocol!r})"
            )

    canonical_order = end_effector.get("canonical_order")
    if canonical_order != "left_then_right":
        raise ValueError(
            f"{episode}: unsupported info.end_effector.canonical_order="
            f"{canonical_order!r}; expected 'left_then_right'"
        )

    for side in ("left", "right"):
        names = end_effector.get(f"{side}_joint_names")
        if (
            not isinstance(names, list)
            or len(names) != contract.hand_dof
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"{episode}: info.end_effector.{side}_joint_names must contain "
                f"{contract.hand_dof} unique non-empty names"
            )

    if contract.raw_type == "inspire":
        expected_semantics = {
            "schema_version": 1,
            "value_unit": "normalized_open_fraction",
            "value_range": [0.0, 1.0],
            "zero_semantics": "fully_closed",
            "one_semantics": "fully_open",
            "canonical_order": "left_then_right",
            "left_joint_names": list(INSPIRE_LEFT_JOINT_NAMES),
            "right_joint_names": list(INSPIRE_RIGHT_JOINT_NAMES),
        }
        for field, expected in expected_semantics.items():
            actual = end_effector.get(field)
            if actual != expected:
                raise ValueError(
                    f"{episode}: info.end_effector.{field}={actual!r} does not "
                    f"match the native Inspire contract ({expected!r})"
                )

    return dict(end_effector)


def validate_episode_contract(
    episode_paths: Sequence[str | Path],
    end_effector: str,
) -> dict[str, Any]:
    """Validate raw episodes and return a compact summary.

    Every frame must contain 7+7 arm positions and the selected number of hand
    positions in both ``states`` and ``actions``.  Explicit provenance, whenever
    present, must agree across every episode.
    """

    try:
        contract = END_EFFECTOR_CONTRACTS[end_effector]
    except KeyError as exc:
        choices = ", ".join(END_EFFECTOR_CONTRACTS)
        raise ValueError(f"Unsupported end effector {end_effector!r}; choose {choices}") from exc

    if not episode_paths:
        raise ValueError("At least one episode path is required")

    expected_lengths = {
        "left_arm": 7,
        "right_arm": 7,
        "left_ee": contract.hand_dof,
        "right_ee": contract.hand_dof,
    }
    reference_provenance: dict[str, Any] | None = None
    provenance_episode: Path | None = None
    total_frames = 0
    legacy_without_provenance = 0

    for episode_arg in episode_paths:
        episode = Path(episode_arg).resolve()
        data_path = episode / "data.json"
        with data_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{data_path}: root must be an object")

        info = payload.get("info")
        if not isinstance(info, Mapping):
            raise ValueError(f"{data_path}: missing or invalid info")
        provenance = _validate_provenance(info.get("end_effector"), contract, episode)
        if provenance is None:
            legacy_without_provenance += 1
        elif reference_provenance is None:
            reference_provenance = provenance
            provenance_episode = episode
        elif provenance != reference_provenance:
            raise ValueError(
                f"{episode}: info.end_effector differs from {provenance_episode}; "
                "do not combine recordings with different hand contracts"
            )

        frames = payload.get("data")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{data_path}: episode contains no frames")

        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError(f"{data_path} frame {frame_index}: frame must be an object")
            context = f"{data_path} frame {frame_index}"
            for section in ("states", "actions"):
                lengths = {
                    key: _qpos_length(
                        frame,
                        section,
                        key,
                        context,
                        require_unit_interval=(
                            contract.raw_type == "inspire" and key in ("left_ee", "right_ee")
                        ),
                    )
                    for key in expected_lengths
                }
                if lengths != expected_lengths:
                    raise ValueError(
                        f"{context}: {section} component lengths {lengths} do not "
                        f"match {contract.selector} {expected_lengths}"
                    )
                vector_dim = sum(lengths.values())
                if vector_dim != contract.vector_dim:
                    raise ValueError(
                        f"{context}: {section} has {vector_dim} values; expected "
                        f"{contract.vector_dim} for {contract.selector}"
                    )
        total_frames += len(frames)

    return {
        "end_effector": contract.selector,
        "hand_dof": contract.hand_dof,
        "vector_dim": contract.vector_dim,
        "episodes": len(episode_paths),
        "frames": total_frames,
        "legacy_episodes_without_provenance": legacy_without_provenance,
        "provenance": reference_provenance,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-effector",
        required=True,
        choices=tuple(END_EFFECTOR_CONTRACTS),
    )
    parser.add_argument("episode", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = validate_episode_contract(args.episode, args.end_effector)
    provenance_note = ""
    if summary["legacy_episodes_without_provenance"]:
        provenance_note = (
            f"; {summary['legacy_episodes_without_provenance']} legacy Dex3 episode(s) "
            "without info.end_effector"
        )
    print(
        f"Validated {summary['episodes']} {summary['end_effector']} episode(s): "
        f"{summary['frames']} frames, {summary['vector_dim']}D state/action"
        f"{provenance_note}."
    )


if __name__ == "__main__":
    main()
