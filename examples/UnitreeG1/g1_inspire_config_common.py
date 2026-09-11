# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared state/action contract for head-camera G1 Inspire configurations."""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


G1_ARM_HAND_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand"]


def make_g1_inspire_config(video: ModalityConfig) -> dict[str, ModalityConfig]:
    """Build a G1 Inspire config while keeping all visual recipes semantically identical."""

    return {
        "video": video,
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=list(G1_ARM_HAND_KEYS),
        ),
        "action": ModalityConfig(
            delta_indices=list(range(32)),
            modality_keys=list(G1_ARM_HAND_KEYS),
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
