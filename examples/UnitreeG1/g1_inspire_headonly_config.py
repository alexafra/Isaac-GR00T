# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T modality configuration for a head-camera G1 with Inspire hands.

The logical modality keys and action semantics match the existing Dex3 setup.
The physical dimensions are deliberately not encoded here: GR00T obtains them
from the dataset's ``meta/modality.json`` slices and normalization statistics.
For Inspire, each hand slice is six-dimensional; see ``modality_inspire.json``.
"""

from g1_inspire_config_common import make_g1_inspire_config
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


g1_inspire_headonly_config = make_g1_inspire_config(
    ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    )
)


register_modality_config(
    g1_inspire_headonly_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
