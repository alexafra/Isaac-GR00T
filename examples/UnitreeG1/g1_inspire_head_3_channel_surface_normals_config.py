# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from g1_inspire_config_common import make_g1_inspire_config
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


g1_inspire_head_3_channel_surface_normals_config = make_g1_inspire_config(
    ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "surface_normals_view"],
    )
)

register_modality_config(
    g1_inspire_head_3_channel_surface_normals_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
