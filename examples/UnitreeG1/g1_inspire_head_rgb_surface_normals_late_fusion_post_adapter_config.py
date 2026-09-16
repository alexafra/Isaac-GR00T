# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Inspire RGB + surface-normal late fusion after the vision-language adapter."""

from g1_inspire_config_common import make_g1_inspire_config
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


g1_inspire_head_rgb_surface_normals_late_fusion_post_adapter_config = make_g1_inspire_config(
    ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "surface_normals_view"],
        post_vision_fusion=True,
        post_vision_fusion_stage="post_vision_language_adapter",
    )
)

register_modality_config(
    g1_inspire_head_rgb_surface_normals_late_fusion_post_adapter_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
