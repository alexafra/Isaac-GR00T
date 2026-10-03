# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Inspire RGB + fixed-Turbo depth concatenated before the vision patch embedding."""

from g1_inspire_config_common import make_g1_inspire_config
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.depth_colormap import FIXED_TURBO_DEPTH_COLORMAP
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VideoChannelSource


g1_inspire_head_6_channel_turbo_depth_fusion_config = make_g1_inspire_config(
    ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "depth_gray_view"],
        channel_fusion=[
            VideoChannelSource("ego_view", (0, 1, 2)),
            VideoChannelSource("depth_gray_view", (0, 1, 2)),
        ],
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )
)

register_modality_config(
    g1_inspire_head_6_channel_turbo_depth_fusion_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
