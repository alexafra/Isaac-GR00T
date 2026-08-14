# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # earlyfusion
# SPDX-License-Identifier: Apache-2.0  # earlyfusion
from examples.UnitreeG1.g1_dex3_head_3_channel_gray_depth_config import (  # earlyfusion
    g1_dex3_head_3_channel_gray_depth_config,  # earlyfusion
)  # earlyfusion
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # earlyfusion
from gr00t.data.embodiment_tags import EmbodimentTag  # earlyfusion
from gr00t.data.types import ModalityConfig, VideoChannelSource  # earlyfusion


g1_dex3_head_4_channel_gray_depth_fusion_config = dict(g1_dex3_head_3_channel_gray_depth_config)  # fmt: skip  # earlyfusion
g1_dex3_head_4_channel_gray_depth_fusion_config["video"] = ModalityConfig(  # earlyfusion
    delta_indices=[0],  # earlyfusion
    modality_keys=["ego_view", "depth_gray_view"],  # earlyfusion
    channel_fusion=[  # earlyfusion
        VideoChannelSource("ego_view", (0, 1, 2)),  # earlyfusion
        VideoChannelSource("depth_gray_view", (0,)),  # earlyfusion
    ],  # earlyfusion
)  # earlyfusion
MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value] = g1_dex3_head_4_channel_gray_depth_fusion_config  # fmt: skip  # earlyfusion
