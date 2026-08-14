# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # earlyfusion
# SPDX-License-Identifier: Apache-2.0  # earlyfusion
from pathlib import Path  # earlyfusion
from unittest.mock import MagicMock, patch  # earlyfusion

import albumentations as A  # earlyfusion
from gr00t.data.types import ModalityConfig, VideoChannelSource  # earlyfusion
from gr00t.data.utils import parse_modality_configs, to_json_serializable  # earlyfusion
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # earlyfusion
import numpy as np  # earlyfusion
from PIL import Image  # earlyfusion


def test_early_fusion_produces_one_channel_first_image():  # earlyfusion
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)  # earlyfusion
    processor.use_albumentations = True  # earlyfusion
    processor.training = True  # earlyfusion
    processor.processor = MagicMock()  # earlyfusion
    processor.processor.apply_chat_template.return_value = "one image"  # earlyfusion
    rgb = np.stack(  # earlyfusion
        [np.full((8, 9), value, np.uint8) for value in (10, 20, 30)],  # earlyfusion
        axis=-1,  # earlyfusion
    )  # earlyfusion
    geometry = np.stack(  # earlyfusion
        [np.full((8, 9), value, np.uint8) for value in (40, 50, 60)],  # earlyfusion
        axis=-1,  # earlyfusion
    )  # earlyfusion
    for channels, expected_channels in [((0,), 4), ((0, 1, 2), 6)]:  # earlyfusion
        result = processor._get_vlm_inputs(  # earlyfusion
            ["ego_view", "depth_view"],  # earlyfusion
            {  # earlyfusion
                "ego_view": [Image.fromarray(rgb)],  # earlyfusion
                "depth_view": [Image.fromarray(geometry)],  # earlyfusion
            },  # earlyfusion
            None,  # earlyfusion
            A.ReplayCompose([]),  # earlyfusion
            "test",  # earlyfusion
            [  # earlyfusion
                VideoChannelSource("ego_view", (0, 1, 2)),  # earlyfusion
                VideoChannelSource("depth_view", channels),  # earlyfusion
            ],  # earlyfusion
        )["vlm_content"]  # earlyfusion
        assert len(result["images"]) == 1  # earlyfusion
        assert result["images"][0].shape == (expected_channels, 8, 9)  # earlyfusion
        assert result["vision_input_channels"] == expected_channels  # earlyfusion


def test_early_fusion_config_roundtrip():  # earlyfusion
    video = ModalityConfig(  # earlyfusion
        [0],  # earlyfusion
        ["ego_view", "depth_view"],  # earlyfusion
        channel_fusion=[  # earlyfusion
            VideoChannelSource("ego_view", (0, 1, 2)),  # earlyfusion
            VideoChannelSource("depth_view", (0,)),  # earlyfusion
        ],  # earlyfusion
    )  # earlyfusion
    serialized = to_json_serializable({"tag": {"video": video}})  # earlyfusion
    restored = parse_modality_configs(serialized)["tag"]["video"]  # earlyfusion
    assert restored.channel_fusion == video.channel_fusion  # earlyfusion
    assert restored.vision_input_channels == 4  # earlyfusion


def test_early_fusion_override_replaces_pretrained_modalities():  # earlyfusion
    video = ModalityConfig(  # earlyfusion
        [0],  # earlyfusion
        ["ego_view", "depth_view"],  # earlyfusion
        channel_fusion=[  # earlyfusion
            VideoChannelSource("ego_view", (0, 1, 2)),  # earlyfusion
            VideoChannelSource("depth_view", (0,)),  # earlyfusion
        ],  # earlyfusion
    )  # earlyfusion
    override = {"fused": {"video": video}}  # earlyfusion
    fixture = Path(__file__).parents[2] / "fixtures" / "processor_config"  # earlyfusion
    with patch.object(  # earlyfusion
        Gr00tN1d7Processor,  # earlyfusion
        "__init__",  # earlyfusion
        return_value=None,  # earlyfusion
    ) as initializer:  # earlyfusion
        Gr00tN1d7Processor.from_pretrained(fixture, modality_configs=override)  # earlyfusion
    assert initializer.call_args.kwargs["modality_configs"] == override  # earlyfusion
