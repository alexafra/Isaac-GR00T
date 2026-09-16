# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock

import albumentations as A
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VideoChannelSource
from gr00t.data.utils import parse_modality_configs, to_json_serializable
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    Gr00tN1d7DataCollator,
    Gr00tN1d7Processor,
    _post_vision_fusion_contract,
)
import numpy as np
from PIL import Image
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _solid_image(values: tuple[int, int, int]) -> Image.Image:
    array = np.stack(
        [np.full((8, 9), value, dtype=np.uint8) for value in values],
        axis=-1,
    )
    return Image.fromarray(array)


def test_post_vision_fusion_config_roundtrip_and_validation():
    video = ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "depth_gray_view"],
        post_vision_fusion=True,
        post_vision_fusion_stage="pre_vision_language_adapter",
    )
    serialized = to_json_serializable({"tag": {"video": video}})
    restored = parse_modality_configs(serialized)["tag"]["video"]

    assert restored.post_vision_fusion is True
    assert restored.post_vision_fusion_stage == "pre_vision_language_adapter"
    assert restored.modality_keys == ["ego_view", "depth_gray_view"]
    assert restored.vision_input_channels == 3

    with pytest.raises(ValueError, match="exactly two ordered"):
        ModalityConfig(
            [0],
            ["ego_view"],
            post_vision_fusion=True,
            post_vision_fusion_stage="pre_vision_language_adapter",
        )
    with pytest.raises(ValueError, match="requires ordered modality_keys"):
        ModalityConfig(
            [0],
            ["surface_normals_view", "ego_view"],
            post_vision_fusion=True,
            post_vision_fusion_stage="pre_vision_language_adapter",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModalityConfig(
            [0],
            ["ego_view", "depth_gray_view"],
            channel_fusion=[
                VideoChannelSource("ego_view", (0, 1, 2)),
                VideoChannelSource("depth_gray_view", (0,)),
            ],
            post_vision_fusion=True,
            post_vision_fusion_stage="pre_vision_language_adapter",
        )
    with pytest.raises(ValueError, match="post_vision_fusion_stage must be"):
        ModalityConfig(
            [0],
            ["ego_view", "depth_gray_view"],
            post_vision_fusion=True,
        )
    with pytest.raises(ValueError, match="must be None"):
        ModalityConfig(
            [0],
            ["ego_view", "depth_gray_view"],
            post_vision_fusion_stage="post_vision_language_adapter",
        )


def test_processor_rejects_mixed_late_fusion_stages():
    configs = {
        "pre": {
            "video": ModalityConfig(
                [0],
                ["ego_view", "depth_gray_view"],
                post_vision_fusion=True,
                post_vision_fusion_stage="pre_vision_language_adapter",
            )
        },
        "post": {
            "video": ModalityConfig(
                [0],
                ["ego_view", "depth_gray_view"],
                post_vision_fusion=True,
                post_vision_fusion_stage="post_vision_language_adapter",
            )
        },
    }

    with pytest.raises(ValueError, match="one post-vision fusion contract"):
        _post_vision_fusion_contract(configs)


@pytest.mark.parametrize(
    ("config_name", "auxiliary_key", "stage"),
    [
        (
            "g1_inspire_head_rgbd_late_fusion_pre_adapter_config",
            "depth_gray_view",
            "pre_vision_language_adapter",
        ),
        (
            "g1_inspire_head_rgbd_late_fusion_post_adapter_config",
            "depth_gray_view",
            "post_vision_language_adapter",
        ),
        (
            "g1_inspire_head_rgb_surface_normals_late_fusion_pre_adapter_config",
            "surface_normals_view",
            "pre_vision_language_adapter",
        ),
        (
            "g1_inspire_head_rgb_surface_normals_late_fusion_post_adapter_config",
            "surface_normals_view",
            "post_vision_language_adapter",
        ),
    ],
)
def test_inspire_post_vision_late_fusion_configs_are_explicit(config_name, auxiliary_key, stage):
    config_path = REPO_ROOT / "examples" / "UnitreeG1" / f"{config_name}.py"
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    previous = MODALITY_CONFIGS.pop(tag, None)
    sys.path.insert(0, str(config_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"test_{config_name}", config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        video = getattr(module, config_name)["video"]
    finally:
        sys.path.remove(str(config_path.parent))
        MODALITY_CONFIGS.pop(tag, None)
        if previous is not None:
            MODALITY_CONFIGS[tag] = previous

    assert video.modality_keys == ["ego_view", auxiliary_key]
    assert video.post_vision_fusion is True
    assert video.post_vision_fusion_stage == stage
    assert video.channel_fusion is None
    assert video.vision_input_channels == 3


def test_post_vision_fusion_keeps_two_3_channel_branches_and_one_prompt_stream():
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor.use_albumentations = True
    processor.training = True
    processor.processor = MagicMock()
    processor.processor.apply_chat_template.return_value = "one destination image"

    result = processor._get_vlm_inputs(
        image_keys=["ego_view", "surface_normals_view"],
        images={
            "ego_view": [_solid_image((10, 20, 30))],
            "surface_normals_view": [_solid_image((40, 50, 60))],
        },
        masks=None,
        image_transform=A.ReplayCompose([]),
        language="test",
        post_vision_fusion=True,
    )["vlm_content"]

    assert len(result["images"]) == 1
    assert len(result["late_fusion_images"]) == 1
    assert np.asarray(result["images"][0]).shape == (8, 9, 3)
    assert np.asarray(result["late_fusion_images"][0]).shape == (8, 9, 3)
    conversation = processor.processor.apply_chat_template.call_args.args[0]
    assert [item["type"] for item in conversation[0]["content"]] == ["image", "text"]


def test_post_vision_fusion_collator_emits_paired_auxiliary_tensors():
    grid = torch.tensor([[1, 16, 16]])
    primary_pixels = torch.full((256, 1176), 1.0)
    auxiliary_pixels = torch.full((256, 1176), 2.0)
    qwen_processor = MagicMock()
    qwen_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": primary_pixels,
        "image_grid_thw": grid,
    }
    qwen_processor.image_processor.return_value = {
        "pixel_values": auxiliary_pixels,
        "image_grid_thw": grid.clone(),
    }

    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.processor = qwen_processor
    collator.vision_input_channels = 3
    collator.vision_channel_layout = None
    collator.post_vision_fusion = True

    primary_image = _solid_image((10, 20, 30))
    auxiliary_image = _solid_image((40, 50, 60))
    batch = collator(
        [
            {
                "vlm_content": {
                    "text": "one destination image",
                    "images": [primary_image],
                    "late_fusion_images": [auxiliary_image],
                    "vision_input_channels": 3,
                }
            }
        ]
    )["inputs"]

    assert batch["pixel_values"] is primary_pixels
    assert batch["image_grid_thw"] is grid
    assert batch["late_fusion_pixel_values"] is auxiliary_pixels
    assert torch.equal(batch["late_fusion_image_grid_thw"], grid)
    assert qwen_processor.call_args.kwargs["images"] == [primary_image]
    assert qwen_processor.image_processor.call_args.kwargs["images"] == [auxiliary_image]


def test_post_vision_fusion_collator_rejects_misaligned_processed_grids():
    grid = torch.tensor([[1, 16, 16]])
    qwen_processor = MagicMock()
    qwen_processor.return_value = {
        "pixel_values": torch.zeros((256, 1176)),
        "image_grid_thw": grid,
    }
    qwen_processor.image_processor.return_value = {
        "pixel_values": torch.zeros((240, 1176)),
        "image_grid_thw": torch.tensor([[1, 15, 16]]),
    }

    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.processor = qwen_processor
    collator.vision_input_channels = 3
    collator.vision_channel_layout = None
    collator.post_vision_fusion = True

    with pytest.raises(ValueError, match="identical processed image_grid_thw"):
        collator(
            [
                {
                    "vlm_content": {
                        "text": "one destination image",
                        "images": [_solid_image((10, 20, 30))],
                        "late_fusion_images": [_solid_image((40, 50, 60))],
                        "vision_input_channels": 3,
                    }
                }
            ]
        )
