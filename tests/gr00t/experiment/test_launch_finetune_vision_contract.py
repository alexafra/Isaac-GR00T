# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gr00t.data.types import ModalityConfig, VideoChannelSource
from gr00t.experiment.launch_finetune import (
    configure_vision_input_contract,
    validate_vision_trainability_contract,
)
import pytest


def _config(video: ModalityConfig):
    return SimpleNamespace(
        data=SimpleNamespace(
            modality_configs={
                "new_embodiment": {"video": video},
                "inactive_pretraining_embodiment": {"video": ModalityConfig([0], ["ego_view"])},
            }
        ),
        model=SimpleNamespace(
            vision_input_channels=3,
            vision_channel_layout=["ego_view:0", "ego_view:1", "ego_view:2"],
            vision_patch_embed_init="original_rgb",
            vision_fusion_mode="none",
            vision_fusion_source_keys=[],
            tune_llm=False,
            tune_visual=False,
            tune_vision_patch_embed=False,
            tune_top_llm_layers=0,
        ),
    )


@pytest.mark.parametrize(
    ("auxiliary_key", "stage"),
    [
        ("depth_gray_view", "pre_vision_language_adapter"),
        ("depth_gray_view", "post_vision_language_adapter"),
        ("surface_normals_view", "pre_vision_language_adapter"),
        ("surface_normals_view", "post_vision_language_adapter"),
    ],
)
def test_launch_maps_post_vision_recipe_to_four_adapter_model_contract(auxiliary_key, stage):
    video = ModalityConfig(
        [0],
        ["ego_view", auxiliary_key],
        post_vision_fusion=True,
        post_vision_fusion_stage=stage,
    )
    config = _config(video)

    configure_vision_input_contract(config, "new_embodiment", video, None)

    assert list(config.data.modality_configs) == ["new_embodiment"]
    assert config.model.vision_input_channels == 3
    assert config.model.vision_channel_layout == [
        "ego_view:0",
        "ego_view:1",
        "ego_view:2",
    ]
    assert config.model.vision_patch_embed_init == "original_rgb"
    assert config.model.vision_fusion_mode == "post_vision_late_fusion"
    assert config.model.vision_fusion_version == 1
    assert config.model.vision_fusion_stage == stage
    assert config.model.vision_fusion_adapter_type == "concat_linear"
    assert config.model.vision_fusion_adapter_count == 4
    assert config.model.vision_fusion_deepstack_layers == [5, 11, 17]
    expected_dims = {
        "pre_vision_language_adapter": (2048, 1024),
        "post_vision_language_adapter": (4096, 2048),
    }
    assert (
        config.model.vision_fusion_adapter_input_dim,
        config.model.vision_fusion_adapter_output_dim,
    ) == expected_dims[stage]
    assert config.model.vision_fusion_source_keys == ["ego_view", auxiliary_key]
    assert config.model.vision_fusion_adapter_init == "rgb50_geo50"
    assert config.model.vision_fusion_rgb_init_scale == pytest.approx(0.5)
    assert config.model.vision_fusion_geometry_init_scale == pytest.approx(0.5)
    assert config.model.vision_fusion_adapter_bias is True
    assert config.model.vision_fusion_adapter_bias_init == "zeros"
    assert config.model.tune_vision_fusion_adapters is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tune_llm", True),
        ("tune_visual", True),
        ("tune_vision_patch_embed", True),
        ("tune_top_llm_layers", 1),
    ],
)
def test_post_vision_recipe_rejects_unfrozen_qwen_parameters(field, value):
    video = ModalityConfig(
        [0],
        ["ego_view", "surface_normals_view"],
        post_vision_fusion=True,
        post_vision_fusion_stage="pre_vision_language_adapter",
    )
    config = _config(video)
    setattr(config.model, field, value)

    configure_vision_input_contract(config, "new_embodiment", video, None)

    with pytest.raises(ValueError, match=field):
        validate_vision_trainability_contract(config, video)


def test_post_vision_recipe_accepts_frozen_qwen_and_trainable_fusion_adapters():
    video = ModalityConfig(
        [0],
        ["ego_view", "surface_normals_view"],
        post_vision_fusion=True,
        post_vision_fusion_stage="post_vision_language_adapter",
    )
    config = _config(video)

    configure_vision_input_contract(config, "new_embodiment", video, None)
    validate_vision_trainability_contract(config, video)


def test_launch_preserves_existing_separate_view_contract():
    video = ModalityConfig([0], ["ego_view", "surface_normals_view"])
    config = _config(video)

    configure_vision_input_contract(config, "new_embodiment", video, None)

    assert list(config.data.modality_configs) == [
        "new_embodiment",
        "inactive_pretraining_embodiment",
    ]
    assert config.model.vision_fusion_mode == "none"
    assert config.model.vision_fusion_source_keys == []


def test_launch_preserves_existing_early_fusion_mapping():
    video = ModalityConfig(
        [0],
        ["ego_view", "depth_gray_view"],
        channel_fusion=[
            VideoChannelSource("ego_view", (0, 1, 2)),
            VideoChannelSource("depth_gray_view", (0,)),
        ],
    )
    config = _config(video)

    configure_vision_input_contract(config, "new_embodiment", video, "rgb_mean")

    assert list(config.data.modality_configs) == ["new_embodiment"]
    assert config.model.vision_input_channels == 4
    assert config.model.vision_channel_layout == [
        "ego_view:0",
        "ego_view:1",
        "ego_view:2",
        "depth_gray_view:0",
    ]
    assert config.model.vision_patch_embed_init == "rgb_mean"
    assert config.model.vision_fusion_mode == "none"


def test_launch_rejects_patch_embed_override_for_post_vision_fusion():
    video = ModalityConfig(
        [0],
        ["ego_view", "depth_gray_view"],
        post_vision_fusion=True,
        post_vision_fusion_stage="pre_vision_language_adapter",
    )

    with pytest.raises(ValueError, match="only valid for early-fusion"):
        configure_vision_input_contract(_config(video), "new_embodiment", video, "zeros")
