# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
from tests.gr00t.model.test_post_vision_late_fusion import (
    POST_ADAPTER,
    PRE_ADAPTER,
    _attach_fake_qwen,
    _FakePostAdapterQwen,
    _FakePreAdapterQwen,
    _late_backbone_shell,
    _late_inputs,
)
import torch


def _enable(backbone):
    backbone.vision_modality_dropout_rgb_prob = 0.1
    backbone.vision_modality_dropout_geometry_prob = 0.1
    backbone.vision_modality_dropout_source_keys = ["ego_view", "surface_normals_view"]
    backbone.set_frozen_modules_to_eval_mode = lambda: None
    backbone.train()
    return backbone


def _metadata(inputs, keep=((0.0, 1.0),), frames=(1,)):
    return {
        **inputs,
        "vision_modality_keep": torch.tensor(keep),
        "vision_modality_num_frames": torch.tensor(frames),
    }


def _mask_shell(merge_size=1, channels=3):
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    visual = types.SimpleNamespace(
        spatial_merge_size=merge_size,
        patch_embed=types.SimpleNamespace(proj=types.SimpleNamespace(in_channels=channels)),
    )
    backbone.model = types.SimpleNamespace(
        model=types.SimpleNamespace(visual=visual), visual=visual
    )
    return _enable(backbone)


def test_token_masks_follow_ragged_episode_time_image_order():
    backbone = _mask_shell()
    inputs = _metadata(
        {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 3]]),
        },
        keep=((0.0, 1.0), (1.0, 0.0)),
        frames=(2, 1),
    )
    assert backbone._modality_token_keep(inputs, separate_views=False, merged=False).tolist() == (
        [[0, 1]] * 3 + [[1, 0]] * 3
    )
    inputs["image_grid_thw"] = torch.tensor(
        [[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 1], [1, 1, 2]]
    )
    assert backbone._modality_token_keep(
        inputs, separate_views=True, merged=False
    ).flatten().tolist() == ([0] + [1] * 2 + [0] * 3 + [1] * 4 + [1] + [0] * 2)


def test_merged_mask_uses_grid_token_counts_not_fixed_resolution():
    backbone = _mask_shell(merge_size=2)
    inputs = _metadata(
        {"input_ids": torch.ones((1, 2)), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    )
    assert backbone._modality_token_keep(inputs, separate_views=False, merged=True).shape == (2, 2)
    inputs["image_grid_thw"] = torch.tensor([[1, 1, 3]])
    with pytest.raises(ValueError, match="spatial-merge"):
        backbone._modality_token_keep(inputs, separate_views=False, merged=True)


@pytest.mark.parametrize("keep", [((0, 0),), ((float("nan"), 1),), ((0.5, 1),), ((1, 2),)])
def test_malformed_or_double_missing_keep_rejected(keep):
    backbone = _mask_shell()
    inputs = _metadata(
        {"input_ids": torch.ones((1, 2)), "image_grid_thw": torch.tensor([[1, 1, 1]])},
        keep=keep,
    )
    with pytest.raises(ValueError):
        backbone._modality_token_keep(inputs, separate_views=False, merged=False)


@pytest.mark.parametrize("frames", [(0,), (-1,), (1, 1), (2,)])
def test_bad_frame_counts_rejected(frames):
    backbone = _mask_shell()
    inputs = _metadata(
        {"input_ids": torch.ones((1, 2)), "image_grid_thw": torch.tensor([[1, 1, 1]])},
        frames=frames,
    )
    with pytest.raises(ValueError):
        backbone._modality_token_keep(inputs, separate_views=False, merged=False)


@pytest.mark.parametrize("channels", [4, 6])
def test_early_fusion_zeroes_normalized_channel_groups_and_their_gradients(channels):
    backbone = _mask_shell(channels=channels)
    pixels = torch.arange(1.0, 2 * channels * 8 + 1).reshape(2, channels * 8).requires_grad_()
    inputs = _metadata(
        {
            "input_ids": torch.ones((2, 2)),
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        },
        keep=((0, 1), (1, 0)),
        frames=(1, 1),
    )
    masked = backbone._mask_early_fusion_pixels(inputs).reshape(2, channels, 8)
    assert torch.count_nonzero(masked[0, :3]) == 0
    assert torch.count_nonzero(masked[1, 3:]) == 0
    assert torch.equal(masked[0, 3:], pixels.reshape(2, channels, 8)[0, 3:])
    assert torch.equal(masked[1, :3], pixels.reshape(2, channels, 8)[1, :3])
    masked.sum().backward()
    assert torch.count_nonzero(pixels.grad.reshape(2, channels, 8)[0, :3]) == 0
    assert torch.count_nonzero(pixels.grad.reshape(2, channels, 8)[1, 3:]) == 0


@pytest.mark.parametrize(
    "stage,qwen_type", [(PRE_ADAPTER, _FakePreAdapterQwen), (POST_ADAPTER, _FakePostAdapterQwen)]
)
@pytest.mark.parametrize("keep", [((0, 1),), ((1, 0),), ((1, 1),)])
def test_late_modality_mask_reaches_all_four_sites_and_gradients(stage, qwen_type, keep):
    backbone = _late_backbone_shell(stage)
    qwen = qwen_type()
    if not hasattr(qwen, "visual"):
        qwen.visual = types.SimpleNamespace(spatial_merge_size=1)
    _attach_fake_qwen(backbone, qwen)
    _enable(backbone)
    primary = torch.tensor([[2.0, 4.0]], requires_grad=True)
    auxiliary = torch.tensor([[8.0, 16.0]], requires_grad=True)
    inputs = _metadata(_late_inputs(primary, auxiliary), keep=keep)
    before = torch.random.get_rng_state().clone()
    result = backbone(inputs)
    assert torch.equal(before, torch.random.get_rng_state())
    expected = 0.5 * (primary * keep[0][0] + auxiliary * keep[0][1])
    assert torch.equal(result.backbone_features[0, 1], expected[0])
    for index, stream in enumerate(qwen.received_deepstack):
        offset = (index + 1) * 10 if stage == POST_ADAPTER else 0
        expected_stream = 0.5 * (
            (primary + offset) * keep[0][0] + (auxiliary + offset) * keep[0][1]
        )
        assert torch.equal(stream, expected_stream)
    result.backbone_features.sum().backward()
    assert torch.equal(primary.grad, torch.full_like(primary, 0.5 * keep[0][0]))
    assert torch.equal(auxiliary.grad, torch.full_like(auxiliary, 0.5 * keep[0][1]))


def test_separate_view_content_removed_before_language_at_all_entry_points():
    backbone = _mask_shell()

    class SeparateQwen(_FakePostAdapterQwen):
        @property
        def language_model(self):
            return self._language

    class Language(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.fail = False

        def forward(self, *, inputs_embeds, deepstack_visual_embeds, **_kwargs):
            qwen.received_deepstack = [
                stream.detach().clone() for stream in deepstack_visual_embeds
            ]
            hidden = self.layers[0](inputs_embeds)
            if self.fail:
                raise RuntimeError("injected language-model failure")
            return types.SimpleNamespace(last_hidden_state=hidden)

    qwen = SeparateQwen()
    qwen._language = Language()
    qwen.visual = types.SimpleNamespace(spatial_merge_size=1)
    backbone.model = types.SimpleNamespace(model=qwen)
    pixels = torch.tensor([[2.0, 4.0], [8.0, 16.0]], requires_grad=True)
    inputs = _metadata(
        {
            "input_ids": torch.tensor([[1, 9, 9, 2]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
    )
    result = backbone._forward_separate_view_modality_dropout(inputs)
    assert result.shape == (1, 4, 2)
    assert torch.count_nonzero(result[0, 1]) == 0
    assert torch.equal(result[0, 2], pixels[1])
    for index, stream in enumerate(qwen.received_deepstack):
        assert torch.count_nonzero(stream[0]) == 0
        assert torch.equal(stream[1], pixels[1] + (index + 1) * 10)
    result.sum().backward()
    assert torch.count_nonzero(pixels.grad[0]) == 0
    assert torch.equal(pixels.grad[1], torch.ones(2))
    assert not qwen.language_model.layers[-1]._forward_hooks
    qwen.language_model.fail = True
    with pytest.raises(RuntimeError, match="injected language-model failure"):
        backbone._forward_separate_view_modality_dropout(inputs)
    assert not qwen.language_model.layers[-1]._forward_hooks


class _CaptureStandard(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.visual = types.SimpleNamespace(
            patch_embed=types.SimpleNamespace(proj=types.SimpleNamespace(in_channels=channels))
        )
        self.config = types.SimpleNamespace(image_token_id=9)
        self.received = None

    def forward(self, **kwargs):
        self.received = kwargs
        return types.SimpleNamespace(hidden_states=[kwargs["pixel_values"]])


@pytest.mark.parametrize("training,enabled", [(False, True), (True, False)])
def test_disabled_or_eval_preserves_standard_path_inputs_and_rng(training, enabled):
    backbone = _mask_shell()
    backbone.model = _CaptureStandard(3)
    backbone.vision_fusion_mode = "none"
    backbone.vision_modality_dropout_rgb_prob = 0.1 if enabled else 0
    backbone.vision_modality_dropout_geometry_prob = 0
    backbone.train(training)
    inputs = {
        "input_ids": torch.tensor([[9]]),
        "attention_mask": torch.ones((1, 1)),
        "pixel_values": torch.ones((1, 2)),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    before = torch.random.get_rng_state().clone()
    backbone(inputs)
    assert torch.equal(before, torch.random.get_rng_state())
    assert backbone.model.received["pixel_values"] is inputs["pixel_values"]
    assert set(backbone.model.received) == {*inputs, "output_hidden_states"}


@pytest.mark.parametrize("channels", [4, 6])
def test_early_forward_consumes_metadata_without_leaking_to_qwen(channels):
    backbone = _mask_shell(channels=channels)
    backbone.model = _CaptureStandard(channels)
    backbone.vision_fusion_mode = "none"
    inputs = _metadata(
        {
            "input_ids": torch.tensor([[9]]),
            "attention_mask": torch.ones((1, 1)),
            "pixel_values": torch.ones((1, channels * 8)),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
    )
    backbone(inputs)
    assert not any(key.startswith("vision_modality_") for key in backbone.model.received)
    packed = backbone.model.received["pixel_values"].reshape(channels, 8)
    assert torch.count_nonzero(packed[:3]) == 0
    assert torch.equal(packed[3:], torch.ones((channels - 3, 8)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_modality_dropout_rgb_prob": -0.1},
        {"vision_modality_dropout_rgb_prob": float("nan")},
        {"vision_modality_dropout_rgb_prob": 0.7, "vision_modality_dropout_geometry_prob": 0.7},
        {"vision_modality_dropout_rgb_prob": 0.1, "vision_modality_dropout_source_keys": []},
    ],
)
def test_invalid_contract_fails_before_backbone_download(overrides):
    with pytest.raises(ValueError, match="dropout"):
        Qwen3Backbone(**overrides)


def test_tiny_real_qwen_separate_keep_both_matches_original_and_dropped_view_cannot_leak():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 3,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1, 2],
        },
        image_token_id=9,
        video_token_id=12,
        vision_start_token_id=10,
        vision_end_token_id=11,
    )
    config._attn_implementation = "eager"
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.model = Qwen3VLForConditionalGeneration(config).eval()
    backbone.vision_fusion_mode = "none"
    _enable(backbone)
    backbone.model.eval().requires_grad_(False)
    pixels = torch.randn((8, 24))
    inputs = {
        "input_ids": torch.tensor([[1, 10, 9, 11, 10, 9, 11, 2]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    with torch.no_grad():
        baseline = backbone.model(**inputs, output_hidden_states=True).hidden_states[-1]
        result = backbone(_metadata(inputs, keep=((1, 1),))).backbone_features
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    with torch.no_grad():
        dropped = backbone(_metadata(inputs)).backbone_features
        changed = pixels.clone()
        changed[:4] = 100 * torch.randn_like(changed[:4])
        changed_result = backbone(_metadata({**inputs, "pixel_values": changed})).backbone_features
    torch.testing.assert_close(dropped, changed_result, rtol=0, atol=0)
    differentiable = pixels.clone().requires_grad_()
    backbone(
        _metadata({**inputs, "pixel_values": differentiable})
    ).backbone_features.sum().backward()
    assert torch.count_nonzero(differentiable.grad[:4]) == 0
    assert torch.count_nonzero(differentiable.grad[4:]) > 0
    assert not backbone.model.model.language_model.layers[-1]._forward_hooks
