# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # earlyfusion
# SPDX-License-Identifier: Apache-2.0  # earlyfusion
import types  # earlyfusion

from gr00t.configs.finetune_config import FinetuneConfig  # earlyfusion
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone  # earlyfusion
import torch  # earlyfusion


def _backbone_with_rgb_patch_embed():  # earlyfusion
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)  # earlyfusion
    torch.nn.Module.__init__(backbone)  # earlyfusion
    patch_embed = torch.nn.Module()  # earlyfusion
    patch_embed.proj = torch.nn.Conv3d(3, 2, kernel_size=(2, 2, 2), bias=True)  # earlyfusion
    patch_embed.in_channels = 3  # earlyfusion
    with torch.no_grad():  # earlyfusion
        patch_embed.proj.weight.copy_(torch.arange(patch_embed.proj.weight.numel()).reshape_as(patch_embed.proj.weight))  # fmt: skip  # earlyfusion
        patch_embed.proj.bias.copy_(torch.tensor([1.25, -2.5]))  # earlyfusion
    visual = torch.nn.Module()  # earlyfusion
    visual.patch_embed = patch_embed  # earlyfusion
    visual.encoder = torch.nn.Linear(2, 2)  # earlyfusion
    visual.config = types.SimpleNamespace(in_channels=3)  # earlyfusion
    language_model = torch.nn.Module()  # earlyfusion
    language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])  # earlyfusion
    model = torch.nn.Module()  # earlyfusion
    model.visual = visual  # earlyfusion
    model.language_model = language_model  # earlyfusion
    model.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(in_channels=3))  # fmt: skip  # earlyfusion
    backbone.model = model  # earlyfusion
    return backbone  # earlyfusion


def test_early_fusion_conv3d_expansion_preserves_pretrained_parameters():  # earlyfusion
    for channels, initialization in ((4, "rgb_mean"), (6, "zeros")):  # earlyfusion
        backbone = _backbone_with_rgb_patch_embed()  # earlyfusion
        old_weight = backbone.model.visual.patch_embed.proj.weight.detach().clone()  # earlyfusion
        old_bias = backbone.model.visual.patch_embed.proj.bias.detach().clone()  # earlyfusion
        backbone.expand_vision_input_channels(channels, initialization)  # earlyfusion
        expanded = backbone.model.visual.patch_embed.proj  # earlyfusion
        assert torch.equal(expanded.weight[:, :3], old_weight)  # earlyfusion
        assert torch.equal(expanded.bias, old_bias)  # earlyfusion
        if channels == 4:  # earlyfusion
            assert torch.equal(expanded.weight[:, 3:4], old_weight.mean(dim=1, keepdim=True))  # fmt: skip  # earlyfusion
        else:  # earlyfusion
            assert torch.count_nonzero(expanded.weight[:, 3:]) == 0  # earlyfusion


def test_early_fusion_patch_embed_can_be_the_only_trainable_visual_layer():  # earlyfusion
    backbone = _backbone_with_rgb_patch_embed()  # earlyfusion
    backbone.expand_vision_input_channels(6, "zeros")  # earlyfusion
    backbone.set_trainable_parameters(False, False, 0, True)  # earlyfusion
    trainable = {name for name, parameter in backbone.named_parameters() if parameter.requires_grad}  # fmt: skip  # earlyfusion
    assert trainable == {"model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias"}  # fmt: skip  # earlyfusion


def test_early_fusion_patch_embed_tuning_is_enabled_by_default():  # earlyfusion
    assert FinetuneConfig.__dataclass_fields__["tune_vision_patch_embed"].default is True  # fmt: skip  # earlyfusion
