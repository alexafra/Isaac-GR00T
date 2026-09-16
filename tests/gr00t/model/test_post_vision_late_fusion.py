# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
import torch


PRE_ADAPTER = "pre_vision_language_adapter"
POST_ADAPTER = "post_vision_language_adapter"


def _late_contract(stage: str = PRE_ADAPTER) -> dict:
    input_dim, output_dim = {
        PRE_ADAPTER: (2048, 1024),
        POST_ADAPTER: (4096, 2048),
    }[stage]
    return {
        "mode": "post_vision_late_fusion",
        "version": 1,
        "stage": stage,
        "adapter_type": "concat_linear",
        "adapter_bias": True,
        "adapter_bias_init": "zeros",
        "adapter_count": 4,
        "deepstack_layers": [5, 11, 17],
        "adapter_input_dim": input_dim,
        "adapter_output_dim": output_dim,
        "source_keys": ["ego_view", "surface_normals_view"],
        "adapter_init": "rgb50_geo50",
        "rgb_init_scale": 0.5,
        "geometry_init_scale": 0.5,
    }


def _late_backbone_shell(stage: str = PRE_ADAPTER, hidden_size: int = 2) -> Qwen3Backbone:
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.vision_fusion_mode = "post_vision_late_fusion"
    backbone.vision_fusion_stage = stage
    backbone.vision_fusion_adapter_bias = True
    backbone.vision_fusion_adapter_bias_init = "zeros"
    backbone.vision_fusion_deepstack_layers = [5, 11, 17]
    backbone.vision_fusion_adapter_input_dim = 2 * hidden_size
    backbone.vision_fusion_adapter_output_dim = hidden_size
    backbone.vision_fusion_adapters = torch.nn.ModuleDict(
        {
            name: torch.nn.Linear(2 * hidden_size, hidden_size, bias=True)
            for name in ("final", "deepstack_0", "deepstack_1", "deepstack_2")
        }
    )
    with torch.no_grad():
        for adapter in backbone.vision_fusion_adapters.values():
            adapter.weight.zero_()
            adapter.weight[:, :hidden_size].diagonal().fill_(0.5)
            adapter.weight[:, hidden_size:].diagonal().fill_(0.5)
            adapter.bias.zero_()
    return backbone


@pytest.mark.parametrize(
    ("stage", "input_dim", "output_dim"),
    [(PRE_ADAPTER, 2048, 1024), (POST_ADAPTER, 4096, 2048)],
)
def test_late_fusion_config_roundtrip_is_explicit_and_opt_in(stage, input_dim, output_dim):
    default = Gr00tN1d7Config()
    assert default.vision_fusion_mode == "none"
    assert default.vision_fusion_source_keys == []

    configured = Gr00tN1d7Config(
        vision_fusion_mode="post_vision_late_fusion",
        vision_fusion_stage=stage,
        vision_fusion_adapter_input_dim=input_dim,
        vision_fusion_adapter_output_dim=output_dim,
        vision_fusion_source_keys=["ego_view", "depth_gray_view"],
    )
    restored = Gr00tN1d7Config.from_dict(configured.to_dict())
    assert restored.vision_fusion_mode == "post_vision_late_fusion"
    assert restored.vision_fusion_version == 1
    assert restored.vision_fusion_stage == stage
    assert restored.vision_fusion_adapter_type == "concat_linear"
    assert restored.vision_fusion_adapter_bias is True
    assert restored.vision_fusion_adapter_bias_init == "zeros"
    assert restored.vision_fusion_adapter_count == 4
    assert restored.vision_fusion_deepstack_layers == [5, 11, 17]
    assert restored.vision_fusion_adapter_input_dim == input_dim
    assert restored.vision_fusion_adapter_output_dim == output_dim
    assert restored.vision_fusion_source_keys == ["ego_view", "depth_gray_view"]
    assert restored.vision_fusion_adapter_init == "rgb50_geo50"
    assert restored.vision_fusion_rgb_init_scale == 0.5
    assert restored.vision_fusion_geometry_init_scale == 0.5


@pytest.mark.parametrize("stage", [PRE_ADAPTER, POST_ADAPTER])
def test_late_fusion_contract_accepts_both_explicit_stages(stage):
    Qwen3Backbone.validate_post_vision_fusion_contract(
        **_late_contract(stage), vision_input_channels=3
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"version": 2}, "version"),
        ({"version": True}, "version"),
        ({"stage": "pre_merger"}, "stage"),
        ({"adapter_type": "average"}, "adapter_type"),
        ({"adapter_bias": False}, "adapter_bias"),
        ({"adapter_bias_init": "random"}, "adapter_bias_init"),
        ({"adapter_count": 1}, "four independent"),
        ({"adapter_count": 4.0}, "four independent"),
        ({"deepstack_layers": [5, 11]}, "deepstack_layers"),
        ({"deepstack_layers": (5, 11, 17)}, "deepstack_layers"),
        ({"adapter_input_dim": 4096}, "dimensions"),
        ({"adapter_output_dim": 2048}, "dimensions"),
        ({"source_keys": ["ego_view"]}, "exactly two"),
        ({"source_keys": ("ego_view", "surface_normals_view")}, "exactly two"),
        ({"source_keys": ["ego_view", "ego_view"]}, "distinct"),
        ({"adapter_init": "rgb90_geo10"}, "adapter_init"),
        ({"rgb_init_scale": 0.9}, "exact serialized scales"),
        ({"geometry_init_scale": 0.1}, "exact serialized scales"),
        ({"geometry_init_scale": float("nan")}, "exact serialized scales"),
        ({"vision_input_channels": 6}, "three-channel"),
    ],
)
def test_late_fusion_contract_rejects_architecture_drift(override, message):
    contract = {**_late_contract(), "vision_input_channels": 3, **override}
    with pytest.raises(ValueError, match=message):
        Qwen3Backbone.validate_post_vision_fusion_contract(**contract)


def _full_dimension_backbone_shell() -> Qwen3Backbone:
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.trainable_params_fp32 = False
    backbone.vision_fusion_mode = "none"
    backbone.vision_fusion_version = 1
    backbone.vision_fusion_stage = PRE_ADAPTER
    backbone.vision_fusion_adapter_type = "concat_linear"
    backbone.vision_fusion_adapter_bias = True
    backbone.vision_fusion_adapter_bias_init = "zeros"
    backbone.vision_fusion_adapter_count = 4
    backbone.vision_fusion_deepstack_layers = [5, 11, 17]
    backbone.vision_fusion_adapter_input_dim = 2048
    backbone.vision_fusion_adapter_output_dim = 1024
    backbone.vision_fusion_source_keys = []
    backbone.vision_fusion_adapter_init = "rgb50_geo50"
    backbone.vision_fusion_rgb_init_scale = 0.5
    backbone.vision_fusion_geometry_init_scale = 0.5
    backbone.vision_fusion_adapters = None
    backbone.tune_llm = False
    backbone.tune_visual = False
    backbone.tune_vision_patch_embed = False
    backbone.tune_top_llm_layers = 0

    visual = torch.nn.Module()
    visual.patch_embed = torch.nn.Module()
    visual.patch_embed.proj = torch.nn.Conv3d(3, 1, 1, bias=False)
    visual.deepstack_visual_indexes = [5, 11, 17]
    visual.deepstack_merger_list = torch.nn.ModuleList(
        [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()]
    )
    model = torch.nn.Module()
    model.visual = visual
    model.anchor = torch.nn.Parameter(torch.zeros(()))
    model.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(hidden_size=1024, out_hidden_size=2048),
        text_config=types.SimpleNamespace(hidden_size=2048),
    )
    backbone.model = model
    return backbone


@pytest.mark.parametrize(
    ("stage", "input_dim", "output_dim", "parameter_count"),
    [
        (PRE_ADAPTER, 2048, 1024, 8_392_704),
        (POST_ADAPTER, 4096, 2048, 33_562_624),
    ],
)
def test_enable_late_fusion_creates_four_independent_weighted_adapters(
    stage, input_dim, output_dim, parameter_count
):
    backbone = _full_dimension_backbone_shell()
    contract = _late_contract(stage)
    backbone.enable_post_vision_late_fusion(**contract, tune_adapters=True)

    adapters = backbone.vision_fusion_adapters
    assert adapters is not None
    assert tuple(adapters) == ("final", "deepstack_0", "deepstack_1", "deepstack_2")
    assert len({id(adapter.weight) for adapter in adapters.values()}) == 4
    assert sum(parameter.numel() for parameter in adapters.parameters()) == parameter_count
    assert not any(parameter.requires_grad for parameter in backbone.model.parameters())
    assert all(parameter.requires_grad for parameter in adapters.parameters())
    for adapter in adapters.values():
        assert adapter.bias is not None
        assert adapter.weight.shape == (output_dim, input_dim)
        assert adapter.bias.shape == (output_dim,)
        assert torch.count_nonzero(adapter.weight[:, :output_dim]) == output_dim
        assert torch.count_nonzero(adapter.weight[:, output_dim:]) == output_dim
        assert torch.equal(
            adapter.weight[:, :output_dim].diagonal(), torch.full((output_dim,), 0.5)
        )
        assert torch.equal(
            adapter.weight[:, output_dim:].diagonal(), torch.full((output_dim,), 0.5)
        )
        assert torch.count_nonzero(adapter.bias) == 0


@pytest.mark.parametrize("stage", [PRE_ADAPTER, POST_ADAPTER])
def test_rgb50_geo50_initialization_is_exact_and_both_blocks_and_bias_learn(stage):
    backbone = _late_backbone_shell(stage=stage, hidden_size=4)
    primary = torch.tensor([[0.2, -0.4, 0.6, 0.8], [1.0, 0.5, -0.3, 0.7], [-0.8, 0.9, 0.4, -0.2]])
    geometry = torch.tensor([[0.7, -0.2, 1.1, 0.3], [-0.5, 0.8, 0.4, 1.2], [0.9, 0.6, -0.7, 0.5]])
    target = torch.tensor([[-0.1, 0.9, 0.2, -0.6], [0.4, -0.7, 0.8, 0.1], [0.3, 0.2, -0.9, 1.0]])

    output = backbone._fuse_vision_streams("final", primary, geometry)
    assert torch.allclose(output, 0.5 * primary + 0.5 * geometry, rtol=0, atol=1e-7)

    torch.nn.functional.mse_loss(output, target).backward()
    adapter = backbone.vision_fusion_adapters["final"]
    rgb_gradient = adapter.weight.grad[:, :4]
    geometry_gradient = adapter.weight.grad[:, 4:]
    assert torch.count_nonzero(rgb_gradient) == rgb_gradient.numel()
    assert torch.count_nonzero(geometry_gradient) == geometry_gradient.numel()
    assert rgb_gradient.std() > 0
    assert geometry_gradient.std() > 0
    assert torch.unique(geometry_gradient).numel() > 1
    assert torch.count_nonzero(adapter.bias.grad) == adapter.bias.numel()


class _IdentityVisionBlock(torch.nn.Module):
    def forward(self, hidden_states, *, cu_seqlens, position_embeddings):
        del cu_seqlens, position_embeddings
        return hidden_states


class _CaptureMerger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, hidden_states):
        self.last_input = hidden_states.detach().clone()
        return hidden_states


class _CapturePatchEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.combined_pixels = None

    def forward(self, pixels):
        self.combined_pixels = pixels.detach().clone()
        return pixels


class _FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.patch_embed = _CapturePatchEmbed()
        self.blocks = torch.nn.ModuleList([_IdentityVisionBlock() for _ in range(18)])
        self.deepstack_visual_indexes = [5, 11, 17]
        self.deepstack_merger_list = torch.nn.ModuleList(
            [_CaptureMerger(), _CaptureMerger(), _CaptureMerger()]
        )
        self.merger = _CaptureMerger()
        self.spatial_merge_size = 1

    @staticmethod
    def fast_pos_embed_interpolate(grid):
        return torch.zeros((int(grid.prod(dim=-1).sum()), 2))

    @staticmethod
    def rot_pos_emb(grid):
        return torch.zeros((int(grid.prod(dim=-1).sum()), 1))


class _FakeQwenBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 2)
        self.rope_deltas = None
        self.received_deepstack = None

    def get_input_embeddings(self):
        return self.embedding

    @staticmethod
    def get_placeholder_mask(input_ids, inputs_embeds, image_features=None, **_kwargs):
        mask = (input_ids == 9).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[mask].numel() != image_features.numel():
            raise ValueError("image/token mismatch")
        return mask, torch.zeros_like(mask)

    @staticmethod
    def get_rope_index(input_ids, image_grid_thw, _video_grid, attention_mask):
        del image_grid_thw, attention_mask
        return torch.zeros((3, *input_ids.shape), dtype=torch.long), torch.zeros((1, 1))

    def language_model(self, *, inputs_embeds, deepstack_visual_embeds, **_kwargs):
        self.received_deepstack = [stream.detach().clone() for stream in deepstack_visual_embeds]
        return types.SimpleNamespace(last_hidden_state=inputs_embeds)


class _FakePreAdapterQwen(_FakeQwenBase):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()


class _FakePostAdapterQwen(_FakeQwenBase):
    def __init__(self):
        super().__init__()
        self.combined_pixels = None

    def get_image_features(self, pixels, grids):
        self.combined_pixels = pixels.detach().clone()
        chunks = tuple(pixels[index : index + 1] for index in range(grids.shape[0]))
        return chunks, [pixels + offset for offset in (10.0, 20.0, 30.0)]


def _attach_fake_qwen(backbone: Qwen3Backbone, qwen: torch.nn.Module) -> None:
    conditional = torch.nn.Module()
    conditional.model = qwen
    conditional.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(hidden_size=2, out_hidden_size=2),
        text_config=types.SimpleNamespace(hidden_size=2),
        image_token_id=9,
    )
    backbone.model = conditional
    backbone.eval()


def _late_inputs(primary, auxiliary):
    return {
        "input_ids": torch.tensor([[1, 9, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values": primary,
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
        "late_fusion_pixel_values": auxiliary,
        "late_fusion_image_grid_thw": torch.tensor([[1, 1, 1]]),
    }


def test_pre_adapter_forward_fuses_before_all_four_existing_qwen_projections():
    backbone = _late_backbone_shell(PRE_ADAPTER)
    with torch.no_grad():
        for scale, adapter in enumerate(backbone.vision_fusion_adapters.values(), start=1):
            adapter.weight[:, 2:].diagonal().fill_(scale)
    qwen = _FakePreAdapterQwen()
    _attach_fake_qwen(backbone, qwen)

    primary = torch.tensor([[1.0, 2.0]])
    auxiliary = torch.tensor([[101.0, 202.0]])
    result = backbone(_late_inputs(primary, auxiliary))

    assert torch.equal(
        qwen.visual.patch_embed.combined_pixels, torch.cat((primary, auxiliary), dim=0)
    )
    expected_final = 0.5 * primary + auxiliary
    assert torch.equal(qwen.visual.merger.last_input, expected_final)
    assert torch.equal(result.backbone_features[0, 1], expected_final[0])
    for index, (merger, stream) in enumerate(
        zip(qwen.visual.deepstack_merger_list, qwen.received_deepstack, strict=True)
    ):
        expected = 0.5 * primary + (index + 2) * auxiliary
        assert torch.equal(merger.last_input, expected)
        assert torch.equal(stream, expected)
    assert result.image_mask.tolist() == [[False, True, False]]


def test_post_adapter_forward_fuses_all_four_projected_qwen_streams():
    backbone = _late_backbone_shell(POST_ADAPTER)
    with torch.no_grad():
        for scale, adapter in enumerate(backbone.vision_fusion_adapters.values(), start=1):
            adapter.weight[:, 2:].diagonal().fill_(scale)
    qwen = _FakePostAdapterQwen()
    _attach_fake_qwen(backbone, qwen)

    primary = torch.tensor([[1.0, 2.0]])
    auxiliary = torch.tensor([[101.0, 202.0]])
    result = backbone(_late_inputs(primary, auxiliary))

    assert torch.equal(qwen.combined_pixels, torch.cat((primary, auxiliary), dim=0))
    assert torch.equal(result.backbone_features[0, 1], (0.5 * primary + auxiliary)[0])
    for index, stream in enumerate(qwen.received_deepstack):
        offset = (index + 1) * 10.0
        scale = index + 2
        assert torch.equal(stream, 0.5 * (primary + offset) + scale * (auxiliary + offset))
    assert result.image_mask.tolist() == [[False, True, False]]


@pytest.mark.parametrize(
    ("stage", "qwen"),
    [(PRE_ADAPTER, _FakePreAdapterQwen), (POST_ADAPTER, _FakePostAdapterQwen)],
)
def test_late_forward_rejects_unaligned_grids_before_vision_compute(stage, qwen):
    backbone = _late_backbone_shell(stage)
    _attach_fake_qwen(backbone, qwen())
    inputs = _late_inputs(torch.zeros((1, 2)), torch.zeros((1, 2)))
    inputs["late_fusion_image_grid_thw"] = torch.tensor([[1, 1, 2]])
    with pytest.raises(ValueError, match="image_grid_thw tensors must be exactly equal"):
        backbone(inputs)


def test_pre_adapter_forward_rejects_runtime_deepstack_layout_drift():
    backbone = _late_backbone_shell(PRE_ADAPTER)
    qwen = _FakePreAdapterQwen()
    qwen.visual.deepstack_visual_indexes = [4, 11, 17]
    _attach_fake_qwen(backbone, qwen)
    with pytest.raises(RuntimeError, match="no longer match the serialized"):
        backbone(_late_inputs(torch.zeros((1, 2)), torch.zeros((1, 2))))


@pytest.mark.parametrize(
    "override",
    [
        {"tune_llm": True},
        {"tune_visual": True},
        {"tune_vision_patch_embed": True},
        {"tune_top_llm_layers": 1},
        {"tune_vision_fusion_adapters": False},
    ],
)
def test_late_fusion_rejects_incompatible_qwen_trainability_before_model_load(override):
    kwargs = {
        "vision_fusion_mode": "post_vision_late_fusion",
        "vision_fusion_source_keys": ["ego_view", "surface_normals_view"],
        **override,
    }
    with pytest.raises(ValueError, match="Late fusion requires|requires tune"):
        Qwen3Backbone(**kwargs)


def test_default_backbone_rejects_late_inputs_instead_of_silently_ignoring_them():
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.vision_fusion_mode = "none"
    backbone.eval()
    with pytest.raises(ValueError, match="non-late-fusion checkpoint"):
        backbone(
            {
                "late_fusion_pixel_values": torch.empty(0),
                "late_fusion_image_grid_thw": torch.empty((0, 3), dtype=torch.long),
            }
        )
