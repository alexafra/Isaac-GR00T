"""Readout parity against real Qwen, gradients, and legacy checkpoint compatibility."""
import types

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.setup import _reconcile_late_fusion_language_readout
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone


@pytest.mark.parametrize("mode", ["pre_final_rmsnorm", "post_final_rmsnorm"])
def test_serialized_readout_roundtrip(mode):
    config = Gr00tN1d7Config(late_fusion_language_readout=mode)
    restored = Gr00tN1d7Config.from_dict(config.to_dict())
    assert restored.late_fusion_language_readout == mode


def test_missing_field_retains_historical_readout():
    assert Gr00tN1d7Config.from_dict({}).late_fusion_language_readout == "post_final_rmsnorm"


def test_base_adopts_new_readout_but_trained_checkpoint_cannot_silently_switch():
    source = Gr00tN1d7Config()
    target = Gr00tN1d7Config(late_fusion_language_readout="pre_final_rmsnorm")
    with pytest.raises(RuntimeError, match="readout mismatch"):
        _reconcile_late_fusion_language_readout(source, target, initializing_late_fusion=False)
    _reconcile_late_fusion_language_readout(source, target, initializing_late_fusion=True)
    assert source.late_fusion_language_readout == "pre_final_rmsnorm"
    _reconcile_late_fusion_language_readout(source, target, initializing_late_fusion=False)
    with pytest.raises(RuntimeError, match="readout mismatch"):
        _reconcile_late_fusion_language_readout(source, Gr00tN1d7Config(), initializing_late_fusion=False)


def test_invalid_readout_fails_before_weight_loading():
    with pytest.raises(ValueError, match="readout"):
        Qwen3Backbone(late_fusion_language_readout="typo")


def test_real_qwen_pre_norm_output_and_input_gradients_match_original_gr00t():
    torch.manual_seed(17)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 64, "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 16, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1, 2],
        },
        image_token_id=9, video_token_id=12, vision_start_token_id=10, vision_end_token_id=11,
    )
    config._attn_implementation = "eager"
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.model = Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    backbone.vision_fusion_mode = "post_vision_late_fusion"
    pixels = torch.randn(4, 24, requires_grad=True)
    inputs = dict(input_ids=torch.tensor([[1, 10, 9, 11, 2]]),
                  attention_mask=torch.ones(1, 5, dtype=torch.long),
                  pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]]))
    standard = backbone.model(**inputs, output_hidden_states=True).hidden_states[-1]
    chunks, streams = backbone.model.model.get_image_features(pixels, inputs["image_grid_thw"])
    kwargs = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                  primary_grid=inputs["image_grid_thw"], fused_final=torch.cat(chunks),
                  fused_deepstack=streams)
    # No field means legacy (important for old objects/checkpoints).
    legacy = backbone._forward_fused_vision_language(**kwargs)
    expected_legacy = backbone.model.model.language_model.norm(standard)
    torch.testing.assert_close(legacy, expected_legacy, rtol=0, atol=0)
    assert not torch.allclose(legacy, standard)
    backbone.late_fusion_language_readout = "post_final_rmsnorm"
    torch.testing.assert_close(backbone._forward_fused_vision_language(**kwargs), legacy, rtol=0, atol=0)
    backbone.late_fusion_language_readout = "pre_final_rmsnorm"
    corrected = backbone._forward_fused_vision_language(**kwargs)
    torch.testing.assert_close(corrected, standard, rtol=0, atol=0)
    coefficients = torch.randn_like(standard)
    expected_grad = torch.autograd.grad((standard * coefficients).sum(), pixels, retain_graph=True)[0]
    actual_grad = torch.autograd.grad((corrected * coefficients).sum(), pixels)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(actual_grad) > 0
    assert not backbone.model.model.language_model.layers[-1]._forward_hooks


def test_capture_hook_removed_on_forward_failure():
    class BrokenLanguage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, **kwargs):
            self.layers[0](kwargs["inputs_embeds"])
            raise RuntimeError("intentional failure")

    from tests.gr00t.model.test_post_vision_late_fusion import _FakePostAdapterQwen, _late_backbone_shell

    class Qwen(_FakePostAdapterQwen):
        @property
        def language_model(self):
            return self.language

    qwen = Qwen()
    qwen.language = BrokenLanguage()
    backbone = _late_backbone_shell()
    backbone.model = types.SimpleNamespace(model=qwen)
    backbone.late_fusion_language_readout = "pre_final_rmsnorm"
    with pytest.raises(RuntimeError, match="intentional failure"):
        backbone._forward_fused_vision_language(
            input_ids=torch.tensor([[1, 9, 2]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
            primary_grid=torch.tensor([[1, 1, 1]]), fused_final=torch.ones(1, 2),
            fused_deepstack=[torch.ones(1, 2) for _ in range(3)],
        )
    assert not qwen.language.layers[-1]._forward_hooks
