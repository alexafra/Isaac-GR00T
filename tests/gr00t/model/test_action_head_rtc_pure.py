"""CPU-only paper RTC tests: no checkpoint weights, CUDA, or robot access."""

import math

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead, sample_initial_action_noise
from gr00t.model.gr00t_n1d7.rtc_pure import (
    rtc_pure_guidance_coefficient,
    rtc_pure_guided_step,
    rtc_pure_weights,
)
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _head(alternate=False):
    config = Gr00tN1d7Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=4,
        max_action_dim=4,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=4,
        max_num_embodiments=2,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=alternate,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        num_timestep_buckets=1000,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )
    return Gr00tN1d7ActionHead(config).eval()


def _inputs(dtype=torch.float32):
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 5, 32, dtype=dtype),
            "backbone_attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "image_mask": torch.tensor([[True, True, False, False, False]]),
        }
    )
    action = BatchFeature(
        data={
            "state": torch.zeros(1, 1, 4, dtype=dtype),
            "action": torch.tensor([[[0.1] * 4, [0.2] * 4, [0.3] * 4, [999.0] * 4]], dtype=dtype),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
        }
    )
    return backbone, action


def _options(**updates):
    result = {
        "rtc_pure": True,
        "rtc_action_dim": 2,
        "action_horizon": 3,
        "rtc_overlap_steps": 2,
        "rtc_frozen_steps": 1,
        "noise_seeds": [123],
    }
    result.update(updates)
    return result


def test_soft_mask_matches_paper_equation_and_ignores_padded_coordinates():
    weights = rtc_pure_weights(
        horizon=8,
        action_dim=4,
        real_action_dim=2,
        overlap_steps=6,
        frozen_steps=2,
        device=torch.device("cpu"),
    )
    expected = [1.0, 1.0]
    expected.extend((6 - i) / 5 * math.expm1((6 - i) / 5) / math.expm1(1) for i in range(2, 6))
    expected.extend([0.0, 0.0])
    torch.testing.assert_close(weights[0, :, 0], torch.tensor(expected))
    torch.testing.assert_close(weights[0, :, 1], torch.tensor(expected))
    assert torch.count_nonzero(weights[:, :, 2:]) == 0


@pytest.mark.parametrize("frozen,overlap", [(0, 0), (0, 4), (4, 4)])
def test_soft_mask_edge_cases(frozen, overlap):
    weights = rtc_pure_weights(
        horizon=4,
        action_dim=2,
        real_action_dim=2,
        overlap_steps=overlap,
        frozen_steps=frozen,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(weights).all()
    assert ((weights >= 0) & (weights <= 1)).all()
    if frozen == overlap:
        assert torch.all(weights[:, :frozen] == 1)
        assert torch.all(weights[:, overlap:] == 0)


@pytest.mark.parametrize(
    "time,expected", [(0.0, 5.0), (0.25, 10 / 3), (0.5, 2.0), (0.75, 10 / 3), (0.99, 5.0)]
)
def test_guidance_coefficient(time, expected):
    assert rtc_pure_guidance_coefficient(time) == pytest.approx(expected)


def test_vjp_matches_analytic_coupled_linear_model_not_detached_velocity():
    matrix = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.7, 0.3]]))
    matrix.grad = torch.full_like(matrix, 17.0)
    x = torch.tensor([[[0.1, -0.2], [0.3, 0.7]]])
    target = torch.tensor([[[0.9, 0.2], [0.5, -0.1]]])
    weights = torch.tensor([[[1.0, 0.0], [0.2, 0.0]]])
    time, step_size = 0.25, 0.25
    velocity = x @ matrix.detach().T
    jacobian = torch.eye(2) + (1 - time) * matrix.detach()
    clean = x + (1 - time) * velocity
    expected_gradient = ((target - clean) * weights) @ jacobian
    expected = x + step_size * (velocity + rtc_pure_guidance_coefficient(time) * expected_gradient)
    result = rtc_pure_guided_step(
        x,
        predict_velocity=lambda latent: latent @ matrix.T,
        target=target,
        weights=weights,
        flow_time=time,
        step_size=step_size,
    )
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad and result.grad_fn is None
    torch.testing.assert_close(matrix.grad, torch.full_like(matrix, 17.0))
    # Padding outputs are not penalized; coupling can still influence any input
    # coordinate through the Jacobian (do not incorrectly mask the VJP again).
    assert torch.count_nonzero(expected_gradient[:, :, 1]) > 0


def test_nonfinite_prediction_fails_closed():
    with pytest.raises(FloatingPointError, match="non-finite"):
        rtc_pure_guided_step(
            torch.zeros(1, 2, 2),
            predict_velocity=lambda latent: latent * float("nan"),
            target=torch.zeros(1, 2, 2),
            weights=torch.ones(1, 2, 2),
            flow_time=0,
            step_size=0.25,
        )


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_tiny_dit_runs_under_outer_inference_mode_without_parameter_grads(alternate, dtype):
    head = _head(alternate).to(dtype)
    parameters = [(parameter, parameter.requires_grad) for parameter in head.parameters()]
    sentinel = next(head.parameters())
    sentinel.grad = torch.ones_like(sentinel)
    with torch.inference_mode():
        backbone, action = _inputs(dtype)
        result = head.get_action(backbone, action, options=_options())
    assert result.rtc_pure_applied is True
    assert result.action_pred.dtype == torch.float32
    assert result.action_pred.shape == (1, 4, 4)
    assert torch.isfinite(result.action_pred).all()
    assert result.action_pred.grad_fn is None and not result.action_pred.requires_grad
    for parameter, requires_grad in parameters:
        assert parameter.requires_grad == requires_grad
        if parameter is sentinel:
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        else:
            assert parameter.grad is None


def test_full_noise_initialization_fp32_target_and_no_prefix_clamp():
    head = _head().to(torch.bfloat16)
    head.num_inference_timesteps = 1
    with torch.no_grad():
        for parameter in head.action_decoder.parameters():
            parameter.zero_()
    backbone, action = _inputs(torch.bfloat16)
    precise_previous = action.action.float()
    precise_previous[:, 1, :2] = 0.1234567
    options = _options(rtc_previous_action_fp32=precise_previous)
    observed_inputs = []
    handle = head.action_encoder.register_forward_pre_hook(
        lambda _module, args: observed_inputs.append(args[0].detach().clone())
    )
    try:
        result = head.get_action(backbone, action, options=options)
    finally:
        handle.remove()
    noise = sample_initial_action_noise(
        batch_size=1,
        action_horizon=4,
        action_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        noise_seeds=[123],
    ).float()
    torch.testing.assert_close(observed_inputs[0].float(), noise, rtol=0, atol=0)
    target = torch.zeros_like(noise)
    target[:, :2] = precise_previous[:, 1:3]
    weights = rtc_pure_weights(
        horizon=4,
        action_dim=4,
        real_action_dim=2,
        overlap_steps=2,
        frozen_steps=1,
        device=torch.device("cpu"),
    )
    expected = noise + 5 * (target - noise) * weights
    torch.testing.assert_close(result.action_pred, expected, rtol=0, atol=0)
    assert not torch.equal(result.action_pred[:, :1, :2], target[:, :1, :2])
    torch.testing.assert_close(result.action_pred[:, :, 2:], noise[:, :, 2:], rtol=0, atol=0)


def test_false_flag_keeps_legacy_outputs_bitwise_identical_and_frozen():
    head = _head()
    backbone, action = _inputs()
    legacy_options = _options(rtc_ramp_rate=6.0)
    legacy_options.pop("rtc_pure")
    default = head.get_action(BatchFeature(data=dict(backbone)), action, options=legacy_options)
    explicit = head.get_action(
        BatchFeature(data=dict(backbone)), action, options={**legacy_options, "rtc_pure": False}
    )
    torch.testing.assert_close(default.action_pred, explicit.action_pred, rtol=0, atol=0)
    torch.testing.assert_close(default.action_pred[:, :1], action.action[:, 1:2], rtol=0, atol=0)
    assert "rtc_pure_applied" not in default and "rtc_pure_applied" not in explicit


def test_bootstrap_without_previous_action_is_unchanged_and_acknowledged():
    head = _head()
    backbone, action = _inputs()
    del action["action"]
    legacy = head.get_action(
        BatchFeature(data=dict(backbone)), action, options={"noise_seeds": [123]}
    )
    pure = head.get_action(BatchFeature(data=dict(backbone)), action, options=_options())
    torch.testing.assert_close(legacy.action_pred, pure.action_pred, rtol=0, atol=0)
    assert pure.rtc_pure_applied is True


@pytest.mark.parametrize(
    "updates",
    [
        {"rtc_frozen_steps": 3},
        {"rtc_action_dim": 0},
        {"rtc_action_dim": 5},
        {"action_horizon": 5},
        {"rtc_overlap_steps": -1},
        {"rtc_frozen_steps": True},
    ],
)
def test_invalid_contract_rejected(updates):
    head = _head()
    with pytest.raises(ValueError, match="RTC pure"):
        head.get_action(*_inputs(), options=_options(**updates))


def test_training_mode_rejected():
    with pytest.raises(ValueError, match="inference-only"):
        _head().train().get_action(*_inputs(), options=_options())


def test_conditioning_features_are_detached_and_encoded_once():
    head = _head(True)
    counts = {"state": 0, "vision": 0}

    def count_state(_module, _args, _output):
        counts["state"] += 1

    def count_vision(_module, _args, _output):
        counts["vision"] += 1

    handles = [
        head.state_encoder.register_forward_hook(count_state),
        head.vlln.register_forward_hook(count_vision),
    ]
    backbone, action = _inputs()
    backbone.backbone_features.requires_grad_(True)
    action.state.requires_grad_(True)
    try:
        result = head.get_action(backbone, action, options=_options())
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"state": 1, "vision": 1}
    assert action.state.grad is None
    assert not result.backbone_features.requires_grad
    assert not result.state_features.requires_grad
