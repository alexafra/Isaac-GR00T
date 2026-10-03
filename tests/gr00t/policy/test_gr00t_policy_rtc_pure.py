"""CPU-only opt-in guidance plumbing; no checkpoints, GPU or robot access."""

from types import SimpleNamespace

import numpy as np
import pytest
from tests.gr00t.policy import test_gr00t_policy as policy_test_support
import torch
from transformers.feature_extraction_utils import BatchFeature


ACTION_KEYS = policy_test_support.ACTION_KEYS
_make_observation = policy_test_support._make_observation
base_policy = policy_test_support.policy


@pytest.fixture
def pure_policy(base_policy):
    base_policy.model.action_head = SimpleNamespace(supports_rtc_pure=True)
    base_policy.model.get_action.return_value = BatchFeature(
        data={"action_pred": torch.zeros(1, 16, 7), "rtc_pure_applied": True}
    )
    return base_policy


def _options(**overrides):
    options = {
        "inference_mode": "rtc",
        "rtc_pure": True,
        "rtc_previous_action": {
            key: np.full((1, 5, 1), index + 0.123456, dtype=np.float32)
            for index, key in enumerate(ACTION_KEYS)
        },
        "rtc_overlap_steps": 4,
        "rtc_frozen_steps": 2,
    }
    options.update(overrides)
    return options


@pytest.mark.parametrize("frozen_steps", [0, 2, 4])
def test_pure_keeps_fp32_targets_without_replacing_decoded_prefix(pure_policy, frozen_steps):
    options = _options(rtc_frozen_steps=frozen_steps)
    decoded = {
        key: np.arange(16, dtype=np.float32).reshape(1, 16, 1) + 10 + index
        for index, key in enumerate(ACTION_KEYS)
    }
    pure_policy.processor.decode_action.side_effect = lambda *_args: {
        key: values.copy() for key, values in decoded.items()
    }
    del pure_policy.model.config.rtc_ramp_rate  # Pure does not use the legacy ramp.
    actions, info = pure_policy.get_action(_make_observation(), options=options)
    model_options = pure_policy.model.get_action.call_args.kwargs["options"]
    assert model_options["rtc_pure"] is True
    assert model_options["rtc_action_dim"] == 7
    assert model_options["action_horizon"] == 5
    assert "rtc_ramp_rate" not in model_options
    target = model_options["rtc_previous_action_fp32"]
    assert target.dtype == torch.float32
    expected = np.concatenate(list(options["rtc_previous_action"].values()), axis=-1)
    np.testing.assert_array_equal(target[:, :5].numpy(), expected)
    assert torch.count_nonzero(target[:, 5:]) == 0
    for key in ACTION_KEYS:
        # Every decoded action, including the high-weight prefix, is untouched.
        np.testing.assert_array_equal(actions[key], decoded[key])
    assert info["rtc_pure_applied"] is True
    assert info["rtc_sampler"] == "pseudoinverse_guidance_v1"
    assert info["rtc_guidance_max_weight"] == 5.0
    assert info["rtc_physical_prefix_restored"] is False
    assert "rtc_ramp_rate" not in info
    assert (
        pure_policy.processor.state_action_processor.apply_action.call_args.kwargs["clip_outliers"]
        is False
    )


@pytest.mark.parametrize("outer_inference_mode", [False, True])
def test_pure_model_context_allows_local_input_backward(pure_policy, outer_inference_mode):
    def model_call(**_kwargs):
        assert not torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()  # Backbone remains no-grad.
        with torch.enable_grad():
            action = torch.ones(1, 16, 7, requires_grad=True)
            gradient = torch.autograd.grad((action.square()).sum(), action)[0]
        return BatchFeature(data={"action_pred": gradient, "rtc_pure_applied": True})

    pure_policy.model.get_action.side_effect = model_call
    with torch.inference_mode(outer_inference_mode):
        pure_policy.get_action(_make_observation(), options=_options())


def test_legacy_context_and_wire_options_remain_unchanged(pure_policy):
    def model_call(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert kwargs["options"] == {
            "action_horizon": 5,
            "rtc_overlap_steps": 4,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 6.0,
        }
        return BatchFeature(data={"action_pred": torch.zeros(1, 16, 7)})

    pure_policy.model.get_action.side_effect = model_call
    options = _options(rtc_pure=False)
    actions, info = pure_policy.get_action(_make_observation(), options=options)
    for key in ACTION_KEYS:
        np.testing.assert_array_equal(
            actions[key][:, :2], options["rtc_previous_action"][key][:, 1:3]
        )
        np.testing.assert_array_equal(actions[key][:, 2:], 0)
    assert "rtc_pure_applied" not in info


@pytest.mark.parametrize("bad", [1, "true", None, np.bool_(True)])
def test_pure_rejects_non_boolean_flag(pure_policy, bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        pure_policy.get_action(_make_observation(), options=_options(rtc_pure=bad))
    pure_policy.model.get_action.assert_not_called()


def test_pure_rejects_legacy_ramp_override(pure_policy):
    with pytest.raises(ValueError, match="only supported by legacy RTC"):
        pure_policy.get_action(_make_observation(), options=_options(rtc_ramp_rate=6.0))


def test_pure_fails_closed_without_model_capability(pure_policy):
    pure_policy.model.action_head.supports_rtc_pure = False
    with pytest.raises(RuntimeError, match="does not support rtc_pure"):
        pure_policy.get_action(_make_observation(), options=_options())
    pure_policy.model.get_action.assert_not_called()


def test_pure_fails_closed_when_model_ignores_flag(pure_policy):
    pure_policy.model.get_action.return_value = BatchFeature(
        data={"action_pred": torch.zeros(1, 16, 7)}
    )
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        pure_policy.get_action(_make_observation(), options=_options())
    pure_policy.processor.decode_action.assert_not_called()


def test_pure_rejects_nonfinite_before_decode(pure_policy):
    pure_policy.model.get_action.return_value["action_pred"][0, -1, -1] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        pure_policy.get_action(_make_observation(), options=_options())
    pure_policy.processor.decode_action.assert_not_called()


def test_pure_reuses_current_state_reencoding_without_clipping(pure_policy):
    observation = _make_observation()
    options = _options()

    def reencode(action, _embodiment, *, state, clip_outliers):
        assert clip_outliers is False
        return {
            key: value if key == "gripper" else value - state[key][-1, :1]
            for key, value in action.items()
        }

    pure_policy.processor.state_action_processor.apply_action.side_effect = reencode
    pure_policy.get_action(observation, options=options)
    target = pure_policy.model.get_action.call_args.kwargs["options"]["rtc_previous_action_fp32"]
    for index, key in enumerate(ACTION_KEYS):
        expected = options["rtc_previous_action"][key][0]
        if key != "gripper":
            expected = expected - observation["state"][key][0, -1, :1]
        np.testing.assert_array_equal(target[0, :5, index : index + 1].numpy(), expected)


@pytest.mark.parametrize("mode", [None, "synchronous"])
def test_pure_cannot_be_silently_ignored_outside_rtc(pure_policy, mode):
    options = {"rtc_pure": True}
    if mode is not None:
        options["inference_mode"] = mode
    with pytest.raises(ValueError, match="RTC options|RTC keys"):
        pure_policy.get_action(_make_observation(), options=options)
