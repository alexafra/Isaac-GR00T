# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Gr00tPolicy: observation validation and inference pipeline.

Uses mocked model and processor to avoid downloading checkpoints.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.data.types import ModalityConfig, VideoChannelSource  # earlyfusion
from gr00t.policy.gr00t_policy import _vision_input_contract  # earlyfusion
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"

VIDEO_KEYS = ["observation.images.rgb.head_256_256", "observation.images.rgb.left_wrist_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


def test_early_fusion_contract_is_exact_and_rejects_model_mismatch():  # earlyfusion
    video = ModalityConfig(delta_indices=[0], modality_keys=["ego_view", "depth_gray_view"], channel_fusion=[VideoChannelSource("ego_view", (0, 1, 2)), VideoChannelSource("depth_gray_view", (0,))])  # fmt: skip  # earlyfusion
    model = SimpleNamespace(vision_input_channels=4, vision_channel_layout=["ego_view:0", "ego_view:1", "ego_view:2", "depth_gray_view:0"], vision_patch_embed_init="rgb_mean")  # fmt: skip  # earlyfusion
    contract = _vision_input_contract(model, video)  # earlyfusion
    assert contract == {"version": 1, "mode": "early_channel_fusion", "input_channels": 4, "channel_layout": model.vision_channel_layout, "patch_embed_init": "rgb_mean", "wire_video_keys": ["ego_view", "depth_gray_view"]}  # fmt: skip  # earlyfusion
    with pytest.raises(ValueError, match="model/processor channel mismatch"):  # earlyfusion
        _vision_input_contract(SimpleNamespace(vision_input_channels=6, vision_channel_layout=model.vision_channel_layout, vision_patch_embed_init="zeros"), video)  # fmt: skip  # earlyfusion


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


@pytest.fixture
def policy():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16
    mock_model.config = SimpleNamespace(
        action_horizon=16,
        max_action_dim=7,
        rtc_ramp_rate=6.0,
    )

    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": torch.randn(1, 16, 7)})
    )

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.state_action_processor = MagicMock()
    mock_processor.state_action_processor.norm_params = {
        EMBODIMENT: {
            "action": {key: {"dim": np.array(1)} for key in ACTION_KEYS},
        }
    }
    mock_processor.state_action_processor.apply_action.side_effect = (
        lambda action, _embodiment_tag, state=None, clip_outliers=None: action
    )
    mock_processor.action_dim = {EMBODIMENT: 7}
    mock_processor.max_action_dim = 7
    mock_processor.max_action_horizon = 16
    mock_processor.eval = MagicMock()
    mock_processor.training = False

    def fake_collator(features):
        batch_size = len(features)
        return BatchFeature(
            data={
                "inputs": {
                    "state": torch.zeros(batch_size, 1, 128),
                    "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
                }
            }
        )

    mock_processor.collator = MagicMock(side_effect=fake_collator)

    def fake_process_observation(observation, embodiment_tag):
        return BatchFeature(
            data={
                "state": torch.randn(1, 1, 128),
                "action_mask": torch.ones(1, 16, 128),
                "embodiment_id": torch.zeros(1, dtype=torch.long),
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 256, 256),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }
        )

    mock_processor.process_observation = MagicMock(side_effect=fake_process_observation)

    def fake_decode_action(action, embodiment_tag, state=None):
        return {k: np.zeros((action.shape[0], 16, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    # Patch both AutoModel and AutoProcessor, and also the processor_config.json check
    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        p = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
        )
    return p


def _make_observation(batch_size=1):
    return {
        "video": {
            k: np.random.randint(0, 255, (batch_size, 1, 256, 256, 3), dtype=np.uint8)
            for k in VIDEO_KEYS
        },
        "state": {
            k: np.random.randn(batch_size, 1, 1).astype(np.float32)
            for k in STATE_KEYS[:-1]  # all except gripper
        }
        | {"gripper": np.random.randn(batch_size, 1, 2).astype(np.float32)},
        "language": {
            LANGUAGE_KEY: [["pick up the apple"]] * batch_size,
        },
    }


class TestGr00tPolicyInit:
    def test_policy_has_model_and_processor(self, policy):
        assert policy.model is not None
        assert policy.processor is not None

    def test_policy_embodiment_tag(self, policy):
        assert policy.embodiment_tag is not None

    def test_policy_can_load_processor_from_separate_path(self):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.config = SimpleNamespace(vision_input_channels=3, vision_channel_layout=["ego_view:0", "ego_view:1", "ego_view:2"], vision_patch_embed_init="original_rgb")  # fmt: skip  # earlyfusion
        mock_processor = MagicMock()
        mock_processor.get_modality_configs.return_value = _build_modality_configs()
        mock_processor.collator = MagicMock()

        with (
            patch("gr00t.policy.gr00t_policy.AutoModel") as mock_auto_model,
            patch("gr00t.policy.gr00t_policy.AutoProcessor") as mock_auto_processor,
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_processor.from_pretrained.return_value = mock_processor

            from gr00t.policy.gr00t_policy import Gr00tPolicy

            Gr00tPolicy(
                embodiment_tag=EMBODIMENT,
                model_path="/models/base",
                processor_path="/runs/custom/processor",
                device="cpu",
            )

        assert mock_auto_model.from_pretrained.call_args.args[0] == Path("/models/base")
        assert mock_auto_processor.from_pretrained.call_args.args[0] == Path(
            "/runs/custom/processor"
        )


class TestGr00tPolicyCheckObservation:
    def test_valid_observation_passes(self, policy):
        obs = _make_observation()
        policy.check_observation(obs)

    def test_missing_video_key_raises(self, policy):
        obs = _make_observation()
        del obs["video"][VIDEO_KEYS[0]]
        with pytest.raises(AssertionError):
            policy.check_observation(obs)

    def test_wrong_video_dtype_raises(self, policy):
        obs = _make_observation()
        obs["video"][VIDEO_KEYS[0]] = obs["video"][VIDEO_KEYS[0]].astype(np.float32)
        with pytest.raises(AssertionError):
            policy.check_observation(obs)


class TestGr00tPolicyGetAction:
    def test_get_action_returns_tuple(self, policy):
        obs = _make_observation()
        result = policy.get_action(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_action_returns_dict(self, policy):
        obs = _make_observation()
        action, info = policy.get_action(obs)
        assert isinstance(action, dict)
        assert isinstance(info, dict)

    def test_synchronous_inference_does_not_add_model_options_or_action_input(self, policy):
        policy.get_action(_make_observation(), options={"inference_mode": "synchronous"})

        call_kwargs = policy.model.get_action.call_args.kwargs
        assert "options" not in call_kwargs
        assert "action" not in call_kwargs["inputs"]

    def test_rtc_reencodes_variable_physical_tail_and_derives_model_options(self, policy):
        tail_horizon = 5
        previous_action = {
            key: np.full((1, tail_horizon, 1), index + 0.25, dtype=np.float32)
            for index, key in enumerate(ACTION_KEYS)
        }

        action, info = policy.get_action(
            _make_observation(),
            options={
                "inference_mode": "rtc",
                "rtc_previous_action": previous_action,
                "rtc_overlap_steps": tail_horizon,
                "rtc_frozen_steps": 2,
            },
        )

        call_kwargs = policy.model.get_action.call_args.kwargs
        assert call_kwargs["options"] == {
            "action_horizon": tail_horizon,
            "rtc_overlap_steps": tail_horizon,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 6.0,
        }
        encoded = call_kwargs["inputs"]["action"]
        assert encoded.shape == (1, 16, 7)
        assert encoded.dtype == torch.bfloat16
        expected_prefix = np.concatenate([previous_action[key][0] for key in ACTION_KEYS], axis=-1)
        np.testing.assert_allclose(encoded[0, :tail_horizon].float().numpy(), expected_prefix)
        assert torch.count_nonzero(encoded[0, tail_horizon:]) == 0
        apply_kwargs = policy.processor.state_action_processor.apply_action.call_args.kwargs
        assert apply_kwargs["clip_outliers"] is False
        for key in ACTION_KEYS:
            np.testing.assert_array_equal(action[key][:, :2], previous_action[key][:, :2])
        assert info == {
            "rtc_applied": True,
            "rtc_previous_action_horizon": tail_horizon,
            "rtc_overlap_steps": tail_horizon,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 6.0,
        }

    def test_rtc_rejects_client_action_horizon(self, policy):
        previous_action = {key: np.zeros((1, 5, 1), dtype=np.float32) for key in ACTION_KEYS}
        with pytest.raises(ValueError, match="derived from rtc_previous_action"):
            policy.get_action(
                _make_observation(),
                options={
                    "inference_mode": "rtc",
                    "rtc_previous_action": previous_action,
                    "rtc_overlap_steps": 5,
                    "rtc_frozen_steps": 2,
                    "action_horizon": 5,
                },
            )

    @pytest.mark.parametrize(
        ("mutate_options", "message"),
        [
            (lambda options: options.update(rtc_overlap_steps=6), "RTC overlap must satisfy"),
            (lambda options: options.update(rtc_frozen_steps=6), "RTC frozen steps must satisfy"),
            (lambda options: options.update(rtc_ramp_rate=float("nan")), "finite positive"),
        ],
    )
    def test_rtc_rejects_invalid_schedule(self, policy, mutate_options, message):
        previous_action = {key: np.zeros((1, 5, 1), dtype=np.float32) for key in ACTION_KEYS}
        options = {
            "inference_mode": "rtc",
            "rtc_previous_action": previous_action,
            "rtc_overlap_steps": 5,
            "rtc_frozen_steps": 2,
        }
        mutate_options(options)

        with pytest.raises(ValueError, match=message):
            policy.get_action(_make_observation(), options=options)

    def test_rtc_rejects_inconsistent_tail_horizons(self, policy):
        previous_action = {key: np.zeros((1, 5, 1), dtype=np.float32) for key in ACTION_KEYS}
        previous_action[ACTION_KEYS[-1]] = np.zeros((1, 4, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="same tail horizon"):
            policy.get_action(
                _make_observation(),
                options={
                    "inference_mode": "rtc",
                    "rtc_previous_action": previous_action,
                    "rtc_overlap_steps": 4,
                    "rtc_frozen_steps": 2,
                },
            )

    def test_rtc_rejects_missing_action_key_and_nonfinite_value(self, policy):
        previous_action = {key: np.zeros((1, 5, 1), dtype=np.float32) for key in ACTION_KEYS}
        del previous_action[ACTION_KEYS[0]]
        with pytest.raises(ValueError, match="must exactly match"):
            policy.get_action(
                _make_observation(),
                options={
                    "inference_mode": "rtc",
                    "rtc_previous_action": previous_action,
                    "rtc_overlap_steps": 5,
                    "rtc_frozen_steps": 2,
                },
            )

        previous_action[ACTION_KEYS[0]] = np.zeros((1, 5, 1), dtype=np.float32)
        previous_action[ACTION_KEYS[1]][0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinity"):
            policy.get_action(
                _make_observation(),
                options={
                    "inference_mode": "rtc",
                    "rtc_previous_action": previous_action,
                    "rtc_overlap_steps": 5,
                    "rtc_frozen_steps": 2,
                },
            )

    def test_policy_options_must_be_a_dictionary(self, policy):
        with pytest.raises(ValueError, match="must be a dictionary"):
            policy.get_action(_make_observation(), options=[])


class _NumpyLanguageSimPolicy:
    def __init__(self):
        self.modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["camera"]),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state"],
            ),
            "action": ModalityConfig(delta_indices=[0], modality_keys=["action"]),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.human.action.task_description"],
            ),
        }
        self.last_observation = None

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observation, options=None):
        self.last_observation = observation
        return {"action": np.zeros((1, 1, 2), dtype=np.float32)}, {}

    def reset(self, options=None):
        return {}


def test_sim_policy_wrapper_accepts_numpy_language_batches():
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    policy = _NumpyLanguageSimPolicy()
    wrapper = Gr00tSimPolicyWrapper(policy)
    observation = {
        "video.camera": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "state.state": np.zeros((1, 1, 3), dtype=np.float32),
        "annotation.human.action.task_description": np.array(["follow the instruction"]),
    }

    action, info = wrapper.get_action(observation)

    assert policy.last_observation["language"][LANGUAGE_KEY] == [["follow the instruction"]]
    assert "action.action" in action
    assert info == {}
