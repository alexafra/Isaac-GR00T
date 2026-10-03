"""CPU-only tests for shared state/visual decisions and legacy compatibility."""

from collections import Counter
from pathlib import Path
import random
from unittest.mock import MagicMock, patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VLAStepData
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.modality_dropout import (
    sample_modality_dropout,
    validate_modality_dropout,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


@pytest.mark.parametrize(
    "args",
    [
        (float("nan"), 0.05, 0.2, "state_first"),
        (0.05, float("inf"), 0.2, "state_first"),
        (-0.1, 0.05, 0.2, "state_first"),
        (0.7, 0.4, 0.2, "independent"),
        (0.05, 0.05, 0.8, "state_first"),
        (0.05, 0.05, 0.2, "unknown"),
    ],
)
def test_reject_invalid_probabilities(args):
    with pytest.raises(ValueError):
        validate_modality_dropout(*args)


def test_state_exclusion_rejects_state_first_but_independent_is_supported():
    with pytest.raises(ValueError, match="fraction of examples"):
        validate_modality_dropout(0.05, 0.05, 0.2, "state_first", True)
    validate_modality_dropout(0.05, 0.05, 0.2, "independent", True)


@pytest.mark.parametrize("policy", ["state_first", "independent"])
def test_empirical_global_probabilities_and_state_marginals(policy):
    rng = random.Random(618)
    results = [
        sample_modality_dropout(0.05, 0.05, 0.2, policy, exclude_state=False, draw=rng.random)
        for _ in range(100_000)
    ]
    counts = Counter(result.vision_keep for result in results)
    for keep, expected in [((1.0, 1.0), 0.9), ((0.0, 1.0), 0.05), ((1.0, 0.0), 0.05)]:
        assert counts[keep] / len(results) == pytest.approx(expected, abs=0.004)
    assert (0.0, 0.0) not in counts
    assert np.mean([r.drop_state_input for r in results]) == pytest.approx(0.2, abs=0.004)
    assert np.mean([not r.keep_state_features for r in results]) == pytest.approx(0.2, abs=0.004)
    assert np.mean(
        [r.drop_state_input or not r.keep_state_features for r in results]
    ) == pytest.approx(0.36, abs=0.004)
    if policy == "state_first":
        assert all(
            not r.drop_state_input and r.keep_state_features
            for r in results
            if r.vision_keep != (1.0, 1.0)
        )


def _processor_shell(enabled=True, training=True):
    tag = EmbodimentTag.NEW_EMBODIMENT
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor.training = training
    processor.modality_configs = {
        tag.value: {
            "video": ModalityConfig([0], ["ego_view", "depth_gray_view"]),
            "state": ModalityConfig([0], ["joint"]),
            "action": ModalityConfig([0], ["joint"]),
        }
    }
    processor.max_state_dim = processor.max_action_dim = processor.max_action_horizon = 1
    processor.exclude_state = False
    processor.state_dropout_prob = 0.2
    processor.vision_modality_dropout_rgb_prob = 0.05 if enabled else 0.0
    processor.vision_modality_dropout_geometry_prob = 0.05 if enabled else 0.0
    processor.vision_modality_dropout_state_policy = "state_first"
    processor.vision_modality_dropout_source_keys = ["ego_view", "depth_gray_view"]
    processor.state_action_processor = MagicMock()
    processor.state_action_processor.apply.return_value = (
        {"joint": np.array([[0.2]], dtype=np.float32)},
        {"joint": np.array([[0.6]], dtype=np.float32)},
    )
    processor.train_image_transform = processor.eval_image_transform = None
    processor.formalize_language = False
    processor.embodiment_id_mapping = {tag.value: 0}
    processor._get_vlm_inputs = MagicMock(return_value={})
    content = VLAStepData(
        images={
            key: [np.zeros((2, 2, 3), dtype=np.uint8)]
            for key in processor.vision_modality_dropout_source_keys
        },
        states={"joint": np.array([[0.7]], dtype=np.float32)},
        actions={"joint": np.array([[0.8]], dtype=np.float32)},
        text="pick up the cup.",
        embodiment=tag,
    )
    return processor, [{"content": content}]


@pytest.mark.parametrize(
    "draws, expected_state, feature_keep, vision_keep",
    [
        ([0.1, 0.9], 0.0, True, [1.0, 1.0]),
        ([0.9, 0.1], 0.2, False, [1.0, 1.0]),
        ([0.9, 0.9, 0.01], 0.2, True, [0.0, 1.0]),
        ([0.9, 0.9, 0.1], 0.2, True, [1.0, 0.0]),
        ([0.9, 0.9, 0.9], 0.2, True, [1.0, 1.0]),
    ],
)
def test_processor_shares_decision_without_changing_action_targets(
    draws, expected_state, feature_keep, vision_keep
):
    processor, messages = _processor_shell()
    with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.random.random", side_effect=draws):
        result = processor(messages)
    assert result["state"].item() == pytest.approx(expected_state)
    assert result["state_feature_keep"].item() is feature_keep
    assert result["vision_modality_keep"].tolist() == vision_keep
    assert result["vision_modality_num_frames"].item() == 1
    assert result["action"].item() == pytest.approx(0.6)
    np.testing.assert_array_equal(
        processor.state_action_processor.apply.call_args.kwargs["state"]["joint"],
        [[np.float32(0.7)]],
    )


@pytest.mark.parametrize("training", [True, False])
def test_disabled_route_preserves_legacy_state_rng_and_emits_no_new_metadata(training):
    processor, messages = _processor_shell(enabled=False, training=training)
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.random.random", return_value=0.1
    ) as draw:
        result = processor(messages)
    # Existing code draws once even in eval; preserve that behavior when disabled.
    assert draw.call_count == 1
    assert result["state"].item() == pytest.approx(0.0 if training else 0.2)
    assert not any(k.startswith("vision_modality") or k == "state_feature_keep" for k in result)


def test_enabled_eval_keeps_inputs_and_omits_training_masks():
    processor, messages = _processor_shell(training=False)
    result = processor(messages)
    assert result["state"].item() == pytest.approx(0.2)
    assert "vision_modality_keep" not in result
    assert "state_feature_keep" not in result


def test_frame_mismatch_fails_closed():
    processor, messages = _processor_shell()
    messages[0]["content"].images["depth_gray_view"] = []
    with pytest.raises(ValueError, match="frame counts"):
        processor(messages)


def test_collator_preserves_boolean_state_and_per_example_visual_metadata():
    processor, messages = _processor_shell()
    samples = [processor(messages), processor(messages)]
    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    result = collator(samples)["inputs"]
    assert result["vision_modality_keep"].shape == (2, 2)
    assert result["vision_modality_num_frames"].dtype == torch.int64
    assert result["state_feature_keep"].dtype == torch.bool


def test_separate_view_override_replaces_pretrained_modalities():
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "processor_config"
    video = ModalityConfig([0], ["ego_view", "depth_gray_view"])
    override = {"new_embodiment": {"video": video}}
    with patch.object(Gr00tN1d7Processor, "__init__", return_value=None) as init:
        Gr00tN1d7Processor.from_pretrained(
            fixture,
            modality_configs=override,
            vision_modality_dropout_rgb_prob=0.05,
            vision_modality_dropout_geometry_prob=0.05,
            vision_modality_dropout_source_keys=["ego_view", "depth_gray_view"],
        )
    assert init.call_args.kwargs["modality_configs"] == override
    assert init.call_args.kwargs["vision_modality_dropout_rgb_prob"] == 0.05
    assert init.call_args.kwargs["vision_modality_dropout_state_policy"] == "independent"


@pytest.mark.parametrize("state_draw", [0.1, 0.9])
def test_independent_mode_keeps_legacy_state_input_and_defers_feature_mask_to_head(state_draw):
    processor, messages = _processor_shell()
    processor.vision_modality_dropout_state_policy = "independent"
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.random.random", side_effect=[state_draw, 0.01]
    ) as draw:
        result = processor(messages)
    assert draw.call_count == 2
    assert result["state"].item() == pytest.approx(0.0 if state_draw < 0.2 else 0.2)
    assert result["vision_modality_keep"].tolist() == [0.0, 1.0]
    assert "state_feature_keep" not in result


def test_action_head_uses_shared_feature_mask_instead_of_drawing_again():
    config = Gr00tN1d7Config(
        backbone_embedding_dim=16,
        hidden_size=16,
        input_embedding_dim=16,
        max_state_dim=2,
        max_action_dim=2,
        action_horizon=2,
        max_num_embodiments=1,
        state_history_length=1,
        state_dropout_prob=1.0,
        use_alternate_vl_dit=False,
        add_pos_embed=False,
        use_vlln=False,
        diffusion_model_cfg={
            "num_layers": 1,
            "num_attention_heads": 1,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 16,
        },
    )
    head = Gr00tN1d7ActionHead(config)
    head.train()
    data = BatchFeature(
        {
            "state": torch.ones(2, 1, 2),
            "action": torch.zeros(2, 2, 2),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
            "action_mask": torch.ones(2, 2, 2),
            "state_feature_keep": torch.tensor([True, False]),
        }
    )
    expected = head.state_encoder(data.state, data.embodiment_id)
    backbone = BatchFeature(
        {
            "backbone_features": torch.ones(2, 3, 16),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
            "image_mask": torch.ones(2, 3, dtype=torch.bool),
        }
    )
    result = head(backbone, data)
    torch.testing.assert_close(result["state_features"][0], expected[0])
    assert torch.count_nonzero(result["state_features"][1]) == 0


def test_enabled_processor_save_reload_retains_recipe_and_statistics(tmp_path):
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    modalities = {
        tag: {
            "video": ModalityConfig([0], ["ego_view", "depth_gray_view"]),
            "state": ModalityConfig([0], ["joint"]),
            "action": ModalityConfig([0], ["joint"]),
            "language": ModalityConfig([0], ["annotation.human.task_description"]),
        }
    }
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor", return_value=MagicMock()
    ):
        processor = Gr00tN1d7Processor(
            modality_configs=modalities,
            image_crop_size=[8, 8],
            image_target_size=[8, 8],
            max_state_dim=1,
            max_action_dim=1,
            max_action_horizon=1,
            state_dropout_prob=0.2,
            vision_modality_dropout_rgb_prob=0.05,
            vision_modality_dropout_geometry_prob=0.05,
            vision_modality_dropout_source_keys=["ego_view", "depth_gray_view"],
        )
        processor.save_pretrained(tmp_path)
        restored = Gr00tN1d7Processor.from_pretrained(tmp_path)
    assert restored.vision_modality_dropout_rgb_prob == 0.05
    assert restored.vision_modality_dropout_geometry_prob == 0.05
    assert restored.vision_modality_dropout_state_policy == "independent"
    assert restored.vision_modality_dropout_source_keys == ["ego_view", "depth_gray_view"]
    assert restored.state_dropout_prob == 0.2
    assert restored.state_action_processor.statistics == processor.state_action_processor.statistics
    restored.eval()
    assert not restored.training
