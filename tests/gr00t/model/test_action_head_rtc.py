# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only temporal-alignment tests for the N1.7 RTC primitive.

These tests intentionally exercise the small action head directly.  They do not
load a checkpoint or touch CUDA.  The deployment scheduler relies on the exact
indexing contract pinned here when it feeds an unconsumed action suffix back to
the policy.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_action_head() -> tuple[Gr00tN1d7ActionHead, Gr00tN1d7Config]:
    config = Gr00tN1d7Config(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=4,
        max_action_dim=4,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=2,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _backbone_output(config: Gr00tN1d7Config) -> BatchFeature:
    return BatchFeature(
        data={
            "backbone_features": torch.randn(1, 5, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(1, 5, dtype=torch.long),
            "image_mask": torch.ones(1, 5, dtype=torch.bool),
        }
    )


def _action_input(config: Gr00tN1d7Config, previous: torch.Tensor) -> BatchFeature:
    return BatchFeature(
        data={
            "state": torch.zeros(1, config.state_history_length, config.max_state_dim),
            "action": previous.clone(),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
            "action_mask": torch.ones_like(previous),
        }
    )


def test_rtc_copies_the_selected_previous_suffix_into_the_new_prefix() -> None:
    """``action_horizon`` is pre-padding length; overlap selects its tail."""

    head, config = _small_action_head()
    previous = torch.tensor(
        [
            [
                [0.10, 0.11, 0.12, 0.13],
                [0.20, 0.21, 0.22, 0.23],
                [0.30, 0.31, 0.32, 0.33],
                [9.00, 9.00, 9.00, 9.00],  # processor-style padding, excluded
            ]
        ],
        dtype=torch.float32,
    )

    result = head.get_action(
        _backbone_output(config),
        _action_input(config, previous),
        options={
            "action_horizon": 3,
            "rtc_overlap_steps": 2,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 6.0,
        },
    )["action_pred"]

    # With a pre-padding horizon of three and overlap of two, the copied prior
    # is previous[1:3], never the padded row and never previous[0:2].  Fully
    # frozen positions receive exactly zero denoising velocity.
    torch.testing.assert_close(result[:, :2], previous[:, 1:3], rtol=0.0, atol=0.0)


def test_rtc_full_leftover_overlap_freezes_the_committed_queue_prefix() -> None:
    """Passing the whole leftover queue preserves its committed first steps."""

    head, config = _small_action_head()
    previous = torch.tensor(
        [
            [
                [-0.30, -0.20, -0.10, 0.00],
                [-0.20, -0.10, 0.00, 0.10],
                [-0.10, 0.00, 0.10, 0.20],
                [0.00, 0.00, 0.00, 0.00],  # processor-style padding
            ]
        ],
        dtype=torch.float32,
    )

    result = head.get_action(
        _backbone_output(config),
        _action_input(config, previous),
        options={
            "action_horizon": 3,
            "rtc_overlap_steps": 3,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 6.0,
        },
    )["action_pred"]

    torch.testing.assert_close(result[:, :2], previous[:, :2], rtol=0.0, atol=0.0)
