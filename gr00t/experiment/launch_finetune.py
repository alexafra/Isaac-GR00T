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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig, validate_vision_modality_dropout_config
from gr00t.data.depth_colormap import (
    FIXED_TURBO_DEPTH_COLORMAP,
    fixed_turbo_depth_colormap_contract,
)
from gr00t.experiment.experiment import run


_POST_VISION_LATE_FUSION = "post_vision_late_fusion"
_PRE_VISION_LANGUAGE_ADAPTER = "pre_vision_language_adapter"
_POST_VISION_LANGUAGE_ADAPTER = "post_vision_language_adapter"
_VISION_FUSION_DEEPSTACK_LAYERS = [5, 11, 17]
_VISION_FUSION_STAGE_DIMS = {
    _PRE_VISION_LANGUAGE_ADAPTER: (2048, 1024),
    _POST_VISION_LANGUAGE_ADAPTER: (4096, 2048),
}


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def configure_vision_input_contract(
    config,
    embodiment_tag: str,
    video_config,
    vision_patch_embed_init: str | None,
) -> None:
    """Map the selected video recipe onto the serialized model vision contract."""

    depth_colormap = video_config.depth_colormap
    depth_colormap_contract = (
        fixed_turbo_depth_colormap_contract() if depth_colormap is not None else None
    )
    config.model.vision_depth_colormap = depth_colormap
    config.model.vision_depth_colormap_lut_sha256 = (
        depth_colormap_contract["lut_sha256"] if depth_colormap_contract is not None else None
    )

    if vision_patch_embed_init is not None and not video_config.channel_fusion:  # fmt: skip  # earlyfusion
        raise ValueError(
            "--vision-patch-embed-init is only valid for early-fusion modality configs"
        )
    if video_config.channel_fusion:  # earlyfusion
        config.data.modality_configs = {
            embodiment_tag: config.data.modality_configs[embodiment_tag]
        }
        config.model.vision_channel_layout = video_config.vision_channel_layout  # earlyfusion
        config.model.vision_input_channels = video_config.vision_input_channels  # earlyfusion
        default_patch_init = "rgb_mean" if config.model.vision_input_channels == 4 else "zeros"
        config.model.vision_patch_embed_init = vision_patch_embed_init or default_patch_init
    elif video_config.post_vision_fusion:
        # Post-vision fusion is one model-wide graph contract. Remove inactive
        # pretrained embodiment recipes so the processor cannot mix fused and
        # non-fused batches, while retaining two separate 3-channel sources.
        config.data.modality_configs = {
            embodiment_tag: config.data.modality_configs[embodiment_tag]
        }
        config.model.vision_input_channels = 3
        config.model.vision_channel_layout = [
            f"{video_config.modality_keys[0]}:{channel}" for channel in range(3)
        ]
        config.model.vision_patch_embed_init = "original_rgb"
        config.model.vision_fusion_mode = _POST_VISION_LATE_FUSION
        config.model.vision_fusion_version = 1
        config.model.vision_fusion_stage = video_config.post_vision_fusion_stage
        config.model.vision_fusion_adapter_type = "concat_linear"
        config.model.vision_fusion_adapter_count = 4
        config.model.vision_fusion_deepstack_layers = list(_VISION_FUSION_DEEPSTACK_LAYERS)
        (
            config.model.vision_fusion_adapter_input_dim,
            config.model.vision_fusion_adapter_output_dim,
        ) = _VISION_FUSION_STAGE_DIMS[video_config.post_vision_fusion_stage]
        config.model.vision_fusion_source_keys = list(video_config.modality_keys)
        config.model.vision_fusion_adapter_init = "rgb50_geo50"
        config.model.vision_fusion_rgb_init_scale = 0.5
        config.model.vision_fusion_geometry_init_scale = 0.5
        config.model.vision_fusion_adapter_bias = True
        config.model.vision_fusion_adapter_bias_init = "zeros"
        config.model.tune_vision_fusion_adapters = True


def configure_vision_modality_dropout_contract(
    config, embodiment_tag: str, video_config, ft_config: FinetuneConfig
) -> None:
    """Bind opt-in masking to a unique ordered RGB/geometry input contract."""
    rgb_prob = ft_config.vision_modality_dropout_rgb_prob
    geometry_prob = ft_config.vision_modality_dropout_geometry_prob
    policy = ft_config.vision_modality_dropout_state_policy
    validate_vision_modality_dropout_config(
        rgb_prob, geometry_prob, policy, ft_config.state_dropout_prob
    )
    config.model.vision_modality_dropout_rgb_prob = rgb_prob
    config.model.vision_modality_dropout_geometry_prob = geometry_prob
    config.model.vision_modality_dropout_state_policy = policy
    config.model.vision_modality_dropout_source_keys = []
    if not rgb_prob + geometry_prob:
        return

    source_keys = list(video_config.modality_keys)
    if source_keys not in (
        ["ego_view", "depth_gray_view"],
        ["ego_view", "surface_normals_view"],
    ):
        raise ValueError(
            "Modality dropout requires exactly ordered paired RGB and geometry keys: "
            "['ego_view', 'depth_gray_view'] or ['ego_view', 'surface_normals_view']"
        )
    # ModalityConfig shares delta_indices across its video keys: a nonempty
    # list therefore defines equal, positive frame counts for both sources.
    if not video_config.delta_indices:
        raise ValueError("Modality dropout requires equal positive frame counts for both views")
    if video_config.channel_fusion:
        gray_depth = (
            source_keys[1] == "depth_gray_view"
            and video_config.depth_colormap != FIXED_TURBO_DEPTH_COLORMAP
        )
        geometry_channels = [0] if gray_depth else [0, 1, 2]
        expected_layout = [f"ego_view:{channel}" for channel in range(3)] + [
            f"{source_keys[1]}:{channel}" for channel in geometry_channels
        ]
        if video_config.vision_channel_layout != expected_layout:
            raise ValueError(
                "Modality dropout requires RGB channels 0,1,2 first, followed by "
                "gray depth channel 0 or Turbo/normals channels 0,1,2"
            )
    config.model.vision_modality_dropout_source_keys = source_keys
    # Separate-view recipes otherwise retain unrelated pretrained embodiments.
    # An active dropout model must have one unambiguous view-order contract.
    config.data.modality_configs = {embodiment_tag: config.data.modality_configs[embodiment_tag]}


def validate_vision_trainability_contract(config, video_config) -> None:
    """Reject trainability flags that would make a late-fusion comparison ambiguous."""

    if not video_config.post_vision_fusion:
        return
    incompatible = {
        "tune_llm": config.model.tune_llm,
        "tune_visual": config.model.tune_visual,
        "tune_vision_patch_embed": config.model.tune_vision_patch_embed,
        "tune_top_llm_layers": config.model.tune_top_llm_layers != 0,
    }
    enabled = [name for name, is_enabled in incompatible.items() if is_enabled]
    if enabled:
        raise ValueError(
            "Late fusion requires the pretrained Qwen backbone and shared 3-channel "
            "patch embedding to remain frozen; disable " + ", ".join(enabled)
        )
    if not config.model.tune_vision_fusion_adapters:
        raise ValueError("Late fusion requires tune_vision_fusion_adapters=True")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    video_config = config.data.modality_configs[embodiment_tag]["video"]  # earlyfusion
    configure_vision_input_contract(
        config,
        embodiment_tag,
        video_config,
        ft_config.vision_patch_embed_init,
    )
    configure_vision_modality_dropout_contract(config, embodiment_tag, video_config, ft_config)
    if ft_config.late_fusion_language_readout == "pre_final_rmsnorm" and not video_config.post_vision_fusion:
        raise ValueError("--late-fusion-language-readout applies only to late-fusion recipes")
    config.model.late_fusion_language_readout = ft_config.late_fusion_language_readout
    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_vision_patch_embed = ft_config.tune_vision_patch_embed  # earlyfusion
    validate_vision_trainability_contract(config, video_config)
    if config.model.vision_input_channels == 6 and config.model.vision_patch_embed_init == "zeros" and not config.model.tune_vision_patch_embed:  # fmt: skip  # earlyfusion
        raise ValueError("6-channel zero-initialized normals require --tune-vision-patch-embed")  # fmt: skip  # earlyfusion
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = ft_config.load_bf16
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    # config.training.optim = "adamw_torch"
    config.training.optim = ft_config.optim
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_final_model_save = ft_config.skip_final_model_save
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
