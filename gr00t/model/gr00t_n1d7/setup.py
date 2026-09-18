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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.depth_colormap import fixed_turbo_depth_colormap_contract
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


def _reconcile_depth_colormap_contract(
    checkpoint_config,
    requested_config,
    *,
    initializing_late_fusion: bool,
) -> None:
    """Bind a base checkpoint to the requested depth representation exactly once."""

    source = (
        getattr(checkpoint_config, "vision_depth_colormap", None),
        getattr(checkpoint_config, "vision_depth_colormap_lut_sha256", None),
    )
    target = (
        getattr(requested_config, "vision_depth_colormap", None),
        getattr(requested_config, "vision_depth_colormap_lut_sha256", None),
    )
    if target[0] is not None:
        expected = fixed_turbo_depth_colormap_contract()
        expected_target = (expected["name"], expected["lut_sha256"])
        if target != expected_target:
            raise RuntimeError(
                "Requested depth-colormap contract is not the pinned fixed-Turbo contract: "
                f"got {target!r}, expected {expected_target!r}"
            )
    elif target[1] is not None:
        raise RuntimeError(
            "Requested depth-colormap LUT digest requires a depth-colormap name"
        )
    if source == target:
        return
    if source == (None, None) and initializing_late_fusion:
        checkpoint_config.vision_depth_colormap = target[0]
        checkpoint_config.vision_depth_colormap_lut_sha256 = target[1]
        return
    raise RuntimeError(
        "Checkpoint depth-colormap contract does not match the requested configuration: "
        f"checkpoint={source!r}, requested={target!r}"
    )


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_vision_patch_embed=self.config.model.tune_vision_patch_embed,  # earlyfusion
                tune_vision_fusion_adapters=self.config.model.tune_vision_fusion_adapters,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            other_missing = [k for k in missing_keys if "mask_token" not in k]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )
            source_channels = model.config.vision_input_channels  # earlyfusion
            target_channels = self.model_config.vision_input_channels  # earlyfusion
            source_layout = list(model.config.vision_channel_layout)  # earlyfusion
            target_layout = list(self.model_config.vision_channel_layout)  # earlyfusion
            source_init = model.config.vision_patch_embed_init  # earlyfusion
            target_init = self.model_config.vision_patch_embed_init  # earlyfusion
            if source_channels != target_channels:  # earlyfusion
                if source_channels != 3 or target_channels not in (4, 6):  # earlyfusion
                    raise RuntimeError(f"Cannot migrate vision input from {source_channels} to {target_channels} channels")  # fmt: skip  # earlyfusion
                model.backbone.expand_vision_input_channels(target_channels, self.model_config.vision_patch_embed_init)  # fmt: skip  # earlyfusion
                model.backbone._apply_vision_patch_embed_channels_last()  # earlyfusion
                model.config.vision_input_channels = target_channels  # earlyfusion
                model.config.vision_channel_layout = target_layout  # earlyfusion
                model.config.vision_patch_embed_init = self.model_config.vision_patch_embed_init  # fmt: skip  # earlyfusion
                model.collator.vision_input_channels = target_channels  # earlyfusion
                model.collator.vision_channel_layout = target_layout  # earlyfusion
            elif source_channels > 3 and (source_layout != target_layout or source_init != target_init):  # fmt: skip  # earlyfusion
                raise RuntimeError(f"Checkpoint vision contract {(source_layout, source_init)} does not match requested {(target_layout, target_init)}")  # fmt: skip  # earlyfusion

            source_fusion = (
                model.config.vision_fusion_mode,
                model.config.vision_fusion_version,
                model.config.vision_fusion_stage,
                model.config.vision_fusion_adapter_type,
                model.config.vision_fusion_adapter_bias,
                model.config.vision_fusion_adapter_bias_init,
                model.config.vision_fusion_adapter_count,
                list(model.config.vision_fusion_deepstack_layers),
                model.config.vision_fusion_adapter_input_dim,
                model.config.vision_fusion_adapter_output_dim,
                list(model.config.vision_fusion_source_keys),
                model.config.vision_fusion_adapter_init,
                model.config.vision_fusion_rgb_init_scale,
                model.config.vision_fusion_geometry_init_scale,
            )
            target_fusion = (
                self.model_config.vision_fusion_mode,
                self.model_config.vision_fusion_version,
                self.model_config.vision_fusion_stage,
                self.model_config.vision_fusion_adapter_type,
                self.model_config.vision_fusion_adapter_bias,
                self.model_config.vision_fusion_adapter_bias_init,
                self.model_config.vision_fusion_adapter_count,
                list(self.model_config.vision_fusion_deepstack_layers),
                self.model_config.vision_fusion_adapter_input_dim,
                self.model_config.vision_fusion_adapter_output_dim,
                list(self.model_config.vision_fusion_source_keys),
                self.model_config.vision_fusion_adapter_init,
                self.model_config.vision_fusion_rgb_init_scale,
                self.model_config.vision_fusion_geometry_init_scale,
            )
            initializing_late_fusion = (
                source_fusion[0] == "none"
                and target_fusion[0] == "post_vision_late_fusion"
            )
            if source_fusion != target_fusion:
                if initializing_late_fusion:
                    model.backbone.enable_post_vision_late_fusion(
                        mode=target_fusion[0],
                        version=target_fusion[1],
                        stage=target_fusion[2],
                        adapter_type=target_fusion[3],
                        adapter_bias=target_fusion[4],
                        adapter_bias_init=target_fusion[5],
                        adapter_count=target_fusion[6],
                        deepstack_layers=target_fusion[7],
                        adapter_input_dim=target_fusion[8],
                        adapter_output_dim=target_fusion[9],
                        source_keys=target_fusion[10],
                        adapter_init=target_fusion[11],
                        rgb_init_scale=target_fusion[12],
                        geometry_init_scale=target_fusion[13],
                        tune_adapters=self.model_config.tune_vision_fusion_adapters,
                    )
                    model.config.vision_fusion_mode = target_fusion[0]
                    model.config.vision_fusion_version = target_fusion[1]
                    model.config.vision_fusion_stage = target_fusion[2]
                    model.config.vision_fusion_adapter_type = target_fusion[3]
                    model.config.vision_fusion_adapter_bias = target_fusion[4]
                    model.config.vision_fusion_adapter_bias_init = target_fusion[5]
                    model.config.vision_fusion_adapter_count = target_fusion[6]
                    model.config.vision_fusion_deepstack_layers = target_fusion[7]
                    model.config.vision_fusion_adapter_input_dim = target_fusion[8]
                    model.config.vision_fusion_adapter_output_dim = target_fusion[9]
                    model.config.vision_fusion_source_keys = target_fusion[10]
                    model.config.vision_fusion_adapter_init = target_fusion[11]
                    model.config.vision_fusion_rgb_init_scale = target_fusion[12]
                    model.config.vision_fusion_geometry_init_scale = target_fusion[13]
                    model.config.tune_vision_fusion_adapters = (
                        self.model_config.tune_vision_fusion_adapters
                    )
                    model.collator.post_vision_fusion = True
                else:
                    raise RuntimeError(
                        "Checkpoint vision-fusion contract does not match the requested "
                        f"configuration: checkpoint={source_fusion!r}, requested={target_fusion!r}"
                    )
            elif source_fusion[0] == "post_vision_late_fusion":
                model.backbone.validate_post_vision_fusion_contract(
                    mode=source_fusion[0],
                    version=source_fusion[1],
                    stage=source_fusion[2],
                    adapter_type=source_fusion[3],
                    adapter_bias=source_fusion[4],
                    adapter_bias_init=source_fusion[5],
                    adapter_count=source_fusion[6],
                    deepstack_layers=source_fusion[7],
                    adapter_input_dim=source_fusion[8],
                    adapter_output_dim=source_fusion[9],
                    source_keys=source_fusion[10],
                    adapter_init=source_fusion[11],
                    rgb_init_scale=source_fusion[12],
                    geometry_init_scale=source_fusion[13],
                    vision_input_channels=source_channels,
                )
            _reconcile_depth_colormap_contract(
                model.config,
                self.model_config,
                initializing_late_fusion=initializing_late_fusion,
            )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(f"Model Config: {model.config}")
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                    f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        letter_box_transform = self.model_config.letter_box_transform
        logging.info("N1.7 letter_box_transform=%s", letter_box_transform)
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        with run_or_wait_on_rank0(label="final_processor_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                    json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                stats_dict = convert_tensors_to_lists(stats)
                with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                    json.dump(stats_dict, f, indent=2)
                logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
