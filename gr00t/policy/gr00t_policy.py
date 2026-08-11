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

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import FINETUNE_ONLY_TAGS, POSTTRAIN_TAGS, EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


_RTC_MODE = "rtc"
_SYNCHRONOUS_MODE = "synchronous"
_RTC_OPTION_KEYS = frozenset(
    {
        "inference_mode",
        "rtc_previous_action",
        "rtc_overlap_steps",
        "rtc_frozen_steps",
        "rtc_ramp_rate",
    }
)
_RTC_REQUIRED_OPTION_KEYS = frozenset(
    {
        "rtc_previous_action",
        "rtc_overlap_steps",
        "rtc_frozen_steps",
    }
)


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _sim_language_batch_to_sequence(value: Any) -> Any:
    """Normalize sim language batches while preserving validation semantics."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, str):
        return [value]
    return value


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        processor_path: str | Path | None = None,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type.
                Accepts an EmbodimentTag enum or a string (resolved case-insensitively).
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            processor_path: Optional processor/statistics directory. When omitted,
                the processor is loaded from ``model_path`` as before. This allows
                evaluating base weights with a finetuning run's embodiment-specific
                processor without copying the base model weights.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation.
        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root.  Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_model_dir = Path(processor_path) if processor_path is not None else model_dir
        processor_dir = (
            processor_model_dir / "processor"
            if (processor_model_dir / "processor").is_dir()
            and not (processor_model_dir / "processor_config.json").exists()
            else processor_model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            # Map raw checkpoint tag values to user-friendly enum names where possible.
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
                else:
                    supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model. "
                    f"See the example READMEs for how to finetune and download checkpoints."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots. "
                    f"Use it with launch_finetune.py, not with the base model directly."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    @staticmethod
    def _rtc_integer_option(options: dict[str, Any], key: str) -> int:
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"RTC option '{key}' must be an integer, got {value!r}")
        return int(value)

    def _prepare_rtc_action_input(
        self,
        options: dict[str, Any] | None,
        states: list[dict[str, np.ndarray]],
    ) -> tuple[
        np.ndarray | None,
        dict[str, Any] | None,
        dict[str, Any],
        dict[str, np.ndarray] | None,
    ]:
        """Validate and encode an RTC continuation request.

        ``rtc_previous_action`` is the unconsumed *physical* action tail at the
        instant the accompanying observation was captured. Re-encoding it against
        that new observation is required for relative-action embodiments: feeding
        back the old normalized deltas would re-anchor them to the new state and
        move the supposedly frozen physical targets.
        """

        if options is None:
            return None, None, {}, None
        if not isinstance(options, dict):
            raise ValueError(f"Policy options must be a dictionary, got {type(options).__name__}")
        if not options:
            return None, None, {}, None

        mode = options.get("inference_mode")
        supplied_rtc_keys = set(options) & (_RTC_OPTION_KEYS - {"inference_mode"})
        if mode is None:
            if supplied_rtc_keys:
                raise ValueError(
                    "RTC options require inference_mode='rtc'; got RTC keys "
                    f"{sorted(supplied_rtc_keys)}"
                )
            # Preserve the historical behavior for unrelated policy options.
            return None, None, {}, None
        if mode == _SYNCHRONOUS_MODE:
            unexpected = set(options) - {"inference_mode"}
            if unexpected:
                raise ValueError(
                    f"Synchronous inference does not accept RTC options: {sorted(unexpected)}"
                )
            return None, None, {}, None
        if mode != _RTC_MODE:
            raise ValueError(
                f"Policy option 'inference_mode' must be 'synchronous' or 'rtc', got {mode!r}"
            )

        unknown = set(options) - _RTC_OPTION_KEYS
        if unknown:
            suffix = (
                " The RTC action horizon is derived from rtc_previous_action and must not "
                "be supplied by the client."
                if "action_horizon" in unknown
                else ""
            )
            raise ValueError(f"Unknown RTC options: {sorted(unknown)}.{suffix}")
        missing = _RTC_REQUIRED_OPTION_KEYS - set(options)
        if missing:
            raise ValueError(f"Missing required RTC options: {sorted(missing)}")

        previous_action = options["rtc_previous_action"]
        if not isinstance(previous_action, dict):
            raise ValueError("RTC option 'rtc_previous_action' must be an action dictionary")

        action_config = self.modality_configs["action"]
        action_keys = list(action_config.modality_keys)
        expected_keys = set(action_keys)
        actual_keys = set(previous_action)
        if actual_keys != expected_keys:
            raise ValueError(
                "RTC previous action keys must exactly match the policy action keys; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )

        batch_size = len(states)
        previous_horizon: int | None = None
        action_dims: dict[str, int] = {}
        normalizer = getattr(self.processor, "state_action_processor", None)
        if normalizer is None or not callable(getattr(normalizer, "apply_action", None)):
            raise RuntimeError(
                "This policy processor cannot re-encode physical actions, so RTC is unsupported"
            )

        norm_params = normalizer.norm_params[self.embodiment_tag.value]["action"]
        for key in action_keys:
            value = previous_action[key]
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    f"RTC previous action '{key}' must be a numpy array, got {type(value).__name__}"
                )
            if value.dtype != np.float32:
                raise ValueError(
                    f"RTC previous action '{key}' must have dtype float32, got {value.dtype}"
                )
            if value.ndim != 3:
                raise ValueError(
                    f"RTC previous action '{key}' must have shape (B, L, D), got {value.shape}"
                )
            expected_dim = int(norm_params[key]["dim"].item())
            expected_prefix = (batch_size,)
            if value.shape[0:1] != expected_prefix or value.shape[2] != expected_dim:
                raise ValueError(
                    f"RTC previous action '{key}' must have shape "
                    f"({batch_size}, L, {expected_dim}), got {value.shape}"
                )
            if previous_horizon is None:
                previous_horizon = value.shape[1]
            elif value.shape[1] != previous_horizon:
                raise ValueError(
                    "All RTC previous action groups must have the same tail horizon; "
                    f"'{key}' has {value.shape[1]}, expected {previous_horizon}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"RTC previous action '{key}' contains NaN or infinity")
            action_dims[key] = expected_dim

        assert previous_horizon is not None  # action configuration must contain at least one key
        trained_horizon = len(action_config.delta_indices)
        if not 1 <= previous_horizon <= trained_horizon:
            raise ValueError(
                "RTC previous action tail horizon must satisfy "
                f"1 <= L <= trained horizon {trained_horizon}, got {previous_horizon}"
            )

        overlap_steps = self._rtc_integer_option(options, "rtc_overlap_steps")
        frozen_steps = self._rtc_integer_option(options, "rtc_frozen_steps")
        if not 1 <= overlap_steps <= previous_horizon:
            raise ValueError(
                "RTC overlap must satisfy "
                f"1 <= rtc_overlap_steps <= previous tail horizon {previous_horizon}, "
                f"got {overlap_steps}"
            )
        if not 0 <= frozen_steps <= overlap_steps:
            raise ValueError(
                "RTC frozen steps must satisfy "
                f"0 <= rtc_frozen_steps <= rtc_overlap_steps {overlap_steps}, "
                f"got {frozen_steps}"
            )

        ramp_rate = options.get("rtc_ramp_rate")
        if ramp_rate is None:
            ramp_rate = getattr(getattr(self.model, "config", None), "rtc_ramp_rate", None)
            if ramp_rate is None:
                raise ValueError(
                    "RTC ramp rate was not supplied and the model config has no rtc_ramp_rate"
                )
        if isinstance(ramp_rate, bool) or not isinstance(ramp_rate, Real):
            raise ValueError(f"RTC ramp rate must be a finite positive number, got {ramp_rate!r}")
        ramp_rate = float(ramp_rate)
        if not math.isfinite(ramp_rate) or ramp_rate <= 0.0:
            raise ValueError(f"RTC ramp rate must be a finite positive number, got {ramp_rate!r}")

        model_config = getattr(self.model, "config", None)
        try:
            model_horizon = int(model_config.action_horizon)
            model_action_dim = int(model_config.max_action_dim)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "The loaded model does not expose action_horizon/max_action_dim; RTC is unsupported"
            ) from exc
        total_action_dim = sum(action_dims.values())
        if model_horizon < trained_horizon or model_action_dim < total_action_dim:
            raise RuntimeError(
                "The loaded model action shape cannot represent this RTC request: "
                f"model=({model_horizon}, {model_action_dim}), "
                f"policy=({trained_horizon}, {total_action_dim})"
            )

        normalized_batch = []
        for batch_index, state in enumerate(states):
            sample_action = {key: previous_action[key][batch_index] for key in action_keys}
            normalized_groups = normalizer.apply_action(
                sample_action,
                self.embodiment_tag.value,
                state=state,
                clip_outliers=False,
            )
            normalized = np.concatenate(
                [np.asarray(normalized_groups[key], dtype=np.float32) for key in action_keys],
                axis=-1,
            )
            if normalized.shape != (previous_horizon, total_action_dim):
                raise RuntimeError(
                    "RTC action normalizer returned an unexpected shape: "
                    f"{normalized.shape}, expected ({previous_horizon}, {total_action_dim})"
                )
            if not np.isfinite(normalized).all():
                raise ValueError("Normalized RTC previous action contains NaN or infinity")
            padded = np.zeros((model_horizon, model_action_dim), dtype=np.float32)
            padded[:previous_horizon, :total_action_dim] = normalized
            normalized_batch.append(padded)

        model_options = {
            "action_horizon": previous_horizon,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": ramp_rate,
        }
        info = {
            "rtc_applied": True,
            "rtc_previous_action_horizon": previous_horizon,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": ramp_rate,
        }
        frozen_physical_prefix = None
        if frozen_steps:
            source_start = previous_horizon - overlap_steps
            frozen_physical_prefix = {
                key: previous_action[key][:, source_start : source_start + frozen_steps, :].copy()
                for key in action_keys
            }
        return np.stack(normalized_batch, axis=0), model_options, info, frozen_physical_prefix

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Check that the expected state key exists in the observation
            # (must happen before indexing — see video validation above)
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Check that the expected language key exists in the observation
            # (must happen before indexing — see video validation above)
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional inference parameters. ``inference_mode='rtc'``
                accepts an unconsumed physical action tail plus RTC overlap/frozen
                settings; synchronous behavior is unchanged when omitted.

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        states = [obs["state"] for obs in unbatched_observations]
        (
            rtc_action_input,
            model_options,
            inference_info,
            frozen_physical_prefix,
        ) = self._prepare_rtc_action_input(options, states)

        # Step 2: Process each observation through the VLA processor
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        if rtc_action_input is not None:
            try:
                collated_inputs["inputs"]["action"] = torch.from_numpy(rtc_action_input)
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    "The policy collator does not expose inputs['action']; RTC is unsupported"
                ) from exc
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            if model_options is None:
                model_pred = self.model.get_action(**collated_inputs)
            else:
                model_pred = self.model.get_action(**collated_inputs, options=model_options)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        if frozen_physical_prefix is not None:
            for key, frozen_targets in frozen_physical_prefix.items():
                unnormalized_action[key][:, : frozen_targets.shape[1], :] = frozen_targets

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, inference_info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language = _sim_language_batch_to_sequence(observation[language_key])

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple, list, or numpy array. "
                f"Got {type(observation[language_key])}"
            )
            assert batched_language, f"Language key '{language_key}' must not be empty"

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    arr = _sim_language_batch_to_sequence(arr)
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
