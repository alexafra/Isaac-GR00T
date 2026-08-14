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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag


class MessageType(Enum):
    START_OF_EPISODE = "start_of_episode"
    END_OF_EPISODE = "end_of_episode"
    EPISODE_STEP = "episode_step"
    IMAGE = "image"
    TEXT = "text"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure returned by datasets, containing raw observation
    and action data that will be processed by the SequenceVLAProcessor.
    """

    # Core data
    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[
        str, np.ndarray
    ]  # state_name -> np.ndarray (dim,) for single step or (horizon, dim) for trajectory
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    masks: dict[str, list[np.ndarray]] | None = None  # view_name -> list[np.ndarray] (H, W)
    text: str | None = None  # Optional task description or instruction
    embodiment: EmbodimentTag = (
        EmbodimentTag.NEW_EMBODIMENT
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = False  # Whether the step is a demonstration. If True, no loss should be computed for this step.

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None


@dataclass(frozen=True)  # earlyfusion
class VideoChannelSource:  # earlyfusion
    key: str  # earlyfusion
    channels: tuple[int, ...]  # earlyfusion

    def __post_init__(self):  # earlyfusion
        object.__setattr__(self, "channels", tuple(self.channels))  # earlyfusion
        if not self.key or not self.channels or len(set(self.channels)) != len(self.channels):  # fmt: skip  # earlyfusion
            raise ValueError(f"Invalid channel-fusion source: {self!r}")  # earlyfusion
        if any(not isinstance(channel, int) or channel not in range(3) for channel in self.channels):  # fmt: skip  # earlyfusion
            raise ValueError(f"Channel-fusion channels must be unique RGB indices 0..2: {self!r}")  # fmt: skip  # earlyfusion


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.

    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys."""
    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys."""
    action_configs: list[ActionConfig] | None = None
    channel_fusion: list[VideoChannelSource] | None = None  # earlyfusion

    @property  # earlyfusion
    def vision_input_channels(self) -> int:  # earlyfusion
        return sum(len(source.channels) for source in self.channel_fusion) if self.channel_fusion else 3  # fmt: skip  # earlyfusion

    @property  # earlyfusion
    def vision_channel_layout(self) -> list[str]:  # earlyfusion
        sources = self.channel_fusion or [VideoChannelSource(self.modality_keys[0], (0, 1, 2))]  # fmt: skip  # earlyfusion
        return [f"{source.key}:{channel}" for source in sources for channel in source.channels]  # fmt: skip  # earlyfusion

    def __post_init__(self):
        """Validate fields and set default values."""
        if self.delta_indices is None or not isinstance(self.delta_indices, list):
            raise ValueError(f"delta_indices must be a non-None list, got {self.delta_indices!r}")
        if (
            self.modality_keys is None
            or not isinstance(self.modality_keys, list)
            or len(self.modality_keys) == 0
        ):
            raise ValueError(f"modality_keys must be a non-empty list, got {self.modality_keys!r}")
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
        if self.channel_fusion is not None:  # earlyfusion
            self.channel_fusion = [  # earlyfusion
                VideoChannelSource(**source) if isinstance(source, dict) else source  # earlyfusion
                for source in self.channel_fusion  # earlyfusion
            ]  # earlyfusion
            source_keys = [source.key for source in self.channel_fusion]  # earlyfusion
            if source_keys != self.modality_keys or self.channel_fusion[0].channels != (0, 1, 2):  # fmt: skip  # earlyfusion
                raise ValueError("channel_fusion must cover modality_keys in order and start with RGB channels 0,1,2")  # fmt: skip  # earlyfusion
            if self.vision_input_channels not in (4, 6):  # earlyfusion
                raise ValueError("channel_fusion must produce exactly 4 or 6 input channels")  # fmt: skip  # earlyfusion
