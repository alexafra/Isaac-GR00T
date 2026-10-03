# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-only visual loading with unchanged shard and timestep sampling.

The full numerical/language dataframe remains available for action horizons.
Only the union of video/mask positions needed by this shard is materialized.
Keep the legacy dataset and global decoder defaults unchanged for evaluation
and inference. The native subclass also survives spawned DataLoader workers.
"""

from operator import index

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset


def required_visual_frame_indices(step_indices, modality_configs, episode_length, allow_padding):
    """Match extract_step_data's positional indexing at the actual retained length."""
    episode_length = index(episode_length)
    if episode_length < 0:
        raise ValueError("episode_length must be nonnegative")
    offsets = set()
    for modality in ("video", "mask"):
        config = modality_configs.get(modality)
        if config is not None and config.modality_keys:
            offsets.update(index(delta) for delta in config.delta_indices)
    if not offsets:
        return []

    required = set()
    for step_index in step_indices:
        step_index = index(step_index)
        for delta in offsets:
            position = step_index + delta
            if not episode_length:
                raise IndexError("Visual frame requested from an empty episode")
            if allow_padding:
                position = max(0, min(position, episode_length - 1))
            elif position < -episode_length or position >= episode_length:
                raise IndexError(
                    f"Visual frame position {position} is outside iloc bounds for "
                    f"episode length {episode_length}"
                )
            elif position < 0:
                position += episode_length
            required.add(position)
    return sorted(required)


class SelectedFrameTrainingDataset(ShardedSingleStepDataset):
    """Same training samples, with bounded CPU decoding and fewer materialized frames."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Local to this training dataset, not a global decoder policy. Preserve
        # any explicit decoder settings provided by an episode-loader subclass.
        self.episode_loader.decoder_kwargs = {
            "num_ffmpeg_threads": 1,
            **(self.episode_loader.decoder_kwargs or {}),
        }

    def get_shard(self, idx: int) -> list:
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            # Keep original order/duplicates; consume one-shot iterables once.
            if iter(step_indices) is step_indices or not hasattr(step_indices, "__len__"):
                step_indices = list(step_indices)
            episode_data = self.episode_loader.load_episode(
                ep_idx,
                frame_indices=(
                    lambda actual_length: required_visual_frame_indices(
                        step_indices,
                        self.modality_configs,
                        actual_length,
                        self.allow_padding,
                    )
                )
                if len(step_indices)
                else None,
            )
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, step_index))
        return datapoints
