"""CPU-only parity checks against the unchanged native full-episode loader.

These tests use real pandas indexing and LeRobotEpisodeLoader.load_episode, with
in-memory storage and a capturing processor. They do not instantiate a model,
open dataset/video files, or claim parity of the complete vision processor.
"""

import importlib
import pickle
import random
from types import SimpleNamespace

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.selected_frame_training_dataset import (
    SelectedFrameTrainingDataset,
    required_visual_frame_indices,
)
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import EmbodimentTag, ModalityConfig
from gr00t.utils import video_utils
import numpy as np
import pandas as pd
import pytest


def _modalities(video=(0,), mask=None, state=(0,), action=(0, 1, 2, 3)):
    result = {
        "video": ModalityConfig(list(video), ["rgb", "depth"]),
        "state": ModalityConfig(list(state), ["hand"]),
        "action": ModalityConfig(list(action), ["hand"]),
        "language": ModalityConfig([0], ["task"]),
    }
    if mask is not None:
        result["mask"] = ModalityConfig(list(mask), ["segmentation"])
    return result


def _array_value(array):
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tolist()


def _run_shard(dataset_class, modalities, episodes, lengths, *, padding=False, steps_kind="list"):
    """Return full captured samples, decoder requests, and final random states."""
    previous_python_state = random.getstate()
    previous_numpy_state = np.random.get_state()
    try:
        random.seed(117)
        np.random.seed(871)
        loader = object.__new__(LeRobotEpisodeLoader)
        loader.modality_configs = modalities
        loader.episodes_metadata = [
            {
                "episode_index": index,
                "length": nominal,
                "tasks": [f"episode-{index}-a", f"episode-{index}-b"],
            }
            for index, (_, nominal) in enumerate(lengths)
        ]

        def parquet(episode_index):
            actual = lengths[episode_index][0]
            return pd.DataFrame(
                {
                    "state.hand": [np.array([episode_index, i]) for i in range(actual)],
                    "action.hand": [np.array([episode_index, i + 100]) for i in range(actual)],
                }
            )

        loader._load_parquet_data = parquet
        decode_events = []

        def decode(modality, episode_index, positions):
            positions = [int(position) for position in positions]
            decode_events.append((modality, episode_index, positions))
            config = modalities.get(modality)
            if config is None:
                return {}
            shape = (2, 3) if modality == "mask" else (2, 3, 3)
            return {
                key: [
                    np.full(shape, episode_index * 100 + position, dtype=np.uint8)
                    for position in positions
                ]
                for key in config.modality_keys
            }

        loader._load_video_data = lambda index, positions: decode("video", index, positions)
        loader._load_mask_data = lambda index, positions: decode("mask", index, positions)
        frame_requests = []
        native_load_episode = loader.load_episode

        def tracked_load_episode(index, frame_indices=None):
            frame_requests.append(frame_indices)
            return native_load_episode(index, frame_indices=frame_indices)

        loader.load_episode = tracked_load_episode
        dataset = object.__new__(dataset_class)
        dataset.episode_loader = loader
        dataset.modality_configs = modalities
        dataset.allow_padding = padding
        dataset.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        dataset.rng = np.random.default_rng(19)

        def convert_steps(steps):
            if steps_kind == "generator":
                return iter(steps)
            if steps_kind == "numpy":
                return np.asarray(steps, dtype=np.int64)
            return list(steps)

        dataset.sharded_episodes = [[(index, convert_steps(steps)) for index, steps in episodes]]

        def process(messages):
            assert len(messages) == 1 and messages[0]["type"] == "episode_step"
            sample = messages[0]["content"]
            return {
                "images": {
                    key: [_array_value(frame) for frame in frames]
                    for key, frames in sample.images.items()
                },
                "masks": {
                    key: [_array_value(frame) for frame in frames]
                    for key, frames in (sample.masks or {}).items()
                },
                "states": {key: _array_value(value) for key, value in sample.states.items()},
                "actions": {key: _array_value(value) for key, value in sample.actions.items()},
                "text": sample.text,
                "embodiment": sample.embodiment,
                "processor_rng": (
                    random.random(),
                    float(np.random.random()),
                    int(dataset.rng.integers(10000)),
                ),
            }

        dataset.processor = process
        result = dataset.get_shard(0)
        numpy_state = np.random.get_state()
        random_states = (
            random.getstate(),
            (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            dataset.rng.bit_generator.state,
        )
        return result, decode_events, random_states, frame_requests
    finally:
        random.setstate(previous_python_state)
        np.random.set_state(previous_numpy_state)


def _assert_parity(*args, **kwargs):
    # Intentionally use the native base class, never a factory alias or a
    # separately rendered copy that might silently become the optimized code.
    assert SelectedFrameTrainingDataset.get_shard is not ShardedSingleStepDataset.get_shard
    original = _run_shard(ShardedSingleStepDataset, *args, **kwargs)
    selected = _run_shard(SelectedFrameTrainingDataset, *args, **kwargs)
    assert original[0] == selected[0]
    assert original[2] == selected[2]
    assert all(request is None for request in original[3])
    return original, selected


@pytest.mark.parametrize("steps_kind", ["list", "numpy", "generator"])
def test_samples_order_duplicates_actions_and_rng_match_legacy(steps_kind):
    original, selected = _assert_parity(
        _modalities(),
        [(1, [5, 1, 5]), (0, [3, 0])],
        [(10, 10), (12, 12)],
        steps_kind=steps_kind,
    )
    assert selected[1][0] == ("video", 1, [1, 5])
    assert selected[1][2] == ("video", 0, [0, 3])
    assert original[1][0] == ("video", 1, list(range(12)))
    assert [sample["states"]["hand"][2][0] for sample in selected[0]] == [
        [1.0, 5.0],
        [1.0, 1.0],
        [1.0, 5.0],
        [0.0, 3.0],
        [0.0, 0.0],
    ]
    assert selected[0][0]["actions"]["hand"][1] == (4, 2)


def test_temporal_video_and_mask_union_preserves_actual_pixel_values():
    _, selected = _assert_parity(
        _modalities(video=(-1, 0, 1), mask=(-2, 2)),
        [(0, [6, 2, 6])],
        [(12, 12)],
        steps_kind="numpy",
    )
    assert selected[1] == [("video", 0, list(range(9))), ("mask", 0, list(range(9)))]
    assert selected[0][0]["images"]["rgb"][0][2][0][0] == [5, 5, 5]
    assert selected[0][0]["masks"]["segmentation"][0][2][0][0] == 4


def test_actual_length_padding_preserves_full_action_horizon():
    _, selected = _assert_parity(
        _modalities(video=(-1, 0, 2), mask=(-3, 1), state=(-1, 0)),
        [(0, [5, 0])],
        [(6, 15)],
        padding=True,
    )
    assert selected[1][0] == ("video", 0, [0, 1, 2, 4, 5])
    assert selected[0][0]["actions"]["hand"][2] == [[0.0, 105.0]] * 4
    assert selected[0][0]["states"]["hand"][2] == [[0.0, 4.0], [0.0, 5.0]]


def test_negative_visual_indices_follow_pandas_iloc_wrapping():
    _, selected = _assert_parity(
        _modalities(video=(-2, -1, 0), mask=(-3,), action=(0,)),
        [(0, [0, -1])],
        [(6, 6)],
    )
    assert selected[1][0] == ("video", 0, [0, 2, 3, 4, 5])


@pytest.mark.parametrize("step", [-7, 6])
def test_out_of_bounds_fail_for_legacy_and_selected(step):
    for dataset_class in (ShardedSingleStepDataset, SelectedFrameTrainingDataset):
        with pytest.raises(IndexError):
            _run_shard(dataset_class, _modalities(action=(0,)), [(0, [step])], [(6, 15)])


@pytest.mark.parametrize("steps_kind", ["list", "generator"])
def test_empty_sample_list_keeps_full_load_and_language_rng(steps_kind):
    original, selected = _assert_parity(
        _modalities(),
        [(0, []), (0, [2])],
        [(6, 6)],
        steps_kind=steps_kind,
    )
    assert selected[3][0] is None
    assert original[1][:2] == selected[1][:2]
    assert selected[1][0] == ("video", 0, list(range(6)))


def test_empty_shard_and_single_frame_padding():
    original, selected = _assert_parity(_modalities(), [], [(0, 0)])
    assert original[0] == selected[0] == []
    _, selected = _assert_parity(
        _modalities(video=(-2, 0, 2), mask=(-9, 9)),
        [(0, [0])],
        [(1, 9)],
        padding=True,
    )
    assert selected[1][0] == ("video", 0, [0])


def test_helper_ignores_nonvisual_horizons_without_mutating_steps():
    steps = np.array([7, 2, 7])
    modalities = _modalities(video=(-1, 0, 2), mask=(-2, 1), action=range(100), state=(-100,))
    assert required_visual_frame_indices(steps, modalities, 12, False) == list(range(10))
    np.testing.assert_array_equal(steps, [7, 2, 7])
    assert required_visual_frame_indices([500], {"action": modalities["action"]}, 0, False) == []


@pytest.mark.parametrize("existing,expected", [(None, 1), ({}, 1), ({"num_ffmpeg_threads": 3}, 3)])
def test_constructor_limits_training_decoder_only_and_preserves_explicit_overrides(
    monkeypatch, existing, expected
):
    global_defaults = dict(video_utils._DEFAULT_DECODER_KWARGS)
    base_calls = []
    old_kwargs = None if existing is None else dict(existing)

    def initialize_base(self, *args, **kwargs):
        base_calls.append((args, kwargs))
        self.episode_loader = SimpleNamespace(decoder_kwargs=existing)

    monkeypatch.setattr(ShardedSingleStepDataset, "__init__", initialize_base)
    dataset = SelectedFrameTrainingDataset("unused-dataset", seed=19)
    assert base_calls == [(("unused-dataset",), {"seed": 19})]
    assert dataset.episode_loader.decoder_kwargs["num_ffmpeg_threads"] == expected
    assert existing == old_kwargs
    assert video_utils._DEFAULT_DECODER_KWARGS == global_defaults


def test_constructor_retains_other_explicit_decoder_options(monkeypatch):
    original = {"dimension_order": "NHWC", "num_ffmpeg_threads": 8}

    def initialize_base(self, *args, **kwargs):
        self.episode_loader = SimpleNamespace(decoder_kwargs=original)

    monkeypatch.setattr(ShardedSingleStepDataset, "__init__", initialize_base)
    dataset = SelectedFrameTrainingDataset()
    assert dataset.episode_loader.decoder_kwargs == original
    assert dataset.episode_loader.decoder_kwargs is not original


def test_class_is_stable_importable_pickle_target_for_spawn_workers():
    # Class-reference pickling exercises the global import path used by spawn;
    # no subprocess, training worker, or CUDA initialization is needed here.
    module = importlib.import_module(SelectedFrameTrainingDataset.__module__)
    assert getattr(module, "SelectedFrameTrainingDataset") is SelectedFrameTrainingDataset
    assert pickle.loads(pickle.dumps(SelectedFrameTrainingDataset)) is SelectedFrameTrainingDataset
    instance = object.__new__(SelectedFrameTrainingDataset)
    instance.sharded_episodes = [[(0, np.array([3, 1, 3]))]]
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is SelectedFrameTrainingDataset
    np.testing.assert_array_equal(restored.sharded_episodes[0][0][1], [3, 1, 3])


def test_normal_training_factory_uses_optimized_class_without_replacing_legacy():
    from gr00t.data.dataset.factory import DatasetFactory

    factory_module = importlib.import_module(DatasetFactory.__module__)
    assert factory_module.ShardedSingleStepDataset is SelectedFrameTrainingDataset
    assert SelectedFrameTrainingDataset is not ShardedSingleStepDataset


def test_shard_membership_weights_and_sampling_rng_are_identical(monkeypatch):
    source_module = importlib.import_module(ShardedSingleStepDataset.__module__)

    class FakeEpisodeLoader:
        def __init__(self, **kwargs):
            self.episode_lengths = [34, 14, 60]
            self.decoder_kwargs = None

        def get_episode_length(self, episode_index):
            return self.episode_lengths[episode_index]

    monkeypatch.setattr(source_module, "LeRobotEpisodeLoader", FakeEpisodeLoader)
    kwargs = {
        "dataset_path": "unused",
        "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        "modality_configs": _modalities(),
        "shard_size": 16,
        "episode_sampling_rate": 0.1,
        "seed": 42,
    }
    legacy = ShardedSingleStepDataset(**kwargs)
    selected = SelectedFrameTrainingDataset(**kwargs)
    np.testing.assert_array_equal(legacy.shard_lengths, selected.shard_lengths)
    assert legacy.rng.bit_generator.state == selected.rng.bit_generator.state
    assert legacy.action_horizon == selected.action_horizon
    for old_shard, new_shard in zip(legacy.sharded_episodes, selected.sharded_episodes):
        assert len(old_shard) == len(new_shard)
        for (old_episode, old_steps), (new_episode, new_steps) in zip(old_shard, new_shard):
            assert old_episode == new_episode
            np.testing.assert_array_equal(old_steps, new_steps)
