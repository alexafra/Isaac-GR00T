from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
import numpy as np
import pandas as pd


def _loader_with_mocked_storage():
    loader = object.__new__(LeRobotEpisodeLoader)
    loader.episodes_metadata = [{"episode_index": 0, "length": 5}]
    loader.modality_configs = {}
    values = [np.asarray([index], dtype=np.float32) for index in range(5)]
    loader._load_parquet_data = lambda _episode_id: pd.DataFrame(
        {"state.arm": values, "action.arm": values}
    )
    requested = []

    def load_video(_episode_id, indices):
        requested.append(indices.copy())
        return {
            "camera": np.stack(
                [np.full((2, 2, 3), index, dtype=np.uint8) for index in indices]
            )
        }

    loader._load_video_data = load_video
    loader._load_mask_data = lambda _episode_id, _indices: {}
    return loader, requested


def test_load_episode_decodes_only_requested_frames_and_keeps_row_alignment():
    loader, requested = _loader_with_mocked_storage()

    episode = loader.load_episode(0, frame_indices=lambda actual_length: range(0, actual_length, 2))

    np.testing.assert_array_equal(requested[0], [0, 2, 4])
    assert len(episode) == 5
    assert episode["video.camera"].iloc[1] is None
    assert episode["video.camera"].iloc[3] is None
    for frame_index in (0, 2, 4):
        np.testing.assert_array_equal(
            episode["video.camera"].iloc[frame_index],
            np.full((2, 2, 3), frame_index, dtype=np.uint8),
        )


def test_item_access_retains_full_episode_decode_behavior():
    loader, requested = _loader_with_mocked_storage()

    episode = loader[0]

    np.testing.assert_array_equal(requested[0], np.arange(5))
    assert all(frame is not None for frame in episode["video.camera"])
