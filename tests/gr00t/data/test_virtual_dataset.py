"""Virtual unions must behave like one physically appended Inspire dataset.

The physical reference is constructed independently, not by copying the builder's
output. Tiny video placeholders test reference integrity without a codec dependency;
the LZ4 test also decodes real compressed pixels through the production reader.
"""

from copy import deepcopy
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.stats import calculate_dataset_statistics, generate_rel_stats, generate_stats
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
import gr00t.data.virtual_dataset as virtual_dataset
from gr00t.data.virtual_dataset import VirtualDatasetError, build_virtual_dataset
from gr00t.utils.lz4_frame_utils import get_plain_lz4_frames_by_indices
import numpy as np
import pandas as pd
import pytest


TAG = EmbodimentTag.NEW_EMBODIMENT
GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 20), "right_hand": (20, 26)}
RGB = "observation.images.ego_view"
NORMALS = "observation.images.surface_normals_view"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
GOALS = ["stack the three red cups.", "pick up the red cup."]
# Identity, frame count, goal. Both tasks recur across source dataset boundaries.
SPECS = [(0, 36, GOALS[0]), (1, 39, GOALS[1]), (2, 35, GOALS[1]), (3, 40, GOALS[0])]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _read_json(path):
    return json.loads(path.read_text())


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _data_path(root, index):
    return root / DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)


def _video_path(root, index, key=RGB):
    return root / VIDEO_PATH.format(episode_chunk=index // 1000, episode_index=index, video_key=key)


def _episode_arrays(identity, length):
    time = np.arange(length, dtype=np.float32)[:, None]
    joints = np.arange(26, dtype=np.float32)[None, :]
    state = (0.04 * time + 0.025 * joints + identity * 1.1).astype(np.float32)
    action = (state + 0.07 * np.sin(time * 0.3 + joints * 0.1) + identity * 0.09).astype(np.float32)
    state[:, 14:] = (0.1 + (time + joints[:, 14:] + identity * 3) % 31 / 40).astype(np.float32)
    action[:, 14:] = (0.05 + (time * 2 + joints[:, 14:] + identity * 7) % 35 / 40).astype(
        np.float32
    )
    # Isolated tails make min/max observably different from q01/q99.
    if identity == 0:
        state[0, :] = -2
        action[0, :] = -3
    if identity == 3:
        action[-1, :] = 8
    return state, action


def _write_dataset(root, specs, task_order=GOALS):
    features = {
        "observation.state": {"dtype": "float32", "shape": [26]},
        "action": {"dtype": "float32", "shape": [26]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        **{
            key: {"dtype": "int64", "shape": [1]}
            for key in ("frame_index", "episode_index", "index", "task_index")
        },
        RGB: {"dtype": "video", "shape": [2, 3, 3]},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Inspire_HeadOnly",
        "total_episodes": len(specs),
        "total_frames": sum(spec[1] for spec in specs),
        "total_tasks": len(task_order),
        "total_videos": len(specs),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{len(specs)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "depth_encoding": {"near_m": 0.25, "far_m": 1.0, "invalid_value": 0},
        "surface_normals_encoding": {"version": 2, "near_m": 0.25, "far_m": 1.0},
        "end_effector": {"type": "inspire", "value_unit": "normalized_open_fraction"},
    }
    modality = {
        kind: {name: {"start": begin, "end": end} for name, (begin, end) in GROUPS.items()}
        for kind in ("state", "action")
    }
    modality["video"] = {"ego_view": {"original_key": RGB}}
    modality["annotation"] = {"human.task_description": {"original_key": "task_index"}}
    _write_json(root / "meta/info.json", info)
    _write_json(root / "meta/modality.json", modality)
    _write_jsonl(
        root / "meta/tasks.jsonl",
        [dict(task_index=i, task=goal) for i, goal in enumerate(task_order)],
    )
    episodes = []
    offset = 0
    for episode, (identity, length, goal) in enumerate(specs):
        states, actions = _episode_arrays(identity, length)
        frame = pd.DataFrame(
            {
                "observation.state": list(states),
                "action": list(actions),
                "timestamp": np.arange(length, dtype=np.float32) / np.float32(30),
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode, dtype=np.int64),
                "index": np.arange(offset, offset + length, dtype=np.int64),
                "task_index": np.full(length, task_order.index(goal), dtype=np.int64),
            }
        )
        path = _data_path(root, episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        video = _video_path(root, episode)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"unchanged-video-{identity}".encode())
        episodes.append(dict(episode_index=episode, length=length, tasks=[goal]))
        offset += length
    _write_jsonl(root / "meta/episodes.jsonl", episodes)
    # Deliberately invalid caches prove the union computes fresh statistics.
    _write_json(root / "meta/stats.json", {"wrong_source_cache": True})
    _write_json(root / "meta/relative_stats.json", {"wrong_source_cache": True})
    return root


def _tree_hash(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def configs(monkeypatch):
    config = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(GROUPS)),
        "action": ModalityConfig(
            delta_indices=list(range(32)),
            modality_keys=list(GROUPS),
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE
                    if "arm" in key
                    else ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for key in GROUPS
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }
    monkeypatch.setitem(MODALITY_CONFIGS, TAG.value, config)
    return config


@pytest.fixture
def union(tmp_path):
    first = _write_dataset(tmp_path / "first", SPECS[:2])
    second = _write_dataset(tmp_path / "second", SPECS[2:], task_order=list(reversed(GOALS)))
    reference = _write_dataset(tmp_path / "physical", SPECS)
    (reference / "meta/stats.json").unlink()
    (reference / "meta/relative_stats.json").unlink()
    before = [_tree_hash(first), _tree_hash(second)]
    output = build_virtual_dataset([first, second], tmp_path / "virtual")
    assert [_tree_hash(first), _tree_hash(second)] == before
    return first, second, reference, Path(output)


def test_shared_goals_keep_all_episodes_and_global_indices(union, configs):
    first, second, reference, output = union
    info = _read_json(output / "meta/info.json")
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 150
    assert info["total_tasks"] == 2
    assert _read_jsonl(output / "meta/tasks.jsonl") == _read_jsonl(reference / "meta/tasks.jsonl")
    loader = LeRobotEpisodeLoader(output, configs)
    for episode, (_, length, goal) in enumerate(SPECS):
        pd.testing.assert_frame_equal(
            pd.read_parquet(_data_path(output, episode)),
            pd.read_parquet(_data_path(reference, episode)),
        )
        assert (
            loader[episode]["language.annotation.human.task_description"].tolist()
            == [goal] * length
        )
        source = first if episode < 2 else second
        source_video = _video_path(source, episode % 2)
        assert _video_path(output, episode).is_symlink()
        assert _video_path(output, episode).resolve() == source_video.resolve()
        assert (
            _video_path(output, episode).read_bytes()
            == _video_path(reference, episode).read_bytes()
        )
    assert not (output / "meta/relative_stats.json").exists()


def test_exact_union_stats_and_per_horizon_relative_stats(union, configs):
    first, second, reference, output = union
    for root in (reference, output):
        generate_stats(root)
        generate_rel_stats(root, TAG)
    assert _read_json(output / "meta/stats.json") == _read_json(reference / "meta/stats.json")
    relative = _read_json(output / "meta/relative_stats.json")
    assert relative == _read_json(reference / "meta/relative_stats.json")
    assert set(relative) == {"left_arm", "right_arm", "__fingerprints__"}
    assert np.asarray(relative["left_arm"]["min"]).shape == (32, 7)
    # Independent complete-window calculation: subtract state at chunk start,
    # not state at each future timestep and never cross an episode boundary.
    chunks = []
    for identity, length, _ in SPECS:
        states, actions = _episode_arrays(identity, length)
        chunks.extend(
            actions[start : start + 32, :7] - states[start, :7] for start in range(length - 31)
        )
    chunks = np.asarray(chunks, dtype=np.float32)
    np.testing.assert_array_equal(relative["left_arm"]["min"], chunks.min(axis=0))
    np.testing.assert_array_equal(relative["left_arm"]["max"], chunks.max(axis=0))
    np.testing.assert_array_equal(relative["left_arm"]["q01"], np.quantile(chunks, 0.01, axis=0))
    # Guard against silently using the mixture loader's min of root quantiles.
    roots_q01 = [
        calculate_dataset_statistics(list(root.glob("data/*/*.parquet")), ["action"])["action"][
            "q01"
        ]
        for root in (first, second)
    ]
    union_q01 = _read_json(output / "meta/stats.json")["action"]["q01"]
    assert not np.array_equal(union_q01, np.minimum(*roots_q01))


def _processor(root, configs):
    generate_stats(root)
    generate_rel_stats(root, TAG)
    loader = LeRobotEpisodeLoader(root, configs)
    return StateActionProcessor(
        {TAG.value: configs},
        {TAG.value: loader.get_dataset_statistics()},
        use_percentiles=True,
        use_relative_action=True,
        clip_outliers=True,
    )


def test_inspire_normalization_quirks_and_exact_processed_tensors(union, configs):
    _, _, reference, output = union
    physical_processor = _processor(reference, configs)
    virtual_processor = _processor(output, configs)
    stats = virtual_processor.statistics[TAG.value]
    params = virtual_processor.norm_params[TAG.value]
    for key in GROUPS:
        np.testing.assert_array_equal(params["state"][key]["min"], stats["state"][key]["q01"])
        expected = (
            stats["relative_action"][key]["min"] if "arm" in key else stats["action"][key]["q01"]
        )
        np.testing.assert_array_equal(params["action"][key]["min"], expected)
    # Preserve current asymmetry: arms use per-horizon relative min/max despite
    # use_percentiles=True, whereas absolute hands use one q01/q99 per joint.
    assert not np.array_equal(
        params["action"]["left_arm"]["min"], stats["relative_action"]["left_arm"]["q01"]
    )
    assert params["action"]["left_arm"]["min"].shape == (32, 7)
    assert params["action"]["left_hand"]["min"].shape == (6,)
    for identity, length, _ in SPECS:
        state, action = _episode_arrays(identity, length)
        for start in range(length - 31):
            states = {
                key: state[start : start + 1, begin:end] for key, (begin, end) in GROUPS.items()
            }
            actions = {
                key: action[start : start + 32, begin:end] for key, (begin, end) in GROUPS.items()
            }
            physical_state = physical_processor.apply_state(states, TAG.value)
            virtual_state = virtual_processor.apply_state(states, TAG.value)
            physical_action = physical_processor.apply_action(actions, TAG.value, states)
            virtual_action = virtual_processor.apply_action(actions, TAG.value, states)
            for key in GROUPS:
                np.testing.assert_array_equal(physical_state[key], virtual_state[key])
                np.testing.assert_array_equal(physical_action[key], virtual_action[key])
            # A changed measured hand state must not re-anchor hand commands.
            changed_state = deepcopy(states)
            changed_state["left_hand"] += 999
            np.testing.assert_array_equal(
                virtual_action["left_hand"],
                virtual_processor.apply_action(actions, TAG.value, changed_state)["left_hand"],
            )
    states = {
        key: np.zeros((1, end - begin), dtype=np.float32) for key, (begin, end) in GROUPS.items()
    }
    actions = {
        key: np.full((32, end - begin), 10000, dtype=np.float32)
        for key, (begin, end) in GROUPS.items()
    }
    clipped = virtual_processor.apply_action(actions, TAG.value, states)
    assert all(np.all(values == 1) for values in clipped.values())
    unclipped = virtual_processor.apply_action(actions, TAG.value, states, clip_outliers=False)
    assert all(np.any(values > 1) for values in unclipped.values())


@pytest.mark.parametrize("sampling_rate", [1.0, 0.5, 0.1])
def test_same_sharding_and_sample_order_as_physical_append(union, configs, sampling_rate):
    _, _, reference, output = union
    generate_stats(reference)
    kwargs = dict(
        embodiment_tag=TAG,
        modality_configs=configs,
        shard_size=7,
        episode_sampling_rate=sampling_rate,
        seed=42,
    )
    physical = ShardedSingleStepDataset(reference, **kwargs)
    virtual = ShardedSingleStepDataset(output, **kwargs)
    np.testing.assert_array_equal(physical.shard_lengths, virtual.shard_lengths)

    def signature(dataset):
        return [
            [(int(episode), steps.tolist()) for episode, steps in shard]
            for shard in dataset.sharded_episodes
        ]

    assert signature(physical) == signature(virtual)
    assert int(sum(virtual.shard_lengths)) == sum(length - 31 for _, length, _ in SPECS)


def _compress_lz4(raw):
    library = ctypes.util.find_library("lz4")
    if library is None:
        pytest.skip("system liblz4 is unavailable")
    lib = ctypes.CDLL(library)
    lib.LZ4F_compressFrameBound.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
    lib.LZ4F_compressFrameBound.restype = ctypes.c_size_t
    lib.LZ4F_compressFrame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.LZ4F_compressFrame.restype = ctypes.c_size_t
    capacity = lib.LZ4F_compressFrameBound(len(raw), None)
    destination = ctypes.create_string_buffer(capacity)
    size = lib.LZ4F_compressFrame(destination, capacity, raw, len(raw), None)
    assert size <= capacity
    return destination.raw[:size]


def _add_lz4(root, specs):
    info = _read_json(root / "meta/info.json")
    info["features"][NORMALS] = {"dtype": "video", "shape": [2, 3, 3]}
    info["surface_normals_lz4"] = {
        "feature_key": NORMALS,
        "storage": "plain_lz4_chunks",
        "root": "normals_lz4",
        "dtype": "uint8",
        "layout": "FHWC",
        "chunk_frames": 32,
    }
    info["total_videos"] *= 2
    _write_json(root / "meta/info.json", info)
    modality = _read_json(root / "meta/modality.json")
    modality["video"]["surface_normals_view"] = {"original_key": NORMALS}
    _write_json(root / "meta/modality.json", modality)
    for episode, (identity, length, _) in enumerate(specs):
        frames = (np.arange(length * 2 * 3 * 3, dtype=np.uint8) + identity).reshape(length, 2, 3, 3)
        directory = root / "normals_lz4" / f"episode_{episode:06d}"
        directory.mkdir(parents=True)
        (directory / "chunk_000000.lz4").write_bytes(_compress_lz4(frames.tobytes()))
        _write_json(
            directory / "index.json",
            {
                "episode_index": episode,
                "dtype": "uint8",
                "layout": "FHWC",
                "transform": "none",
                "frame_count": length,
                "chunks": [
                    {
                        "filename": "chunk_000000.lz4",
                        "start_frame": 0,
                        "frame_count": length,
                        "height": 2,
                        "width": 3,
                        "channels": 3,
                        "uncompressed_bytes": frames.nbytes,
                    }
                ],
            },
        )
        video = _video_path(root, episode, NORMALS)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"normals-backup-{identity}".encode())


def test_lz4_indexes_remap_but_pixels_and_chunks_are_unchanged(tmp_path):
    first = _write_dataset(tmp_path / "first", SPECS[:2])
    second = _write_dataset(tmp_path / "second", SPECS[2:])
    _add_lz4(first, SPECS[:2])
    _add_lz4(second, SPECS[2:])
    before = [_tree_hash(first), _tree_hash(second)]
    output = Path(build_virtual_dataset([first, second], tmp_path / "virtual"))
    info = _read_json(output / "meta/info.json")
    storage = output / info["surface_normals_lz4"]["root"]
    for global_episode, (_, length, _) in enumerate(SPECS):
        source = first if global_episode < 2 else second
        local_episode = global_episode % 2
        index = _read_json(storage / f"episode_{global_episode:06d}/index.json")
        assert index["episode_index"] == global_episode
        frames = [length - 1, 0, 3, 3]
        np.testing.assert_array_equal(
            get_plain_lz4_frames_by_indices(storage, global_episode, frames),
            get_plain_lz4_frames_by_indices(source / "normals_lz4", local_episode, frames),
        )
    assert [_tree_hash(first), _tree_hash(second)] == before


def test_lz4_storage_does_not_require_replaced_normals_mp4s(tmp_path):
    source = _write_dataset(tmp_path / "source", SPECS[:1])
    _add_lz4(source, SPECS[:1])
    _video_path(source, 0, NORMALS).unlink()
    output = Path(build_virtual_dataset([source], tmp_path / "virtual"))
    info = _read_json(output / "meta/info.json")
    storage = output / info["surface_normals_lz4"]["root"]
    np.testing.assert_array_equal(
        get_plain_lz4_frames_by_indices(storage, 0, [0, 1]),
        get_plain_lz4_frames_by_indices(source / "normals_lz4", 0, [0, 1]),
    )


@pytest.mark.parametrize("backup_exists", [True, False])
def test_lz4_view_omits_optional_h264_backup_even_when_source_advertises_it(
    tmp_path, backup_exists
):
    source = _write_dataset(tmp_path / "source", SPECS[:1])
    _add_lz4(source, SPECS[:1])
    _video_path(source, 0, NORMALS).unlink()
    info = _read_json(source / "meta/info.json")
    info["surface_normals_lz4"]["h264_backup"] = "surface_normals_h264_backup"
    _write_json(source / "meta/info.json", info)
    backup = source / "surface_normals_h264_backup/episode_000000.mp4"
    if backup_exists:
        backup.parent.mkdir()
        backup.write_bytes(b"optional-archival-normals")
    before = _tree_hash(source)

    output = Path(build_virtual_dataset([source], tmp_path / "virtual"))

    output_info = _read_json(output / "meta/info.json")
    assert "h264_backup" not in output_info["surface_normals_lz4"]
    assert not (output / "surface_normals_h264_backup").exists()
    assert not _video_path(output, 0, NORMALS).exists()
    assert _tree_hash(source) == before
    assert build_virtual_dataset([source], output, reuse=True) == output
    manifest = _read_json(output / "meta/virtual_dataset.json")
    assert manifest["schema_version"] == 1
    # Keep legacy fingerprints for existing source backups; an absent optional
    # backup does not prevent creating/reusing a view with working LZ4 normals.
    assert manifest["splits"][""][0]["media_files"] == 3 + int(backup_exists)
    storage = output / output_info["surface_normals_lz4"]["root"]
    np.testing.assert_array_equal(
        get_plain_lz4_frames_by_indices(storage, 0, [0, 1]),
        get_plain_lz4_frames_by_indices(source / "normals_lz4", 0, [0, 1]),
    )


def _add_depth(root, specs):
    info = _read_json(root / "meta/info.json")
    for key, folder in (
        ("raw_depth_encoding", "raw_depths"),
        ("aligned_depth_encoding", "aligned_depths"),
    ):
        info[key] = {
            "storage": "lossless_png",
            "dtype": "uint16",
            "shape": [2, 3],
            "scale_m_per_unit": 0.001,
            "invalid_value": 0,
            "total_files": sum(length for _, length, _ in specs),
            "path": folder
            + "/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
        }
        for episode, (identity, length, _) in enumerate(specs):
            directory = root / folder / "chunk-000" / f"episode_{episode:06d}"
            directory.mkdir(parents=True)
            for frame in range(length):
                (directory / f"frame_{frame:06d}.png").write_bytes(
                    f"{key}-{identity}-{frame}".encode()
                )
    _write_json(root / "meta/info.json", info)


@pytest.mark.parametrize("missing_middle", [False, True])
def test_raw_and_aligned_depth_remap_without_copying(tmp_path, missing_middle):
    first = _write_dataset(tmp_path / "first", SPECS[:1])
    second = _write_dataset(tmp_path / "second", SPECS[1:2])
    _add_depth(first, SPECS[:1])
    _add_depth(second, SPECS[1:2])
    output = tmp_path / "virtual"
    if missing_middle:
        (second / "aligned_depths/chunk-000/episode_000000/frame_000010.png").unlink()
        with pytest.raises(VirtualDatasetError):
            build_virtual_dataset([first, second], output)
        assert not output.exists()
        assert not list(tmp_path.glob(".virtual.building-*"))
        return
    build_virtual_dataset([first, second], output)
    for folder in ("raw_depths", "aligned_depths"):
        target = output / folder / "chunk-000/episode_000001"
        source = second / folder / "chunk-000/episode_000000"
        assert target.is_symlink()
        assert target.resolve() == source.resolve()
        assert (target / "frame_000010.png").read_bytes() == (
            source / "frame_000010.png"
        ).read_bytes()
    info = _read_json(output / "meta/info.json")
    assert info["raw_depth_encoding"]["total_files"] == 75
    assert info["aligned_depth_encoding"]["total_files"] == 75


def test_failed_build_cleans_stage_and_preserves_sources(tmp_path, monkeypatch):
    source = _write_dataset(tmp_path / "source", SPECS[:1])
    before = _tree_hash(source)

    def fail_write(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(virtual_dataset.pq, "write_table", fail_write)
    output = tmp_path / "virtual"
    with pytest.raises(OSError, match="simulated full disk"):
        build_virtual_dataset([source], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".virtual.building-*"))
    assert _tree_hash(source) == before


def test_existing_output_never_overwritten(tmp_path):
    source = _write_dataset(tmp_path / "source", SPECS[:1])
    output = tmp_path / "existing-user-files"
    output.mkdir()
    (output / "keep.txt").write_text("user-owned content")
    before = _tree_hash(output)
    with pytest.raises(VirtualDatasetError):
        build_virtual_dataset([source], output, reuse=True)
    assert _tree_hash(output) == before


@pytest.mark.parametrize(
    "failure", ["duplicate_root", "schema", "normals_version", "missing_video"]
)
def test_invalid_sources_fail_without_publishing_output(tmp_path, failure):
    first = _write_dataset(tmp_path / "first", SPECS[:2])
    second = _write_dataset(tmp_path / "second", SPECS[2:])
    if failure == "duplicate_root":
        second = first
    elif failure in {"schema", "normals_version"}:
        info = _read_json(second / "meta/info.json")
        if failure == "schema":
            info["features"]["action"]["shape"] = [25]
        else:
            info["surface_normals_encoding"]["version"] = 1
        _write_json(second / "meta/info.json", info)
    else:
        _video_path(second, 1).unlink()
    output = tmp_path / "must-not-publish"
    with pytest.raises((VirtualDatasetError, FileNotFoundError)):
        build_virtual_dataset([first, second], output)
    assert not output.exists()


def test_reuse_is_explicit_and_rejects_changed_source(union):
    first, second, _, output = union
    with pytest.raises((VirtualDatasetError, FileExistsError)):
        build_virtual_dataset([first, second], output)
    assert Path(build_virtual_dataset([first, second], output, reuse=True)) == output
    original_hash = _tree_hash(output)
    source_parquet = _data_path(second, 0)
    frame = pd.read_parquet(source_parquet)
    replacement = frame.loc[0, "action"].copy()
    replacement[0] += 0.125
    frame.at[0, "action"] = replacement
    frame.to_parquet(source_parquet, index=False)
    with pytest.raises(VirtualDatasetError):
        build_virtual_dataset([first, second], output, reuse=True)
    assert _tree_hash(output) == original_hash


def test_split_roots_preserve_splits_without_validation_statistics_leakage(tmp_path):
    sources = [tmp_path / "first", tmp_path / "second"]
    for source_index, source in enumerate(sources):
        for split_index, split in enumerate(("train", "validation", "test")):
            _write_dataset(
                source / split, [(source_index + split_index * 10, 35, GOALS[source_index])]
            )
    output = Path(build_virtual_dataset(sources, tmp_path / "virtual"))
    for split in ("train", "validation", "test"):
        info = _read_json(output / split / "meta/info.json")
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 70
        assert info["total_tasks"] == 2
    train_stats = _read_json(output / "train/meta/stats.json")
    validation_stats = _read_json(output / "validation/meta/stats.json")
    assert train_stats["action"]["max"][0] < validation_stats["action"]["min"][0]


def test_selected_train_validation_builds_without_discovering_test(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for index, split in enumerate(("train", "validation")):
        _write_dataset(source / split, [(index, 35, GOALS[0])])
    forbidden = source / "test"
    original_stat, original_open = Path.stat, Path.open

    def guarded_stat(path, *args, **kwargs):
        assert not path.is_relative_to(forbidden), "unselected test path was discovered"
        return original_stat(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        assert not path.is_relative_to(forbidden), "unselected test file was opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    output = build_virtual_dataset(
        [source], tmp_path / "virtual", splits=("train", "validation"), require_splits=True
    )
    assert set(_read_json(output / "virtual_dataset.json")["splits"]) == {"train", "validation"}
    assert not (output / "test").exists()
    assert build_virtual_dataset(
        [source], output, splits=("train", "validation"), reuse=True
    ) == output


def test_selected_splits_require_every_requested_split_in_every_source(tmp_path):
    sources = [tmp_path / "complete", tmp_path / "train-only"]
    for source in sources:
        _write_dataset(source / "train", SPECS[:1])
    _write_dataset(sources[0] / "validation", SPECS[1:2])
    output = tmp_path / "virtual"
    with pytest.raises(VirtualDatasetError, match="Missing selected split 'validation'"):
        build_virtual_dataset(sources, output, splits=("train", "validation"))
    assert not output.exists()
    with pytest.raises(VirtualDatasetError, match="all train/validation/test"):
        build_virtual_dataset([sources[0]], output, require_splits=True)
    assert not output.exists()


@pytest.mark.parametrize("splits", [[], ["train", "train"], ["unknown"], "train"])
def test_invalid_split_selections_fail_before_source_access(tmp_path, splits):
    with pytest.raises(VirtualDatasetError, match="unique train/validation/test"):
        build_virtual_dataset([tmp_path / "missing-source"], tmp_path / "view", splits=splits)


def test_selecting_train_alone_needs_no_other_splits(tmp_path):
    source = tmp_path / "source"
    _write_dataset(source / "train", SPECS[:1])
    output = build_virtual_dataset([source], tmp_path / "virtual", splits=["train"])
    assert set(_read_json(output / "virtual_dataset.json")["splits"]) == {"train"}
    assert not (output / "validation").exists()
    assert not (output / "test").exists()


def _split_source_with_provenance(root, hashes, identity_offset=0):
    records = []
    for split_index, (split, digest) in enumerate(hashes.items()):
        _write_dataset(
            root / split, [(identity_offset + split_index, 35, GOALS[0])], task_order=[GOALS[0]]
        )
        records.append(
            {
                "split": split,
                "split_episode": "episode_000000",
                "final_episode_index": 0,
                "frame_count": 35,
                "goal": GOALS[0],
                "data_json_sha256": digest,
            }
        )
    _write_json(
        root / "provenance/merge_manifest.json",
        {
            "version": 1,
            "generator": "append_lerobot2.py",
            "episodes": records,
        },
    )
    return root


@pytest.mark.parametrize("duplicate_split", ["train", "validation"])
def test_provenance_duplicate_episodes_across_roots_or_splits_are_rejected(
    tmp_path, duplicate_split
):
    first = _split_source_with_provenance(
        tmp_path / "first", {"train": "a" * 64, "validation": "b" * 64}
    )
    other_hashes = {"train": "c" * 64, "validation": "d" * 64}
    other_hashes[duplicate_split] = "a" * 64
    second = _split_source_with_provenance(tmp_path / "second", other_hashes, identity_offset=10)
    output = tmp_path / "virtual"
    with pytest.raises(VirtualDatasetError, match="(?i)duplicate|overlap"):
        build_virtual_dataset([first, second], output)
    assert not output.exists()


def test_shared_goal_distinct_provenance_is_allowed_and_nested_unions_keep_identity(tmp_path):
    first = _split_source_with_provenance(
        tmp_path / "first", {"train": "a" * 64, "validation": "b" * 64}
    )
    second = _split_source_with_provenance(
        tmp_path / "second", {"train": "c" * 64, "validation": "d" * 64}, identity_offset=10
    )
    output = Path(build_virtual_dataset([first, second], tmp_path / "virtual"))
    # One shared language goal, four genuinely distinct demonstrations.
    for split, expected in (("train", {"a" * 64, "c" * 64}), ("validation", {"b" * 64, "d" * 64})):
        info = _read_json(output / split / "meta/info.json")
        assert info["total_tasks"] == 1
        assert info["total_episodes"] == 2
        provenance = _read_jsonl(output / split / "meta/virtual_episode_sources.jsonl")
        assert {episode["data_json_sha256"] for episode in provenance} == expected
    # The original raw identity must survive the view, otherwise adding a component
    # already present in that view would silently double its training frequency.
    with pytest.raises(VirtualDatasetError, match="(?i)duplicate|overlap"):
        build_virtual_dataset([output, first], tmp_path / "nested-duplicate")
    third = _split_source_with_provenance(
        tmp_path / "third", {"train": "e" * 64, "validation": "f" * 64}, identity_offset=20
    )
    nested = Path(build_virtual_dataset([output, third], tmp_path / "nested-distinct"))
    provenance = _read_jsonl(nested / "train/meta/virtual_episode_sources.jsonl")
    assert {episode["data_json_sha256"] for episode in provenance} == {"a" * 64, "c" * 64, "e" * 64}
