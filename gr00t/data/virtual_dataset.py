"""Build a single logical LeRobot 2.1 dataset without copying its media.

Only the small Parquet indices and metadata are rewritten. Source order and episode
order are preserved, so the ordinary single-root loader, sharder and exact statistics
calculation see the same examples as a physical append. Media remain read-only source
dependencies: keep the source directories in place for the lifetime of the view.

    python -m gr00t.data.virtual_dataset --sources OLD NEW --output VIEW --reuse

Sources must be either all individual datasets or all split roots. This deliberately
does not use ShardedMixtureDataset's per-root weights or merged percentile statistics.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
MANIFEST = "virtual_dataset.json"
DATA_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATTERN = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


class VirtualDatasetError(ValueError):
    """Sources cannot safely be represented as one ordinary dataset."""


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def _path(root: Path, pattern: str, episode: int, chunks: int, **keys: Any) -> Path:
    return root / pattern.format(episode_index=episode, episode_chunk=episode // chunks, **keys)


def _safe_destination(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise VirtualDatasetError(f"Output-relative path required, got {relative!r}")
    return path


def _link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise VirtualDatasetError(f"Missing source media: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _publish_noreplace(stage: Path, destination: Path) -> None:
    """Linux atomic rename with no overwrite, including a concurrent empty directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as exc:
        raise VirtualDatasetError("Atomic view publication requires Linux renameat2") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _split_paths(root: Path, splits: Sequence[str] | None = None) -> dict[str, Path]:
    if (root / "meta/info.json").is_file():
        if splits is not None:
            raise VirtualDatasetError(f"Selected splits require a split root, not a dataset: {root}")
        return {"": root}
    result = {}
    for name in ("train", "validation", "test") if splits is None else splits:
        candidates = ("validation", "validate") if name == "validation" else (name,)
        found = [
            root / candidate
            for candidate in candidates
            if (root / candidate / "meta/info.json").is_file()
        ]
        if len(found) > 1:
            raise VirtualDatasetError(f"Both validation and validate datasets exist in {root}")
        if found:
            result[name] = found[0]
        elif splits is not None:
            raise VirtualDatasetError(f"Missing selected split {name!r} in {root}")
    if not result:
        raise VirtualDatasetError(f"Not a LeRobot dataset or split root: {root}")
    return result


def _semantic_info(info: dict[str, Any]) -> dict[str, Any]:
    """Compare data contracts, not sample counts or storage locations."""
    fields = (
        "robot_type",
        "fps",
        "features",
        "depth_encoding",
        "surface_normals_encoding",
        "camera_calibration",
        "end_effector",
    )
    result = copy.deepcopy({key: info.get(key) for key in fields})
    # How identical calibration numbers were discovered is provenance, not an
    # encoding difference (recorded episode metadata versus converter profile).
    # Keep the complete calibration values, identity and fingerprint strict.
    calibration = result.get("camera_calibration")
    if isinstance(calibration, dict):
        calibration.pop("source", None)
    normals = result.get("surface_normals_encoding")
    if isinstance(normals, dict) and isinstance(normals.get("camera_calibration"), dict):
        normals["camera_calibration"].pop("source", None)
    for key in ("raw_depth_encoding", "aligned_depth_encoding", "surface_normals_lz4"):
        config = info.get(key)
        result[key] = (
            None
            if config is None
            else {
                k: v
                for k, v in config.items()
                if k not in {"total_files", "root", "path", "h264_backup"}
            }
        )
    result["has_mask"] = bool(info.get("mask_path"))
    return result


def _without_normals_backup(info: dict[str, Any]) -> dict[str, Any]:
    """Keep the training contract while dropping optional archival media."""
    result = copy.deepcopy(info)
    if isinstance(result.get("surface_normals_lz4"), dict):
        result["surface_normals_lz4"].pop("h264_backup", None)
    return result


def _load_provenance(
    root: Path, owner: Path, split: str, episodes: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[Path]]:
    virtual = root / "meta/virtual_episode_sources.jsonl"
    if virtual.is_file():
        records = _read_jsonl(virtual)
        paths = [virtual]
        indices = [int(record["episode_index"]) for record in records]
    else:
        manifest = owner / "provenance/merge_manifest.json"
        if not manifest.is_file():
            manifest = owner / "split_manifest.json"
        if not manifest.is_file():
            return {}, []
        records = _read_json(manifest).get("episodes", [])
        if split:
            records = [
                record
                for record in records
                if ("validation" if record.get("split") == "validate" else record.get("split"))
                == split
            ]
        paths = [manifest]
        if records and all(
            isinstance(record.get("final_episode_index"), int) for record in records
        ):
            indices = [int(record["final_episode_index"]) for record in records]
        elif records:
            if len(records) != len(episodes):
                raise VirtualDatasetError(f"Incomplete legacy episode provenance: {manifest}")
            try:
                records.sort(
                    key=lambda record: int(
                        re.fullmatch(r"episode_(\d+)", str(record["split_episode"]))[1]
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise VirtualDatasetError(f"Invalid legacy episode provenance: {manifest}") from exc
            if len({record["split_episode"] for record in records}) != len(records):
                raise VirtualDatasetError(f"Duplicate episode provenance: {manifest}")
            indices = list(range(len(records)))
        else:
            indices = []
        if manifest.name == "split_manifest.json" and len(records) != len(episodes):
            raise VirtualDatasetError(f"Incomplete split provenance: {manifest}")
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= len(episodes) for index in indices
    ):
        raise VirtualDatasetError(f"Invalid episode provenance indices in {paths[0]}")
    indexed = dict(zip(indices, records, strict=True))
    for index, record in indexed.items():
        if "frame_count" in record and int(record["frame_count"]) != int(episodes[index]["length"]):
            raise VirtualDatasetError(f"Episode provenance frame count mismatch in {paths[0]}")
        digest = record.get("data_json_sha256")
        if digest is not None and (
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise VirtualDatasetError(f"Invalid episode provenance hash in {paths[0]}")
    return indexed, paths


def _load_source(root: Path, owner: Path, split: str) -> dict[str, Any]:
    info = _read_json(root / "meta/info.json")
    if info.get("codebase_version") not in {"v2.1", "2.1"}:
        raise VirtualDatasetError(f"Only LeRobot v2.1 is supported: {root}")
    if int(info.get("chunks_size", 0)) <= 0:
        raise VirtualDatasetError(f"Invalid chunks_size in {root}")
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    tasks = _read_jsonl(root / "meta/tasks.jsonl")
    modality = _read_json(root / "meta/modality.json")
    if not episodes or len(episodes) != int(info["total_episodes"]):
        raise VirtualDatasetError(f"Empty or inconsistent episode metadata in {root}")
    indices = [int(ep["episode_index"]) for ep in episodes]
    if indices != list(range(len(episodes))):
        raise VirtualDatasetError(f"Episode indices must be ordered and contiguous in {root}")
    if any(int(ep["length"]) <= 0 for ep in episodes) or sum(
        int(ep["length"]) for ep in episodes
    ) != int(info["total_frames"]):
        raise VirtualDatasetError(f"Inconsistent frame counts in {root}")
    task_ids = [int(task["task_index"]) for task in tasks]
    if len(set(task_ids)) != len(task_ids) or any(
        not isinstance(task.get("task"), str) for task in tasks
    ):
        raise VirtualDatasetError(f"Duplicate task IDs or invalid task text in {root}")
    if len(tasks) != int(info["total_tasks"]):
        raise VirtualDatasetError(f"Inconsistent task count in {root}")
    # A cache referring to trajectory names cannot be concatenated by blindly copying
    # it. Current Inspire datasets do not have this optional initialization cache.
    if (root / "meta/initial_actions.npz").exists():
        raise VirtualDatasetError(
            f"initial_actions.npz needs an explicit remap; use physical append: {root}"
        )
    if (
        not split
        and root.name in {"train", "test", "validation", "validate"}
        and not (owner / "provenance/merge_manifest.json").exists()
        and not (owner / "split_manifest.json").exists()
    ):
        owner, split = root.parent, "validation" if root.name == "validate" else root.name
    provenance, provenance_paths = _load_provenance(root, owner, split, episodes)
    return {
        "root": root,
        "info": info,
        "episodes": episodes,
        "tasks": tasks,
        "modality": modality,
        "provenance": provenance,
        "provenance_paths": provenance_paths,
    }


def _fingerprint(source: dict[str, Any]) -> dict[str, Any]:
    """Hash small numeric/contract inputs; track media size+mtime without decoding.

    Derived statistics caches are intentionally excluded: training may regenerate
    those without changing the underlying demonstration or invalidating this view.
    Present H.264 backups retain legacy fingerprint compatibility, but missing
    backups do not prevent construction of a training view from its LZ4 data.
    """
    root, info = source["root"], source["info"]
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        digest.update(name.encode())
        digest.update((root / "meta" / name).read_bytes())
    for path in source["provenance_paths"]:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    tracked_roots = set()
    for episode in source["episodes"]:
        index = int(episode["episode_index"])
        parquet = _path(root, info["data_path"], index, int(info["chunks_size"]))
        if not parquet.is_file():
            raise VirtualDatasetError(f"Missing source Parquet: {parquet}")
        digest.update(parquet.read_bytes())
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                video = _path(
                    root,
                    info["video_path"],
                    index,
                    int(info["chunks_size"]),
                    video_key=key,
                )
                if (
                    info.get("surface_normals_lz4", {}).get("feature_key") == key
                    and not video.exists()
                ):
                    continue
                tracked_roots.add(video.parent)
        for name in ("raw_depth_encoding", "aligned_depth_encoding"):
            if info.get(name):
                tracked_roots.add(
                    _path(
                        root,
                        info[name]["path"],
                        index,
                        int(info["chunks_size"]),
                        frame_index=0,
                    ).parent
                )
    if info.get("surface_normals_lz4"):
        config = info["surface_normals_lz4"]
        tracked_roots.add(root / config["root"])
        if config.get("h264_backup") and (root / config["h264_backup"]).is_dir():
            tracked_roots.add(root / config["h264_backup"])
    if info.get("mask_path"):
        for key, config in source["modality"].get("mask", {}).items():
            original = config.get("original_key", key)
            for episode in source["episodes"]:
                tracked_roots.add(
                    _path(
                        root,
                        info["mask_path"],
                        int(episode["episode_index"]),
                        int(info["chunks_size"]),
                        mask_key=original,
                        video_key=original,
                    ).parent
                )
    # Walk each directory only once, even if many episode videos share it.
    tracked_roots = {
        p for p in tracked_roots if not any(q != p and q in p.parents for q in tracked_roots)
    }
    count = 0
    for directory in sorted(tracked_roots):
        if not directory.is_dir():
            raise VirtualDatasetError(f"Missing source media directory: {directory}")
        for current, dirs, files in os.walk(directory, followlinks=True):
            dirs.sort()
            for name in sorted(files):
                path = Path(current) / name
                stat = path.stat()
                digest.update(str(path).encode())
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}".encode())
                if name == "index.json":
                    digest.update(path.read_bytes())
                count += 1
    return {"path": str(root), "sha256": digest.hexdigest(), "media_files": count}


def _remap_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    location = table.schema.get_field_index(name)
    if location < 0:
        raise VirtualDatasetError(f"Missing Parquet column {name}")
    field = table.schema.field(location)
    return table.set_column(location, field, pa.array(values, type=field.type))


def _link_lz4(
    source: dict[str, Any],
    destination: Path,
    info: dict[str, Any],
    old: int,
    new: int,
    length: int,
) -> None:
    config = source["info"]["surface_normals_lz4"]
    if config.get("storage") != "plain_lz4_chunks":
        raise VirtualDatasetError(f"Unsupported normals storage: {config}")
    source_episode = source["root"] / config["root"] / f"episode_{old:06d}"
    index = _read_json(source_episode / "index.json")
    if int(index.get("episode_index", -1)) != old or int(index.get("frame_count", -1)) != length:
        raise VirtualDatasetError(f"Inconsistent LZ4 episode index: {source_episode}")
    if any(
        index.get(key) != value
        for key, value in {
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
        }.items()
    ):
        raise VirtualDatasetError(f"Unsupported LZ4 encoding: {source_episode}")
    target = destination / info["surface_normals_lz4"]["root"] / f"episode_{new:06d}"
    next_frame = 0
    for chunk in index["chunks"]:
        if int(chunk["start_frame"]) != next_frame or int(chunk["frame_count"]) <= 0:
            raise VirtualDatasetError(f"Non-contiguous LZ4 chunks: {source_episode}")
        next_frame += int(chunk["frame_count"])
        filename = chunk["filename"]
        _link(source_episode / filename, _safe_destination(target, filename))
    if next_frame != length:
        raise VirtualDatasetError(f"Wrong LZ4 frame count: {source_episode}")
    index["episode_index"] = new
    _write_json(target / "index.json", index)


def _build_direct(
    sources: list[dict[str, Any]],
    destination: Path,
    split: str,
    signatures: list[dict[str, Any]],
) -> None:
    info = _without_normals_backup(sources[0]["info"])
    chunks = int(info["chunks_size"])
    info["data_path"], info["video_path"] = DATA_PATTERN, VIDEO_PATTERN
    if info.get("surface_normals_lz4"):
        info["surface_normals_lz4"]["root"] = "surface_normals_lz4"
    for name, folder in (
        ("raw_depth_encoding", "raw_depths"),
        ("aligned_depth_encoding", "aligned_depths"),
    ):
        if info.get(name):
            info[name]["path"] = (
                f"{folder}/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}/frame_{{frame_index:06d}}.png"
            )
    if info.get("mask_path"):
        suffix = Path(info["mask_path"]).suffix
        info["mask_path"] = (
            f"masks/chunk-{{episode_chunk:03d}}/{{mask_key}}/episode_{{episode_index:06d}}{suffix}"
        )
    tasks: dict[str, int] = {}
    episodes, provenance = [], []
    frame_offset = 0
    for source in sources:
        task_map = {}
        for task in source["tasks"]:
            text = task["task"]
            tasks.setdefault(text, len(tasks))
            task_map[int(task["task_index"])] = tasks[text]
        original_info = source["info"]
        original_chunks = int(original_info["chunks_size"])
        for episode in source["episodes"]:
            old, new, length = (
                int(episode["episode_index"]),
                len(episodes),
                int(episode["length"]),
            )
            source_parquet = _path(source["root"], original_info["data_path"], old, original_chunks)
            table = pq.read_table(source_parquet)
            if table.num_rows != length:
                raise VirtualDatasetError(f"Wrong Parquet length: {source_parquet}")
            if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)) or not np.all(
                table["episode_index"].to_numpy() == old
            ):
                raise VirtualDatasetError(f"Incorrect frame/episode indices: {source_parquet}")
            try:
                mapped_tasks = [task_map[int(value)] for value in table["task_index"].to_pylist()]
            except KeyError as exc:
                raise VirtualDatasetError(f"Unknown task ID in {source_parquet}") from exc
            table = _remap_column(table, "task_index", mapped_tasks)
            table = _remap_column(table, "episode_index", np.full(length, new, dtype=np.int64))
            table = _remap_column(
                table,
                "index",
                np.arange(frame_offset, frame_offset + length, dtype=np.int64),
            )
            target = _path(destination, DATA_PATTERN, new, chunks)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            for key, feature in info["features"].items():
                if feature["dtype"] != "video":
                    continue
                original_video = _path(
                    source["root"],
                    original_info["video_path"],
                    old,
                    original_chunks,
                    video_key=key,
                )
                if (
                    info.get("surface_normals_lz4", {}).get("feature_key") == key
                    and not original_video.exists()
                ):
                    continue  # LZ4 storage can replace, rather than accompany, the video.
                _link(
                    original_video,
                    _path(destination, VIDEO_PATTERN, new, chunks, video_key=key),
                )
            if info.get("surface_normals_lz4"):
                _link_lz4(source, destination, info, old, new, length)
            for name in ("raw_depth_encoding", "aligned_depth_encoding"):
                if info.get(name):
                    src = _path(
                        source["root"],
                        original_info[name]["path"],
                        old,
                        original_chunks,
                        frame_index=0,
                    )
                    dst = _path(destination, info[name]["path"], new, chunks, frame_index=0)
                    # Only the episode directory is reindexed; frame basenames must agree.
                    for frame in (0, length - 1):
                        a = _path(
                            source["root"],
                            original_info[name]["path"],
                            old,
                            original_chunks,
                            frame_index=frame,
                        )
                        b = _path(
                            destination,
                            info[name]["path"],
                            new,
                            chunks,
                            frame_index=frame,
                        )
                        if a.parent != src.parent or a.name != b.name or not a.is_file():
                            raise VirtualDatasetError(
                                f"Unsupported or incomplete depth sequence: {a}"
                            )
                    available = {entry.name for entry in src.parent.iterdir() if entry.is_file()}
                    expected = {
                        _path(destination, info[name]["path"], new, chunks, frame_index=frame).name
                        for frame in range(length)
                    }
                    if not expected.issubset(available):
                        raise VirtualDatasetError(f"Incomplete depth sequence: {src.parent}")
                    _link(src.parent, dst.parent)
            if info.get("mask_path"):
                for key, config in source["modality"].get("mask", {}).items():
                    original = config.get("original_key", key)
                    _link(
                        _path(
                            source["root"],
                            original_info["mask_path"],
                            old,
                            original_chunks,
                            mask_key=original,
                            video_key=original,
                        ),
                        _path(
                            destination,
                            info["mask_path"],
                            new,
                            chunks,
                            mask_key=original,
                            video_key=original,
                        ),
                    )
            updated = copy.deepcopy(episode)
            updated["episode_index"] = new
            episodes.append(updated)
            origin = copy.deepcopy(source["provenance"].get(old, {}))
            origin.update(
                episode_index=new,
                source=str(source["root"]),
                source_episode_index=old,
                split=split or "train",
                split_episode=f"episode_{new:06d}",
                final_episode_index=new,
                final_index_start=frame_offset,
                final_index_end_exclusive=frame_offset + length,
                frame_count=length,
            )
            provenance.append(origin)
            frame_offset += length
    info.update(
        total_episodes=len(episodes),
        total_frames=frame_offset,
        total_tasks=len(tasks),
        total_chunks=math.ceil(len(episodes) / chunks),
        total_videos=len(episodes)
        * sum(feature["dtype"] == "video" for feature in info["features"].values()),
        splits={split or "train": f"0:{len(episodes)}"},
    )
    for name in ("raw_depth_encoding", "aligned_depth_encoding"):
        if info.get(name):
            info[name]["total_files"] = frame_offset
    _write_json(destination / "meta/info.json", info)
    _write_json(destination / "meta/modality.json", sources[0]["modality"])
    _write_jsonl(destination / "meta/episodes.jsonl", episodes)
    _write_jsonl(
        destination / "meta/tasks.jsonl",
        [{"task_index": index, "task": text} for text, index in tasks.items()],
    )
    _write_jsonl(destination / "meta/virtual_episode_sources.jsonl", provenance)
    _write_json(
        destination / "meta" / MANIFEST,
        {"schema_version": SCHEMA_VERSION, "sources": signatures},
    )
    # Never combine per-source percentile caches. Calculate from the actual union,
    # in the exact same order and float32 precision as the normal training pipeline.
    from gr00t.data.stats import generate_stats

    generate_stats(destination)


def build_virtual_dataset(
    sources: Sequence[str | Path],
    output: str | Path,
    *,
    reuse: bool = False,
    require_splits: bool = False,
    splits: Sequence[str] | None = None,
) -> Path:
    """Atomically publish an indexed view; refuse overwrite or stale reuse.

    Split membership is never recomputed. All sources must have the same split set
    (``validate`` is accepted as an alias for ``validation``). Source order determines
    the same episode order a successive physical append would produce.
    Explicit ``splits`` requires only those splits in every source and never
    discovers or opens unselected split directories. Omission preserves legacy
    discovery, including the all-three-splits ``require_splits`` contract.
    """
    if splits is not None:
        if (
            isinstance(splits, (str, bytes))
            or not isinstance(splits, Sequence)
            or not splits
            or any(name not in ("train", "validation", "test") for name in splits)
            or len(set(splits)) != len(splits)
        ):
            raise VirtualDatasetError("splits must contain unique train/validation/test names")
        splits = tuple(splits)
    roots = [Path(source).expanduser().resolve(strict=True) for source in sources]
    destination = Path(output).expanduser().absolute()
    if not roots or len(set(roots)) != len(roots):
        raise VirtualDatasetError(
            "Provide at least one source; duplicate source roots are not allowed"
        )
    for root in roots:
        if (
            destination.resolve() == root
            or root in destination.resolve().parents
            or destination.resolve() in root.parents
        ):
            raise VirtualDatasetError(
                "The output and source directories must be separate, not nested"
            )
    layouts = [_split_paths(root, splits) for root in roots]
    names = list(layouts[0])
    if require_splits and splits is None and "" in names:
        raise VirtualDatasetError(
            "This runner requires split roots containing train/validation/test"
        )
    if require_splits and splits is None and set(names) != {"train", "validation", "test"}:
        raise VirtualDatasetError("This runner requires all train/validation/test splits")
    if any(set(layout) != set(names) for layout in layouts[1:]):
        raise VirtualDatasetError(
            "All sources must have identical split sets; direct and split roots cannot mix"
        )
    grouped = {
        name: [
            _load_source(layout[name], root, name)
            for root, layout in zip(roots, layouts, strict=True)
        ]
        for name in names
    }
    flat = [source for group in grouped.values() for source in group]
    resolved_splits = [source["root"].resolve() for source in flat]
    if len(set(resolved_splits)) != len(resolved_splits):
        raise VirtualDatasetError(
            "Duplicate source datasets across roots or splits are not allowed"
        )
    seen_hashes: dict[str, str] = {}
    for source in flat:
        for index, record in source["provenance"].items():
            digest = record.get("data_json_sha256")
            if digest is None:
                continue
            location = f"{source['root']} episode {index}"
            if digest in seen_hashes:
                raise VirtualDatasetError(
                    f"Duplicate source episode hash (overlap or split leakage): {seen_hashes[digest]} and {location}"
                )
            seen_hashes[digest] = location
    reference = flat[0]
    for source in flat[1:]:
        if (
            _semantic_info(source["info"]) != _semantic_info(reference["info"])
            or source["modality"] != reference["modality"]
        ):
            raise VirtualDatasetError(
                f"Incompatible schema, encoding, calibration or modality: {source['root']}"
            )
    signatures = {
        name: [_fingerprint(source) for source in group] for name, group in grouped.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "split-root" if names != [""] else "dataset",
        "sources": [str(root) for root in roots],
        "splits": signatures,
    }
    manifest_path = destination / (MANIFEST if names != [""] else f"meta/{MANIFEST}")
    if destination.exists() or destination.is_symlink():
        if reuse and manifest_path.is_file() and _read_json(manifest_path) == manifest:
            for name in names:
                view = destination / name
                if not (view / "meta/stats.json").is_file():
                    raise VirtualDatasetError(f"Incomplete view (missing stats): {view}")
            return destination
        raise VirtualDatasetError(
            f"Output exists; refusing overwrite or stale reuse: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        for name, group in grouped.items():
            _build_direct(group, stage / name, name, signatures[name])
        if names != [""]:
            records = [
                record
                for name in names
                for record in _read_jsonl(stage / name / "meta/virtual_episode_sources.jsonl")
            ]
            _write_json(
                stage / "provenance/merge_manifest.json",
                {"version": 1, "generator": "gr00t.data.virtual_dataset", "episodes": records},
            )
        _write_json(stage / (MANIFEST if names != [""] else f"meta/{MANIFEST}"), manifest)
        # Catch concurrent dataset conversion changes before publishing a view whose
        # copied numeric data no longer matches its source contract/media snapshot.
        if signatures != {
            name: [_fingerprint(source) for source in group] for name, group in grouped.items()
        }:
            raise VirtualDatasetError(
                "Sources changed while building; retry after conversion finishes"
            )
        if destination.exists() or destination.is_symlink():
            raise VirtualDatasetError(f"Output appeared during build: {destination}")
        _publish_noreplace(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse only if all source fingerprints still match",
    )
    parser.add_argument(
        "--require-splits",
        action="store_true",
        help="Require train/validation/test split roots unless --splits selects a subset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        help="Build only these required splits; do not discover or read other split directories",
    )
    args = parser.parse_args()
    try:
        output = build_virtual_dataset(
            args.sources,
            args.output,
            reuse=args.reuse,
            require_splits=args.require_splits,
            splits=args.splits,
        )
    except (VirtualDatasetError, OSError, KeyError) as exc:
        parser.exit(2, f"virtual dataset: {exc}\n")
    print(f"Virtual dataset ready: {output}")


if __name__ == "__main__":
    main()
