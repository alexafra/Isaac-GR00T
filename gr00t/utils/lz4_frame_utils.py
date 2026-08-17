"""Read independently LZ4-framed uint8 image chunks."""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


class _Lz4FrameDecoder:
    VERSION = 100

    def __init__(self) -> None:
        library_name = ctypes.util.find_library("lz4") or "liblz4.so.1"
        self.lib = ctypes.CDLL(library_name)
        self.lib.LZ4F_createDecompressionContext.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
        self.lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
        self.lib.LZ4F_freeDecompressionContext.restype = ctypes.c_size_t
        self.lib.LZ4F_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.LZ4F_decompress.restype = ctypes.c_size_t
        self.lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
        self.lib.LZ4F_isError.restype = ctypes.c_uint
        self.lib.LZ4F_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.LZ4F_getErrorName.restype = ctypes.c_char_p

    def _check(self, code: int) -> int:
        if self.lib.LZ4F_isError(code):
            name = self.lib.LZ4F_getErrorName(code).decode("utf-8", "replace")
            raise RuntimeError(f"liblz4 error: {name}")
        return code

    def decompress(self, compressed: bytes, expected_bytes: int) -> bytes:
        context = ctypes.c_void_p()
        self._check(
            self.lib.LZ4F_createDecompressionContext(
                ctypes.byref(context), self.VERSION
            )
        )
        destination = ctypes.create_string_buffer(expected_bytes)
        source_pointer = ctypes.c_char_p(compressed)
        source_base = ctypes.cast(source_pointer, ctypes.c_void_p).value
        if source_base is None:
            raise RuntimeError("Could not obtain compressed-buffer address")

        source_offset = 0
        destination_offset = 0
        try:
            while True:
                source_size = ctypes.c_size_t(len(compressed) - source_offset)
                destination_size = ctypes.c_size_t(
                    expected_bytes - destination_offset
                )
                result = self._check(
                    self.lib.LZ4F_decompress(
                        context,
                        ctypes.c_void_p(
                            ctypes.addressof(destination) + destination_offset
                        ),
                        ctypes.byref(destination_size),
                        ctypes.c_void_p(source_base + source_offset),
                        ctypes.byref(source_size),
                        None,
                    )
                )
                source_offset += source_size.value
                destination_offset += destination_size.value
                if result == 0:
                    break
                if source_size.value == 0 and destination_size.value == 0:
                    raise RuntimeError("liblz4 decompression made no progress")
        finally:
            self._check(self.lib.LZ4F_freeDecompressionContext(context))

        if source_offset != len(compressed):
            raise RuntimeError("LZ4 frame contains unread trailing bytes")
        if destination_offset != expected_bytes:
            raise RuntimeError(
                f"LZ4 decoded {destination_offset} bytes; expected {expected_bytes}"
            )
        return destination.raw


@lru_cache(maxsize=1)
def _decoder() -> _Lz4FrameDecoder:
    return _Lz4FrameDecoder()


@lru_cache(maxsize=256)
def _read_index(index_path: str) -> dict:
    with open(index_path, encoding="utf-8") as stream:
        return json.load(stream)


def get_plain_lz4_frames_by_indices(
    storage_root: str | Path,
    episode_index: int,
    indices: list[int] | np.ndarray,
) -> np.ndarray:
    """Return requested FHWC frames from independently compressed episode chunks."""

    storage_root = Path(storage_root)
    episode_dir = storage_root / f"episode_{episode_index:06d}"
    index = _read_index(str(episode_dir / "index.json"))
    if int(index.get("episode_index", -1)) != episode_index:
        raise ValueError(
            f"LZ4 index episode mismatch in {episode_dir}: "
            f"{index.get('episode_index')} != {episode_index}"
        )
    if index.get("dtype") != "uint8":
        raise ValueError(
            f"Expected uint8 LZ4 chunks in {episode_dir}; found {index.get('dtype')}"
        )
    if index.get("layout") != "FHWC" or index.get("transform") != "none":
        raise ValueError(
            f"Expected plain FHWC LZ4 chunks in {episode_dir}; "
            f"found layout={index.get('layout')} transform={index.get('transform')}"
        )

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError(f"Frame indices must be one-dimensional: {requested.shape}")
    if len(requested) == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    if np.any(requested < 0):
        raise IndexError("Frame indices must be non-negative")

    first_chunk = index["chunks"][0]
    height = int(first_chunk["height"])
    width = int(first_chunk["width"])
    channels = int(first_chunk["channels"])
    output = np.empty((len(requested), height, width, channels), dtype=np.uint8)
    filled = np.zeros(len(requested), dtype=bool)
    expected_start = 0

    for chunk in index["chunks"]:
        start = int(chunk["start_frame"])
        frame_count = int(chunk["frame_count"])
        if start != expected_start:
            raise ValueError(
                f"Non-contiguous LZ4 chunks in {episode_dir}: "
                f"expected start {expected_start}, found {start}"
            )
        expected_start += frame_count
        expected_bytes = frame_count * height * width * channels
        if int(chunk["uncompressed_bytes"]) != expected_bytes:
            raise ValueError(
                f"Invalid uncompressed size in {episode_dir}/{chunk['filename']}: "
                f"{chunk['uncompressed_bytes']} != {expected_bytes}"
            )
        positions = np.flatnonzero(
            (requested >= start) & (requested < start + frame_count)
        )
        if len(positions) == 0:
            continue

        compressed = (episode_dir / chunk["filename"]).read_bytes()
        decoded = _decoder().decompress(
            compressed,
            expected_bytes,
        )
        frames = np.frombuffer(decoded, dtype=np.uint8).reshape(
            frame_count,
            height,
            width,
            channels,
        )
        output[positions] = frames[requested[positions] - start]
        filled[positions] = True

    if not np.all(filled):
        missing = requested[~filled].tolist()
        raise IndexError(
            f"Frames {missing} are outside the stored chunks for episode {episode_index}"
        )
    return output
