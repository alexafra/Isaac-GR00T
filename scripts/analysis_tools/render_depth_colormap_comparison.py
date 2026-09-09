#!/usr/bin/env python3
"""Render fixed-metric depth with several standard colour maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLOUR_MAPS = (
    ("current_gray", "Current grayscale", "gray"),
    ("turbo", "Turbo", "turbo"),
    ("viridis", "Viridis", "viridis"),
    ("cividis", "Cividis", "cividis"),
    ("plasma", "Plasma", "plasma"),
    ("inferno", "Inferno", "inferno"),
    ("magma", "Magma", "magma"),
    ("jet", "Jet (legacy)", "jet"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colourize_depth(
    depth_u16: np.ndarray,
    *,
    scale_m_per_unit: float,
    near_m: float,
    far_m: float,
    colour_map: str,
) -> np.ndarray:
    """Map depth to RGB while reserving exact black for invalid sensor pixels."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected HxW uint16 depth, got {depth.shape=} and {depth.dtype=}")
    if not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0:
        raise ValueError("scale_m_per_unit must be finite and positive")
    if not np.isfinite(near_m) or not np.isfinite(far_m) or far_m <= near_m:
        raise ValueError("near_m and far_m must be finite, with far_m > near_m")

    valid = depth != 0
    depth_m = depth.astype(np.float32) * scale_m_per_unit
    normalized = np.clip((depth_m - near_m) / (far_m - near_m), 0.0, 1.0)

    if colour_map == "gray":
        rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = 1 + np.rint(254 * normalized[valid]).astype(np.uint8)
        rgb[:] = gray[..., None]
        return rgb

    if colour_map not in colormaps:
        raise ValueError(f"Unknown Matplotlib colour map: {colour_map}")
    rgb = colormaps[colour_map](normalized, bytes=True)[..., :3]
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-m-per-unit", type=float, required=True)
    parser.add_argument("--near-m", type=float, default=0.25)
    parser.add_argument("--far-m", type=float, default=1.0)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    depth = np.asarray(Image.open(args.depth))
    if depth.dtype != np.uint16:
        raise SystemExit(f"Expected uint16 depth PNG, got {depth.dtype}")
    if rgb.shape[:2] != depth.shape:
        raise SystemExit(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape}")

    rendered: list[tuple[str, str, np.ndarray]] = []
    for stem, title, colour_map in COLOUR_MAPS:
        image = colourize_depth(
            depth,
            scale_m_per_unit=args.scale_m_per_unit,
            near_m=args.near_m,
            far_m=args.far_m,
            colour_map=colour_map,
        )
        output = args.output_dir / f"{stem}.png"
        Image.fromarray(image).save(output, compress_level=6)
        rendered.append((stem, title, image))

    panels = [("rgb_reference", "RGB reference", rgb), *rendered]
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for axis, (_, title, image) in zip(axes.flat, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.axis("off")
    invalid_fraction = float(np.count_nonzero(depth == 0) / depth.size)
    figure.suptitle(
        f"Stack {args.episode}, frame {args.frame:06d} — identical fixed depth scale",
        fontsize=18,
        fontweight="bold",
    )
    figure.supxlabel(
        f"Near = {args.near_m:.2f} m  →  Far = {args.far_m:.2f} m; "
        f"invalid sensor depth stays black ({invalid_fraction:.2%} of pixels)",
        fontsize=13,
    )
    comparison_path = args.output_dir / "depth_colormap_comparison.png"
    figure.savefig(comparison_path, dpi=160, facecolor="white")
    plt.close(figure)

    valid_depth_m = depth[depth != 0].astype(np.float64) * args.scale_m_per_unit
    manifest = {
        "source": {
            "episode": args.episode,
            "frame_index": args.frame,
            "rgb_path": str(args.rgb.resolve()),
            "rgb_sha256": _sha256(args.rgb),
            "depth_path": str(args.depth.resolve()),
            "depth_sha256": _sha256(args.depth),
        },
        "encoding": {
            "scale_m_per_unit": args.scale_m_per_unit,
            "near_m": args.near_m,
            "far_m": args.far_m,
            "invalid_source_value": 0,
            "invalid_output_rgb": [0, 0, 0],
            "mapping_direction": "near_to_far",
            "colour_maps": [stem for stem, _, _ in COLOUR_MAPS],
            "matplotlib_version": matplotlib.__version__,
        },
        "frame": {
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "invalid_pixels": int(np.count_nonzero(depth == 0)),
            "invalid_fraction": invalid_fraction,
            "valid_depth_min_m": float(valid_depth_m.min()),
            "valid_depth_max_m": float(valid_depth_m.max()),
        },
        "outputs": {path.name: _sha256(path) for path in sorted(args.output_dir.glob("*.png"))},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(comparison_path)


if __name__ == "__main__":
    main()
