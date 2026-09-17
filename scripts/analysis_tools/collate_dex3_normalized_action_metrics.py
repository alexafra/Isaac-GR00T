#!/usr/bin/env python3
"""Collate and plot the six isolated Dex3 normalized-metric evaluations.

The command only reads the explicitly allowlisted children of ``--input-root``.
It refuses overlapping input/output trees, snapshots every input file before and
after generation, and publishes the report only after proving that all source
hashes and mtimes are unchanged.  It never follows the original-model paths
recorded inside the isolated provenance files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPORT_PARENT = Path(
    "/home/alex/Development/Models/dex3/dex3_normalized_action_metrics_exec_hor_8"
)
DEFAULT_INPUT_ROOT = DEFAULT_REPORT_PARENT / "models"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPORT_PARENT / "reports"

OUTPUT_COLUMNS = (
    "model",
    "checkpoint",
    "mse_total",
    "mse_hands",
    "mse_arms",
    "mae_total",
    "mae_hands",
    "mae_arms",
)
SCOPE_COLUMNS = (
    ("total", "overall", "Total"),
    ("hands", "hands", "Hands"),
    ("arms", "arms", "Arms"),
)


@dataclass(frozen=True)
class Dex3ModelSpec:
    key: str
    label: str
    directory: str


@dataclass(frozen=True)
class MetricFamilySpec:
    key: str
    title: str
    formula_version: str
    source_filename: str
    mae_column: str
    mse_column: str


@dataclass(frozen=True)
class SourceFileEvidence:
    relative_path: str
    size: int
    mtime_ns: int
    sha256: str


# This is deliberately an allowlist rather than filesystem discovery.  The
# tuple order is the canonical row and legend order in every output.
DEX3_MODEL_SPECS: tuple[Dex3ModelSpec, ...] = (
    Dex3ModelSpec("simple_tasks_rgb", "Simple tasks: RGB", "simple_tasks_rgb"),
    Dex3ModelSpec(
        "simple_tasks_rgb_surface_normals",
        "Simple tasks: RGB + surface normals",
        "simple_tasks_rgb_surface_normals",
    ),
    Dex3ModelSpec("simple_tasks_rgbd", "Simple tasks: RGB-D", "simple_tasks_rgbd"),
    Dex3ModelSpec("full_stack_rgb", "Full stack: RGB", "full_stack_rgb"),
    Dex3ModelSpec(
        "full_stack_rgb_surface_normals",
        "Full stack: RGB + surface normals",
        "full_stack_rgb_surface_normals",
    ),
    Dex3ModelSpec("full_stack_rgbd", "Full stack: RGB-D", "full_stack_rgbd"),
)

PROCESSOR_CONTRACT = MetricFamilySpec(
    key="processor_contract",
    title=(
        "Dex3 processor-contract validation error "
        "(training/live normalization and clipping)"
    ),
    formula_version="processor_training_coordinates_v1",
    source_filename="normalized_metrics_by_scope.csv",
    mae_column="mae",
    mse_column="mse",
)
Q01_Q99_DIAGNOSTIC = MetricFamilySpec(
    key="q01_q99_diagnostic",
    title="Dex3 q01-q99 diagnostic validation error (unclipped)",
    formula_version="unclipped_q01_q99_minus1_plus1_v1",
    source_filename="unclipped_q01_q99_metrics_by_scope.csv",
    mae_column="diagnostic_unclipped_q01_q99_mae",
    mse_column="diagnostic_unclipped_q01_q99_mse",
)
METRIC_FAMILIES = (PROCESSOR_CONTRACT, Q01_Q99_DIAGNOSTIC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_child(root: Path, relative: str) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Allowlisted path escapes input root {root}: {relative}") from exc
    return candidate


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.expanduser().resolve()
    second = second.expanduser().resolve()
    return first == second or first in second.parents or second in first.parents


def validate_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Resolve and validate path-disjoint input and output trees."""

    resolved_input = input_root.expanduser().resolve()
    resolved_output = output_root.expanduser().resolve()
    if not resolved_input.is_dir():
        raise FileNotFoundError(f"Isolated Dex3 input root does not exist: {resolved_input}")
    if _paths_overlap(resolved_input, resolved_output):
        raise ValueError(
            "Input and output roots must be path-disjoint; refusing any report location "
            f"inside or above the isolated source tree: input={resolved_input}, "
            f"output={resolved_output}"
        )
    return resolved_input, resolved_output


def snapshot_source_tree(input_root: Path) -> tuple[SourceFileEvidence, ...]:
    """Hash every regular input file and record its nanosecond mtime."""

    input_root = input_root.expanduser().resolve()
    evidence: list[SourceFileEvidence] = []
    for spec in DEX3_MODEL_SPECS:
        model_dir = _resolved_child(input_root, spec.directory)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing isolated model directory: {model_dir}")
        for path in sorted(model_dir.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"Symlinks are not allowed in isolated input trees: {path}")
            if not path.is_file():
                continue
            stat = path.stat()
            evidence.append(
                SourceFileEvidence(
                    relative_path=str(path.relative_to(input_root)),
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=_sha256(path),
                )
            )
    if not evidence:
        raise RuntimeError(f"No files found beneath isolated input root: {input_root}")
    return tuple(evidence)


def source_tree_digest(evidence: Sequence[SourceFileEvidence]) -> str:
    serialized = json.dumps(
        [asdict(item) for item in evidence],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_provenance(model_dir: Path) -> Mapping[str, object]:
    path = model_dir / "normalized_metrics_provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing isolated normalization provenance: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Normalization provenance must be a JSON object: {path}")
    return value


def _validate_formula_version(
    model_key: str,
    provenance: Mapping[str, object],
    family: MetricFamilySpec,
) -> None:
    field = "formula_version" if family.key == "processor_contract" else "diagnostic_formula_version"
    actual = provenance.get(field)
    if actual != family.formula_version:
        raise RuntimeError(
            f"{model_key} has unexpected {family.key} formula version: "
            f"expected={family.formula_version!r}, actual={actual!r}"
        )


def _load_model_family(
    input_root: Path,
    model: Dex3ModelSpec,
    family: MetricFamilySpec,
) -> pd.DataFrame:
    model_dir = _resolved_child(input_root, model.directory)
    provenance = _load_provenance(model_dir)
    _validate_formula_version(model.key, provenance, family)
    path = _resolved_child(model_dir, family.source_filename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing isolated {family.key} scope table: {path}")
    required = {
        "split",
        "checkpoint_step",
        "scope",
        family.mae_column,
        family.mse_column,
    }
    table = pd.read_csv(path)
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"{model.key} {family.key} table is missing: {sorted(missing)}")
    selected = table[
        (table["split"].astype(str) == "validation")
        & table["scope"].astype(str).isin(scope for _, scope, _ in SCOPE_COLUMNS)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"{model.key} {family.key} has no validation scope rows")
    selected["checkpoint_step"] = pd.to_numeric(
        selected["checkpoint_step"], errors="raise"
    ).astype(int)
    if (selected["checkpoint_step"] < 0).any():
        raise RuntimeError(f"{model.key} {family.key} contains a negative checkpoint")
    for source_column in (family.mae_column, family.mse_column):
        selected[source_column] = pd.to_numeric(selected[source_column], errors="raise")
        values = selected[source_column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError(
                f"{model.key} {family.key} {source_column} must be finite and nonnegative"
            )
    if selected.duplicated(["checkpoint_step", "scope"], keep=False).any():
        raise RuntimeError(
            f"{model.key} {family.key} has duplicate checkpoint/scope validation rows"
        )
    expected_scopes = {scope for _, scope, _ in SCOPE_COLUMNS}
    scope_sets = selected.groupby("checkpoint_step")["scope"].agg(lambda values: set(values))
    incomplete = {
        int(step): sorted(expected_scopes - scopes)
        for step, scopes in scope_sets.items()
        if scopes != expected_scopes
    }
    if incomplete:
        raise RuntimeError(
            f"{model.key} {family.key} lacks overall/arms/hands rows: {incomplete}"
        )

    rows: list[dict[str, object]] = []
    for checkpoint, checkpoint_frame in selected.groupby("checkpoint_step", sort=True):
        scoped = checkpoint_frame.set_index("scope")
        row: dict[str, object] = {"model": model.key, "checkpoint": int(checkpoint)}
        for output_scope, source_scope, _ in SCOPE_COLUMNS:
            row[f"mse_{output_scope}"] = float(scoped.at[source_scope, family.mse_column])
            row[f"mae_{output_scope}"] = float(scoped.at[source_scope, family.mae_column])
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)


def load_metric_families(input_root: Path) -> dict[str, pd.DataFrame]:
    """Load both validation metric families in the fixed six-model order."""

    input_root = input_root.expanduser().resolve()
    result: dict[str, pd.DataFrame] = {}
    for family in METRIC_FAMILIES:
        frames = [_load_model_family(input_root, model, family) for model in DEX3_MODEL_SPECS]
        family_frame = pd.concat(frames, ignore_index=True)
        family_frame["model"] = pd.Categorical(
            family_frame["model"],
            categories=[model.key for model in DEX3_MODEL_SPECS],
            ordered=True,
        )
        family_frame = family_frame.sort_values(["model", "checkpoint"], kind="stable")
        family_frame["model"] = family_frame["model"].astype("object")
        result[family.key] = family_frame.reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)

    canonical_pairs = set(
        map(tuple, result[PROCESSOR_CONTRACT.key][["model", "checkpoint"]].to_numpy())
    )
    diagnostic_pairs = set(
        map(tuple, result[Q01_Q99_DIAGNOSTIC.key][["model", "checkpoint"]].to_numpy())
    )
    if canonical_pairs != diagnostic_pairs:
        raise RuntimeError(
            "Processor-contract and q01-q99 diagnostic tables contain different "
            "model/checkpoint pairs"
        )
    return result


def select_best_checkpoints(frame: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    """Select one whole checkpoint per model, resolving exact ties to the lower step."""

    if selection_metric not in {"mse", "mae"}:
        raise ValueError("selection_metric must be 'mse' or 'mae'")
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Metric table must use the exact output schema: {list(OUTPUT_COLUMNS)}"
        )
    rows = []
    for model in DEX3_MODEL_SPECS:
        model_frame = frame[frame["model"].astype(str) == model.key].copy()
        if model_frame.empty:
            raise RuntimeError(f"Metric table is missing fixed model: {model.key}")
        chosen = model_frame.sort_values(
            [f"{selection_metric}_total", "checkpoint"], kind="stable"
        ).iloc[0]
        # Copy the full source row so every scope/metric is guaranteed to come
        # from the same selected checkpoint.
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)


def _shared_limits(values: np.ndarray, *, logarithmic: bool) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Plot limits require at least one finite value")
    minimum = float(values.min())
    maximum = float(values.max())
    if logarithmic:
        if minimum <= 0:
            raise ValueError("Logarithmic plot limits require positive values")
        low = float(np.log10(minimum))
        high = float(np.log10(maximum))
        span = max(high - low, 0.2)
        padding = max(span * 0.06, 0.04)
        return 10 ** (low - padding), 10 ** (high + padding)
    span = maximum - minimum
    padding = max(span * 0.06, maximum * 0.02, 1e-12)
    return max(0.0, minimum - padding), maximum + padding


def build_checkpoint_figure(
    frame: pd.DataFrame,
    *,
    family: MetricFamilySpec,
    model_keys: Sequence[str],
    include_checkpoint_zero: bool,
    subject: str,
) -> tuple[object, np.ndarray]:
    """Build a 2x3 MAE/MSE plot with row-wise shared y-axis limits."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    known_keys = {model.key for model in DEX3_MODEL_SPECS}
    model_keys = tuple(model_keys)
    if not model_keys or len(model_keys) != len(set(model_keys)):
        raise ValueError("model_keys must be a nonempty sequence of unique keys")
    unknown = set(model_keys) - known_keys
    if unknown:
        raise ValueError(f"Unknown Dex3 model keys: {sorted(unknown)}")
    plotted = frame[frame["model"].astype(str).isin(model_keys)].copy()
    if not include_checkpoint_zero:
        plotted = plotted[plotted["checkpoint"] != 0].copy()
    actual_keys = set(plotted["model"].astype(str))
    if actual_keys != set(model_keys):
        raise RuntimeError(
            f"Plot data lacks selected models: {sorted(set(model_keys) - actual_keys)}"
        )
    labels = {model.key: model.label for model in DEX3_MODEL_SPECS}
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey="row")
    row_values: dict[str, list[np.ndarray]] = {"mae": [], "mse": []}
    for column, (output_scope, _, scope_label) in enumerate(SCOPE_COLUMNS):
        for model_key in model_keys:
            model_frame = plotted[plotted["model"].astype(str) == model_key].sort_values(
                "checkpoint"
            )
            steps = model_frame["checkpoint"].to_numpy(dtype=float) / 1000.0
            for row, metric in enumerate(("mae", "mse")):
                values = model_frame[f"{metric}_{output_scope}"].to_numpy(dtype=float)
                row_values[metric].append(values)
                axes[row, column].plot(
                    steps,
                    values,
                    marker="o",
                    linestyle="-",
                    label=labels[model_key],
                )
        axes[0, column].set_title(f"{scope_label} MAE")
        axes[1, column].set_title(f"{scope_label} MSE")
        for row in range(2):
            axes[row, column].set_xlabel("Checkpoint step (thousands)")
            axes[row, column].grid(alpha=0.25)

    for row, metric in enumerate(("mae", "mse")):
        combined = np.concatenate(row_values[metric])
        logarithmic = bool((combined > 0).all())
        limits = _shared_limits(combined, logarithmic=logarithmic)
        for column in range(3):
            axes[row, column].set_yscale("log" if logarithmic else "linear")
            axes[row, column].set_ylim(limits)
    axes[0, 0].set_ylabel("Normalized MAE")
    axes[1, 0].set_ylabel("Normalized MSE")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(3, len(model_keys)),
        fontsize=8.5,
    )
    zero_text = "including checkpoint 0" if include_checkpoint_zero else "excluding checkpoint 0"
    figure.suptitle(f"{subject}: {family.title} ({zero_text})")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    return figure, axes


def _write_figure(figure: object, path: Path) -> None:
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=160)
        plt.close(figure)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(figure)
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Plot is not a valid PNG: {path}")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_family_report(
    frame: pd.DataFrame,
    family: MetricFamilySpec,
    output_dir: Path,
) -> dict[str, str]:
    family_dir = output_dir / family.key
    all_checkpoints_path = family_dir / "all_checkpoints.csv"
    best_mse_path = family_dir / "best_by_mse.csv"
    best_mae_path = family_dir / "best_by_mae.csv"
    _atomic_csv(frame, all_checkpoints_path)
    _atomic_csv(select_best_checkpoints(frame, selection_metric="mse"), best_mse_path)
    _atomic_csv(select_best_checkpoints(frame, selection_metric="mae"), best_mae_path)

    model_keys = tuple(model.key for model in DEX3_MODEL_SPECS)
    plot_paths = {
        "comparison_with_checkpoint_0": family_dir / "comparison_with_checkpoint_0.png",
        "comparison_without_checkpoint_0": family_dir / "comparison_without_checkpoint_0.png",
    }
    for include_zero, key in (
        (True, "comparison_with_checkpoint_0"),
        (False, "comparison_without_checkpoint_0"),
    ):
        figure, _ = build_checkpoint_figure(
            frame,
            family=family,
            model_keys=model_keys,
            include_checkpoint_zero=include_zero,
            subject="All six Dex3 models",
        )
        _write_figure(figure, plot_paths[key])

    per_model_dir = family_dir / "per_model"
    for model in DEX3_MODEL_SPECS:
        figure, _ = build_checkpoint_figure(
            frame,
            family=family,
            model_keys=(model.key,),
            include_checkpoint_zero=True,
            subject=model.label,
        )
        _write_figure(figure, per_model_dir / f"{model.key}_checkpoints.png")
    return {
        "all_checkpoints": str(all_checkpoints_path.relative_to(output_dir)),
        "best_by_mse": str(best_mse_path.relative_to(output_dir)),
        "best_by_mae": str(best_mae_path.relative_to(output_dir)),
        **{key: str(path.relative_to(output_dir)) for key, path in plot_paths.items()},
        "per_model_plot_directory": str(per_model_dir.relative_to(output_dir)),
    }


def _output_evidence(output_dir: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            result.append(
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return result


def generate_report(input_root: Path, output_root: Path) -> Path:
    """Generate a new report tree without mutating any isolated source file."""

    input_root, output_root = validate_roots(input_root, output_root)
    if output_root.exists():
        raise FileExistsError(
            f"Report output already exists; choose a new --output-root: {output_root}"
        )
    before = snapshot_source_tree(input_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        frames = load_metric_families(input_root)
        family_outputs = {
            family.key: _write_family_report(frames[family.key], family, staging)
            for family in METRIC_FAMILIES
        }
        after = snapshot_source_tree(input_root)
        if before != after:
            raise RuntimeError(
                "Isolated source tree hash/mtime evidence changed during report generation; "
                "the staged report will not be published"
            )
        provenance = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "scripts.analysis_tools.collate_dex3_normalized_action_metrics",
            "validation_only": True,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "model_order": [asdict(model) for model in DEX3_MODEL_SPECS],
            "scope_mapping": {
                output_scope: source_scope for output_scope, source_scope, _ in SCOPE_COLUMNS
            },
            "output_columns": list(OUTPUT_COLUMNS),
            "selection": {
                "best_by_mse": "minimum validation mse_total",
                "best_by_mae": "minimum validation mae_total",
                "tie_break": "lowest checkpoint step",
                "all_reported_metrics_from_selected_checkpoint": True,
            },
            "plotting": {
                "checkpoint_markers": "circles for every observed checkpoint",
                "checkpoint_0_variants": ["included where available", "excluded"],
                "y_axis": (
                    "logarithmic when every plotted value in a metric row is positive; "
                    "otherwise linear"
                ),
                "shared_limits": "total/hands/arms share bounds within each MAE or MSE row",
            },
            "metric_families": {
                family.key: {
                    "title": family.title,
                    "formula_version": family.formula_version,
                    "source_filename": family.source_filename,
                    "source_columns": {"mae": family.mae_column, "mse": family.mse_column},
                    "outputs": family_outputs[family.key],
                }
                for family in METRIC_FAMILIES
            },
            "source_tree": {
                "file_count": len(before),
                "digest_before": source_tree_digest(before),
                "digest_after": source_tree_digest(after),
                "hashes_and_mtimes_unchanged": True,
                "files": [asdict(item) for item in before],
            },
            "outputs": _output_evidence(staging),
        }
        _atomic_json(provenance, staging / "provenance.json")
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_root


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = generate_report(args.input_root, args.output_root)
    print(f"Dex3 normalized metric report written to {output_root}")


if __name__ == "__main__":
    main()
