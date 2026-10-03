#!/usr/bin/env python3
"""Calculate unit-independent action errors from saved open-loop predictions.

Canonical metrics reproduce the action coordinates seen during training:

* relative arms are first re-anchored to the measured state at the beginning
  of each executed action chunk, then scaled with horizon-specific
  ``relative_action.min/max``;
* absolute hands are scaled with ``action.q01/q99``;
* ground truth and prediction are normalized and clipped independently to
  ``[-1, +1]`` before their error is calculated.

The earlier all-q01/q99, unclipped error scaling remains available as an
explicit diagnostic because it is useful for exposing out-of-range tails::

    diagnostic_error = 2 * (prediction - target) / (q99 - q01)

The standalone CLI rebuilds canonical aggregate CSVs with processor-coordinate
headline columns, explicit ``raw_*`` counterparts, and explicit
``diagnostic_unclipped_q01_q99_*`` columns.  It never rewrites or removes the
frame-level ``*_frame_predictions.csv.gz`` evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


CANONICAL_FORMULA_VERSION = "processor_training_coordinates_v1"
DIAGNOSTIC_FORMULA_VERSION = "unclipped_q01_q99_minus1_plus1_v1"
# Backwards-compatible import name. It now identifies the canonical/default
# metric contract rather than the explicitly named diagnostic contract.
FORMULA_VERSION = CANONICAL_FORMULA_VERSION
DEFAULT_RANGE_FLOOR = 1e-8
PREDICTION_FILENAMES = {
    "validation": "validation_frame_predictions.csv.gz",
    "test": "test_frame_predictions.csv.gz",
    "train_probe": "train_probe_frame_predictions.csv.gz",
}
JOINT_PATTERN = re.compile(r"^(?P<group>[A-Za-z0-9_.-]+)\[(?P<index>[0-9]+)\]$")
NORMALIZED_METRIC_COLUMNS = (
    "mae",
    "mse",
    "rmse",
    "median_absolute_error",
    "p95_absolute_error",
    "bias",
    "max_absolute_error",
)
RAW_METRIC_COLUMNS = tuple(f"raw_{name}" for name in NORMALIZED_METRIC_COLUMNS)
DIAGNOSTIC_PREFIX = "diagnostic_unclipped_q01_q99"
DIAGNOSTIC_METRIC_COLUMNS = tuple(
    f"{DIAGNOSTIC_PREFIX}_{name}" for name in NORMALIZED_METRIC_COLUMNS
)


def _numeric_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def _resolve_embodiment_statistics(
    statistics: dict[str, Any], embodiment_tag: str | None
) -> tuple[str, dict[str, Any]]:
    if not isinstance(statistics, dict) or not statistics:
        raise ValueError("Statistics file must contain a non-empty JSON object")

    if embodiment_tag is None:
        if len(statistics) != 1:
            raise ValueError(
                "Statistics contain multiple embodiments; pass --embodiment-tag explicitly"
            )
        resolved_tag = str(next(iter(statistics)))
    else:
        candidates = [embodiment_tag, embodiment_tag.lower(), embodiment_tag.upper()]
        matching = [candidate for candidate in candidates if candidate in statistics]
        if not matching:
            casefold_matches = [
                str(key) for key in statistics if str(key).casefold() == embodiment_tag.casefold()
            ]
            matching = casefold_matches
        if len(set(matching)) != 1:
            raise KeyError(
                f"Embodiment {embodiment_tag!r} is not uniquely present; "
                f"available={sorted(str(key) for key in statistics)}"
            )
        resolved_tag = str(matching[0])

    selected = statistics[resolved_tag]
    if not isinstance(selected, dict):
        raise ValueError(f"Statistics for {resolved_tag!r} must be an object")
    return resolved_tag, selected


class ActionErrorNormalizer:
    """Validated processor-coordinate contract plus the former q01/q99 diagnostic."""

    def __init__(
        self,
        statistics: dict[str, Any],
        *,
        execution_horizon: int,
        embodiment_tag: str | None = None,
        range_floor: float = DEFAULT_RANGE_FLOOR,
        statistics_path: Path | None = None,
    ) -> None:
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if not np.isfinite(range_floor) or range_floor <= 0:
            raise ValueError("range_floor must be finite and positive")

        resolved_tag, selected = _resolve_embodiment_statistics(statistics, embodiment_tag)
        self.statistics = selected
        self.embodiment_tag = resolved_tag
        self.execution_horizon = int(execution_horizon)
        self.range_floor = float(range_floor)
        self.statistics_path = statistics_path

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        execution_horizon: int,
        embodiment_tag: str | None = None,
        range_floor: float = DEFAULT_RANGE_FLOOR,
    ) -> "ActionErrorNormalizer":
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing training statistics: {path}")
        with path.open(encoding="utf-8") as handle:
            statistics = json.load(handle)
        return cls(
            statistics,
            execution_horizon=execution_horizon,
            embodiment_tag=embodiment_tag,
            range_floor=range_floor,
            statistics_path=path,
        )

    def _group_statistics(
        self,
        group: str,
        *,
        diagnostic_unclipped_q01_q99: bool,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return the exact lower/upper bounds for one metric contract."""

        if group.endswith("_arm"):
            statistics_group = "relative_action"
            bound_names = ("q01", "q99") if diagnostic_unclipped_q01_q99 else ("min", "max")
            expected_dimensions = 2
        elif group.endswith("_hand"):
            statistics_group = "action"
            bound_names = ("q01", "q99")
            expected_dimensions = 1
        else:
            raise ValueError(
                f"Unsupported action group {group!r}; expected a *_arm or *_hand group"
            )

        lower_name, upper_name = bound_names
        source = f"{statistics_group}.{lower_name}/{upper_name}"
        try:
            group_stats = self.statistics[statistics_group][group]
            lower = _numeric_array(
                group_stats[lower_name],
                label=f"{statistics_group}.{group}.{lower_name}",
            )
            upper = _numeric_array(
                group_stats[upper_name],
                label=f"{statistics_group}.{group}.{upper_name}",
            )
        except KeyError as exc:
            raise KeyError(
                f"Training statistics do not define "
                f"{statistics_group}.{group}.{lower_name}/{upper_name}"
            ) from exc
        if lower.shape != upper.shape:
            raise ValueError(
                f"{statistics_group}.{group} {lower_name} shape {lower.shape} differs from "
                f"{upper_name} {upper.shape}"
            )
        if lower.ndim != expected_dimensions:
            expected = (
                "[action_horizon, dimensions]" if expected_dimensions == 2 else "[dimensions]"
            )
            raise ValueError(f"{source}.{group} must be {expected}, got {lower.shape}")
        if expected_dimensions == 2 and lower.shape[0] < self.execution_horizon:
            raise ValueError(
                f"{source}.{group} has {lower.shape[0]} horizon positions, fewer than "
                f"execution_horizon={self.execution_horizon}"
            )
        return lower, upper, source

    def _prepare_predictions(
        self, raw_predictions: pd.DataFrame, *, copy: bool
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Validate raw evidence and add unit-independent row identity columns."""

        required = {
            "checkpoint_step",
            "trajectory",
            "frame",
            "joint",
            "ground_truth",
            "prediction",
            "error",
            "absolute_error",
        }
        missing = required - set(raw_predictions.columns)
        if missing:
            raise ValueError("Raw prediction CSV is missing columns: " + ", ".join(sorted(missing)))
        if raw_predictions.empty:
            raise ValueError("Raw prediction CSV contains no rows")

        result = raw_predictions.copy() if copy else raw_predictions
        numeric_columns = [
            "checkpoint_step",
            "trajectory",
            "frame",
            "ground_truth",
            "prediction",
            "error",
            "absolute_error",
        ]
        for column in numeric_columns:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Raw prediction column {column!r} contains NaN or infinity")

        prediction = result["prediction"].to_numpy(dtype=np.float64)
        ground_truth = result["ground_truth"].to_numpy(dtype=np.float64)
        raw_error = result["error"].to_numpy(dtype=np.float64)
        raw_absolute_error = result["absolute_error"].to_numpy(dtype=np.float64)
        # CSV serialization preserves the separately computed float32 error, but
        # recomputing it from serialized prediction/target can differ by roughly
        # one float32 ULP.
        if not np.allclose(raw_error, prediction - ground_truth, rtol=1e-5, atol=5e-7):
            raise ValueError("Raw prediction error values disagree with prediction - ground_truth")
        if not np.allclose(raw_absolute_error, np.abs(raw_error), rtol=1e-6, atol=1e-8):
            raise ValueError("Raw prediction absolute_error values disagree with abs(error)")

        frame_values = result["frame"].to_numpy(dtype=np.float64)
        if np.any(frame_values < 0) or not np.all(frame_values == np.floor(frame_values)):
            raise ValueError("Raw prediction frame indices must be non-negative integers")
        frame_values = frame_values.astype(np.int64)

        unique_labels = result["joint"].astype(str).unique()
        parsed_labels = {}
        invalid_labels = []
        for label in unique_labels:
            match = JOINT_PATTERN.fullmatch(label)
            if match is None:
                invalid_labels.append(label)
            else:
                parsed_labels[label] = (match.group("group"), int(match.group("index")))
        if invalid_labels:
            raise ValueError(f"Invalid action joint labels: {sorted(invalid_labels)}")
        group_by_label = {label: values[0] for label, values in parsed_labels.items()}
        index_by_label = {label: values[1] for label, values in parsed_labels.items()}
        result["joint_group"] = result["joint"].astype(str).map(group_by_label).astype("category")
        result["joint_index"] = (
            result["joint"].astype(str).map(index_by_label).to_numpy(dtype=np.int16)
        )
        result["side"] = (
            result["joint_group"].astype(str).str.extract(r"^(left|right)_", expand=False)
        )
        if result["side"].isna().any():
            invalid = result.loc[result["side"].isna(), "joint_group"].unique()
            raise ValueError(f"Action groups do not declare left/right side: {sorted(invalid)}")
        result["body_part"] = np.where(
            result["joint_group"].astype(str).str.endswith("_arm"),
            "arms",
            np.where(result["joint_group"].astype(str).str.endswith("_hand"), "hands", ""),
        )
        result["side"] = result["side"].astype("category")
        result["body_part"] = result["body_part"].astype("category")
        if (result["body_part"] == "").any():
            invalid = result.loc[result["body_part"] == "", "joint_group"].unique()
            raise ValueError(f"Unsupported action groups: {sorted(invalid)}")

        horizon_positions = frame_values % self.execution_horizon
        result["horizon_position"] = horizon_positions
        return result, horizon_positions

    def _selected_bounds(
        self,
        result: pd.DataFrame,
        horizon_positions: np.ndarray,
        *,
        diagnostic_unclipped_q01_q99: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select each row's training-only bound without constructing frame copies."""

        row_count = len(result)
        lower_rows = np.empty(row_count, dtype=np.float64)
        upper_rows = np.empty(row_count, dtype=np.float64)
        sources = np.empty(row_count, dtype=object)

        for group in result["joint_group"].drop_duplicates():
            mask = result["joint_group"].to_numpy() == group
            indices = result.loc[mask, "joint_index"].to_numpy(dtype=np.int64)
            lower, upper, source = self._group_statistics(
                str(group),
                diagnostic_unclipped_q01_q99=diagnostic_unclipped_q01_q99,
            )
            width = lower.shape[-1]
            if np.any(indices >= width):
                raise IndexError(
                    f"{group} joint index exceeds statistics width {width}: "
                    f"{sorted(set(indices[indices >= width].tolist()))}"
                )
            if str(group).endswith("_arm"):
                positions = horizon_positions[mask]
                lower_rows[mask] = lower[positions, indices]
                upper_rows[mask] = upper[positions, indices]
            else:
                lower_rows[mask] = lower[indices]
                upper_rows[mask] = upper[indices]
            sources[mask] = source
        return lower_rows, upper_rows, sources

    def _effective_ranges(
        self,
        lower_rows: np.ndarray,
        upper_rows: np.ndarray,
        *,
        contract_name: str,
        processor_zero_range: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_ranges = upper_rows - lower_rows
        if not np.all(np.isfinite(raw_ranges)):
            raise ValueError(f"Selected {contract_name} ranges contain NaN or infinity")
        if np.any(raw_ranges < 0):
            bad = np.flatnonzero(raw_ranges < 0)[:5]
            raise ValueError(
                f"Selected {contract_name} upper bounds are below lower bounds at row(s) "
                f"{bad.tolist()}"
            )
        floored = raw_ranges < self.range_floor
        if processor_zero_range:
            # This is the mask used by normalize_values_minmax. Constant
            # channels are represented as zero, not divided by a tiny range.
            floored |= np.isclose(upper_rows, lower_rows)
        return raw_ranges, np.maximum(raw_ranges, self.range_floor), floored

    def _relative_arm_anchors(
        self,
        result: pd.DataFrame,
        horizon_positions: np.ndarray,
        arm_mask: np.ndarray,
    ) -> np.ndarray:
        """Find the observed arm state used to anchor every executed action chunk."""

        if "measured_state" not in result:
            raise ValueError(
                "Processor-coordinate arm metrics require the saved measured_state column"
            )
        arm_states = pd.to_numeric(result.loc[arm_mask, "measured_state"], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(arm_states)):
            raise ValueError(
                "Processor-coordinate arm metrics require a finite measured_state for every arm row"
            )

        key_columns = [
            column
            for column in ("split", "checkpoint_step", "trajectory", "joint")
            if column in result
        ]
        if not {"checkpoint_step", "trajectory", "joint"}.issubset(key_columns):
            raise ValueError(
                "Cannot reconstruct arm anchors without checkpoint/trajectory/joint keys"
            )

        frame_values = result["frame"].to_numpy(dtype=np.int64)
        chunk_starts = frame_values - horizon_positions
        anchor_mask = arm_mask & (horizon_positions == 0)
        anchor_columns = [*key_columns, "frame", "measured_state"]
        anchors = result.loc[anchor_mask, anchor_columns].copy()
        anchors = anchors.rename(columns={"frame": "chunk_start"})
        anchor_keys = [*key_columns, "chunk_start"]
        if anchors.duplicated(anchor_keys).any():
            duplicates = anchors.loc[anchors.duplicated(anchor_keys, keep=False), anchor_keys]
            raise ValueError(
                "Raw prediction rows contain duplicate arm chunk anchors: "
                f"{duplicates.head(3).to_dict(orient='records')}"
            )

        anchor_index = pd.MultiIndex.from_frame(anchors[anchor_keys])
        anchor_lookup = pd.Series(
            pd.to_numeric(anchors["measured_state"], errors="raise").to_numpy(dtype=np.float64),
            index=anchor_index,
        )
        arm_rows = result.loc[arm_mask]
        requested_frame = arm_rows[key_columns].copy()
        requested_frame["chunk_start"] = chunk_starts[arm_mask]
        requested_index = pd.MultiIndex.from_frame(requested_frame[anchor_keys])
        selected = anchor_lookup.reindex(requested_index)
        if selected.isna().any():
            missing = requested_frame.loc[selected.isna().to_numpy(), anchor_keys]
            raise ValueError(
                "Cannot reconstruct processor-coordinate arm actions because chunk-start "
                f"measured_state rows are missing: {missing.head(3).to_dict(orient='records')}"
            )
        return selected.to_numpy(dtype=np.float64)

    def _add_unclipped_q01_q99_diagnostic(
        self, result: pd.DataFrame, horizon_positions: np.ndarray
    ) -> None:
        lower, upper, sources = self._selected_bounds(
            result,
            horizon_positions,
            diagnostic_unclipped_q01_q99=True,
        )
        raw_ranges, effective_ranges, floored = self._effective_ranges(
            lower,
            upper,
            contract_name="diagnostic q01/q99",
        )
        raw_error = result["error"].to_numpy(dtype=np.float64)
        diagnostic_error = 2.0 * raw_error / effective_ranges
        if not np.all(np.isfinite(diagnostic_error)):
            raise ValueError("Unclipped q01/q99 diagnostic errors contain NaN or infinity")

        result[f"{DIAGNOSTIC_PREFIX}_source"] = pd.Categorical(sources)
        result[f"{DIAGNOSTIC_PREFIX}_lower_bound"] = lower
        result[f"{DIAGNOSTIC_PREFIX}_upper_bound"] = upper
        result[f"{DIAGNOSTIC_PREFIX}_range"] = raw_ranges
        result[f"{DIAGNOSTIC_PREFIX}_effective_range"] = effective_ranges
        result[f"{DIAGNOSTIC_PREFIX}_range_floored"] = floored
        result[f"{DIAGNOSTIC_PREFIX}_error"] = diagnostic_error
        result[f"{DIAGNOSTIC_PREFIX}_absolute_error"] = np.abs(diagnostic_error)

    def normalize_unclipped_q01_q99_predictions(
        self, raw_predictions: pd.DataFrame, *, copy: bool = True
    ) -> pd.DataFrame:
        """Add the former all-q01/q99, no-clipping diagnostic contract explicitly."""

        result, horizon_positions = self._prepare_predictions(raw_predictions, copy=copy)
        self._add_unclipped_q01_q99_diagnostic(result, horizon_positions)
        return result

    def normalize_predictions(
        self, raw_predictions: pd.DataFrame, *, copy: bool = True
    ) -> pd.DataFrame:
        """Add canonical processor-coordinate and legacy diagnostic error evidence."""

        result, horizon_positions = self._prepare_predictions(raw_predictions, copy=copy)
        self._add_unclipped_q01_q99_diagnostic(result, horizon_positions)

        lower, upper, sources = self._selected_bounds(
            result,
            horizon_positions,
            diagnostic_unclipped_q01_q99=False,
        )
        raw_ranges, effective_ranges, floored = self._effective_ranges(
            lower,
            upper,
            contract_name="processor-coordinate",
            processor_zero_range=True,
        )

        ground_truth_coordinates = result["ground_truth"].to_numpy(dtype=np.float64).copy()
        prediction_coordinates = result["prediction"].to_numpy(dtype=np.float64).copy()
        arm_mask = result["body_part"].to_numpy() == "arms"
        anchors = np.full(len(result), np.nan, dtype=np.float64)
        if np.any(arm_mask):
            arm_anchors = self._relative_arm_anchors(result, horizon_positions, arm_mask)
            anchors[arm_mask] = arm_anchors
            ground_truth_coordinates[arm_mask] -= arm_anchors
            prediction_coordinates[arm_mask] -= arm_anchors

        ground_truth_unclipped = np.zeros(len(result), dtype=np.float64)
        prediction_unclipped = np.zeros(len(result), dtype=np.float64)
        variable_range = ~floored
        ground_truth_unclipped[variable_range] = (
            2.0
            * (ground_truth_coordinates[variable_range] - lower[variable_range])
            / effective_ranges[variable_range]
            - 1.0
        )
        prediction_unclipped[variable_range] = (
            2.0
            * (prediction_coordinates[variable_range] - lower[variable_range])
            / effective_ranges[variable_range]
            - 1.0
        )
        normalized_ground_truth = np.clip(ground_truth_unclipped, -1.0, 1.0)
        normalized_prediction = np.clip(prediction_unclipped, -1.0, 1.0)
        normalized_error = normalized_prediction - normalized_ground_truth
        if not np.all(np.isfinite(normalized_error)):
            raise ValueError("Processor-coordinate action errors contain NaN or infinity")

        result["normalization_source"] = pd.Categorical(sources)
        result["normalization_coordinate"] = pd.Categorical(
            np.where(arm_mask, "relative_to_chunk_start_measured_state", "absolute_action")
        )
        result["normalization_anchor"] = anchors
        result["normalization_lower_bound"] = lower
        result["normalization_upper_bound"] = upper
        result["normalization_range"] = raw_ranges
        result["normalization_effective_range"] = effective_ranges
        result["normalization_range_floored"] = floored
        result["normalization_ground_truth_unclipped"] = ground_truth_unclipped
        result["normalization_prediction_unclipped"] = prediction_unclipped
        result["normalization_ground_truth_clipped"] = np.abs(ground_truth_unclipped) > 1.0
        result["normalization_prediction_clipped"] = np.abs(prediction_unclipped) > 1.0
        result["normalized_ground_truth"] = normalized_ground_truth
        result["normalized_prediction"] = normalized_prediction
        result["normalized_error"] = normalized_error
        result["normalized_absolute_error"] = np.abs(normalized_error)
        return result


def _metric_values(error: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64)
    if error.size == 0 or not np.all(np.isfinite(error)):
        raise ValueError("Metric errors must be non-empty and finite")
    absolute = np.abs(error)
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(absolute)),
    }


def error_metrics(
    normalized_error: np.ndarray,
    raw_error: np.ndarray,
    diagnostic_unclipped_q01_q99_error: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize canonical, raw, and optional former diagnostic coordinates."""

    normalized_absolute = np.abs(normalized_error)
    raw_absolute = np.abs(raw_error)
    normalized_mse = float(np.mean(normalized_error**2))
    raw_mse = float(np.mean(raw_error**2))
    metrics = {
        "mae": float(np.mean(normalized_absolute)),
        "mse": normalized_mse,
        "rmse": float(np.sqrt(normalized_mse)),
        "median_absolute_error": float(np.median(normalized_absolute)),
        "p95_absolute_error": float(np.percentile(normalized_absolute, 95)),
        "bias": float(np.mean(normalized_error)),
        "max_absolute_error": float(np.max(normalized_absolute)),
        "raw_mae": float(np.mean(raw_absolute)),
        "raw_mse": raw_mse,
        "raw_rmse": float(np.sqrt(raw_mse)),
        "raw_median_absolute_error": float(np.median(raw_absolute)),
        "raw_p95_absolute_error": float(np.percentile(raw_absolute, 95)),
        "raw_bias": float(np.mean(raw_error)),
        "raw_max_absolute_error": float(np.max(raw_absolute)),
    }
    if diagnostic_unclipped_q01_q99_error is not None:
        diagnostic = _metric_values(diagnostic_unclipped_q01_q99_error)
        metrics.update({f"{DIAGNOSTIC_PREFIX}_{key}": value for key, value in diagnostic.items()})
    return metrics


def _validate_equal_dimension_weighting(frame: pd.DataFrame) -> None:
    counts = frame.groupby("joint", sort=False, observed=True).size()
    if counts.empty:
        raise ValueError("Metric scope contains no joint rows")
    if counts.nunique() != 1:
        raise ValueError(
            "Per-dimension weighting requires every joint to have the same sample count; "
            f"counts range from {int(counts.min())} to {int(counts.max())}"
        )


def _summarize_group(frame: pd.DataFrame) -> dict[str, Any]:
    _validate_equal_dimension_weighting(frame)
    row = {
        "episodes": int(frame["trajectory"].nunique()),
        "frames": int(len(frame[["trajectory", "frame"]].drop_duplicates())),
        "dimensions": int(frame["joint"].nunique()),
        "samples": int(len(frame)),
        "floored_samples": int(frame["normalization_range_floored"].sum()),
        f"{DIAGNOSTIC_PREFIX}_floored_samples": int(
            frame[f"{DIAGNOSTIC_PREFIX}_range_floored"].sum()
        ),
    }
    row.update(
        error_metrics(
            frame["normalized_error"].to_numpy(dtype=np.float64),
            frame["error"].to_numpy(dtype=np.float64),
            frame[f"{DIAGNOSTIC_PREFIX}_error"].to_numpy(dtype=np.float64),
        )
    )
    return row


def metric_scopes(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield one view at a time so full-frame scope copies do not accumulate."""

    yield "overall", frame
    for name in ("arms", "hands"):
        scoped = frame[frame["body_part"] == name]
        if not scoped.empty:
            yield name, scoped
    for name in ("left", "right"):
        scoped = frame[frame["side"] == name]
        if not scoped.empty:
            yield name, scoped
    for group in frame["joint_group"].drop_duplicates():
        scoped = frame[frame["joint_group"] == group]
        if not scoped.empty:
            yield str(group), scoped


def summarize_by_scope(normalized_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, checkpoint_step), checkpoint in normalized_predictions.groupby(
        ["split", "checkpoint_step"], sort=True
    ):
        for scope, scoped in metric_scopes(checkpoint):
            rows.append(
                {
                    "split": str(split),
                    "checkpoint_step": int(checkpoint_step),
                    "scope": scope,
                    **_summarize_group(scoped),
                }
            )
    return pd.DataFrame(rows)


def summarize_per_episode(normalized_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, checkpoint_step, trajectory), episode in normalized_predictions.groupby(
        ["split", "checkpoint_step", "trajectory"], sort=True
    ):
        for scope, scoped in metric_scopes(episode):
            rows.append(
                {
                    "split": str(split),
                    "checkpoint_step": int(checkpoint_step),
                    "trajectory": int(trajectory),
                    "scope": scope,
                    **_summarize_group(scoped),
                }
            )
    return pd.DataFrame(rows)


def summarize_per_joint(normalized_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, checkpoint_step, joint), joint_frame in normalized_predictions.groupby(
        ["split", "checkpoint_step", "joint"], sort=True, observed=True
    ):
        sources = joint_frame["normalization_source"].unique()
        if len(sources) != 1:
            raise ValueError(f"Joint {joint} has multiple normalization sources: {sources}")
        row = {
            "split": str(split),
            "checkpoint_step": int(checkpoint_step),
            "joint": str(joint),
            "joint_group": str(joint_frame["joint_group"].iloc[0]),
            "joint_index": int(joint_frame["joint_index"].iloc[0]),
            "side": str(joint_frame["side"].iloc[0]),
            "body_part": str(joint_frame["body_part"].iloc[0]),
            "normalization_source": str(sources[0]),
            "normalization_range_min": float(joint_frame["normalization_range"].min()),
            "normalization_range_max": float(joint_frame["normalization_range"].max()),
            f"{DIAGNOSTIC_PREFIX}_range_min": float(
                joint_frame[f"{DIAGNOSTIC_PREFIX}_range"].min()
            ),
            f"{DIAGNOSTIC_PREFIX}_range_max": float(
                joint_frame[f"{DIAGNOSTIC_PREFIX}_range"].max()
            ),
            **_summarize_group(joint_frame),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def saved_prediction_paths(evaluation_dir: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for checkpoint_dir in sorted(
        evaluation_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    ):
        for split, filename in PREDICTION_FILENAMES.items():
            path = checkpoint_dir / filename
            if not path.is_file():
                continue
            paths.append((split, path))
    if not paths:
        raise FileNotFoundError(
            f"No saved frame predictions were found under {evaluation_dir}/checkpoint-*"
        )
    return paths


def load_saved_predictions(evaluation_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load all prediction files; intended only for small tests and interactive inspection."""

    frames = []
    paths = []
    for split, path in saved_prediction_paths(evaluation_dir):
        frame = pd.read_csv(path, compression="gzip")
        frame["split"] = split
        frames.append(frame)
        paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def calculate_metric_tables(
    *,
    evaluation_dir: Path,
    normalizer: ActionErrorNormalizer,
    model_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Calculate one run's tables while holding only one checkpoint/split in memory."""

    evaluation_dir = evaluation_dir.expanduser().resolve()
    model_dir = (
        model_dir.expanduser().resolve()
        if model_dir is not None
        else evaluation_dir.parent.resolve()
    )
    scope_frames = []
    episode_frames = []
    joint_frames = []
    input_paths = []
    for split, path in saved_prediction_paths(evaluation_dir):
        raw_predictions = pd.read_csv(
            path,
            compression="gzip",
            usecols=[
                "checkpoint_step",
                "trajectory",
                "frame",
                "joint",
                "ground_truth",
                "prediction",
                "error",
                "absolute_error",
                "measured_state",
            ],
            dtype={
                "checkpoint_step": np.int32,
                "trajectory": np.int32,
                "frame": np.int32,
                "joint": "category",
                "ground_truth": np.float64,
                "prediction": np.float64,
                "error": np.float64,
                "absolute_error": np.float64,
                "measured_state": np.float64,
            },
        )
        raw_predictions["split"] = split
        normalized = normalizer.normalize_predictions(raw_predictions, copy=False)
        scope_frames.append(summarize_by_scope(normalized))
        episode_frames.append(summarize_per_episode(normalized))
        joint_frames.append(summarize_per_joint(normalized))
        input_paths.append(path)

    metadata = {"model_name": model_dir.name, "model_dir": str(model_dir)}
    tables = {
        "scopes": pd.concat(scope_frames, ignore_index=True),
        "episodes": pd.concat(episode_frames, ignore_index=True),
        "joints": pd.concat(joint_frames, ignore_index=True),
    }
    for table in tables.values():
        for column, value in reversed(tuple(metadata.items())):
            table.insert(0, column, value)
    return tables, input_paths


def canonical_metric_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Select canonical overall and per-joint views with normalized headline columns."""

    checkpoint = tables["scopes"].loc[tables["scopes"]["scope"] == "overall"].copy()
    episode = tables["episodes"].loc[tables["episodes"]["scope"] == "overall"].copy()
    # Preserve legacy leading positions: shell reports read checkpoint MAE/MSE
    # from fields 5/6. New raw/evidence fields are deliberately appended.
    checkpoint_columns = [
        "split",
        "checkpoint_step",
        "episodes",
        "frames",
        "mae",
        "mse",
        "median_absolute_error",
        "p95_absolute_error",
        "rmse",
        "bias",
        "max_absolute_error",
        "raw_mae",
        "raw_mse",
        "raw_median_absolute_error",
        "raw_p95_absolute_error",
        "raw_rmse",
        "raw_bias",
        "raw_max_absolute_error",
        *DIAGNOSTIC_METRIC_COLUMNS,
        "dimensions",
        "samples",
        "floored_samples",
        f"{DIAGNOSTIC_PREFIX}_floored_samples",
    ]
    episode_columns = [
        "split",
        "checkpoint_step",
        "trajectory",
        "frames",
        "mae",
        "mse",
        "rmse",
        "bias",
        "max_absolute_error",
        "median_absolute_error",
        "p95_absolute_error",
        "raw_mae",
        "raw_mse",
        "raw_rmse",
        "raw_bias",
        "raw_max_absolute_error",
        "raw_median_absolute_error",
        "raw_p95_absolute_error",
        *DIAGNOSTIC_METRIC_COLUMNS,
        "dimensions",
        "samples",
        "floored_samples",
        f"{DIAGNOSTIC_PREFIX}_floored_samples",
    ]
    joint_columns = [
        "split",
        "checkpoint_step",
        "joint",
        "mae",
        "mse",
        "rmse",
        "bias",
        "max_absolute_error",
        "median_absolute_error",
        "p95_absolute_error",
        "raw_mae",
        "raw_mse",
        "raw_rmse",
        "raw_bias",
        "raw_max_absolute_error",
        "raw_median_absolute_error",
        "raw_p95_absolute_error",
        *DIAGNOSTIC_METRIC_COLUMNS,
        "joint_group",
        "joint_index",
        "side",
        "body_part",
        "normalization_source",
        "normalization_range_min",
        "normalization_range_max",
        f"{DIAGNOSTIC_PREFIX}_range_min",
        f"{DIAGNOSTIC_PREFIX}_range_max",
        "samples",
        "floored_samples",
        f"{DIAGNOSTIC_PREFIX}_floored_samples",
    ]
    checkpoint = checkpoint.reindex(columns=checkpoint_columns).sort_values(
        ["split", "checkpoint_step"]
    )
    episode = episode.reindex(columns=episode_columns).sort_values(
        ["split", "checkpoint_step", "trajectory"]
    )
    joints = (
        tables["joints"]
        .reindex(columns=joint_columns)
        .sort_values(["split", "checkpoint_step", "joint"])
    )
    return {"checkpoints": checkpoint, "episodes": episode, "joints": joints}


def diagnostic_metric_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return explicit projections of the preserved unclipped q01/q99 diagnostic."""

    common_evidence = [
        "episodes",
        "frames",
        "dimensions",
        "samples",
        f"{DIAGNOSTIC_PREFIX}_floored_samples",
        *DIAGNOSTIC_METRIC_COLUMNS,
    ]
    scope_columns = [
        "model_name",
        "model_dir",
        "split",
        "checkpoint_step",
        "scope",
        *common_evidence,
    ]
    episode_columns = [
        "model_name",
        "model_dir",
        "split",
        "checkpoint_step",
        "trajectory",
        "scope",
        *common_evidence,
    ]
    joint_columns = [
        "model_name",
        "model_dir",
        "split",
        "checkpoint_step",
        "joint",
        "joint_group",
        "joint_index",
        "side",
        "body_part",
        f"{DIAGNOSTIC_PREFIX}_range_min",
        f"{DIAGNOSTIC_PREFIX}_range_max",
        *common_evidence,
    ]
    return {
        "scopes": tables["scopes"].reindex(columns=scope_columns),
        "episodes": tables["episodes"].reindex(columns=episode_columns),
        "joints": tables["joints"].reindex(columns=joint_columns),
    }


def calculate_run_scope_metrics(
    *,
    evaluation_dir: Path,
    normalizer: ActionErrorNormalizer,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the small, collator-ready per-run scope table."""

    tables, _ = calculate_metric_tables(
        evaluation_dir=evaluation_dir,
        normalizer=normalizer,
        model_dir=model_dir,
    )
    return tables["scopes"]


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_metric_tables(
    *,
    evaluation_dir: Path,
    normalizer: ActionErrorNormalizer,
    model_dir: Path | None = None,
    update_canonical: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Read saved predictions from ``evaluation_dir`` and write derived tables elsewhere.

    ``output_dir=None`` preserves the historical in-place behaviour used by the
    training launchers.  Supplying ``output_dir`` enables isolated mode: the
    destination must be path-disjoint from the source evaluation tree, and
    every CSV/JSON produced by this function is written only below that
    destination.  In particular, source aggregate CSVs, plots, and prediction
    archives are never opened for writing in isolated mode.
    """

    evaluation_dir = evaluation_dir.expanduser().resolve()
    isolated_output = output_dir is not None
    if output_dir is None:
        resolved_output_dir = evaluation_dir
    else:
        resolved_output_dir = output_dir.expanduser().resolve()
        # A child would still mutate the source tree; a parent would make the
        # purported output root contain the source.  Require true siblings (or
        # otherwise disjoint roots) so an isolated invocation cannot be
        # mistaken for an in-place one.
        if (
            resolved_output_dir == evaluation_dir
            or resolved_output_dir.is_relative_to(evaluation_dir)
            or evaluation_dir.is_relative_to(resolved_output_dir)
        ):
            raise ValueError(
                "Explicit output_dir must be path-disjoint from evaluation_dir; "
                f"source={evaluation_dir}, output={resolved_output_dir}"
            )
    tables, input_paths = calculate_metric_tables(
        evaluation_dir=evaluation_dir,
        normalizer=normalizer,
        model_dir=model_dir,
    )
    canonical = canonical_metric_tables(tables)
    diagnostic = diagnostic_metric_tables(tables)

    outputs = {
        "scopes": resolved_output_dir / "normalized_metrics_by_scope.csv",
        "episodes": resolved_output_dir / "normalized_metrics_per_episode.csv",
        "joints": resolved_output_dir / "normalized_metrics_per_joint.csv",
        "diagnostic_scopes": resolved_output_dir / "unclipped_q01_q99_metrics_by_scope.csv",
        "diagnostic_episodes": resolved_output_dir / "unclipped_q01_q99_metrics_per_episode.csv",
        "diagnostic_joints": resolved_output_dir / "unclipped_q01_q99_metrics_per_joint.csv",
        "provenance": resolved_output_dir / "normalized_metrics_provenance.json",
    }
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(tables["scopes"], outputs["scopes"])
    _write_csv_atomic(tables["episodes"], outputs["episodes"])
    _write_csv_atomic(tables["joints"], outputs["joints"])
    _write_csv_atomic(diagnostic["scopes"], outputs["diagnostic_scopes"])
    _write_csv_atomic(diagnostic["episodes"], outputs["diagnostic_episodes"])
    _write_csv_atomic(diagnostic["joints"], outputs["diagnostic_joints"])
    if update_canonical:
        outputs.update(
            {
                "metrics_by_checkpoint": resolved_output_dir / "metrics_by_checkpoint.csv",
                "checkpoint_metric_summary": resolved_output_dir / "checkpoint_metric_summary.csv",
                "metrics_per_episode": resolved_output_dir / "metrics_per_episode.csv",
                "metrics_per_joint": resolved_output_dir / "metrics_per_joint.csv",
            }
        )
        _write_csv_atomic(canonical["checkpoints"], outputs["metrics_by_checkpoint"])
        _write_csv_atomic(canonical["checkpoints"], outputs["checkpoint_metric_summary"])
        _write_csv_atomic(canonical["episodes"], outputs["metrics_per_episode"])
        _write_csv_atomic(canonical["joints"], outputs["metrics_per_joint"])

    statistics_digest = None
    if normalizer.statistics_path is not None:
        statistics_digest = hashlib.sha256(normalizer.statistics_path.read_bytes()).hexdigest()
    provenance = {
        "formula_version": CANONICAL_FORMULA_VERSION,
        "formula": (
            "normalized_error = clip(normalize(prediction_coordinate), -1, 1) - "
            "clip(normalize(ground_truth_coordinate), -1, 1)"
        ),
        "clip_normalized_operands": True,
        "clip_normalized_error": False,
        "normalized_operand_range": [-1.0, 1.0],
        "normalized_error_range": [-2.0, 2.0],
        "range_floor": normalizer.range_floor,
        "processor_zero_range_behavior": (
            "bounds considered equal by numpy.isclose (or below range_floor) normalize to zero"
        ),
        "arm_statistics": "relative_action.min/max, selected per frame % execution_horizon",
        "arm_anchor": (
            "measured_state at frame - frame % execution_horizon, shared by prediction and "
            "ground truth before independent normalization and clipping"
        ),
        "hand_statistics": "action.q01/q99 (absolute hand command)",
        "diagnostic_formula_version": DIAGNOSTIC_FORMULA_VERSION,
        "diagnostic_formula": (
            "diagnostic_unclipped_q01_q99_error = 2 * "
            "(prediction - ground_truth) / effective_q01_q99_range"
        ),
        "diagnostic_clip_normalized_operands": False,
        "diagnostic_clip_normalized_error": False,
        "diagnostic_arm_statistics": (
            "relative_action.q01/q99, selected per frame % execution_horizon; common anchor "
            "cancels because operands are not clipped"
        ),
        "diagnostic_hand_statistics": "action.q01/q99 (absolute hand command)",
        "execution_horizon": normalizer.execution_horizon,
        "embodiment_tag": normalizer.embodiment_tag,
        "statistics_path": (
            str(normalizer.statistics_path) if normalizer.statistics_path is not None else None
        ),
        "statistics_sha256": statistics_digest,
        "source_evaluation_dir": str(evaluation_dir),
        "output_dir": str(resolved_output_dir),
        "isolated_output": isolated_output,
        "input_prediction_files": [str(path.resolve()) for path in input_paths],
        "input_prediction_files_rewritten": False,
        "source_aggregate_csvs_rewritten": not isolated_output and update_canonical,
        "source_pngs_rewritten": False,
        "canonical_summaries_updated": update_canonical,
        "headline_metrics": list(NORMALIZED_METRIC_COLUMNS),
        "retained_physical_unit_metrics": list(RAW_METRIC_COLUMNS),
        "retained_unclipped_q01_q99_diagnostic_metrics": list(DIAGNOSTIC_METRIC_COLUMNS),
        "metric_families": {
            "processor_contract": {
                "formula_version": CANONICAL_FORMULA_VERSION,
                "columns": list(NORMALIZED_METRIC_COLUMNS),
            },
            "q01_q99_diagnostic": {
                "formula_version": DIAGNOSTIC_FORMULA_VERSION,
                "columns": list(DIAGNOSTIC_METRIC_COLUMNS),
            },
            "raw_joint_units": {"columns": list(RAW_METRIC_COLUMNS)},
        },
    }
    _write_json_atomic(provenance, outputs["provenance"])
    return outputs


def default_statistics_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "experiment_cfg" / "dataset_statistics.json",
        run_dir / "processor" / "statistics.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "Could not find training statistics; checked "
            + ", ".join(str(path) for path in candidates)
        )
    return existing[0]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Defaults to RUN_DIR/evaluation_exec_hor_EXECUTION_HORIZON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write every generated CSV/JSON below this path while reading predictions from "
            "--evaluation-dir. The explicit destination must be path-disjoint from the source. "
            "Omit only for backwards-compatible in-place launcher behaviour."
        ),
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        help="Training-only dataset_statistics.json (auto-detected from --run-dir by default).",
    )
    parser.add_argument("--embodiment-tag")
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--range-floor", type=float, default=DEFAULT_RANGE_FLOOR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    evaluation_dir = (
        args.evaluation_dir.expanduser().resolve()
        if args.evaluation_dir is not None
        else run_dir / f"evaluation_exec_hor_{args.execution_horizon}"
    )
    statistics_path = (
        args.statistics_path.expanduser().resolve()
        if args.statistics_path is not None
        else default_statistics_path(run_dir)
    )
    normalizer = ActionErrorNormalizer.from_file(
        statistics_path,
        execution_horizon=args.execution_horizon,
        embodiment_tag=args.embodiment_tag,
        range_floor=args.range_floor,
    )
    outputs = write_metric_tables(
        evaluation_dir=evaluation_dir,
        normalizer=normalizer,
        model_dir=run_dir,
        output_dir=args.output_dir,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
