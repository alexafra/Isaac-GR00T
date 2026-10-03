"""Training-only, mutually exclusive RGB/geometry dropout decisions.

Probabilities describe fractions of ALL examples, not just state-kept examples.
The default independent policy retains the original processor and action-head
state masks. Only the optional state_first policy samples both together so visual
dropout can depend on them.
"""

from collections.abc import Callable
import math
from typing import NamedTuple


class ModalityDropoutDecision(NamedTuple):
    drop_state_input: bool
    keep_state_features: bool
    vision_keep: tuple[float, float]


def validate_modality_dropout(
    rgb_prob: float,
    geometry_prob: float,
    state_prob: float,
    state_policy: str,
    exclude_state: bool = False,
) -> None:
    for name, probability in (
        ("RGB", rgb_prob),
        ("geometry", geometry_prob),
        ("state", state_prob),
    ):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{name} dropout probability must be finite and in [0, 1]")
    if rgb_prob + geometry_prob > 1:
        raise ValueError("RGB and geometry dropout probabilities must sum to at most 1")
    if state_policy not in ("state_first", "independent"):
        raise ValueError("Modality dropout state policy must be state_first or independent")
    if state_policy == "state_first" and rgb_prob + geometry_prob > 0:
        available = 0.0 if exclude_state else (1 - state_prob) ** 2
        if rgb_prob + geometry_prob > available:
            raise ValueError(
                "Global modality dropout probabilities exceed the fraction of examples "
                f"with both state masks retained ({available:.6f})"
            )


def sample_modality_dropout(
    rgb_prob: float,
    geometry_prob: float,
    state_prob: float,
    state_policy: str,
    *,
    exclude_state: bool,
    draw: Callable[[], float],
) -> ModalityDropoutDecision:
    """Sample two legacy state masks and at most one visual-modality mask."""
    drop_state_input = draw() < state_prob or exclude_state
    keep_state_features = draw() >= state_prob
    eligible = not drop_state_input and keep_state_features
    if state_policy == "state_first" and not eligible:
        return ModalityDropoutDecision(drop_state_input, keep_state_features, (1.0, 1.0))
    divisor = (1 - state_prob) ** 2 if state_policy == "state_first" else 1.0
    choice = draw()
    if choice < rgb_prob / divisor:
        vision_keep = (0.0, 1.0)
    elif choice < (rgb_prob + geometry_prob) / divisor:
        vision_keep = (1.0, 0.0)
    else:
        vision_keep = (1.0, 1.0)
    return ModalityDropoutDecision(drop_state_input, keep_state_features, vision_keep)
