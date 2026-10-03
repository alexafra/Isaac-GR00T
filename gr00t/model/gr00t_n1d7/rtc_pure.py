"""Inference-only RTC guidance from arXiv:2506.07339v2, equations 2--5.

Unlike the legacy GR00T sampler, this does not insert or clamp old actions in
the noisy latent. Only the action inputs receive gradients; learned weights
and training objectives are unchanged.
"""

from collections.abc import Callable
import math

import torch


RTC_PURE_MAX_GUIDANCE = 5.0


def rtc_pure_weights(
    *,
    horizon: int,
    action_dim: int,
    real_action_dim: int,
    overlap_steps: int,
    frozen_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Paper Eq. 5 as [1, horizon, padded_dim], with no padded-output penalty."""
    if not 0 <= frozen_steps <= overlap_steps <= horizon:
        raise ValueError("RTC pure requires 0 <= frozen_steps <= overlap_steps <= horizon")
    if not 0 < real_action_dim <= action_dim:
        raise ValueError("RTC pure requires 0 < rtc_action_dim <= padded action dimension")
    weights = torch.zeros((1, horizon, action_dim), device=device, dtype=torch.float32)
    weights[:, :frozen_steps, :real_action_dim] = 1.0
    index = torch.arange(frozen_steps, overlap_steps, device=device, dtype=torch.float32)
    c = (overlap_steps - index) / (overlap_steps - frozen_steps + 1)
    weights[:, frozen_steps:overlap_steps, :real_action_dim] = (
        c * torch.expm1(c) / math.expm1(1.0)
    )[None, :, None]
    return weights


def rtc_pure_guidance_coefficient(flow_time: float) -> float:
    """Clipped Eq. 2 coefficient, explicitly handling its limit at tau=0."""
    if not 0.0 <= flow_time < 1.0:
        raise ValueError("RTC pure flow_time must be in [0, 1)")
    if flow_time == 0.0:
        return RTC_PURE_MAX_GUIDANCE
    remaining = 1.0 - flow_time
    r_squared = remaining**2 / (flow_time**2 + remaining**2)
    return min(RTC_PURE_MAX_GUIDANCE, remaining / (flow_time * r_squared))


def rtc_pure_guided_step(
    actions: torch.Tensor,
    *,
    predict_velocity: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    weights: torch.Tensor,
    flow_time: float,
    step_size: float,
) -> torch.Tensor:
    """One fp32 Euler/VJP step; its autograd graph never spans timesteps.

    Caller must enable gradients outside inference_mode. The cotangent is
    detached: differentiating the error itself would not be the paper's VJP.
    ``autograd.grad`` requests only input gradients, never parameter .grad.
    """
    coefficient = rtc_pure_guidance_coefficient(flow_time)
    latent = actions.detach().float().requires_grad_(True)
    velocity = predict_velocity(latent).float()
    if velocity.shape != latent.shape:
        raise ValueError("RTC pure predicted velocity must match the action chunk shape")
    clean_estimate = latent + (1.0 - flow_time) * velocity
    error = ((target - clean_estimate) * weights).detach()
    guidance = torch.autograd.grad(
        clean_estimate,
        latent,
        grad_outputs=error,
        create_graph=False,
        retain_graph=False,
    )[0]
    updated = latent.detach() + step_size * (velocity.detach() + coefficient * guidance)
    if not torch.isfinite(updated).all():
        raise FloatingPointError("RTC pure produced non-finite action values; refusing prediction")
    return updated.detach()
