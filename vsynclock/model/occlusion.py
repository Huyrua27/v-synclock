# -*- coding: utf-8 -*-
"""Occlusion-aware helpers. Task #18 (WS2), addresses the occlusion research gap.

Two mechanisms, both driven by per-keypoint visibility v ∈ [0,1] of shape (B,T,J):
  1. Input gating   — zero-out unreliable keypoints before embedding.
  2. Attention bias — down-weight frames whose keypoints are mostly occluded, by adding
                      log(reliability) to the attention logits over those key frames.
When visibility is None (e.g. clean synthetic data) both are no-ops.
"""
from __future__ import annotations
import torch


def gate_keypoints(kp: torch.Tensor, vis: torch.Tensor | None) -> torch.Tensor:
    """kp: (B,T,J,2), vis: (B,T,J) or None -> gated kp."""
    if vis is None:
        return kp
    return kp * vis.unsqueeze(-1)


def reliability_bias(vis: torch.Tensor | None, eps: float = 1e-3) -> torch.Tensor | None:
    """Per-frame additive attention bias log(mean_j vis) : (B,T) or None."""
    if vis is None:
        return None
    rel = vis.mean(dim=-1).clamp_min(eps)          # (B,T)
    return torch.log(rel)
