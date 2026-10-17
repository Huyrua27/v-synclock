# -*- coding: utf-8 -*-
"""Hierarchical loss: coarse cross-entropy (integer bin) + fine residual + direct regression.

Balances "which frame" (coarse) against "sub-frame phase" (fine) so the model does not
collapse onto the integer part. Task #15 (WS2).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def hierarchical_loss(logits, dt_soft, dt_gt, centers, w_coarse=0.2, w_reg=1.0):
    """Coarse cross-entropy on the integer lag + fine parabolic read-out at the GT bin.

    The coarse term keeps the correlation profile peaked at the right lag. The fine term
    is the *differentiable version of exactly what inference does*: parabolic sub-frame
    interpolation around the peak. Supervising the soft-argmax instead (the previous
    version) fights the CE term — as the profile sharpens the soft-argmax collapses to
    the integer bin, and the leftover regression gradient warps the local curvature,
    which is precisely what the parabolic refinement reads. Task #15/#16 (WS2)."""
    K = (centers.numel() - 1) // 2
    n = logits.shape[-1]
    # Soft coarse target: Δt=+0.4 puts the true peak *between* bins 0 and +1 with
    # near-equal correlation — a hard integer label would order the CE gradient to
    # suppress the neighbor and skew the very curvature the fine read-out needs.
    lo = torch.floor(dt_gt.clamp(-K, K - 1e-6))
    frac = (dt_gt.clamp(-K, K) - lo).clamp(0.0, 1.0)                  # (B,)
    lo_bin = (lo + K).long().clamp(0, n - 2)
    soft = torch.zeros_like(logits)
    soft.scatter_(1, lo_bin[:, None], (1.0 - frac)[:, None])
    soft.scatter_add_(1, (lo_bin + 1)[:, None], frac[:, None])
    ce = -(soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    target_bin = (torch.round(dt_gt).clamp(-K, K) + K).long()         # (B,)
    k = target_bin.clamp(1, n - 2)
    y0 = logits.gather(1, (k - 1)[:, None]).squeeze(1)
    y1 = logits.gather(1, k[:, None]).squeeze(1)
    y2 = logits.gather(1, (k + 1)[:, None]).squeeze(1)
    # A proper peak has negative curvature; the clamp keeps the division stable
    # and penalizes non-peaked profiles through the resulting delta error.
    den = torch.clamp(y0 - 2 * y1 + y2, max=-1e-3)
    delta = (0.5 * (y0 - y2) / den).clamp(-1.0, 1.0)
    dt_fine = centers[k] + delta
    fine = F.smooth_l1_loss(dt_fine, dt_gt)

    total = w_coarse * ce + w_reg * fine
    return total, {"ce": ce.item(), "fine": fine.item()}
