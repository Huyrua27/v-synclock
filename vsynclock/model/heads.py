# -*- coding: utf-8 -*-
"""Hierarchical head over a learned cross-correlation profile.

Coarse: a distribution over integer lags read from the profile (softmax with a learned
temperature). Fine: analytic parabolic interpolation around the peak recovers the sub-frame
phase — the same estimator classical CC uses, but on transformer-shaped features. Task #14 (WS2).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class HierarchicalHead(nn.Module):
    def __init__(self, max_offset: int = 8):
        super().__init__()
        self.K = max_offset
        self.nbins = 2 * max_offset + 1
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("centers",
                             torch.arange(-max_offset, max_offset + 1).float())

    def forward(self, corr: torch.Tensor):
        """corr: (B, nbins) -> (dt_soft, logits). dt_soft is the differentiable soft-argmax."""
        logits = corr / self.temp.clamp_min(0.05)
        prob = logits.softmax(-1)
        dt_soft = (prob * self.centers).sum(-1)
        return dt_soft, logits

    @torch.no_grad()
    def dt_hard(self, corr: torch.Tensor):
        """Discrete peak + parabolic sub-frame refinement (inference)."""
        k = corr.argmax(-1)
        km1 = (k - 1).clamp(0, self.nbins - 1)
        kp1 = (k + 1).clamp(0, self.nbins - 1)
        y0 = corr.gather(1, km1[:, None]).squeeze(1)
        y1 = corr.gather(1, k[:, None]).squeeze(1)
        y2 = corr.gather(1, kp1[:, None]).squeeze(1)
        den = y0 - 2 * y1 + y2
        delta = torch.where(den.abs() > 1e-6, 0.5 * (y0 - y2) / den,
                            torch.zeros_like(den)).clamp(-0.5, 0.5)
        edge = (k == 0) | (k == self.nbins - 1)                  # no parabola at edges
        delta = torch.where(edge, torch.zeros_like(delta), delta)
        return self.centers[k] + delta
