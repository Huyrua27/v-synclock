# -*- coding: utf-8 -*-
"""Continuous Temporal Encoding (Rotary Position Embedding on the time axis).

Positions may be *fractional* (real timestamps), so a model trained at one frame rate
generalizes to another — the phase of the video is encoded as a differentiable rotation.
Task #12 (WS2).
"""
from __future__ import annotations
import torch


def rope_cache(pos: torch.Tensor, dim: int, base: float = 10000.0):
    """pos: (T,) float positions (frame index or seconds). Returns cos,sin of (1,1,T,dim)."""
    assert dim % 2 == 0, "head_dim must be even for RoPE"
    device, dtype = pos.device, torch.float32
    inv_freq = base ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)  # (dim/2,)
    freqs = pos.to(dtype)[:, None] * inv_freq[None, :]        # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                    # (T, dim)
    return emb.cos()[None, None], emb.sin()[None, None]        # (1,1,T,dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B,H,T,D); cos/sin: (1,1,T,D)."""
    return x * cos + _rotate_half(x) * sin


def sinusoidal_pe(T: int, dim: int, device=None) -> torch.Tensor:
    """Absolute sin/cos positional encoding (baseline for the RoPE ablation). Returns (T,dim)."""
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-i * (torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(T, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
