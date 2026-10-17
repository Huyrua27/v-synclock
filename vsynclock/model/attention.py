# -*- coding: utf-8 -*-
"""Attention blocks: temporal self-attention (per view) and cross-view attention.

Cross-view attention exposes the softmax(QKᵀ) matrix as an interpretable *soft-alignment*
between the two views' frames. Tasks #11, #13 (WS2).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .rope import apply_rope


class MultiHeadAttention(nn.Module):
    """Generic MHA supporting self/cross attention, optional RoPE, optional additive key bias.

    `rope` is a precomputed (cos, sin) pair (see rope_cache) shared across all blocks — passing
    it in avoids rebuilding the cache on every attention call.
    """

    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.use_rope = use_rope
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, xq, xk, rope=None, key_bias=None, return_attn=False):
        B, Tq, _ = xq.shape
        Tk = xk.shape[1]
        q = self.q(xq).view(B, Tq, self.h, self.dh).transpose(1, 2)   # (B,h,Tq,dh)
        k = self.k(xk).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(B, Tk, self.h, self.dh).transpose(1, 2)
        if self.use_rope and rope is not None:
            cos, sin = rope                                          # (1,1,T,dh)
            q = apply_rope(q, cos[:, :, :Tq], sin[:, :, :Tq])
            k = apply_rope(k, cos[:, :, :Tk], sin[:, :, :Tk])
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)         # (B,h,Tq,Tk)
        if key_bias is not None:                                     # (B,Tk) reliability bias
            att = att + key_bias[:, None, None, :]
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, Tq, -1)
        out = self.o(out)
        return (out, att.mean(1)) if return_attn else (out, None)    # attn: (B,Tq,Tk)


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * mult), nn.GELU(),
                                 nn.Linear(d_model * mult, d_model))

    def forward(self, x): return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm temporal self-attention block."""

    def __init__(self, d_model, n_heads, use_rope=True):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, use_rope)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, rope=None, key_bias=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, rope=rope, key_bias=key_bias)
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class CrossViewBlock(nn.Module):
    """View A queries view B → returns updated A features and soft-alignment matrix."""

    def __init__(self, d_model, n_heads, use_rope=True):
        super().__init__()
        self.nq = nn.LayerNorm(d_model)
        self.nk = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, use_rope)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, xa, xb, rope=None, key_bias_b=None):
        a, S = self.attn(self.nq(xa), self.nk(xb), rope=rope,
                         key_bias=key_bias_b, return_attn=True)
        xa = xa + a
        xa = xa + self.ff(self.n2(xa))
        return xa, S                                                # S: (B,Ta,Tb)
