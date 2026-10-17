# -*- coding: utf-8 -*-
"""Keypoint-pair dataset for training. Wraps the benchmark desync generator.

On real data, swap `synthetic.make_trajectory` for Panoptic-projected 2D trajectories;
the (kp_a, kp_b, Δt) contract stays the same. Task #15 (WS2).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from ..benchmark import synthetic
from ..benchmark.desync import inject_offset


class KeypointPairDataset(Dataset):
    """Generates (kp_a, kp_b, dt) pairs on the fly with random sub-frame offsets."""

    def __init__(self, n=4000, T=90, J=17, fps=30.0, max_offset=6.0,
                 speeds=(0.5, 1.5, 4.0, 9.0), seed=0, occlusion_p=0.0):
        self.n, self.T, self.J, self.fps = n, T, J, fps
        self.max_offset, self.speeds = max_offset, speeds
        self.seed, self.occlusion_p = seed, occlusion_p

    def __len__(self): return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100000 + i)
        sp = float(self.speeds[i % len(self.speeds)])
        a = synthetic.make_trajectory(T=self.T, J=self.J, fps=self.fps,
                                      seed=self.seed * 100000 + i, speed=sp)
        dt = float(rng.uniform(-self.max_offset, self.max_offset))
        b = inject_offset(a, dt)
        ka = torch.tensor(a.xy, dtype=torch.float32)
        kb = torch.tensor(b.xy, dtype=torch.float32)
        va = vb = torch.tensor([])                            # empty = no visibility
        if self.occlusion_p > 0:                              # simulate occlusion
            va = (torch.rand(self.T, self.J) > self.occlusion_p).float()
            vb = (torch.rand(self.T, self.J) > self.occlusion_p).float()
        return ka, kb, torch.tensor(dt, dtype=torch.float32), va, vb


def collate(batch):
    ka = torch.stack([b[0] for b in batch])
    kb = torch.stack([b[1] for b in batch])
    dt = torch.stack([b[2] for b in batch])
    has_vis = batch[0][3].numel() > 0
    va = torch.stack([b[3] for b in batch]) if has_vis else None
    vb = torch.stack([b[4] for b in batch]) if has_vis else None
    return ka, kb, dt, va, vb


class MixedPairDataset:
    """Interleave several pair datasets (e.g. Panoptic + AIST++) into one of length n.

    Sample i comes from source i % k at index i // k, so each source keeps its own
    deterministic per-index sampling."""

    def __init__(self, sources, n):
        self.sources, self.n = list(sources), int(n)
        self.occlusion_p = self.sources[0].occlusion_p

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = len(self.sources)
        return self.sources[i % k][i // k]
