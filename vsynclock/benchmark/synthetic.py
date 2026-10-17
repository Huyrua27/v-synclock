# -*- coding: utf-8 -*-
"""Synthetic keypoint trajectories for demos and tests without Panoptic.

Each joint is a sum of smooth sinusoids: human-like motion with smooth derivatives (good for
sub-frame interpolation). Used for (1) an out-of-the-box demo, (2) unit tests, (3) augmentation.

Owner: P1 (WS1).
"""
from __future__ import annotations
import numpy as np
from ..core.types import KeypointSequence, SyncSample
from .desync import inject_offset


def make_trajectory(T=90, J=17, fps=30.0, seed=0, speed=1.0) -> KeypointSequence:
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    xy = np.zeros((T, J, 2))
    for j in range(J):
        for c in range(2):
            freqs = rng.uniform(0.2, 1.6, 3) * speed        # faster = higher frequency
            amps = rng.uniform(15, 70, 3)
            ph = rng.uniform(0, 2 * np.pi, 3)
            sig = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, ph))
            xy[:, j, c] = sig + 320 + j * 4
    return KeypointSequence(xy, fps, name=f"synth{seed}")


def mean_speed_px_s(seq: KeypointSequence) -> float:
    """Characteristic speed = mean joint speed (px/s)."""
    d = np.linalg.norm(np.diff(seq.xy, axis=0), axis=-1)   # (T-1,J)
    return float(d.mean() * seq.fps)


def make_dataset(n=200, T=90, fps=30.0, seed=0, max_offset=2.0,
                 speeds=(0.5, 1.5, 4.0, 9.0)) -> list[SyncSample]:
    """Make n samples (a, shifted b, dt_gt) spread over speeds, with random offsets."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sp = float(speeds[i % len(speeds)])
        a = make_trajectory(T=T, fps=fps, seed=seed * 1000 + i, speed=sp)
        dt = float(rng.uniform(-max_offset, max_offset))
        b = inject_offset(a, dt)
        out.append(SyncSample(a=a, b=b, dt_gt_frames=dt,
                              velocity=mean_speed_px_s(a),
                              meta={"speed_factor": sp}))
    return out
