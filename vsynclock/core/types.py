# -*- coding: utf-8 -*-
"""Shared data types used by every module.

These dataclasses are the contract between modules; keep their signatures stable,
since the benchmark harness, the model and the real-capture pipeline all import them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class KeypointSequence:
    """2D keypoint trajectory of ONE camera over time.

    xy         : ndarray (T, J, 2), T frames, J joints, (x, y) in pixels.
    fps        : frames per second.
    timestamps : (T,) seconds, optional; webcams store their host timestamps here.
    name       : label for debugging (e.g. 'iphone1', 'cam0').
    """
    xy: np.ndarray
    fps: float
    timestamps: Optional[np.ndarray] = None
    name: str = ""

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=np.float64)
        assert self.xy.ndim == 3 and self.xy.shape[2] == 2, \
            f"xy must be (T,J,2), got {self.xy.shape}"

    @property
    def T(self) -> int: return self.xy.shape[0]
    @property
    def J(self) -> int: return self.xy.shape[1]


@dataclass
class SyncResult:
    """Result of one synchronization method for one camera pair."""
    dt_frames: float            # offset of b relative to a, in frames (fractional, signed)
    confidence: float = 1.0     # [0, 1], method-specific

    def dt_seconds(self, fps: float) -> float:
        return self.dt_frames / fps


@dataclass
class SyncSample:
    """One benchmark sample: a pair (a, b) and its ground-truth label."""
    a: KeypointSequence
    b: KeypointSequence
    dt_gt_frames: float         # injected true offset (label)
    velocity: float = 0.0       # characteristic speed (for bucketing)
    meta: dict = field(default_factory=dict)
