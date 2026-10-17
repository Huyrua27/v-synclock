# -*- coding: utf-8 -*-
"""Common interface for every synchronization method.

Baselines and the ContinuSyncFormer model inherit SyncMethod and implement `predict`,
so the benchmark harness runs every method with the same loop.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from .types import KeypointSequence, SyncResult


class SyncMethod(ABC):
    """Estimate the time offset between two keypoint sequences of one scene seen from two views."""
    name: str = "base"

    @abstractmethod
    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        """Return a SyncResult with dt_frames = offset of `b` relative to `a` (frames).

        Sign convention: dt_frames > 0 means `b` lags `a` by dt_frames frames.
        This is the label produced by benchmark.desync.inject_offset(), so a correct method
        recovers exactly the injected value.
        """
        raise NotImplementedError
