# -*- coding: utf-8 -*-
"""Floor baseline: ZeroOffset (predicts dt = 0). Every method must beat it."""
from __future__ import annotations
from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult


class ZeroOffset(SyncMethod):
    name = "Zero(sàn)"

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        return SyncResult(dt_frames=0.0, confidence=0.0)
