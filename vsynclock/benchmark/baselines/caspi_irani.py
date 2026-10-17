"""Affine temporal alignment baseline ``t_source = alpha*t_video + beta``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult
from ..interpolation import cubic_sample
from .cross_correlation import lag_scores, motion_features, refined_peak


@dataclass(frozen=True)
class AffineEstimate:
    alpha: float
    beta: float
    score: float


def _warp_to_reference(seq: KeypointSequence, alpha: float) -> KeypointSequence:
    source_time = np.arange(seq.T, dtype=np.float64)
    reference_time = source_time / float(alpha)
    valid = reference_time <= seq.T - 1
    xy = cubic_sample(seq.xy, reference_time[valid])
    return KeypointSequence(xy, seq.fps, name=f"{seq.name}/alpha-{alpha:.6f}")


class CaspiIrani(SyncMethod):
    name = "Caspi-Irani"

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.98, 1.02),
        alpha_steps: int = 21,
        max_lag: int = 20,
        features: str = "speed",
    ):
        if alpha_steps < 3:
            raise ValueError("alpha_steps must be at least three")
        self.alpha_range = tuple(float(value) for value in alpha_range)
        self.alpha_steps = int(alpha_steps)
        self.max_lag = int(max_lag)
        self.features = features

    def estimate(self, a: KeypointSequence, b: KeypointSequence) -> AffineEstimate:
        features_a = motion_features(a, self.features)
        best = AffineEstimate(alpha=1.0, beta=0.0, score=-np.inf)
        for alpha in np.linspace(*self.alpha_range, self.alpha_steps):
            warped_b = _warp_to_reference(b, float(alpha))
            features_b = motion_features(warped_b, self.features)
            lags, scores = lag_scores(features_a, features_b, self.max_lag)
            beta, _ = refined_peak(lags, scores)
            peak = float(np.max(scores))
            if peak > best.score:
                best = AffineEstimate(float(alpha), float(beta), peak)
        return best

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        estimate = self.estimate(a, b)
        confidence = float(np.clip((estimate.score + 1.0) / 2.0, 0.0, 1.0))
        return SyncResult(dt_frames=estimate.beta, confidence=confidence)
