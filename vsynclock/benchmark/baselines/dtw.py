"""Dynamic Time Warping baseline with bounded oversampled alignment."""
from __future__ import annotations

import numpy as np

from ...core.interfaces import SyncMethod
from ...core.types import KeypointSequence, SyncResult
from ..interpolation import cubic_sample
from .cross_correlation import motion_features


def _oversample(features: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("oversample_factor must be at least one")
    if factor == 1:
        return features
    target = np.linspace(0.0, len(features) - 1.0, (len(features) - 1) * factor + 1)
    return cubic_sample(features, target)


def _dtw_path(a: np.ndarray, b: np.ndarray, radius: int) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    parent = np.full((n + 1, m + 1), -1, dtype=np.int8)
    for i in range(1, n + 1):
        lower = max(1, i - radius)
        upper = min(m, i + radius)
        for j in range(lower, upper + 1):
            distance = float(np.mean((a[i - 1] - b[j - 1]) ** 2))
            candidates = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            choice = int(np.argmin(candidates))
            cost[i, j] = distance + candidates[choice]
            parent[i, j] = choice
    if not np.isfinite(cost[n, m]):
        raise ValueError("DTW band is too narrow to connect both sequences")

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choice = int(parent[i, j])
        if choice == 0:
            i, j = i - 1, j - 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


class DTW(SyncMethod):
    name = "DTW"

    def __init__(
        self,
        oversample_factor: int = 4,
        max_warp_frames: float = 20.0,
        max_frames: int | None = 240,
        features: str = "speed",
    ):
        self.oversample_factor = int(oversample_factor)
        self.max_warp_frames = float(max_warp_frames)
        self.max_frames = max_frames
        self.features = features

    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        features_a, features_b = motion_features(a, self.features), motion_features(b, self.features)
        if float(np.std(features_a)) < 1e-10 or float(np.std(features_b)) < 1e-10:
            raise ValueError("insufficient motion for DTW")
        if self.max_frames is not None:
            features_a = features_a[: self.max_frames]
            features_b = features_b[: self.max_frames]
        up_a = _oversample(features_a, self.oversample_factor)
        up_b = _oversample(features_b, self.oversample_factor)
        radius = max(
            abs(len(up_a) - len(up_b)),
            int(np.ceil(self.max_warp_frames * self.oversample_factor)),
        )
        path = _dtw_path(up_a, up_b, radius)
        offsets = np.asarray([i - j for i, j in path], dtype=np.float64)
        trim = max(1, len(offsets) // 10)
        core = offsets[trim:-trim] if len(offsets) > 2 * trim else offsets
        estimate = float(np.median(core) / self.oversample_factor)
        dispersion = float(np.median(np.abs(core - np.median(core))))
        confidence = float(1.0 / (1.0 + dispersion / self.oversample_factor))
        return SyncResult(dt_frames=estimate, confidence=confidence)
