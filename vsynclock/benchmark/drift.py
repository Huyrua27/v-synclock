# -*- coding: utf-8 -*-
"""Time-varying (drift) desynchronization and sliding-window affine recovery.

Real multi-device rigs do not share a clock: camera B's frame ``t`` captures the
scene at continuous reference time ``alpha*t + beta`` (alpha != 1 => the offset
drifts across the take). A single-lag method (cross-correlation) cannot represent
this; the offset it applies is right at one instant and wrong everywhere else.

Recovering drift requires estimating a *local* offset in short sliding windows and
fitting a line through them. Short windows are exactly where classical CC becomes
unreliable and a learned estimator wins -- so drift is the regime where the learned
model has a structural advantage.  Task #24 / B3 (for 3DV).
"""
from __future__ import annotations

import numpy as np

from ..core.types import KeypointSequence
from .interpolation import cubic_sample


def drift_offset(index: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """True extra time of B relative to A at each index: (alpha-1)*t + beta."""
    return (alpha - 1.0) * np.asarray(index, dtype=np.float64) + beta


def warp_stream(clean_xy: np.ndarray, alpha: float, beta: float,
                length: int, start: int = 0) -> np.ndarray:
    """Camera-B observation stream under drift: obs[t] = clean(alpha*(start+t)+beta).

    ``clean_xy`` is the full clean trajectory; sampling is cubic with edge
    extrapolation so the caller only needs a modest margin.
    """
    idx = alpha * (start + np.arange(length, dtype=np.float64)) + beta
    return cubic_sample(clean_xy, idx, extrapolate=True)


def theilsen_line(centers: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Robust slope/intercept (median of pairwise slopes) -- rejects the
    catastrophic single-window failures that CC produces on short windows."""
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(centers) & np.isfinite(values)
    centers, values = centers[ok], values[ok]
    if len(centers) < 2:
        return 0.0, float(values.mean()) if len(values) else 0.0
    slopes = []
    for i in range(len(centers)):
        dx = centers[i + 1:] - centers[i]
        dy = values[i + 1:] - values[i]
        m = dy[dx != 0] / dx[dx != 0]
        slopes.append(m)
    slope = float(np.median(np.concatenate(slopes)))
    intercept = float(np.median(values - slope * centers))
    return slope, intercept


def shrink_slope(centers, values, sigma_prior: float, window: int, stride: int) -> tuple[float, float]:
    """Theil-Sen line whose slope is shrunk towards zero by a Gaussian prior N(0, sigma_prior^2)
    (frames per frame). The slope's standard error is a robust OLS-style estimate, inflated by
    sqrt(window / stride) because overlapping windows give correlated residuals."""
    c = np.asarray(centers, dtype=np.float64); v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(v); c, v = c[ok], v[ok]
    s, b = theilsen_line(c, v)
    if len(c) < 3:
        return 0.0, float(np.median(v)) if len(v) else 0.0
    r = v - (s * c + b)
    se = 1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(np.sum((c - c.mean()) ** 2) + 1e-12)
    se *= np.sqrt(window / stride)
    s2 = s * sigma_prior ** 2 / (sigma_prior ** 2 + se ** 2)
    return float(s2), float(np.median(v - s2 * c))


def ztest_slope(centers, values, window: int, stride: int, z: float = 1.96) -> tuple[float, float]:
    """Theil-Sen line whose slope is kept only if |slope| / se >= z (se as in shrink_slope);
    otherwise slope 0 and the median offset (constant timing)."""
    c = np.asarray(centers, dtype=np.float64); v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(v); c, v = c[ok], v[ok]
    s, b = theilsen_line(c, v)
    if len(c) < 3:
        return 0.0, float(np.median(v)) if len(v) else 0.0
    r = v - (s * c + b)
    se = 1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(np.sum((c - c.mean()) ** 2) + 1e-12)
    se *= np.sqrt(window / stride)
    if abs(s) >= z * se:
        return float(s), float(b)
    return 0.0, float(np.median(v))


def sliding_offsets(predict, a_stream: np.ndarray, b_stream: np.ndarray,
                    fps: float, window: int, stride: int):
    """Local offset per window via ``predict(a_seq, b_seq) -> dt_frames``.

    Returns (centers, offsets) over windows where prediction succeeds.
    """
    n = min(len(a_stream), len(b_stream))
    centers, offsets = [], []
    for s in range(0, n - window + 1, stride):
        a = KeypointSequence(a_stream[s:s + window].astype(np.float64), fps, name="a")
        b = KeypointSequence(b_stream[s:s + window].astype(np.float64), fps, name="b")
        try:
            dt = float(predict(a, b))
        except Exception:
            continue
        centers.append(s + window / 2.0)
        offsets.append(dt)
    return np.asarray(centers), np.asarray(offsets)


def _hampel(values: np.ndarray, k: int = 2, n_sigma: float = 2.5) -> np.ndarray:
    """Replace outliers by the local median (rejects CC's catastrophic windows)."""
    v = np.asarray(values, dtype=np.float64).copy()
    out = v.copy()
    for i in range(len(v)):
        lo, hi = max(0, i - k), min(len(v), i + k + 1)
        window = v[lo:hi]
        med = np.median(window)
        mad = 1.4826 * np.median(np.abs(window - med)) + 1e-9
        if abs(v[i] - med) > n_sigma * mad:
            out[i] = med
    return out


def fit_drift(predict, a_stream, b_stream, fps, window, stride, mode="line"):
    """Sliding-window local offsets -> per-index offset estimate.

    mode="line": robust affine fit (Theil-Sen) -- optimal when drift is linear.
    mode="curve": outlier-reject then interpolate the local offsets -- tracks
    *non-linear* drift that a global line cannot represent. Only reliable when the
    local estimates are trustworthy on short windows (i.e. with the learned model).
    """
    centers, offsets = sliding_offsets(predict, a_stream, b_stream, fps, window, stride)
    n = min(len(a_stream), len(b_stream))
    idx = np.arange(n, dtype=np.float64)
    if len(centers) < 2:
        return np.full(n, float(np.median(offsets)) if len(offsets) else 0.0), centers, offsets
    if mode == "curve":
        cleaned = _hampel(offsets)
        return np.interp(idx, centers, cleaned), centers, offsets
    slope, intercept = theilsen_line(centers, offsets)
    return slope * idx + intercept, centers, offsets


def peak_in_range(lags, scores, center=None, radius=None) -> float:
    """Argmax of a lag profile (+ parabolic refinement), optionally restricted to
    |lag - center| <= radius. Restricting the search rejects the aliased peaks that
    periodic motion (gait, running, rhythmic punches) produces one period away."""
    lags = np.asarray(lags, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64).copy()
    if center is not None and radius is not None:
        out = np.abs(lags - center) > radius
        if out.all():                                  # prior outside the profile
            out[np.argmin(np.abs(lags - center))] = False
        s[out] = -np.inf
    k = int(np.argmax(s))
    delta = 0.0
    if 0 < k < len(s) - 1 and np.isfinite(s[k - 1]) and np.isfinite(s[k + 1]):
        den = s[k - 1] - 2 * s[k] + s[k + 1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5 * (s[k - 1] - s[k + 1]) / den, -0.5, 0.5))
    return float(lags[k] + delta)


def fit_drift_c2f(profile, a_stream, b_stream, fps, window, stride,
                  coarse_window=80, coarse_stride=20, radius=3.0, prior="long",
                  min_peak=None, min_kept=3):
    """Coarse-to-fine drift recovery with any lag-profile estimator.

    profile(a_seq, b_seq) -> (lags, scores). Pass 1 runs the estimator on long windows
    (more non-repetitive motion, fewer aliased peaks) and fits a robust line. Pass 2
    re-estimates on short windows, searching only within +-radius of that prior, then
    refits. After coarse alignment the residual offset is small and varies slowly, so a
    narrow search loses nothing and removes peaks one motion period away.

    prior="long"      : the long-window line is the prior (original c2f).
    prior="consensus" : two candidate priors -- the long-window line and the Theil-Sen
        line of the free short-window estimates -- are scored by how many free
        short-window estimates fall within +-radius of them (RANSAC-style support), and
        the better-supported one is used. This guards against a long-window prior that
        is itself aliased (long repetitive gestures), which otherwise locks the narrow
        search onto the wrong peak.
    min_peak: observability gate. Short windows whose profile peak is below this value
        (no shared timing evidence) are ignored both when scoring prior support and in the
        final line fit; if fewer than `min_kept` windows pass, the most confident
        `min_kept` windows are used instead.
    """
    n = min(len(a_stream), len(b_stream))
    idx = np.arange(n, dtype=np.float64)
    cw = int(min(coarse_window, max(window, n // 2)))
    free = lambda a, b: peak_in_range(*profile(a, b))
    c0, o0 = sliding_offsets(free, a_stream, b_stream, fps, cw, coarse_stride)
    if len(c0) >= 2:
        slope, inter = theilsen_line(c0, o0)
    else:
        slope, inter = 0.0, (float(np.median(o0)) if len(o0) else 0.0)

    centers, profiles = [], []                         # short-window profiles, computed once
    for s in range(0, n - window + 1, stride):
        a = KeypointSequence(a_stream[s:s + window].astype(np.float64), fps)
        b = KeypointSequence(b_stream[s:s + window].astype(np.float64), fps)
        try:
            profiles.append(profile(a, b))
        except Exception:
            continue
        centers.append(s + window / 2.0)
    centers = np.asarray(centers)
    if len(centers) < 2:
        return slope * idx + inter, centers, np.asarray([])

    use = np.ones(len(centers), dtype=bool)
    if min_peak is not None:
        peaks = np.asarray([np.max(sc) for _, sc in profiles])
        use = peaks >= min_peak
        if use.sum() < min_kept:
            use = np.zeros(len(centers), dtype=bool)
            use[np.argsort(-peaks)[:min_kept]] = True

    if prior == "consensus":
        free_off = np.asarray([peak_in_range(l, sc) for l, sc in profiles])
        cands = [(slope, inter), theilsen_line(centers[use], free_off[use])]
        support = [np.mean(np.abs(free_off[use] - (m * centers[use] + q)) <= radius) for m, q in cands]
        slope, inter = cands[int(np.argmax(support))]   # ties keep the long-window prior

    offsets = np.asarray([peak_in_range(l, sc, slope * c + inter, radius)
                          for c, (l, sc) in zip(centers, profiles)])
    s2, i2 = theilsen_line(centers[use], offsets[use])
    return s2 * idx + i2, centers, offsets


def warp_offsets(clean_xy: np.ndarray, delta: np.ndarray, base: float) -> np.ndarray:
    """General (possibly non-linear) drift: obs[t] = clean(base + t + delta[t])."""
    idx = base + np.arange(len(delta), dtype=np.float64) + np.asarray(delta, dtype=np.float64)
    return cubic_sample(clean_xy, idx, extrapolate=True)


def sine_offset(index: np.ndarray, amp: float, period: float, beta: float) -> np.ndarray:
    """Non-linear (oscillating) offset: a global line cannot fit this."""
    return amp * np.sin(2.0 * np.pi * np.asarray(index, dtype=np.float64) / period) + beta
