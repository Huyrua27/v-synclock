# -*- coding: utf-8 -*-
"""Simulated 2D pose-estimator noise, to bridge clean Panoptic projections toward
real detector output (HRNet / ViTPose).  Task #24 (realism without real capture).

Real 2D keypoint detectors are not exact: every joint carries a few-pixel
localization error, joints occasionally fail grossly (left/right swaps, jumps), and
some are missed entirely under occlusion. We inject the first two here (they keep the
coordinates finite); missed detections are modeled separately as visibility dropout
(the dataset's ``occlusion_p``). Applying this to both views makes the benchmark a far
more honest stand-in for real footage than clean projected keypoints.
"""
from __future__ import annotations

import numpy as np


def apply_detector_noise(xy: np.ndarray, rng, noise_px: float = 0.0,
                         outlier_p: float = 0.0, outlier_px: float = 25.0,
                         corr_frames: float = 5.0) -> np.ndarray:
    """Return a finite noisy copy of ``xy`` (T, J, 2) in pixels.

    noise_px    : std of per-joint localization error (pixels).
    corr_frames : temporal correlation length of that error, in frames. Real 2D
                  detectors mislocalize a joint *consistently* across nearby frames,
                  so the error is temporally smooth; pure per-frame iid noise (0)
                  is an unrealistic worst case that erases the motion signal. The
                  absolute error std stays ``noise_px``; only its frame-to-frame
                  jitter shrinks with ``corr_frames``.
    outlier_p   : per-joint-per-frame probability of a gross detector failure.
    outlier_px  : std of the extra displacement applied to those outlier joints.
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    if noise_px > 0:
        e = rng.normal(0.0, 1.0, size=out.shape)
        w = int(round(corr_frames))
        if w > 1 and out.shape[0] > w:                        # temporal smoothing
            k = np.ones(w) / w
            e = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, e)
            e /= (e.std() + 1e-9)                             # restore unit std
        out += noise_px * e
    if outlier_p > 0:
        hit = rng.random(out.shape[:-1]) < outlier_p          # (T, J)
        n = int(hit.sum())
        if n:
            out[hit] += rng.normal(0.0, outlier_px, size=(n, 2))
    return out


# COCO-19 (Panoptic order) limb groups and left/right pairs.
_LIMBS = {"l_arm": (4, 5), "r_arm": (10, 11), "l_leg": (7, 8), "r_leg": (13, 14)}
_LR = [(3, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (15, 17), (16, 18)]
_ARMS_LR, _LEGS_LR = _LR[:3], _LR[3:6]


def _bridge(v: np.ndarray, lo: int, hi: int) -> None:
    """Fill v[lo:hi] (T, ...) like the inference-time gap filler: linear between the
    neighbouring valid samples, holding the edge value when the gap touches a border."""
    T = len(v)
    left = v[lo - 1] if lo > 0 else None
    right = v[hi] if hi < T else None
    if left is None and right is None:
        return
    if left is None:
        v[lo:hi] = right
    elif right is None:
        v[lo:hi] = left
    else:
        w = (np.arange(lo, hi) - (lo - 1)) / (hi - (lo - 1))
        v[lo:hi] = left + w.reshape((-1,) + (1,) * (v.ndim - 1)) * (right - left)


def apply_mediapipe_like(xy: np.ndarray, rng, jitter_frac: float = 0.003,
                         limb_drop_p: float = 0.5, swap_p: float = 0.05,
                         frame_drop_p: float = 0.2, outlier_p: float = 0.002,
                         outlier_frac: float = 0.1, iid_frac: float = 0.0) -> np.ndarray:
    """Corrupt clean (T, J=19, 2) keypoints with failure modes typical of monocular
    single-person 2D detectors such as MediaPipe Pose, then gap-fill exactly as the
    real-data pipeline does. Magnitudes are relative to the person's height, so they
    transfer across resolutions. Generic defaults -- not fitted to any capture.

    jitter_frac  : std of temporally correlated localization error (x height). MediaPipe's
                   video mode smooths landmarks over time, so jitter is small and slow;
                   1% of height already swamps the slow Panoptic motion (CC 1.06 -> 1.97 fr).
    iid_frac     : extra per-frame white jitter (x height).
    limb_drop_p  : P(one limb is missed for a long span) -- self-occlusion in side views;
                   the span is bridged linearly, i.e. the limb looks frozen/ramped.
    swap_p       : P(a short left/right swap of arms, legs or the whole body).
    frame_drop_p : P(1-4 consecutive frames with no detection at all).
    outlier_p    : per-joint-per-frame gross error probability (outlier_frac x height).
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    T, J, _ = out.shape
    height = float(np.median(out[..., 1].max(1) - out[..., 1].min(1))) or 100.0

    out = apply_detector_noise(out, rng, noise_px=jitter_frac * height, corr_frames=5)
    if iid_frac > 0:
        out += rng.normal(0.0, iid_frac * height, size=out.shape)

    if rng.random() < swap_p:                                     # left/right confusion
        pairs = [_ARMS_LR, _LEGS_LR, _LR][rng.integers(3)]
        lo = int(rng.integers(0, T - 2)); hi = min(T, lo + int(rng.integers(2, 11)))
        for l, r in pairs:
            out[lo:hi, [l, r]] = out[lo:hi, [r, l]]

    if outlier_p > 0:
        hit = rng.random((T, J)) < outlier_p
        if hit.any():
            out[hit] += rng.normal(0.0, outlier_frac * height, size=(int(hit.sum()), 2))

    if rng.random() < limb_drop_p:                                # long self-occlusion
        names = list(_LIMBS)
        limb = _LIMBS[names[rng.choice(4, p=[0.35, 0.35, 0.15, 0.15])]]
        span = int(rng.integers(max(2, T // 4), T + 1))
        lo = int(rng.integers(0, T - span + 1))
        for j in limb:
            _bridge(out[:, j], lo, lo + span)

    if rng.random() < frame_drop_p:                               # missed frames
        n = int(rng.integers(1, 5)); lo = int(rng.integers(0, max(1, T - n)))
        _bridge(out, lo, min(T, lo + n))
    return out
