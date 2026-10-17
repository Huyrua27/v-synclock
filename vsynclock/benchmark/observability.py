# -*- coding: utf-8 -*-
"""Observability scores computed from a lag-correlation profile.

A pair of views can only be synchronized from motion if the two streams share timing
evidence. When they do, the lag profile has one dominant, sharp peak; when they do not
(wide viewpoint difference, scarce motion) it is flat or has several comparable peaks,
and the argmax is arbitrary. These scores summarize the profile, so they work with any
estimator that exposes one (classical CC, the learned CSF head).
"""
from __future__ import annotations

import numpy as np


def profile_scores(lags, scores, exclusion: int = 2, n: int | None = None) -> dict:
    """Candidate observability scores for one lag profile (higher = more trustworthy).

    peak      : profile value at the argmax (for CC on z-scored features ~ a correlation)
    contrast  : (peak - median) / std of the profile
    ratio     : peak margin over the best competing peak at least `exclusion` lags away,
                (peak - second) / (|peak| + |second| + eps), in [-1, 1]
    sharpness : negative discrete curvature at the peak, normalised by profile std
    entropy   : 1 - normalised entropy of softmax(profile / std)  (1 = fully concentrated)
    peak_z    : peak * sqrt(n), n = window length -- chance correlation shrinks as 1/sqrt(n),
                so this keeps one threshold meaningful across window lengths
    """
    lags = np.asarray(lags, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    k = int(np.argmax(s))
    peak = float(s[k])
    sd = float(np.std(s)) + 1e-9
    far = np.abs(lags - lags[k]) >= exclusion
    second = float(np.max(s[far])) if far.any() else float(np.min(s))
    if 0 < k < len(s) - 1:
        curv = float(s[k - 1] - 2 * s[k] + s[k + 1])
    else:
        curv = 0.0
    p = np.exp((s - s.max()) / sd)
    p /= p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return {"peak": peak,
            "contrast": (peak - float(np.median(s))) / sd,
            "ratio": (peak - second) / (abs(peak) + abs(second) + 1e-9),
            "sharpness": -curv / sd,
            "entropy": 1.0 - ent,
            "peak_z": peak * np.sqrt(n) if n else float("nan")}


def auc(score, positive):
    """Probability that a random positive has a higher score than a random negative
    (Mann-Whitney U), ties counted half. Here positives are *successes*."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])                  # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
