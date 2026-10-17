# -*- coding: utf-8 -*-
"""Linear (DLT) multi-view triangulation for the sub-frame-sync downstream study.

Pinhole model, self-consistent with :func:`vsynclock.panoptic.project_to_2d` when it is
called without distortion — so at zero desync the round-trip project→triangulate is
exact and any MPJPE is attributable purely to temporal misalignment.
"""
from __future__ import annotations

import numpy as np


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = K [R | t], shape (3, 4). Pinhole: x ~ P @ [X; 1]."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R, t])


def triangulate_dlt(points_2d: list[np.ndarray], mats: list[np.ndarray]) -> np.ndarray:
    """Triangulate ``(P, 2)`` pixel observations from ``len(mats)`` views.

    points_2d[v]: (P, 2) pixels in view v. mats[v]: (3, 4) projection matrix.
    Returns (P, 3) world points. Points must be finite in every supplied view;
    caller masks invalid joints beforehand.
    """
    n_views = len(mats)
    if n_views < 2:
        raise ValueError("triangulation needs at least two views")
    P = points_2d[0].shape[0]
    A = np.empty((P, 2 * n_views, 4), dtype=np.float64)
    for v, (xy, M) in enumerate(zip(points_2d, mats)):
        x = xy[:, 0][:, None]
        y = xy[:, 1][:, None]
        A[:, 2 * v] = x * M[2] - M[0]
        A[:, 2 * v + 1] = y * M[2] - M[1]
    # Batched homogeneous least squares: smallest right-singular vector per point.
    _, _, Vh = np.linalg.svd(A)
    X = Vh[:, -1, :]
    return X[:, :3] / X[:, 3:4]


def triangulate_masked(points: np.ndarray, mats, valid: np.ndarray) -> np.ndarray:
    """Triangulate with per-point variable view sets (for occlusion).

    points: (P, V, 2) pixels; mats: V projection matrices; valid: (P, V) bool.
    A point is recovered from the views where it is valid (>= 2 needed); points
    with fewer than two valid views are returned as NaN. Points are grouped by
    visibility pattern so each group still runs a single batched DLT.
    """
    from collections import defaultdict

    P, V, _ = points.shape
    M = [np.asarray(m, dtype=np.float64) for m in mats]
    out = np.full((P, 3), np.nan, dtype=np.float64)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(P):
        pattern = tuple(bool(b) for b in valid[i])
        if sum(pattern) >= 2:
            groups[pattern].append(i)
    for pattern, rows in groups.items():
        views = [v for v, b in enumerate(pattern) if b]
        idx = np.asarray(rows)
        sub = points[idx][:, views, :]                       # (n, nv, 2)
        X = triangulate_dlt([sub[:, j, :] for j in range(len(views))],
                            [M[v] for v in views])
        out[idx] = X
    return out


def mpjpe(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean per-joint position error over finite rows (same world unit as input)."""
    valid = np.isfinite(estimate).all(-1) & np.isfinite(truth).all(-1)
    if not valid.any():
        return float("nan")
    return float(np.linalg.norm(estimate[valid] - truth[valid], axis=-1).mean())
