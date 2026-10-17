# -*- coding: utf-8 -*-
"""Reader for the "Surrey" multi-camera calibration format (TotalCapture calibration.cal).

    num_cameras distortion_order
    per camera: min_row max_row min_col max_col / fx fy cx cy / distortion params /
                3x3 R / 3x1 t      with p_cam = R p_world + t
"""
from __future__ import annotations

import itertools

import numpy as np


def read_surrey(path: str) -> dict[int, dict]:
    tok = open(path, encoding="utf-8").read().split()
    vals = iter(float(x) for x in tok)
    n_cam, n_dist = int(next(vals)), int(next(vals))
    cams = {}
    for c in range(1, n_cam + 1):
        roi = [next(vals) for _ in range(4)]
        fx, fy, cx, cy = (next(vals) for _ in range(4))
        dist = [next(vals) for _ in range(n_dist)]
        R = np.array([next(vals) for _ in range(9)]).reshape(3, 3)
        t = np.array([next(vals) for _ in range(3)])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        cams[c] = dict(K=K, R=R, t=t, dist=np.array(dist), roi=roi,
                       center=-R.T @ t, axis=R[2])            # optical axis in world coords
    return cams


def pair_angles(cams: dict[int, dict], target: np.ndarray | None = None) -> dict[str, dict]:
    """Per camera pair: angle between optical axes, and angle between the directions from a
    target point (default: point closest to all optical axes) to the two camera centres."""
    if target is None:                                         # least-squares intersection of axes
        A, b = np.zeros((3, 3)), np.zeros(3)
        for c in cams.values():
            P = np.eye(3) - np.outer(c["axis"], c["axis"])
            A += P; b += P @ c["center"]
        target = np.linalg.solve(A, b)
    out = {}
    for u, v in itertools.combinations(sorted(cams), 2):
        cu, cv = cams[u], cams[v]
        ax = np.degrees(np.arccos(np.clip(cu["axis"] @ cv["axis"], -1, 1)))
        du, dv = cu["center"] - target, cv["center"] - target
        dv_ = np.degrees(np.arccos(np.clip(du @ dv / np.linalg.norm(du) / np.linalg.norm(dv), -1, 1)))
        out[f"{u}-{v}"] = dict(axis_deg=float(ax), view_deg=float(dv_))
    return out
