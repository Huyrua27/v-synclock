# -*- coding: utf-8 -*-
"""Build a ground-truth drift curve from periodic on-set flashes.

Flashes are physically simultaneous, so the k-th flash marks the same real-world
instant in every camera. If camera c sees it at sub-frame position f_c^k and the
reference camera at f_r^k, the true offset of c relative to the reference at that
instant is ``f_c^k - f_r^k`` (frames, at a common nominal fps). Interpolating these
per-flash offsets over the take yields the ground-truth drift delta_c(t) that the
sync methods must recover on the real 2D keypoints.  Task #24 (real data).
"""
from __future__ import annotations

import numpy as np


def match_flashes(flash_frames: dict[str, list[float]], reference: str):
    """Order-match flashes across cameras; every camera must see the same count."""
    counts = {c: len(v) for c, v in flash_frames.items()}
    k = counts[reference]
    bad = {c: n for c, n in counts.items() if n != k}
    if bad:
        raise ValueError(
            f"cameras detected different flash counts than the reference ({k}): {bad}. "
            "Re-run detection with a matching --n-flashes, or clean false peaks."
        )
    return k


def build_gt_drift(flash_frames: dict[str, list[float]], reference: str,
                   n_frames: int):
    """Per-camera ground-truth offset curve delta_c(t), t in [0, n_frames).

    Returns dict cam -> {"flash_ref": [...], "offset": [...], "curve": ndarray(n_frames)}.
    The reference camera has an all-zero curve by construction.
    """
    match_flashes(flash_frames, reference)
    ref = np.asarray(flash_frames[reference], dtype=np.float64)
    order = np.argsort(ref)
    ref = ref[order]
    idx = np.arange(n_frames, dtype=np.float64)
    out = {}
    for cam, frames in flash_frames.items():
        fc = np.asarray(frames, dtype=np.float64)[order]
        offset = fc - ref                                  # frames, per flash
        # Linear interpolation between flashes; flat hold beyond the ends.
        curve = np.interp(idx, ref, offset)
        out[cam] = {"flash_ref": ref.tolist(), "offset": offset.tolist(),
                    "curve": curve}
    return out


def drift_rate(gt_cam) -> float:
    """Least-squares slope of the offset over the take (frames per frame)."""
    ref = np.asarray(gt_cam["flash_ref"], dtype=np.float64)
    off = np.asarray(gt_cam["offset"], dtype=np.float64)
    if len(ref) < 2:
        return 0.0
    A = np.vstack([ref, np.ones_like(ref)]).T
    slope, _ = np.linalg.lstsq(A, off, rcond=None)[0]
    return float(slope)
