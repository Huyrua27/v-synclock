"""Metrics and frozen velocity-bucket utilities for synchronization benchmarks."""
from __future__ import annotations

import numpy as np

from ..core.types import KeypointSequence


def _arr(values):
    return np.asarray(values, dtype=np.float64)


def _check_pair(pred, gt) -> tuple[np.ndarray, np.ndarray]:
    pred_array, gt_array = _arr(pred), _arr(gt)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred_array.shape} vs {gt_array.shape}")
    if pred_array.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isfinite(pred_array).all() or not np.isfinite(gt_array).all():
        raise ValueError("metrics do not accept NaN or infinite values")
    return pred_array, gt_array


def frm_err(pred, gt) -> float:
    pred_array, gt_array = _check_pair(pred, gt)
    return float(np.mean(np.abs(pred_array - gt_array)))


def accin(pred, gt, tau=0.1) -> float:
    pred_array, gt_array = _check_pair(pred, gt)
    return float(np.mean(np.abs(pred_array - gt_array) <= float(tau)))


def accex(pred, gt, i=1) -> float:
    pred_array, gt_array = _check_pair(pred, gt)
    return float(
        np.mean(np.abs(np.round(pred_array) - np.round(gt_array)) <= int(i))
    )


def _fps(fps, shape) -> np.ndarray:
    fps_array = np.asarray(fps, dtype=np.float64)
    if fps_array.ndim == 0:
        fps_array = np.full(shape, float(fps_array))
    else:
        fps_array = np.broadcast_to(fps_array, shape)
    if np.any(~np.isfinite(fps_array)) or np.any(fps_array <= 0):
        raise ValueError("fps must be finite and positive")
    return fps_array


def mae_ms(pred, gt, fps) -> float:
    pred_array, gt_array = _check_pair(pred, gt)
    return float(np.mean(np.abs(pred_array - gt_array) * 1000.0 / _fps(fps, pred_array.shape)))


def rmse_ms(pred, gt, fps) -> float:
    pred_array, gt_array = _check_pair(pred, gt)
    errors_ms = (pred_array - gt_array) * 1000.0 / _fps(fps, pred_array.shape)
    return float(np.sqrt(np.mean(errors_ms**2)))


def summary(pred, gt, fps, taus=(0.1, 0.25, 0.5)) -> dict:
    result = {"n": len(pred), "Frm.err": round(frm_err(pred, gt), 4)}
    for tau in taus:
        result[f"Accin@{tau}"] = round(accin(pred, gt, tau), 4)
    result["Accex@1"] = round(accex(pred, gt, 1), 4)
    result["MAE_ms"] = round(mae_ms(pred, gt, fps), 3)
    result["RMSE_ms"] = round(rmse_ms(pred, gt, fps), 3)
    return result


def sequence_velocity_px_s(seq: KeypointSequence) -> float:
    displacement = np.linalg.norm(np.diff(seq.xy, axis=0), axis=-1)
    if not np.isfinite(displacement).any():
        raise ValueError("velocity cannot be computed from an all-missing sequence")
    return float(np.nanmedian(displacement) * seq.fps)


def fit_velocity_edges(velocities, quantiles=(0.25, 0.5, 0.75)) -> tuple[float, ...]:
    values = _arr(velocities)
    if values.size < 4 or not np.isfinite(values).all():
        raise ValueError("at least four finite training velocities are required")
    edges = tuple(float(value) for value in np.quantile(values, quantiles))
    if len(set(edges)) != len(edges):
        raise ValueError("training velocities do not define distinct bucket edges")
    return edges


def by_velocity_bucket(
    pred,
    gt,
    vel,
    fps,
    edges=(1.5, 5.0, 15.0),
    tau=0.1,
) -> dict:
    pred_array, gt_array = _check_pair(pred, gt)
    velocities = _arr(vel)
    if velocities.shape != pred_array.shape:
        raise ValueError("velocity shape must match pred/gt")
    fps_array = _fps(fps, pred_array.shape)
    bounds = [-np.inf, *edges, np.inf]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    output = {}
    for index, label in enumerate(labels):
        mask = (velocities >= bounds[index]) & (velocities < bounds[index + 1])
        if not mask.any():
            output[label] = {"n": 0}
        else:
            output[label] = {
                "n": int(mask.sum()),
                f"Accin@{tau}": round(accin(pred_array[mask], gt_array[mask], tau), 4),
                "Frm.err": round(frm_err(pred_array[mask], gt_array[mask]), 4),
                "MAE_ms": round(
                    mae_ms(pred_array[mask], gt_array[mask], fps_array[mask]), 3
                ),
            }
    return output
