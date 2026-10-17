"""Shared evaluation harness for every synchronization method."""
from __future__ import annotations

from typing import Iterable

from ..core.interfaces import SyncMethod
from ..core.types import SyncSample
from . import metrics


def run(samples: list[SyncSample], methods: Iterable[SyncMethod]) -> dict:
    if not samples:
        raise ValueError("evaluation requires at least one sample")
    results = {}
    for method in methods:
        preds, gts, velocities, frame_rates = [], [], [], []
        for sample in samples:
            result = method.predict(sample.a, sample.b)
            preds.append(result.dt_frames)
            gts.append(sample.dt_gt_frames)
            velocities.append(sample.velocity)
            frame_rates.append(sample.a.fps)
        results[method.name] = (preds, gts, velocities, frame_rates)
    return results


def table(results: dict) -> str:
    columns = [
        "Method",
        "n",
        "Frm.err",
        "Accin@0.1",
        "Accin@0.25",
        "MAE_ms",
        "RMSE_ms",
    ]
    lines = ["  ".join(f"{column:>12}" for column in columns), "-" * 92]
    for name, (pred, gt, _, fps) in results.items():
        values = metrics.summary(pred, gt, fps)
        row = [
            name[:12],
            values["n"],
            values["Frm.err"],
            values["Accin@0.1"],
            values["Accin@0.25"],
            values["MAE_ms"],
            values["RMSE_ms"],
        ]
        lines.append("  ".join(f"{str(value):>12}" for value in row))
    return "\n".join(lines)


def bucket_table(results: dict, edges) -> str:
    lines = [
        "Accin@0.1 by velocity bucket (px/s); edges fitted on training data: "
        + ", ".join(f"{edge:.3f}" for edge in edges)
    ]
    for name, (pred, gt, velocity, fps) in results.items():
        buckets = metrics.by_velocity_bucket(pred, gt, velocity, fps, edges=edges)
        cells = "  ".join(
            f"{label}:{values.get('Accin@0.1', '-')}(n{values['n']})"
            for label, values in buckets.items()
        )
        lines.append(f"  {name:14} {cells}")
    return "\n".join(lines)
