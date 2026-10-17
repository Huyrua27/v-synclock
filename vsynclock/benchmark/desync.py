"""Sub-frame keypoint-domain desynchronization with explicit source lineage."""
from __future__ import annotations

import math

import numpy as np

from ..core.types import KeypointSequence, SyncSample
from .interpolation import cubic_sample


def _validate_sequence(seq: KeypointSequence) -> None:
    if seq.T < 4:
        raise ValueError("at least four frames are required for cubic interpolation")
    if not np.isfinite(seq.xy).all():
        raise ValueError("sequence contains non-finite keypoints; clean or mask it first")


def _sample(seq: KeypointSequence, positions: np.ndarray) -> np.ndarray:
    _validate_sequence(seq)
    return cubic_sample(seq.xy, positions)


def inject_offset(seq: KeypointSequence, dt_frames: float) -> KeypointSequence:
    """Return a same-length shifted sequence for compatibility.

    Boundary values use linear extrapolation. Benchmark construction should use
    :func:`make_sample`, which crops to the common valid interval instead.
    """
    _validate_sequence(seq)
    src_t = np.arange(seq.T, dtype=np.float64)
    xy = cubic_sample(seq.xy, src_t + float(dt_frames), extrapolate=True)
    timestamps = (
        None
        if seq.timestamps is None
        else seq.timestamps + float(dt_frames) / float(seq.fps)
    )
    return KeypointSequence(
        xy,
        seq.fps,
        timestamps=timestamps,
        name=f"{seq.name}[{dt_frames:+.3f}f]",
    )


def make_sample(
    seq: KeypointSequence,
    dt_frames: float,
    *,
    source_sequence: str | None = None,
    velocity: float = 0.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> SyncSample:
    """Create a leakage-auditable pair without boundary extrapolation artifacts."""
    _validate_sequence(seq)
    dt = float(dt_frames)
    start = int(math.ceil(max(0.0, -dt)))
    stop = int(math.floor(min(seq.T - 1.0, seq.T - 1.0 - dt))) + 1
    if stop - start < 4:
        raise ValueError(f"offset {dt} leaves fewer than four common frames")

    base_t = np.arange(start, stop, dtype=np.float64)
    a_xy = _sample(seq, base_t)
    b_xy = _sample(seq, base_t + dt)
    if noise_std:
        rng = np.random.default_rng(seed)
        a_xy = a_xy + rng.normal(0.0, noise_std, a_xy.shape)
        b_xy = b_xy + rng.normal(0.0, noise_std, b_xy.shape)

    timestamps = base_t / float(seq.fps)
    source = source_sequence or seq.name
    meta = {
        "source_sequence": source,
        "source_start_frame": start,
        "source_stop_frame": stop,
    }
    return SyncSample(
        a=KeypointSequence(a_xy, seq.fps, timestamps=timestamps, name=f"{source}/A"),
        b=KeypointSequence(b_xy, seq.fps, timestamps=timestamps, name=f"{source}/B"),
        dt_gt_frames=dt,
        velocity=float(velocity),
        meta=meta,
    )


def make_pair(seq: KeypointSequence, dt_frames: float):
    sample = make_sample(seq, dt_frames)
    return sample.a, sample.b, sample.dt_gt_frames


def split_by_sequence(
    samples: list[SyncSample],
    test_ratio: float = 0.3,
    seed: int = 42,
    sequence_key: str = "source_sequence",
) -> tuple[list[SyncSample], list[SyncSample]]:
    """Split on immutable source-sequence IDs and reject unverifiable lineage."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be strictly between 0 and 1")
    missing = [index for index, sample in enumerate(samples) if sequence_key not in sample.meta]
    if missing:
        raise ValueError(f"samples missing meta[{sequence_key!r}]: {missing[:5]}")

    sequence_names = sorted({str(sample.meta[sequence_key]) for sample in samples})
    if len(sequence_names) < 2:
        raise ValueError("sequence-level split requires at least two source sequences")
    rng = np.random.default_rng(seed)
    rng.shuffle(sequence_names)
    n_test = min(len(sequence_names) - 1, max(1, round(len(sequence_names) * test_ratio)))
    test_names = set(sequence_names[:n_test])
    train_names = set(sequence_names[n_test:])
    if train_names & test_names:
        raise AssertionError("source-sequence leakage detected")

    train = [sample for sample in samples if sample.meta[sequence_key] in train_names]
    test = [sample for sample in samples if sample.meta[sequence_key] in test_names]
    for sample in train:
        sample.meta["split"] = "train"
    for sample in test:
        sample.meta["split"] = "test"
    return train, test
