"""Small dependency-free cubic Hermite interpolation for uniformly sampled signals."""
from __future__ import annotations

import numpy as np


def cubic_sample(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    extrapolate: bool = False,
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    query = np.asarray(positions, dtype=np.float64)
    if samples.ndim < 1 or len(samples) < 2:
        raise ValueError("cubic_sample requires at least two source samples")
    if not extrapolate and (np.any(query < 0) or np.any(query > len(samples) - 1)):
        raise ValueError("query positions fall outside the source interval")

    result = np.empty((len(query), *samples.shape[1:]), dtype=np.float64)
    inside = (query >= 0) & (query <= len(samples) - 1)
    if inside.any():
        q = query[inside]
        left = np.floor(q).astype(int)
        left = np.minimum(left, len(samples) - 2)
        u = q - left
        derivatives = np.gradient(samples, axis=0, edge_order=2)
        reshape = (len(u),) + (1,) * (samples.ndim - 1)
        u = u.reshape(reshape)
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        result[inside] = (
            h00 * samples[left]
            + h10 * derivatives[left]
            + h01 * samples[left + 1]
            + h11 * derivatives[left + 1]
        )
    below = query < 0
    if below.any():
        shape = (int(below.sum()),) + (1,) * (samples.ndim - 1)
        result[below] = samples[0] + query[below].reshape(shape) * (
            samples[1] - samples[0]
        )
    above = query > len(samples) - 1
    if above.any():
        shape = (int(above.sum()),) + (1,) * (samples.ndim - 1)
        distance = (query[above] - (len(samples) - 1)).reshape(shape)
        result[above] = samples[-1] + distance * (samples[-1] - samples[-2])
    return result
