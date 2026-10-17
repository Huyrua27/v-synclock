"""CMU Panoptic Studio body-pose loader and calibrated 3D-to-2D projection."""
from __future__ import annotations

import glob
import json
import os
from typing import Iterable

import numpy as np

from ..core.types import KeypointSequence


def project_to_2d(
    pts3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dist_coef: np.ndarray | None = None,
) -> np.ndarray:
    """Project world points shaped ``(..., 3)`` using the Panoptic camera model."""
    points = np.asarray(pts3d, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"pts3d must end in dimension 3, got {points.shape}")

    original_shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    camera = flat @ np.asarray(R, dtype=np.float64).T
    camera += np.asarray(t, dtype=np.float64).reshape(1, 3)

    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 0)
    normalized = np.full((len(flat), 2), np.nan, dtype=np.float64)
    normalized[valid] = camera[valid, :2] / z[valid, None]

    if dist_coef is not None:
        coeffs = np.zeros(5, dtype=np.float64)
        supplied = np.asarray(dist_coef, dtype=np.float64).reshape(-1)
        coeffs[: min(5, len(supplied))] = supplied[:5]
        x = normalized[:, 0].copy()
        y = normalized[:, 1].copy()
        r2 = x * x + y * y
        radial = 1 + coeffs[0] * r2 + coeffs[1] * r2**2 + coeffs[4] * r2**3
        normalized[:, 0] = (
            x * radial + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
        )
        normalized[:, 1] = (
            y * radial + coeffs[2] * (r2 + 2 * y * y) + 2 * coeffs[3] * x * y
        )

    homogeneous = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    pixels = homogeneous @ np.asarray(K, dtype=np.float64).T
    return pixels[:, :2].reshape(*original_shape, 2)


def _camera_key(camera: dict) -> tuple[int, int]:
    return int(camera["panel"]), int(camera["node"])


def load_calibration(seq_dir: str) -> dict[tuple[int, int], dict]:
    files = sorted(glob.glob(os.path.join(seq_dir, "calibration_*.json")))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one calibration_*.json in {seq_dir}, found {len(files)}"
        )
    with open(files[0], encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = {_camera_key(camera): camera for camera in payload.get("cameras", [])}
    if not cameras:
        raise ValueError(f"no cameras in {files[0]}")
    return cameras


def _select_person_id(frames: list[dict], requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    counts: dict[int, int] = {}
    for frame in frames:
        for body in frame.get("bodies", []):
            body_id = int(body["id"])
            counts[body_id] = counts.get(body_id, 0) + 1
    if not counts:
        raise ValueError("no bodies found in Panoptic pose frames")
    return max(counts, key=counts.get)


def load_pose3d(
    seq_dir: str,
    person_id: int | None = None,
    min_confidence: float = 0.1,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load COCO-19 poses as ``xyz, confidence, frame_indices, person_id``."""
    pattern = os.path.join(
        seq_dir, "hdPose3d_stage1_coco19", "body3DScene_*.json"
    )
    files = sorted(glob.glob(pattern))
    if max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"no pose frames matching {pattern}")

    frames = []
    frame_indices = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            try:
                frames.append(json.load(handle))
            except json.JSONDecodeError:
                # Some official archives ship truncated/empty frame files
                # (e.g. 160422_haggling1 frame 11542) — treat as a bodyless
                # frame so indices stay aligned instead of failing the load.
                frames.append({"bodies": []})
        frame_indices.append(int(os.path.splitext(os.path.basename(path))[0].split("_")[-1]))

    selected_id = _select_person_id(frames, person_id)
    xyz = np.full((len(frames), 19, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((len(frames), 19), dtype=np.float64)
    for index, frame in enumerate(frames):
        body = next(
            (item for item in frame.get("bodies", []) if int(item["id"]) == selected_id),
            None,
        )
        if body is None:
            continue
        joints = np.asarray(body["joints19"], dtype=np.float64).reshape(19, 4)
        confidence[index] = joints[:, 3]
        valid = joints[:, 3] >= min_confidence
        xyz[index, valid] = joints[valid, :3]
    return xyz, confidence, np.asarray(frame_indices), selected_id


def _interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    filled = np.asarray(values, dtype=np.float64).copy()
    time = np.arange(len(filled))
    for joint in range(filled.shape[1]):
        for coordinate in range(filled.shape[2]):
            series = filled[:, joint, coordinate]
            valid = np.isfinite(series)
            if valid.sum() < 2:
                continue
            candidate = np.interp(time, time[valid], series[valid])
            missing = ~valid
            starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
            ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
            for start, end in zip(starts, ends):
                internal = start > 0 and end < len(series) - 1
                if internal and end - start + 1 <= max_gap:
                    series[start : end + 1] = candidate[start : end + 1]
    return filled


def load_sequence(
    seq_dir: str,
    camera_keys: Iterable[tuple[int, int]] | None = None,
    person_id: int | None = None,
    fps: float = 29.97,
    min_confidence: float = 0.1,
    max_frames: int | None = None,
    max_interpolation_gap: int = 5,
) -> dict[str, KeypointSequence]:
    """Return synchronized HD-camera sequences for one Panoptic sequence."""
    cameras = load_calibration(seq_dir)
    if camera_keys is None:
        selected = [key for key in cameras if key[0] == 0]
    else:
        selected = [tuple(key) for key in camera_keys]
    missing = [key for key in selected if key not in cameras]
    if missing:
        raise KeyError(f"camera keys absent from calibration: {missing}")

    xyz, _, frame_indices, selected_id = load_pose3d(
        seq_dir,
        person_id=person_id,
        min_confidence=min_confidence,
        max_frames=max_frames,
    )
    outputs: dict[str, KeypointSequence] = {}
    timestamps = frame_indices.astype(np.float64) / float(fps)
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    for key in selected:
        camera = cameras[key]
        pixels = project_to_2d(
            xyz,
            camera["K"],
            camera["R"],
            camera["t"],
            camera.get("distCoef"),
        )
        pixels = _interpolate_short_gaps(pixels, max_interpolation_gap)
        name = camera.get("name") or f"{key[0]:02d}_{key[1]:02d}"
        outputs[name] = KeypointSequence(
            pixels,
            fps,
            timestamps=timestamps.copy(),
            name=f"{seq_name}/{name}/person-{selected_id}",
        )
    return outputs
