# -*- coding: utf-8 -*-
"""Panoptic-projected keypoint-pair dataset (benchmark B1). Task #16 (WS2).

Builds (kp_a, kp_b, Δt) training pairs from real CMU Panoptic 2D trajectories:
view A is a raw clip from one HD camera, view B is a *different* camera resampled
at ``t + Δt`` (cubic, no extrapolation) — the same cross-view construction the WS1
harness uses. Splits follow the WS1 config (by physical source sequence, never by
frame), so train/validation/test can never share motion.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..benchmark.detector_noise import apply_detector_noise, apply_mediapipe_like
from ..benchmark.interpolation import cubic_sample
from ..panoptic import load_sequence

# Sequence → split assignment, mirroring configs/ws1_final.yaml (WS1, leakage-safe).
SPLITS = {
    "train": ("171204_pose1_sample", "160906_band4", "160906_band1"),
    "validation": ("160906_ian5",),
    "test": ("160422_haggling1", "160226_haggling1"),
}
FPS = 29.97
# HD-camera nodes (panel 0). Dense enough that small/medium-angle pairs exist.
CAMERA_NODES = (0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28)


def _fill_gaps(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaNs per joint/coord; return (filled, per-frame-finite mask)."""
    finite_frame = np.isfinite(xy).all(axis=(1, 2))
    filled = xy.copy()
    time = np.arange(len(xy))
    for j in range(xy.shape[1]):
        for c in range(2):
            series = filled[:, j, c]
            valid = np.isfinite(series)
            if 0 < valid.sum() < len(series):
                filled[:, j, c] = np.interp(time, time[valid], series[valid])
    return filled, finite_frame


def _window_valid(mask: np.ndarray, length: int) -> np.ndarray:
    """valid[s] == True iff mask[s : s+length] is all-True."""
    if length > len(mask):
        return np.zeros(0, dtype=bool)
    cum = np.cumsum(np.concatenate(([0], mask.astype(np.int64))))
    return (cum[length:] - cum[:-length]) == length


def _person_ids(seq_dir: str, max_frames: int | None, sample_step: int = 25,
                min_presence: float = 0.5, max_persons: int = 4) -> list[int]:
    """Body ids present in ≥ ``min_presence`` of sampled pose frames."""
    files = sorted(glob.glob(
        os.path.join(seq_dir, "hdPose3d_stage1_coco19", "body3DScene_*.json")
    ))[:max_frames][::sample_step]
    counts: dict[int, int] = {}
    for path in files:
        with open(path, encoding="utf-8") as handle:
            frame = json.load(handle)
        for body in frame.get("bodies", []):
            counts[int(body["id"])] = counts.get(int(body["id"]), 0) + 1
    keep = [pid for pid, c in counts.items() if c >= min_presence * len(files)]
    keep.sort(key=lambda pid: -counts[pid])
    return keep[:max_persons] or sorted(counts, key=counts.get)[-1:]


def _pair_angles(seq_dir: str, keys: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Viewing-angle separation (deg) between camera pairs, about the dome center."""
    from ..panoptic import load_calibration
    cameras = load_calibration(seq_dir)
    centers = {}
    for key in keys:
        cam = cameras[key]
        R = np.asarray(cam["R"], dtype=np.float64)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
        centers[key] = -R.T @ t
    angles = {}
    for i, ki in enumerate(keys):
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            a, b = centers[ki], centers[kj]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            angles[(i, j)] = math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))
    return angles


class PanopticPairDataset(Dataset):
    """Cross-view (kp_a, kp_b, dt) pairs from Panoptic-projected 2D trajectories."""

    def __init__(self, root="data/panoptic_raw", split="train", n=4000, T=72,
                 max_offset=6.0, stride=12, seed=0, max_frames=None,
                 camera_nodes=CAMERA_NODES, occlusion_p=0.0,
                 min_velocity=30.0, max_pair_angle=60.0, augment=None,
                 noise_px=0.0, outlier_p=0.0, outlier_px=25.0, mediapipe=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
        self.n, self.T, self.max_offset = n, T, float(max_offset)
        self.seed, self.occlusion_p = seed, occlusion_p
        self.noise_px, self.outlier_p, self.outlier_px = noise_px, outlier_p, outlier_px
        self.mediapipe = mediapipe
        self.augment = (split == "train") if augment is None else augment
        margin = int(math.ceil(self.max_offset)) + 2  # cubic support + no extrapolation
        self.margin = margin

        # views[k] = list[(cam_name, filled_xy)] ; clips = (view_idx, start, pairs)
        self.views: list[list[tuple[str, np.ndarray]]] = []
        self.clips: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        skipped_static = 0
        for seq_id in SPLITS[split]:
            seq_dir = os.path.join(root, seq_id)
            keys = self._present_nodes(seq_dir, camera_nodes)
            # Pairs are restricted to nearby viewpoints: past ~60° the shared
            # motion signal collapses and cross-view training degenerates.
            angles = _pair_angles(seq_dir, sorted(keys))
            allowed = {p for p, deg in angles.items() if deg <= max_pair_angle}
            if not allowed:  # degenerate rig: keep the 3 closest pairs
                allowed = set(sorted(angles, key=angles.get)[:3])

            # Every sufficiently-present person is its own view group — band
            # sequences are mostly static per person, so one person is too little.
            for person_id in _person_ids(seq_dir, max_frames):
                cams = load_sequence(seq_dir, camera_keys=keys, fps=FPS,
                                     max_frames=max_frames, person_id=person_id)
                paired_cams = {c for pair in allowed for c in pair}
                filled, masks, speeds = [], [], []
                for c, (name, seq) in enumerate(sorted(cams.items())):
                    xy, mask = _fill_gaps(seq.xy)
                    masks.append(mask)
                    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (F-1,J)
                    speeds.append(np.median(step, axis=1) * FPS)         # px/s
                    if c not in paired_cams:
                        xy = np.empty((0, 0, 2))  # unused view: don't hold data
                    filled.append(
                        (f"{seq_id}/p{person_id}/{name}", xy.astype(np.float32))
                    )
                view_index = len(self.views)
                self.views.append(filled)

                F = len(masks[0])
                span = T + 2 * margin
                valid = [_window_valid(m, span) for m in masks]
                for s in range(0, max(0, F - span + 1), stride):
                    cam_ids = {c for c, v in enumerate(valid) if v[s]}
                    pairs = tuple(
                        p for p in allowed if p[0] in cam_ids and p[1] in cam_ids
                    )
                    if not pairs:
                        continue
                    # Near-static clips carry no alignment signal — drop them.
                    vel = float(np.mean([
                        np.median(speeds[c][s:s + span - 1]) for c in cam_ids
                    ]))
                    if vel < min_velocity:
                        skipped_static += 1
                        continue
                    self.clips.append((view_index, s + margin, pairs))
        if not self.clips:
            raise RuntimeError(
                f"no usable clips for split={split!r} under {root} "
                f"({skipped_static} static windows dropped) — check data / "
                "lower min_velocity, or run `finalize_ws1.py prepare` first"
            )

    @staticmethod
    def _present_nodes(seq_dir, camera_nodes):
        from ..panoptic import load_calibration
        available = load_calibration(seq_dir)
        keys = [(0, node) for node in camera_nodes if (0, node) in available]
        if len(keys) < 2:  # fall back to whatever HD cameras exist
            keys = sorted(k for k in available if k[0] == 0)[:6]
        return keys

    def __len__(self):
        return self.n

    @staticmethod
    def _affine(xy: np.ndarray, rng) -> np.ndarray:
        """Random static per-view similarity transform — Δt-invariant."""
        theta = rng.uniform(-25, 25) * math.pi / 180
        scale = rng.uniform(0.7, 1.3)
        rot = scale * np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
        center = xy.mean(axis=(0, 1), keepdims=True)
        out = (xy - center) @ rot.T + center + rng.uniform(-40, 40, size=(1, 1, 2))
        return out + rng.normal(0.0, 1.5, size=xy.shape)  # ~1.5px detector noise

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100_000 + i)
        view_index, start, pairs = self.clips[i % len(self.clips)]
        ia, ib = pairs[rng.integers(len(pairs))]
        if rng.random() < 0.5:  # both view orders
            ia, ib = ib, ia
        dt = float(rng.uniform(-self.max_offset, self.max_offset))

        xy_a = self.views[view_index][ia][1]
        xy_b = self.views[view_index][ib][1]
        # Interpolate on a local window only — the full sequence can be ~10k
        # frames and cubic_sample differentiates everything it is handed.
        lo, hi = start - self.margin, start + self.T + self.margin
        local_t = np.arange(self.margin, self.margin + self.T, dtype=np.float64)
        clip_a = xy_a[start:start + self.T].astype(np.float64)
        clip_b = cubic_sample(xy_b[lo:hi], local_t + dt)
        if self.augment:
            clip_a = self._affine(clip_a, rng)
            clip_b = self._affine(clip_b, rng)
        if self.noise_px or self.outlier_p:      # simulate 2D-detector noise per view
            clip_a = apply_detector_noise(clip_a, rng, self.noise_px, self.outlier_p, self.outlier_px)
            clip_b = apply_detector_noise(clip_b, rng, self.noise_px, self.outlier_p, self.outlier_px)
        if self.mediapipe:                       # MediaPipe-like failures, per view
            clip_a = apply_mediapipe_like(clip_a, rng)
            clip_b = apply_mediapipe_like(clip_b, rng)
        ka = torch.tensor(clip_a, dtype=torch.float32)
        kb = torch.tensor(clip_b, dtype=torch.float32)
        va = vb = torch.tensor([])
        if self.occlusion_p > 0:
            J = ka.shape[1]
            # Seed the mask from the per-sample rng (not global torch.rand) so a
            # given (seed, i) yields the SAME occlusion every run — required to
            # compare the standard vs occlusion-aware model on identical masks.
            va = torch.from_numpy(
                (rng.random((self.T, J)) > self.occlusion_p).astype(np.float32))
            vb = torch.from_numpy(
                (rng.random((self.T, J)) > self.occlusion_p).astype(np.float32))
        return ka, kb, torch.tensor(dt, dtype=torch.float32), va, vb
