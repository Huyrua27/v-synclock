# -*- coding: utf-8 -*-
"""AIST++ cross-view keypoint pairs (periodic dance motion for training).

Panoptic (social interaction, music) has almost no rhythmic motion, so a model trained on
it never sees the aliasing that periodic movement (gait, running, dance, combos) creates
one period away. AIST++ (Li et al., ICCV 2021; annotations CC BY 4.0) provides 3D COCO-17
keypoints of 1,408 dance sequences and 9 calibrated cameras per setting. We project the
3D keypoints into each camera (with its lens distortion), remap to the COCO-19 layout,
resample 60 -> 30 fps, and build (kp_a, kp_b, dt) pairs with exactly the same sampling,
augmentation and corruption as PanopticPairDataset (whose __getitem__ is reused).

Splits are dancer-disjoint (dNN in the sequence name); AIST++ is used for training and
validation only -- the real-capture test never touches it.

Data:  data/aistpp/{keypoints3d/*.pkl, cameras/setting*.json, cameras/mapping.txt}
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import re

import numpy as np

from .panoptic_dataset import PanopticPairDataset, _window_valid

SPLITS = {"train": range(1, 25), "validation": range(25, 28), "test": range(28, 31)}
SRC_FPS, FPS = 60, 30

# COCO-19 (Panoptic order) <- COCO-17 index, or midpoint of two.
_MID = {0: (5, 6), 2: (11, 12)}                                  # neck, body centre
_DIRECT = {1: 0, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 15, 9: 6, 10: 8, 11: 10,
           12: 12, 13: 14, 14: 16, 15: 1, 16: 3, 17: 2, 18: 4}


def coco17_to_coco19(k: np.ndarray) -> np.ndarray:
    out = np.empty(k.shape[:-2] + (19, k.shape[-1]), dtype=np.float64)
    for j, s in _DIRECT.items():
        out[..., j, :] = k[..., s, :]
    for j, (a, b) in _MID.items():
        out[..., j, :] = 0.5 * (k[..., a, :] + k[..., b, :])
    return out


def _project(X: np.ndarray, cam: dict) -> np.ndarray:
    """X (F, J, 3) world (cm) -> (F, J, 2) distorted pixels."""
    import cv2
    pts, _ = cv2.projectPoints(X.reshape(-1, 3).astype(np.float64),
                               np.asarray(cam["rotation"], dtype=np.float64),
                               np.asarray(cam["translation"], dtype=np.float64),
                               np.asarray(cam["matrix"], dtype=np.float64),
                               np.asarray(cam["distortions"], dtype=np.float64))
    xy = pts.reshape(X.shape[0], X.shape[1], 2)
    W, H = cam["size"]
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    xy[~inside] = np.nan                                       # out of frame -> missing
    return xy


def _centre(cam):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(cam["rotation"], dtype=np.float64))
    return -R.T @ np.asarray(cam["translation"], dtype=np.float64)


class AISTPairDataset(PanopticPairDataset):
    """Same pair construction as PanopticPairDataset, on AIST++ projections."""

    def __init__(self, root="data/aistpp", split="train", n=4000, T=20, max_offset=6.0,
                 stride=12, seed=0, occlusion_p=0.0, min_velocity=30.0,
                 max_pair_angle=110.0, augment=None, noise_px=0.0, outlier_p=0.0,
                 outlier_px=25.0, mediapipe=False, max_sequences=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
        self.n, self.T, self.max_offset = n, T, float(max_offset)
        self.seed, self.occlusion_p = seed, occlusion_p
        self.noise_px, self.outlier_p, self.outlier_px = noise_px, outlier_p, outlier_px
        self.mediapipe = mediapipe
        self.augment = (split == "train") if augment is None else augment
        self.margin = margin = int(math.ceil(self.max_offset)) + 2

        mapping = dict(line.split() for line in
                       open(os.path.join(root, "cameras", "mapping.txt"), encoding="utf-8")
                       if line.strip())
        cams_by_setting, angles_by_setting = {}, {}
        files = sorted(glob.glob(os.path.join(root, "keypoints3d", "*.pkl")))
        dancers = set(SPLITS[split])
        files = [f for f in files
                 if int(re.search(r"_d(\d+)_", os.path.basename(f)).group(1)) in dancers]
        if max_sequences:
            files = files[:max_sequences]

        self.views, self.clips = [], []
        for f in files:
            name = os.path.basename(f)[:-4]
            setting = mapping.get(name.replace("_cAll_", "_cAll_"), None)
            if setting is None:
                continue
            if setting not in cams_by_setting:
                cams = json.load(open(os.path.join(root, "cameras", f"{setting}.json")))
                centres = [_centre(c) for c in cams]
                ang = {}
                for i in range(len(cams)):
                    for j in range(i + 1, len(cams)):
                        a, b = centres[i], centres[j]   # about the stage origin
                        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                        ang[(i, j)] = math.degrees(math.acos(np.clip(cos, -1, 1)))
                cams_by_setting[setting] = cams
                angles_by_setting[setting] = {p for p, d in ang.items() if d <= max_pair_angle}
            cams, allowed = cams_by_setting[setting], angles_by_setting[setting]
            with open(f, "rb") as h:
                d = pickle.load(h)
            X = d.get("keypoints3d_optim", d["keypoints3d"])[::SRC_FPS // FPS]
            X = coco17_to_coco19(X)
            views, masks, speeds = [], [], []
            for c, cam in enumerate(cams):
                xy = _project(X, cam)
                mask = np.isfinite(xy).all(axis=(1, 2))
                step = np.linalg.norm(np.diff(np.nan_to_num(xy), axis=0), axis=-1)
                speeds.append(np.median(step, axis=1) * FPS)
                masks.append(mask)
                views.append((f"{name}/{cam['name']}", np.nan_to_num(xy).astype(np.float32)))
            view_index = len(self.views)
            self.views.append(views)
            F, span = len(X), T + 2 * margin
            valid = [_window_valid(m, span) for m in masks]
            for s in range(0, max(0, F - span + 1), stride):
                ids = {c for c, v in enumerate(valid) if v[s]}
                pairs = tuple(p for p in allowed if p[0] in ids and p[1] in ids)
                if not pairs:
                    continue
                vel = float(np.mean([np.median(speeds[c][s:s + span - 1]) for c in ids]))
                if vel < min_velocity:
                    continue
                self.clips.append((view_index, s + margin, pairs))
        if not self.clips:
            raise RuntimeError(f"no AIST++ clips for split={split!r} under {root}")
