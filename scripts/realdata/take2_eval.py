# -*- coding: utf-8 -*-
"""Take 2: V-SyncLock on five independently clocked devices, scored against flash ground truth.

Every camera is resampled on its own clock (iPhone container pts, webcam host timestamps)
to a uniform 30 Hz grid. The flash fit (data/take2/flash_gt.json, iphone1 reference) gives
for every camera c the line  t_c = t_ref + o_c(t_ref);  for a pair (a, b) the true lag in
frames at ref time t is 30 * (o_b(t) - o_a(t)).

Protocol = the TotalCapture one of the paper: non-overlapping 10-s windows of every camera
pair, one whole-take integer coarse offset per pair from motion-energy correlation (the
pipeline's coarse stage), then each estimator on the coarse-aligned window.

    python scripts/realdata/take2_eval.py --out data/take2/rows_L300.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from realdata_eval import cc_profile, coarse_offset, csf_predictor, load_host_times, motion_signal  # noqa: E402

from vsynclock.benchmark.baselines import CrossCorrelation  # noqa: E402
from vsynclock.benchmark.interpolation import cubic_sample  # noqa: E402
from vsynclock.core.types import KeypointSequence  # noqa: E402
from vsynclock.model.panoptic_dataset import _fill_gaps  # noqa: E402

FPS = 30.0
CAMS = ["iphone1", "iphone2", "iphone3", "webcam0", "webcam1"]


def load(take_dir, pose_dir, name):
    """-> xy on a uniform 30 Hz grid of the camera's own clock, visibility on that grid."""
    raw = np.load(os.path.join(pose_dir, f"{name}.npz"))["xy"].astype(np.float64)
    t = load_host_times(take_dir, name)
    if t is None:
        t = np.load(os.path.join(os.path.dirname(pose_dir), f"pts_{name}.npy"))
    t = t - t[0] if name.startswith("webcam") else t     # webcam t_rel starts at 0; pts as stored
    assert len(t) == len(raw), (name, len(t), len(raw))
    vis = np.isfinite(raw).all(-1)
    filled = _fill_gaps(raw)[0]
    grid = np.arange(np.ceil(t[0] * FPS), np.floor(t[-1] * FPS)) / FPS
    src = np.interp(grid, t, np.arange(len(t)))
    xy = cubic_sample(filled, src, extrapolate=True)
    lo, hi = np.floor(src).astype(int), np.minimum(np.ceil(src).astype(int), len(raw) - 1)
    return xy, vis[lo] & vis[hi], int(round(grid[0] * FPS))   # frame k of the array = clock frame k + k0


def gt_offset(gt, cam, t_ref):
    if cam == "iphone1":
        return 0.0 * t_ref
    g = gt[cam]
    return g["offset_mid_s"] + g["slope_ms_per_min"] / 6e4 * (t_ref - g["t_mid"])


def peak(lags, sc):
    i = int(np.argmax(sc))
    if 0 < i < len(sc) - 1:
        a, b, c = sc[i - 1], sc[i], sc[i + 1]
        den = a - 2 * b + c
        return float(lags[i] + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0))
    return float(lags[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take-dir", default="captures/take2")
    ap.add_argument("--pose-dir", default="data/realdata/s2_t2/pose")
    ap.add_argument("--gt", default="data/take2/flash_gt.json")
    ap.add_argument("--clip-len", type=int, default=300)
    ap.add_argument("--start-sec", type=float, default=30.0, help="skip the flash/calibration lead-in")
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--router-delta", type=float, default=3.0)
    ap.add_argument("--out", default="data/take2/rows_L300.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    gt = json.load(open(a.gt, encoding="utf-8"))
    data = {c: load(a.take_dir, a.pose_dir, c) for c in CAMS}
    for c in CAMS:
        print(f"{c:>8}: {len(data[c][0])} grid frames, full-body {data[c][1].all(1).mean():.0%}")

    est = {
        "CC-speed": lambda x, y: CrossCorrelation(8, "speed").predict(x, y).dt_frames,
        "CC": lambda x, y: CrossCorrelation(8, "speed+vyb").predict(x, y).dt_frames,
        "CSF-speed": csf_predictor("checkpoints/csf_b1_t20.pt"),
        "CSF": csf_predictor("checkpoints/csf_b1_t20_vy.pt"),
    }
    prof = cc_profile(8, "speed+vyb")
    n = a.clip_len
    rows = []
    for ca, cb in itertools.combinations(CAMS, 2):
        (A, va, ka), (B, vb, kb) = data[ca], data[cb]
        # pair GT on the ref clock -> frames of a's grid map to frames of b's grid
        t_ref_of_a = lambda fa: (fa + ka) / FPS - gt_offset(gt, ca, (fa + ka) / FPS)     # approx (o_a slowly varying)
        lag_true = lambda fa: FPS * (gt_offset(gt, cb, t_ref_of_a(fa)) - gt_offset(gt, ca, t_ref_of_a(fa))) + ka - kb
        sa, sb = motion_signal(A), motion_signal(B)
        L = coarse_offset(sa, sb, 400, int(a.start_sec * FPS), None)   # b[t + L] ~ a[t]
        true_mid = lag_true(len(A) / 2)
        print(f"{ca}-{cb}: coarse {L} vs truth {true_mid:.2f} frames")
        for s0 in range(int(a.start_sec * FPS), len(A) - n, n):
            if s0 + L < 0 or s0 + L + n + 8 > len(B):
                continue
            wa, wb = va[s0:s0 + n].mean(), vb[s0 + L:s0 + L + n].mean()
            if min(wa, wb) < a.min_valid:
                continue
            X, Y = A[s0:s0 + n], B[s0 + L:s0 + L + n]
            x, y = KeypointSequence(X, FPS), KeypointSequence(Y, FPS)
            truth = float(lag_true(s0 + n / 2) - L)       # residual after the coarse stage
            r = dict(pair=f"{ca}-{cb}", s0=int(s0), t_ref=float(t_ref_of_a(s0 + n / 2)), coarse=int(L),
                     truth=truth, valid=float(min(wa, wb)),
                     speed=float(np.nanmedian(np.linalg.norm(np.diff(X, axis=0), axis=-1))))
            for m, f in est.items():
                try:
                    r[m] = float(f(x, y))
                except Exception:
                    r[m] = float("nan")
            # router: temporal consistency of CC over the four quarters
            try:
                full = peak(*prof(x, y))
                q = n // 4
                qs = []
                for i in range(4):
                    try:
                        qs.append(peak(*prof(KeypointSequence(X[i * q:(i + 1) * q], FPS),
                                             KeypointSequence(Y[i * q:(i + 1) * q], FPS))))
                    except ValueError:
                        pass
                D = float(np.median(np.abs(np.array(qs) - full))) if qs else np.inf
            except ValueError:
                D = np.inf
            r["D"] = D
            r["Router"] = r["CSF"] if D >= a.router_delta else r["CC"]
            rows.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w", encoding="utf-8"))
    print(f"{len(rows)} rows -> {a.out}")
    # summary (sign checked: estimators report the lag in the TotalCapture convention)
    ms = list(est) + ["Router"]
    for sign in (+1, -1):
        e = {m: np.array([abs(r[m] - sign * r["truth"]) for r in rows]) for m in ms}
        print(f"sign {sign:+d}: " + "  ".join(f"{m} {np.nanmean(v):.3f} ({np.nanmean(v > 1):.0%}>1)" for m, v in e.items()))


if __name__ == "__main__":
    main()
