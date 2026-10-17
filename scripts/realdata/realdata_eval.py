# -*- coding: utf-8 -*-
"""Real multi-device evaluation without timing ground truth (Sec. 6 of the paper).

Pipeline per take:
  1. Load COCO-19 2D poses (extract_pose2d.py). Cameras with a host-timestamp CSV
     (webcams from record_webcams.py) are first resampled onto a uniform 30-fps host
     timeline -- a real system uses timestamps when a device provides them; V-SyncLock only has
     to remove what timestamps cannot (independent device clocks, unknown start).
  2. Coarse integer alignment of every camera to the reference by global motion
     cross-correlation.
  3. Each method estimates the residual sub-frame offset trajectory delta_c(t).
  4. Resample, undistort, triangulate (DLT over the views where the joint was really
     detected -- gap-filled joints are never triangulated), and score:
       * reproj  : mean reprojection error over all views (px)
       * LOO     : leave-one-camera-out transfer error -- triangulate from the other
                   views, project into the held-out camera (px). More sensitive to sync.
     both overall and on the fastest quartile of frames, where desync matters most.

    python scripts/realdata/realdata_eval.py --take-dir captures/take1 \
        --pose-dir data/realdata/s1_t1/pose --calib data/realdata/s1_t1/calib.json \
        --ref iphone_cam1 --start-sec 130
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import cv2
import torch

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation
from vsynclock.benchmark.drift import fit_drift, fit_drift_c2f
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from vsynclock.panoptic.triangulation import projection_matrix, triangulate_masked

FPS = 30.0


# ----------------------------------------------------------------------------- loading
def load_host_times(take_dir, name):
    path = os.path.join(take_dir, f"{name}_timestamps.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))[1:]
    return np.array([int(r[1]) for r in rows], dtype=np.float64) / 1e9


def load_camera(pose_dir, take_dir, name):
    """-> (filled xy (T,J,2), visible mask (T,J), used_host_ts)."""
    raw = np.load(os.path.join(pose_dir, f"{name}.npz"))["xy"].astype(np.float64)
    vis = np.isfinite(raw).all(-1)
    filled = _fill_gaps(raw)[0]
    t = load_host_times(take_dir, name)
    if t is None or len(t) != len(raw):
        return filled, vis, False
    t = t - t[0]
    src = np.interp(np.arange(0, t[-1], 1.0 / FPS), t, np.arange(len(t)))  # fractional idx
    xy = cubic_sample(filled, src, extrapolate=True)
    lo, hi = np.floor(src).astype(int), np.minimum(np.ceil(src).astype(int), len(raw) - 1)
    return xy, vis[lo] & vis[hi], True


# ---------------------------------------------------------------------- coarse alignment
def motion_signal(xy):
    s = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    v = np.concatenate([[0], np.nanmedian(s, axis=1)])
    v = np.nan_to_num(v)
    return (v - v.mean()) / (v.std() + 1e-9)


def coarse_offset(ref_sig, sig, max_lag, r0=0, r1=None):
    """Integer lag L maximizing correlation of ref[t] with sig[t + L], t in [r0, r1).

    Restricting t to the performance segment matters: the calibration part has several
    people moving, which yields spurious peaks."""
    best, blag = -np.inf, 0
    r1 = len(ref_sig) if r1 is None else r1
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(r0, -lag), min(r1, len(sig) - lag)
        if hi - lo < 300:
            continue
        c = float(np.dot(ref_sig[lo:hi], sig[lo + lag:hi + lag]) / (hi - lo))
        if c > best:
            best, blag = c, lag
    return blag


# --------------------------------------------------------------------------- geometry
def undistort(xy, cam):
    K = np.asarray(cam["K"]); d = np.asarray(cam["dist"])
    flat = xy.reshape(-1, 1, 2)
    ok = np.isfinite(flat).all(-1).ravel()
    out = np.full_like(flat, np.nan)
    if ok.any():
        out[ok] = cv2.undistortPoints(flat[ok].astype(np.float64), K, d, P=K)
    return out.reshape(xy.shape)


def project(X, P):
    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1) @ P.T
    return Xh[:, :2] / Xh[:, 2:3]


def evaluate(views, valid, mats, fast):
    """views (T,J,V,2) undistorted px; valid (T,J,V). -> dict of errors (px)."""
    T, J, V, _ = views.shape
    P = views.reshape(-1, V, 2); M = valid.reshape(-1, V)
    X = triangulate_masked(P, mats, M)
    fast_rows = np.repeat(fast, J)
    rep, rep_f, loo, loo_f = [], [], [], []
    for v in range(V):
        e = np.linalg.norm(project(X, mats[v]) - P[:, v], axis=-1)
        ok = M[:, v] & np.isfinite(e)
        rep.append(e[ok]); rep_f.append(e[ok & fast_rows])
        keep = M.copy(); keep[:, v] = False
        Xo = triangulate_masked(P, mats, keep)
        e = np.linalg.norm(project(Xo, mats[v]) - P[:, v], axis=-1)
        ok = M[:, v] & np.isfinite(e)
        loo.append(e[ok]); loo_f.append(e[ok & fast_rows])
    cat = lambda z: float(np.mean(np.concatenate(z)))
    med = lambda z: float(np.median(np.concatenate(z)))
    return {"reproj": cat(rep), "reproj_fast": cat(rep_f), "loo": cat(loo), "loo_fast": cat(loo_f),
            "loo_med": med(loo), "loo_fast_med": med(loo_f),
            "per_cam_loo": [float(np.median(z)) for z in loo]}


# ---------------------------------------------------------------------------- methods
def _finite(xy):
    """(T,J,2) -> (1,T,J,2) float tensor with missing joints gap-filled, as CC does
    internally; a NaN reaching the network would corrupt the whole lag profile."""
    xy = np.asarray(xy, dtype=np.float64)
    if not np.isfinite(xy).all():
        xy = np.nan_to_num(_fill_gaps(xy)[0])
    return torch.tensor(xy, dtype=torch.float32)[None]


def csf_predictor(checkpoint):
    from vsynclock.model.continusyncformer import load_csf
    model = load_csf(checkpoint)

    def predict(a, b):
        with torch.no_grad():
            return float(model(_finite(a.xy), _finite(b.xy))["dt_hard"][0])
    return predict


def csf_profile(checkpoint):
    """CSF's learned correlation profile over integer lags (for coarse-to-fine search)."""
    from vsynclock.model.continusyncformer import load_csf
    model = load_csf(checkpoint)
    lags = model.head.centers.numpy().astype(np.float64)

    def profile(a, b):
        with torch.no_grad():
            corr = model(_finite(a.xy), _finite(b.xy))["corr"][0]
        return lags, corr.numpy().astype(np.float64)
    return profile


def cc_profile(max_lag=8, features="speed"):
    from vsynclock.benchmark.baselines.cross_correlation import lag_scores, motion_features
    return lambda a, b: lag_scores(motion_features(a, features), motion_features(b, features), max_lag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--take-dir", required=True, help="raw capture dir (for *_timestamps.csv)")
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--start-sec", type=float, default=130, help="motion segment start (ref time)")
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-lag", type=int, default=400)
    ap.add_argument("--no-host-ts", action="store_true", help="ignore webcam timestamps")
    ap.add_argument("--prior", choices=("long", "consensus"), default="long",
                    help="coarse-to-fine prior: long-window line, or best-supported candidate")
    ap.add_argument("--out", default=None, help="write per-method results JSON here")
    a = ap.parse_args()

    calib = json.load(open(a.calib, encoding="utf-8"))["cameras"]
    cams = [a.ref] + sorted(c for c in calib if c != a.ref)
    data = {}
    for c in cams:
        xy, vis, used = load_camera(a.pose_dir, "" if a.no_host_ts else a.take_dir, c)
        data[c] = (xy, vis)
        print(f"{c:>12}: {len(xy)} frames{'  (resampled on host timestamps)' if used else ''}")

    s0 = int(a.start_sec * FPS)
    s1 = len(data[a.ref][0]) if a.end_sec is None else int(a.end_sec * FPS)
    sig = {c: motion_signal(data[c][0]) for c in cams}
    coarse = {a.ref: 0}
    for c in cams[1:]:
        coarse[c] = coarse_offset(sig[a.ref], sig[c], a.max_lag, s0, s1)
    print("coarse offsets vs ref (frames):", {c: coarse[c] for c in cams[1:]})
    for c in cams:
        s0 = max(s0, -coarse[c] + 3)
        s1 = min(s1, len(data[c][0]) - coarse[c] - 3)
    idx = np.arange(s0, s1, dtype=np.float64)
    n = len(idx)
    print(f"evaluation segment: ref frames {s0}-{s1} ({n / FPS:.1f}s)\n")

    ref = data[a.ref][0][s0:s1]
    base = {c: cubic_sample(data[c][0], idx + coarse[c], extrapolate=True) for c in cams[1:]}

    cc = CrossCorrelation(max_lag=8, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.997, 1.003), alpha_steps=61, max_lag=8)
    csf = csf_predictor(a.checkpoint)
    csfp, ccp = csf_profile(a.checkpoint), cc_profile(8, a.features)
    cc_p = lambda x, y: cc.predict(x, y).dt_frames
    seq = lambda z: KeypointSequence(z.astype(np.float64), FPS)
    t = np.arange(n, dtype=np.float64)
    methods = {
        "Coarse only": lambda b: np.zeros(n),
        "CC-const": lambda b: np.full(n, cc.predict(seq(ref), seq(b)).dt_frames),
        "Caspi-style": lambda b: (lambda e: (e.alpha - 1) * t + e.beta)(caspi.estimate(seq(ref), seq(b))),
        "CC-slide": lambda b: fit_drift(cc_p, ref, b, FPS, a.window, a.stride)[0],
        "CC-c2f": lambda b: fit_drift_c2f(ccp, ref, b, FPS, a.window, a.stride, prior=a.prior)[0],
        "CSF-slide": lambda b: fit_drift(csf, ref, b, FPS, a.window, a.stride)[0],
        "CSF-c2f": lambda b: fit_drift_c2f(csfp, ref, b, FPS, a.window, a.stride, prior=a.prior)[0],
    }

    # visibility on the ref timeline (nearest source frames)
    def vis_on(c, extra):
        v = data[c][1]
        src = idx + coarse.get(c, 0) + extra
        lo = np.clip(np.floor(src).astype(int), 0, len(v) - 1)
        hi = np.clip(np.ceil(src).astype(int), 0, len(v) - 1)
        return v[lo] & v[hi]

    mats = [projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in cams]
    speed = np.nanmedian(np.linalg.norm(np.diff(ref, axis=0, prepend=ref[:1]), axis=-1), axis=1)
    fast = speed >= np.nanpercentile(speed, 75)

    results = {}
    print(f"{'method':>12} | {'reproj':>7} {'fast':>7} | {'LOO':>7} {'fast':>7} | "
          f"{'LOOmed':>7} {'fast':>7} | per-camera drift (frames/min)")
    for name, fn in methods.items():
        streams, valids, slopes = [undistort(ref, calib[a.ref])], [data[a.ref][1][s0:s1]], {}
        for c in cams[1:]:
            d = fn(base[c])
            slopes[c] = float((d[-1] - d[0]) / (n / FPS / 60))
            streams.append(undistort(cubic_sample(base[c], t - d, extrapolate=True), calib[c]))
            valids.append(vis_on(c, -d))
        views = np.stack(streams, axis=2); valid = np.stack(valids, axis=2)
        r = evaluate(views, valid, mats, fast)
        r["drift_fpm"] = slopes
        results[name] = r
        sl = " ".join(f"{c.split('_')[-1]}:{s:+.2f}" for c, s in slopes.items())
        print(f"{name:>12} | {r['reproj']:7.2f} {r['reproj_fast']:7.2f} | {r['loo']:7.2f} {r['loo_fast']:7.2f} | "
              f"{r['loo_med']:7.2f} {r['loo_fast_med']:7.2f} | {sl}")
    print("\nmedian LOO per held-out camera (px):", cams)
    for name, r in results.items():
        print(f"{name:>12}: " + " ".join(f"{e:7.2f}" for e in r["per_cam_loo"]))
    if a.out:
        json.dump({"cams": cams, "coarse": coarse, "segment": [s0, s1], "results": results},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
