# -*- coding: utf-8 -*-
"""Semi-real drift benchmark: real views, real detector noise, exact timing labels.

Two webcams recorded by tools/record_webcams.py share the laptop's monotonic clock, so
after resampling both onto that host clock they are synchronized up to USB delivery
latency (a few ms). We then impose a known clock drift on one of them, exactly as in the
Panoptic drift protocol (alpha = 1 + d/N, beta = 0.5), and score how well each method
recovers delta(t) = (alpha-1) t + beta.  Unlike Panoptic, the keypoints come from
MediaPipe on real footage, so detector noise and self-occlusion are real.

    python scripts/realdata/semireal_webcam.py --take-dir captures/take1 \
        --pose-dir data/realdata/s1_t1/pose --cam-a webcam0 --cam-b webcam1 --start-sec 130
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation
from vsynclock.benchmark.drift import drift_offset, fit_drift, fit_drift_c2f, warp_stream
from vsynclock.core.types import KeypointSequence
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.model.panoptic_dataset import _fill_gaps

FPS = 30.0


def host_stream(take_dir, pose_dir, name, t_origin):
    xy = _fill_gaps(np.load(os.path.join(pose_dir, f"{name}.npz"))["xy"].astype(np.float64))[0]
    rows = list(csv.reader(open(os.path.join(take_dir, f"{name}_timestamps.csv"), encoding="utf-8-sig")))[1:]
    t = np.array([int(r[1]) for r in rows], dtype=np.float64) / 1e9 - t_origin
    grid = np.arange(np.ceil(t[0] * FPS), np.floor(t[-1] * FPS)) / FPS   # shared host grid
    src = np.interp(grid, t, np.arange(len(t)))
    return cubic_sample(xy, src, extrapolate=True), int(round(grid[0] * FPS))


def first_ts(take_dir, name):
    with open(os.path.join(take_dir, f"{name}_timestamps.csv"), encoding="utf-8-sig") as f:
        next(f)
        return int(next(f).split(",")[1]) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--take-dir", required=True)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--cam-a", default="webcam0")
    ap.add_argument("--cam-b", default="webcam1")
    ap.add_argument("--checkpoints", nargs="+",
                    default=["checkpoints/csf_b1_t20.pt", "checkpoints/csf_b1_t20_noisy.pt"])
    ap.add_argument("--start-sec", type=float, default=130)
    ap.add_argument("--end-sec", type=float, default=None, help="dev/test split boundary")
    ap.add_argument("--clip", type=int, default=200)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--drifts", type=float, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--per-clip", action="store_true")
    ap.add_argument("--gate", default=None,
                    help="observability_choice json: gate short windows by profile peak (c2f only)")
    ap.add_argument("--prior", choices=("long", "consensus"), default="long",
                    help="coarse-to-fine prior: long-window line, or best-supported candidate")
    ap.add_argument("--dump-clips", default="", help="write per-clip errors (JSON) for bootstrap CIs")
    ap.add_argument("--radius", type=float, default=3.0, help="c2f local search radius (frames)")
    ap.add_argument("--min-speed", type=float, default=1.0, help="skip near-static clips (px/frame)")
    a = ap.parse_args()
    import json
    GATE = ({k: v["threshold"] for k, v in json.load(open(a.gate)).items()} if a.gate else {})

    t0 = min(first_ts(a.take_dir, a.cam_a), first_ts(a.take_dir, a.cam_b))
    A, offA = host_stream(a.take_dir, a.pose_dir, a.cam_a, t0)
    B, offB = host_stream(a.take_dir, a.pose_dir, a.cam_b, t0)
    # align both onto absolute host frame index h: A[h - offA], B[h - offB]
    h0 = max(offA, offB, int(a.start_sec * FPS))
    h1 = min(offA + len(A), offB + len(B)) - 10
    if a.end_sec is not None:
        h1 = min(h1, int(a.end_sec * FPS))
    N, margin = a.clip, 6

    sys.path.insert(0, os.path.dirname(__file__))
    from realdata_eval import cc_profile, csf_predictor, csf_profile
    cc = CrossCorrelation(max_lag=8, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.985, 1.015), alpha_steps=31, max_lag=8)
    cc_p = lambda x, y: cc.predict(x, y).dt_frames
    preds = {os.path.basename(c).replace(".pt", ""): csf_predictor(c)
             for c in a.checkpoints if os.path.exists(c)}

    clips = []
    for s in range(h0, h1 - N - 2 * margin, N):
        a_clip = A[s - offA: s - offA + N]
        speed = np.nanmedian(np.linalg.norm(np.diff(a_clip, axis=0), axis=-1))
        if speed >= a.min_speed:
            clips.append(s)
    print(f"{a.cam_a} vs {a.cam_b}: {len(clips)} clips of {N} frames (host-synchronized pair)\n")

    profs = {os.path.basename(c).replace(".pt", ""): csf_profile(c)
             for c in a.checkpoints if os.path.exists(c)}
    ccp = cc_profile(8, a.features)
    names = (["CC-const", "Caspi-style", "CC-slide", "CC-c2f"] + [f"CSF-slide[{k}]" for k in preds]
             + [f"CSF-c2f[{k}]" for k in profs])
    res = {d: {m: [] for m in names} for d in a.drifts}
    idx = np.arange(N, dtype=np.float64)
    for d in a.drifts:
        alpha = 1.0 + d / N
        for s in clips:
            a_stream = A[s - offA: s - offA + N]
            b_full = B[s - offB - margin: s - offB + N + 3 * margin]
            b_stream = warp_stream(b_full, alpha, a.beta + margin, N)
            truth = drift_offset(idx, alpha, a.beta)
            err = lambda est: float(np.mean(np.abs(est - truth)))
            sa, sb = KeypointSequence(a_stream, FPS), KeypointSequence(b_stream, FPS)
            res[d].setdefault("_clip", []).append(s)
            res[d]["CC-const"].append(err(np.full(N, cc.predict(sa, sb).dt_frames)))
            e = caspi.estimate(sa, sb)
            res[d]["Caspi-style"].append(err((e.alpha - 1) * idx + e.beta))
            res[d]["CC-slide"].append(err(fit_drift(cc_p, a_stream, b_stream, FPS, a.window, a.stride)[0]))
            res[d]["CC-c2f"].append(err(fit_drift_c2f(ccp, a_stream, b_stream, FPS, a.window, a.stride,
                                                      radius=a.radius, prior=a.prior, min_peak=GATE.get("CC"))[0]))
            for k, p in preds.items():
                res[d][f"CSF-slide[{k}]"].append(err(fit_drift(p, a_stream, b_stream, FPS, a.window, a.stride)[0]))
            for k, p in profs.items():
                res[d][f"CSF-c2f[{k}]"].append(err(fit_drift_c2f(p, a_stream, b_stream, FPS, a.window,
                                                               a.stride, radius=a.radius, prior=a.prior, min_peak=GATE.get("CSF"))[0]))

    print("drift recovery error E_drift (frames), mean ± std over clips")
    print(f"{'drift':>5} | " + " | ".join(f"{m:>22}" for m in names))
    for d in a.drifts:
        print(f"{d:5.1f} | " + " | ".join(f"{np.mean(res[d][m]):9.3f} ± {np.std(res[d][m]):5.3f}      " for m in names))
    if a.dump_clips:
        import json
        json.dump({str(d): {m: res[d][m] for m in names} for d in a.drifts},
                  open(a.dump_clips, "w"), indent=1)
        print("wrote per-clip errors ->", a.dump_clips)
    if a.per_clip:
        print("\nper-clip error at the largest drift (start frame: " + ", ".join(names) + ")")
        d = a.drifts[-1]
        for k, s in enumerate(res[d]["_clip"]):
            print(f"{s:6d}: " + "  ".join(f"{res[d][m][k]:.2f}" for m in names))


if __name__ == "__main__":
    main()
