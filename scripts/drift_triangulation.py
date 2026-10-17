# -*- coding: utf-8 -*-
"""Downstream 3D under DRIFT: does time-varying desync corrupt triangulation over a
take, and does sliding-window learned correction hold the 3D steady?  Task #24 / B3.

Non-reference cameras drift (obs_k[t] = clean_k(alpha_k*t + beta_k)). We triangulate
per frame after four corrections and report MPJPE (mm):

  naive     : no correction (drifted observations)
  CC-const  : one cross-correlation lag for the whole take (cannot track drift)
  CSF-slide : ContinuSyncFormer in short sliding windows + robust line fit  [ours]
  oracle    : the true per-frame offset

The key figure is MPJPE vs time: CC-const's error grows across the take as the drift
accumulates, while the sliding learned correction stays flat. Use --dump to write the
per-frame curves for plotting.

    python scripts/drift_triangulation.py --checkpoint checkpoints/csf_b1_t20.pt --window 20
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch

from vsynclock.benchmark.baselines import CrossCorrelation
from vsynclock.benchmark.detector_noise import apply_detector_noise
from vsynclock.benchmark.drift import fit_drift, fit_drift_c2f
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from vsynclock.panoptic import load_calibration, load_pose3d, project_to_2d
from vsynclock.panoptic.triangulation import mpjpe, projection_matrix, triangulate_masked

FPS = 29.97
FEATURES = "speed"          # set from --features in main()
DEFAULT_SEQUENCES = ("160422_haggling1", "160226_haggling1")


def csf_predictor(checkpoint):
    from vsynclock.model.continusyncformer import load_csf
    model = load_csf(checkpoint)

    def predict(a, b):
        ka = torch.tensor(a.xy, dtype=torch.float32)[None]
        kb = torch.tensor(b.xy, dtype=torch.float32)[None]
        with torch.no_grad():
            return float(model(ka, kb)["dt_hard"][0])
    return predict


def run(args):
    cc = CrossCorrelation(features=FEATURES)
    cc_predict = lambda a, b: cc.predict(a, b).dt_frames
    csf_predict = csf_predictor(args.checkpoint) if os.path.exists(args.checkpoint) else None

    drifts = [float(x) for x in args.drifts.split(",")]
    beta = args.beta
    N, W, S = args.clip_len, args.window, args.stride
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "realdata"))
    from realdata_eval import cc_profile, csf_profile
    cc_prof = cc_profile(cc.max_lag, FEATURES)
    csf_prof = csf_profile(args.checkpoint) if csf_predict else None
    methods = ["naive", "CC-const", "CC-slide", "CC-c2f", "oracle"] + \
              (["CSF-slide", "CSF-c2f"] if csf_predict else [])
    results = {d: {m: [] for m in methods} for d in drifts}
    curves = {m: [] for m in methods}          # per-frame MPJPE for the largest drift
    keys3 = [(0, n) for n in args.cameras]
    edge = int(math.ceil(max(abs(x) for x in drifts) + abs(beta))) + 2
    n_clips = 0

    for seq_id in args.sequences:
        seq_dir = os.path.join(args.root, seq_id)
        cams = load_calibration(seq_dir)
        keys = [k for k in keys3 if k in cams] or sorted(k for k in cams if k[0] == 0)[:3]
        xyz, _, _, _ = load_pose3d(seq_dir, max_frames=args.max_frames)
        mats = {k: projection_matrix(cams[k]["K"], cams[k]["R"], cams[k]["t"]) for k in keys}
        clean = {k: _fill_gaps(project_to_2d(xyz, cams[k]["K"], cams[k]["R"], cams[k]["t"]))[0]
                 for k in keys}
        if args.noise_px or args.outlier_p:      # realistic per-view detector noise
            nrng = np.random.default_rng(1234)
            clean = {k: apply_detector_noise(v, nrng, args.noise_px, args.outlier_p)
                     for k, v in clean.items()}
        finite = np.isfinite(xyz).all(axis=(1, 2))
        for k in keys:
            finite &= np.isfinite(clean[k]).all(axis=(1, 2))
        margin = edge + 2
        span = N + 2 * margin
        cum = np.cumsum(np.concatenate([[0], finite.astype(int)]))
        step = np.linalg.norm(np.diff(clean[keys[0]], axis=0), axis=-1)
        speed = np.concatenate([[0], np.median(step, axis=1)]) * FPS
        starts = [s for s in range(0, len(xyz) - span + 1, args.stride_clip)
                  if cum[s + span] - cum[s] == span
                  and float(np.median(speed[s + margin:s + margin + N])) >= args.min_speed]
        starts = starts[:args.max_clips]
        n_clips += len(starts)
        others = keys[1:]
        mat_list = [mats[k] for k in keys]
        score = slice(edge, N - edge)               # interior (avoid resample edges)
        truth_idx = np.arange(N)

        for s0 in starts:
            win = slice(s0, s0 + span)
            a_win = clean[keys[0]][win]
            ref_stream = a_win[margin:margin + N]
            truth = xyz[s0 + margin:s0 + margin + N]

            for d in drifts:
                alpha = 1.0 + d / N
                t = np.arange(N, dtype=np.float64)
                obs = {k: cubic_sample(clean[k][win], margin + alpha * t + beta,
                                       extrapolate=True) for k in others}
                delta_true = {k: (alpha - 1.0) * t + beta for k in others}
                # Per-camera per-frame offset estimates.
                est = {"naive": {k: np.zeros(N) for k in others},
                       "oracle": {k: delta_true[k] for k in others}}
                for k in others:
                    est.setdefault("CC-const", {})[k] = np.full(
                        N, cc.predict(KeypointSequence(ref_stream, FPS),
                                      KeypointSequence(obs[k], FPS)).dt_frames)
                    cc_curve, _, _ = fit_drift(cc_predict, ref_stream, obs[k], FPS, W, S)
                    est.setdefault("CC-slide", {})[k] = cc_curve
                    est.setdefault("CC-c2f", {})[k] = fit_drift_c2f(
                        cc_prof, ref_stream, obs[k], FPS, W, S, radius=3.0, prior=args.prior)[0]
                    if csf_predict:
                        curve, _, _ = fit_drift(csf_predict, ref_stream, obs[k], FPS, W, S)
                        est.setdefault("CSF-slide", {})[k] = curve
                        est.setdefault("CSF-c2f", {})[k] = fit_drift_c2f(
                            csf_prof, ref_stream, obs[k], FPS, W, S, radius=3.0, prior=args.prior)[0]

                def triangulate(method):
                    views = [ref_stream]
                    for k in others:
                        views.append(cubic_sample(obs[k], t - est[method][k],
                                                  extrapolate=True))
                    pts = np.stack(views, axis=2)          # (N,19,V,2)
                    valid = np.isfinite(pts).all(-1)
                    P = pts.reshape(-1, len(keys), 2)
                    X = triangulate_masked(P, mat_list, valid.reshape(-1, len(keys)))
                    return X.reshape(N, 19, 3)

                for m in methods:
                    est3d = triangulate(m)
                    results[d][m].append(mpjpe(est3d[score], truth[score]))
                    if d == drifts[-1]:
                        per = [mpjpe(est3d[i:i + 1], truth[i:i + 1]) for i in truth_idx]
                        curves[m].append(per)

    print(f"\nDrift triangulation MPJPE (mm), mean +/- std over clips -- {n_clips} clips, "
          f"clip_len={N}, window={W}, cams={args.cameras}")
    print(f"{'drift(fr)':>10} " + "".join(f"{m:>16}" for m in methods))
    for d in drifts:
        row = f"{d:>10.2f} "
        for m in methods:
            vals = [10.0 * v for v in results[d][m] if np.isfinite(v)]
            if vals:
                row += f"{np.mean(vals):>9.2f}+-{np.std(vals):<5.2f}"
            else:
                row += f"{'-':>16}"
        print(row)

    if args.dump_clips:
        import json
        json.dump({str(d): {m: [10.0 * v for v in results[d][m]] for m in methods} for d in drifts},
                  open(args.dump_clips, "w"), indent=1)
        print("wrote per-clip MPJPE ->", args.dump_clips)
    if not any(curves[m] for m in methods):
        print("\n(no clips matched -- relax --min-speed / raise --max-frames)")
        return
    # Drift-growth signature: MPJPE in the first vs last third at the largest drift.
    print(f"\nMPJPE over time at drift={drifts[-1]:.1f} (mm): first-third -> last-third")
    for m in methods:
        arr = 10.0 * np.nanmean(np.array(curves[m]), axis=0)
        third = len(arr) // 3
        print(f"  {m:>10}: {np.nanmean(arr[:third]):.2f} -> {np.nanmean(arr[-third:]):.2f}")

    if args.dump:
        out = {m: (10.0 * np.nanmean(np.array(curves[m]), axis=0)).tolist() for m in methods}
        import json
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"drift": drifts[-1], "mpjpe_mm_over_frame": out}, f, indent=2)
        print(f"\nwrote per-frame curves -> {args.dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--cameras", type=int, nargs="+", default=[2, 15, 26])
    ap.add_argument("--drifts", default="0,0.5,1.0,2.0")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--clip-len", type=int, default=200)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--stride-clip", type=int, default=240)
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--min-speed", type=float, default=40.0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--prior", choices=("long", "consensus"), default="long",
                    help="coarse-to-fine prior: long-window line, or best-supported candidate")
    ap.add_argument("--dump-clips", default="", help="write per-clip MPJPE (mm, JSON) for bootstrap CIs")
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--outlier-p", type=float, default=0.0)
    args = ap.parse_args()
    global FEATURES
    FEATURES = args.features
    run(args)


if __name__ == "__main__":
    main()
