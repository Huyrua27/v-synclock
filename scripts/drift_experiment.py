# -*- coding: utf-8 -*-
"""Drift experiment: recover a time-varying (affine) inter-camera offset.

Camera B drifts: obs_b[t] = clean_b(alpha*t + beta), so the offset grows across the
take. We compare how well each method recovers the offset function delta(t):

  CC-const   : one cross-correlation lag over the whole clip (ignores drift)
  Caspi-Irani: classical affine grid search (alpha, beta)
  CC-slide   : cross-correlation in short sliding windows + robust line fit
  CSF-slide  : ContinuSyncFormer in short sliding windows + robust line fit  [ours]

Metric: mean |delta_est(t) - delta_true(t)| over the take, in frames (lower better).
Because CC fails on short windows, CC-slide's line is corrupted by outliers; the
learned estimator stays reliable, so CSF-slide should win in the drift regime that
a single lag cannot represent at all.

    python scripts/drift_experiment.py --checkpoint checkpoints/csf_b1_t32.pt --window 32
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

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation
from vsynclock.benchmark.detector_noise import apply_detector_noise
from vsynclock.benchmark.drift import drift_offset, fit_drift, fit_drift_c2f, warp_stream
from vsynclock.core.types import KeypointSequence
from vsynclock.panoptic import load_calibration, load_pose3d, project_to_2d
from vsynclock.model.panoptic_dataset import _fill_gaps

FPS = 29.97
FEATURES = "speed"          # set from --features in main()
DEFAULT_SEQUENCES = ("160422_haggling1", "160226_haggling1")


def nearby_pair(cams):
    """A moderate-baseline in-distribution camera pair (for the sync task)."""
    keys = sorted(k for k in cams if k[0] == 0)
    return keys[0], keys[5] if len(keys) > 5 else keys[1]


def csf_predictor(checkpoint):
    from vsynclock.model.continusyncformer import load_csf
    model = load_csf(checkpoint)

    def predict(a: KeypointSequence, b: KeypointSequence):
        ka = torch.tensor(a.xy, dtype=torch.float32)[None]
        kb = torch.tensor(b.xy, dtype=torch.float32)[None]
        with torch.no_grad():
            return float(model(ka, kb)["dt_hard"][0])
    return predict


def caspi_curve(caspi, a_stream, b_stream, n):
    a = KeypointSequence(a_stream.astype(np.float64), FPS, name="a")
    b = KeypointSequence(b_stream.astype(np.float64), FPS, name="b")
    est = caspi.estimate(a, b)
    idx = np.arange(n, dtype=np.float64)
    return (est.alpha - 1.0) * idx + est.beta


def run(args):
    import json
    GATE = ({k: v["threshold"] for k, v in json.load(open(args.gate)).items()} if args.gate else {})
    cc = CrossCorrelation(features=FEATURES)
    caspi = CaspiIrani(alpha_range=(0.985, 1.015), alpha_steps=31, max_lag=12)
    cc_predict = lambda a, b: cc.predict(a, b).dt_frames
    csf_predict = csf_predictor(args.checkpoint) if os.path.exists(args.checkpoint) else None

    drifts = [float(x) for x in args.drifts.split(",")]   # total drift over the clip
    beta = args.beta
    N, W, S = args.clip_len, args.window, args.stride
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "realdata"))
    from realdata_eval import cc_profile, csf_profile
    cc_prof = cc_profile(cc.max_lag, FEATURES)
    csf_prof = csf_profile(args.checkpoint) if csf_predict else None
    methods = (["CC-const", "Caspi-Irani", "CC-slide", "CC-c2f"]
               + (["CSF-slide", "CSF-c2f"] if csf_predict else []))
    results = {d: {m: [] for m in methods} for d in drifts}
    n_clips = 0

    for seq_id in args.sequences:
        seq_dir = os.path.join(args.root, seq_id)
        cams = load_calibration(seq_dir)
        ka_key, kb_key = nearby_pair(cams)
        xyz, _, _, _ = load_pose3d(seq_dir, max_frames=args.max_frames)
        ca = cams[ka_key]; cb = cams[kb_key]
        clean_a, _ = _fill_gaps(project_to_2d(xyz, ca["K"], ca["R"], ca["t"]))
        clean_b, _ = _fill_gaps(project_to_2d(xyz, cb["K"], cb["R"], cb["t"]))
        if args.noise_px or args.outlier_p:      # realistic per-view detector noise
            nrng = np.random.default_rng(1234)
            clean_a = apply_detector_noise(clean_a, nrng, args.noise_px, args.outlier_p)
            clean_b = apply_detector_noise(clean_b, nrng, args.noise_px, args.outlier_p)
        finite = np.isfinite(clean_a).all(axis=(1, 2)) & np.isfinite(clean_b).all(axis=(1, 2))
        margin = int(math.ceil(max(abs(d) for d in drifts) + abs(beta))) + 3
        span = N + 2 * margin
        cum = np.cumsum(np.concatenate([[0], finite.astype(int)]))
        step = np.linalg.norm(np.diff(clean_a, axis=0), axis=-1)
        speed = np.concatenate([[0], np.median(step, axis=1)]) * FPS
        starts = [s for s in range(0, len(xyz) - span + 1, args.stride_clip)
                  if cum[s + span] - cum[s] == span
                  and float(np.median(speed[s + margin:s + margin + N])) >= args.min_speed]
        starts = starts[:args.max_clips]
        n_clips += len(starts)

        for s0 in starts:
            win = slice(s0, s0 + span)                 # s0 already includes margin
            a_stream = clean_a[win][margin:margin + N]
            idx = np.arange(N, dtype=np.float64)
            for d in drifts:
                alpha = 1.0 + d / N
                # obs_b on the clip grid: local index maps into the window source.
                b_stream = warp_stream(clean_b[win], alpha, beta + margin,
                                       length=N, start=0)
                delta_true = drift_offset(idx, alpha, beta)

                def err(delta_est):
                    return float(np.mean(np.abs(delta_est - delta_true)))

                results[d]["CC-const"].append(
                    err(np.full(N, cc.predict(
                        KeypointSequence(a_stream, FPS), KeypointSequence(b_stream, FPS)
                    ).dt_frames)))
                results[d]["Caspi-Irani"].append(
                    err(caspi_curve(caspi, a_stream, b_stream, N)))
                est, _, _ = fit_drift(cc_predict, a_stream, b_stream, FPS, W, S)
                results[d]["CC-slide"].append(err(est))
                est, _, _ = fit_drift_c2f(cc_prof, a_stream, b_stream, FPS, W, S, radius=args.radius,
                                       prior=args.prior, min_peak=GATE.get("CC"))
                results[d]["CC-c2f"].append(err(est))
                if csf_predict:
                    est, _, _ = fit_drift(csf_predict, a_stream, b_stream, FPS, W, S)
                    results[d]["CSF-slide"].append(err(est))
                    est, _, _ = fit_drift_c2f(csf_prof, a_stream, b_stream, FPS, W, S,
                                              radius=args.radius, prior=args.prior, min_peak=GATE.get("CSF"))
                    results[d]["CSF-c2f"].append(err(est))

    print(f"\nDrift recovery -- mean +/- std over clips of |delta_est - delta_true| "
          f"(frames), lower better")
    print(f"{n_clips} clips, clip_len={N}, window={W}, stride={S}, beta={beta}")
    print(f"{'drift(fr)':>10} " + "".join(f"{m:>18}" for m in methods))
    for d in drifts:
        row = f"{d:>10.2f} "
        for m in methods:
            vals = [v for v in results[d][m] if np.isfinite(v)]
            if vals:
                row += f"{np.mean(vals):>10.4f}+-{np.std(vals):<6.4f}"
            else:
                row += f"{'-':>18}"
        print(row)
    print("\n(drift = total offset change across the clip; CC-const cannot track it.)")
    if args.dump_clips:
        import json
        json.dump({str(d): results[d] for d in drifts}, open(args.dump_clips, "w"), indent=1)
        print("wrote per-clip errors ->", args.dump_clips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t32.pt")
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--drifts", default="0,0.5,1.0,2.0")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--clip-len", type=int, default=200)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--stride-clip", type=int, default=240)
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--min-speed", type=float, default=40.0)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--outlier-p", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=3.0, help="coarse-to-fine search radius")
    ap.add_argument("--gate", default=None,
                    help="observability_choice json: gate short windows by profile peak (c2f only)")
    ap.add_argument("--prior", choices=("long", "consensus"), default="long",
                    help="coarse-to-fine prior: long-window line, or best-supported candidate")
    ap.add_argument("--dump-clips", default="", help="write per-clip errors (JSON) for bootstrap CIs")
    args = ap.parse_args()
    global FEATURES
    FEATURES = args.features
    run(args)


if __name__ == "__main__":
    main()
