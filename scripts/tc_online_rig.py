# -*- coding: utf-8 -*-
"""Streaming V-SyncLock (vsynclock.online.StreamingSynchronizer) on the eight-camera TotalCapture rig.

Each test sequence (first --max-seconds, 30 fps after decimation) is replayed in one-second
chunks. Every non-reference camera gets its own clock: delta_c(t) = beta_c + r_c t with beta_c
uniform in [-1.5, 1.5] frames and r_c uniform in [-3, 3] frames/min (fixed seed), applied exactly
by cubic resampling of the 60-fps keypoints. After every refit the current lines correct the
frames that arrive next (causal); the same synchronizer run once on the whole sequence gives
the offline reference.

Reported: mean |offset error| of placed cameras by elapsed time, share of cameras placed, MPJPE
of the triangulation of the placed cameras against the genlocked triangulation of all eight
(after the first fit), and the wall-clock time of each refit (one CPU thread).

    python scripts/tc_online_rig.py --out docs/paper/clips/vy/tc_online_rig.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.benchmark.surrey_calib import read_surrey
from vsynclock.model.panoptic_dataset import _fill_gaps
from vsynclock.online import StreamingSynchronizer, cc_profile
from vsynclock.panoptic.triangulation import mpjpe, projection_matrix, triangulate_masked
from tc_benchmark import load_sequences

FPS = 30.0
BINS = [(0, 10), (10, 30), (30, 60), (60, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--max-seconds", type=float, default=120)
    ap.add_argument("--features", default="speed+vyb")
    ap.add_argument("--priors", type=float, nargs="+", default=[0, 2], help="0 = no slope prior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first n sequences (smoke test)")
    ap.add_argument("--variant", choices=("B0", "fix", "vote"), default="fix",
                    help="B0: cache as first computed, all pairs always; fix: stale-window re-centring + inactive cameras")
    ap.add_argument("--drift-max", type=float, default=3.0, help="per-camera drift uniform in +-this (frames/min)")
    ap.add_argument("--lead-in", type=float, default=0.0,
                    help="seconds of near-static lead-in (first pose + 1 px jitter) before the motion (stress test)")
    ap.add_argument("--lead-mode", choices=("static", "desync"), default="static",
                    help="static: first pose + 1 px jitter; desync: real motion from a different random time per camera")
    ap.add_argument("--stop-cams", type=int, default=0, help="number of non-reference cameras that stop recording")
    ap.add_argument("--stop-at", type=float, default=40.0, help="... after this many seconds of the motion")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    VARIANT = dict(recenter=a.variant == "fix", inactive_after=5.0 if a.variant == "fix" else None,
                   vote=a.variant == "vote")
    os.chdir(ROOT)
    calib = read_surrey(os.path.join(a.kp_dir, "calibration.cal"))
    seqs = load_sequences(a.kp_dir)
    rng = np.random.default_rng(a.seed)
    prof = cc_profile(a.features)
    modes = [f"{k}_p{p:g}" for p in a.priors for k in ("causal", "offline")] + ["uncorrected"]
    res = {m: {"bins": [[] for _ in BINS], "placed": [], "mpjpe": []} for m in modes}
    timing = {f"p{p:g}": [] for p in a.priors}
    recomputed = {}
    for (subj, seq), cams in sorted(seqs.items())[:a.limit or None]:
        cl = sorted(cams)
        if len(cl) != 8:
            continue
        ref, others = cl[0], cl[1:]
        T60 = min(len(cams[c]) for c in cl)
        beta = {c: rng.uniform(-1.5, 1.5) for c in others}
        rate = {c: rng.uniform(-a.drift_max, a.drift_max) / (60 * FPS) for c in others}
        stopped = set(rng.choice(others, a.stop_cams, replace=False).tolist()) if a.stop_cams else set()
        M = 20                                                  # 60-fps margin for |delta| <= 7.5 frames
        N = min(int((T60 - 2 * M) / 2) - 8, int(a.max_seconds * FPS))
        t = np.arange(N, dtype=np.float64)
        truth = {c: beta[c] + rate[c] * t for c in others}
        filled = {c: _fill_gaps(cams[c])[0] for c in cl}
        valid = {c: np.isfinite(cams[c]).all(-1) for c in cl}
        obs = {ref: filled[ref][M:M + 2 * N:2]}
        for c in others:
            obs[c] = cubic_sample(filled[c], M + 2 * (t + truth[c]), extrapolate=True)
        mats = {c: projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in cl}
        vmask = {c: valid[c][M:M + 2 * N:2] for c in cl}
        # stress tests: a near-static lead-in (no timing information) and cameras that stop recording
        Lf = int(a.lead_in * FPS)
        if Lf or stopped:
            jit = np.random.default_rng(a.seed + 1)
            for c in cl:
                o = obs[c].copy()
                if c in stopped:
                    o[int(a.stop_at * FPS):] = np.nan
                if a.lead_mode == "static":
                    lead = np.repeat(o[:1], Lf, 0) + jit.normal(0, 1.0, (Lf,) + o.shape[1:])
                else:                                            # misleading: real motion, no common timing
                    s_c = int(jit.integers(0, max(1, len(o) - Lf)))
                    lead = o[s_c:s_c + Lf].copy()
                obs[c] = np.concatenate([lead, o])
            N0, N = N, N + Lf
            t = np.arange(N, dtype=np.float64)
            truth = {c: beta[c] + rate[c] * (t - Lf) for c in others}
        live = {c: np.isfinite(obs[c]).any(axis=(1, 2)) & (np.arange(N) >= Lf) for c in others}

        def tri(views):                                          # views: {cam: (N,19,2)}
            cs = sorted(views)
            pts = np.stack([views[c] for c in cs], 2).reshape(-1, len(cs), 2)
            val = np.stack([vmask[c] for c in cs], 2).reshape(-1, len(cs))
            return triangulate_masked(pts, [mats[c] for c in cs], val).reshape(N, -1, 3)
        after = t >= 10 * FPS

        def score(name, delta):                                  # delta: {cam: (N,) estimate or NaN}
            errs = []
            for c in others:
                e = np.abs(delta[c] - truth[c])
                e[~live[c]] = np.nan                              # score frames the camera recorded (after any lead-in)
                errs.append(e)
            E = np.array(errs)                                   # (7, N)
            for i, (b0, b1) in enumerate(BINS):
                sel = (t >= b0 * FPS) & (t < b1 * FPS)
                if sel.any():
                    v = E[:, sel]
                    res[name]["bins"][i].append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
            res[name]["placed"].append(float(np.isfinite(E[:, after & (t >= Lf)]).sum() / max(1, sum(live[c][after].sum() for c in others))))
            if Lf or stopped:                                    # 3D only for the unmodified replay
                res[name]["mpjpe"].append(np.nan); return
            views = {ref: obs[ref]}
            for c in others:
                if np.isfinite(delta[c][after]).all():
                    views[c] = cubic_sample(obs[c], t - np.nan_to_num(delta[c]), extrapolate=True)
            X = tri(views) if len(views) >= 2 else None
            Xr = tri({c: filled[c][M:M + 2 * N:2] for c in views})        # genlocked, same cameras
            res[name]["mpjpe"].append(1000 * mpjpe(X[after], Xr[after]) if X is not None else np.nan)

        score("uncorrected", {c: np.zeros(N) for c in others})
        for p in a.priors:
            sync = StreamingSynchronizer(cl, ref, FPS, profile=prof, prior=(p or None), **VARIANT)
            delta = {c: np.full(N, np.nan) for c in others}
            last = None
            for s0 in range(0, N, int(FPS)):
                s1 = min(N, s0 + int(FPS))
                if last is not None:                             # correct the frames that arrive now
                    for c in others:
                        delta[c][s0:s1] = sync.offset(c, t[s0:s1])
                for c in cl:
                    sync.push(c, obs[c][s0:s1])
                if sync.maybe_update() is not None:
                    last = True
            timing[f"p{p:g}"] += sync.timing
            recomputed.setdefault(f"p{p:g}", []).append(sync.n_recomputed)
            score(f"causal_p{p:g}", delta)
            off = StreamingSynchronizer(cl, ref, FPS, profile=prof, prior=(p or None), **VARIANT)
            for c in cl:
                off.push(c, obs[c])
            off.update()
            score(f"offline_p{p:g}", {c: off.offset(c, t) for c in others})
        print(f"  {subj}_{seq}: {N / FPS:.0f} s", flush=True)

    out = {"args": vars(a), "features": a.features, "bins": [list(b) for b in BINS], "results": res,
           "timing": {k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)), max=float(np.max(v)), n=len(v))
                      for k, v in timing.items()}}
    print(f"\nmean |offset error| (frames) of placed cameras by elapsed time; placed share and MPJPE (mm) after 10 s")
    print(f"{'mode':>14s} " + " ".join(f"{f'{b0}-{b1 if b1 < 1e8 else 0:g}s':>8s}" for b0, b1 in BINS) + f" {'placed':>7s} {'MPJPE':>7s}")
    for m in modes:
        d = res[m]
        print(f"{m:>14s} " + " ".join(f"{np.nanmean(v) if len(v) else np.nan:8.2f}" for v in d["bins"])
              + f" {np.mean(d['placed']):7.0%} {np.nanmean(d['mpjpe']):7.1f}")
    out["recomputed_windows_per_sequence"] = {k: float(np.mean(v)) for k, v in recomputed.items()}
    print("re-centred windows per sequence:", out["recomputed_windows_per_sequence"])
    for k, v in out["timing"].items():
        print(f"refit time ({k}, 8 cameras, 28 pairs): mean {v['mean'] * 1000:.0f} ms, p95 {v['p95'] * 1000:.0f} ms, "
              f"max {v['max'] * 1000:.0f} ms, every 5 s of video")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
