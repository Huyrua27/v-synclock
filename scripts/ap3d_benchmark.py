# -*- coding: utf-8 -*-
"""Real-image fractional-offset benchmark on AthletePose3D running (120 fps, 4 hardware-
synchronized cameras), using MediaPipe keypoints extracted per video (Colab notebook).

Temporal decimation turns hardware-synchronized 120-fps streams into 30-fps streams with an
exactly known fractional offset: camera A keeps frames s0 + 4m, camera B keeps frames
s0 + 4j + k + 4m, so B lags A by delta = j + k/4 frames at 30 fps -- on real images, with a
real detector, without any interpolation. Every camera pair of every trial is evaluated at
every offset; the statistical unit is the trial (and, most strictly, the athlete).

Inputs: data/ap3d/keypoints/<split>_<subject>_Running_<n>_cam_<c>.npz  (xy (T,19,2), vis, fps, src)
The keypoints are derived from AthletePose3D (custom non-commercial license): keep them private.

    python scripts/ap3d_benchmark.py --kp-dir data/ap3d/keypoints --out docs/paper/clips/ap3d_rows.json
"""
import argparse
import collections
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "realdata"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import csf_predictor

FPS_OUT = 30.0
NAME = re.compile(r"(?P<split>\w+?_set)_(?P<subj>S\d+)_Running_(?P<trial>\d+)_cam_(?P<cam>\d+)\.npz$")


def load_trials(kp_dir, min_valid):
    trials = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(kp_dir, "*.npz")):
        m = NAME.search(os.path.basename(f))
        if not m:
            continue
        d = np.load(f)
        xy = d["xy"].astype(np.float64)
        valid = np.isfinite(xy).all(-1).mean()                    # joint-frame detection rate
        trials[(m["subj"], m["split"], int(m["trial"]))][int(m["cam"])] = (xy, valid, float(d["fps"]))
    out, dropped = {}, collections.Counter()
    for key, cams in trials.items():
        ok = {c: v for c, v in cams.items() if v[1] >= min_valid}
        dropped["camera below detection threshold"] += len(cams) - len(ok)
        if len(ok) < 2:
            dropped["trial with <2 usable cameras"] += 1
            continue
        out[key] = ok
    return out, dropped


def decimated(xy, start, step, n):
    return _fill_gaps(xy[start:start + step * n:step])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/ap3d/keypoints")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--step", type=int, default=4, help="decimation factor (120 -> 30 fps)")
    ap.add_argument("--int-offsets", type=int, nargs="+", default=[-1, 0, 1])
    ap.add_argument("--min-valid", type=float, default=0.5, help="min detection rate per camera")
    ap.add_argument("--min-len", type=int, default=24, help="min frames at 30 fps")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--methods", nargs="+", default=["CC", "DTW", "Caspi-style", "CSF"],
                    help="subset of CC, CC-vy (speed + signed vertical velocity), DTW, Caspi-style, CSF")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)

    trials, dropped = load_trials(a.kp_dir, a.min_valid)
    fps_seen = collections.Counter(round(v[2]) for cams in trials.values() for v in cams.values())
    print(f"{len(trials)} usable trials from {len({k[0] for k in trials})} athletes; fps in files: "
          f"{dict(fps_seen)}; dropped: {dict(dropped)}")

    cc = CrossCorrelation(max_lag=8, features=a.features)
    dtw = DTW(oversample_factor=4, max_warp_frames=8.0, max_frames=None, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.98, 1.02), alpha_steps=21, max_lag=8, features=a.features)
    cc_vy = CrossCorrelation(max_lag=8, features="speed+vy")
    csf = csf_predictor(a.checkpoint) if "CSF" in a.methods else None
    methods = {
        "CC": lambda x, y: cc.predict(x, y).dt_frames,
        "CC-vy": lambda x, y: cc_vy.predict(x, y).dt_frames,
        "DTW": lambda x, y: dtw.predict(x, y).dt_frames,
        # affine fit -> its mean offset over the clip (beta alone is the offset at frame 0)
        "Caspi-style": lambda x, y: (lambda e: e.beta + (e.alpha - 1) * (x.T - 1) / 2)(caspi.estimate(x, y)),
        "CSF": csf,
    }
    methods = {m: methods[m] for m in a.methods}

    S = a.step
    s0 = S * max(0, -min(a.int_offsets))                      # room for negative shifts
    rows, length_mismatch = [], 0
    for (subj, split, trial), cams in sorted(trials.items()):
        lens = {c: len(v[0]) for c, v in cams.items()}
        if len(set(lens.values())) > 1:
            length_mismatch += 1
        T = min(lens.values())
        for ca, cb in itertools.combinations(sorted(cams), 2):
            for j in a.int_offsets:
                for k in range(S):
                    shift = S * j + k
                    n = (T - max(s0, s0 + shift)) // S
                    if n < a.min_len:
                        continue
                    A = decimated(cams[ca][0], s0, S, n)
                    B = decimated(cams[cb][0], s0 + shift, S, n)
                    truth = j + k / S
                    sa, sb = KeypointSequence(A, FPS_OUT), KeypointSequence(B, FPS_OUT)
                    for name, fn in methods.items():
                        try:
                            err = abs(float(fn(sa, sb)) - truth)
                        except Exception:
                            err = float("nan")
                        rows.append(dict(subj=subj, split=split, trial=trial, pair=f"{ca}-{cb}",
                                         j=j, k=k, truth=truth, n=n, method=name, err=err))
    print(f"{len(rows) // len(methods)} pair-offset samples; trials with unequal camera lengths: "
          f"{length_mismatch}")

    rng = np.random.default_rng(0)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["method"]][(r["subj"], r["split"], r["trial"])].append(r["err"])
    print(f"\n{'method':12s} {'trial-mean':>10s} {'95% CI (trials)':>18s} {'median':>7s} "
          f"{'<=0.1':>6s} {'<=0.25':>6s} {'fail(>1)':>8s}   per athlete")
    for m, per_trial in by.items():
        tm = np.array([np.nanmean(v) for v in per_trial.values()])
        b = tm[rng.integers(0, len(tm), (a.n_boot, len(tm)))].mean(1)
        allerr = np.array([e for v in per_trial.values() for e in v])
        allerr = allerr[np.isfinite(allerr)]
        subj_means = {s: np.nanmean([np.nanmean(v) for k_, v in per_trial.items() if k_[0] == s])
                      for s in sorted({k_[0] for k_ in per_trial})}
        print(f"{m:12s} {tm.mean():10.3f} [{np.percentile(b, 2.5):6.3f}, {np.percentile(b, 97.5):6.3f}] "
              f"{np.median(allerr):7.3f} {np.mean(allerr <= .1):6.0%} {np.mean(allerr <= .25):6.0%} "
              f"{np.mean(allerr > 1):8.0%}   " + " ".join(f"{s}:{v:.3f}" for s, v in subj_means.items()))

    print("\nmean error by fractional part k/4 (all trials):")
    for m in methods:
        e = {k: np.nanmean([r["err"] for r in rows if r["method"] == m and r["k"] == k]) for k in range(S)}
        print(f"  {m:12s} " + "  ".join(f"{k}/{S}:{v:.3f}" for k, v in e.items()))
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
