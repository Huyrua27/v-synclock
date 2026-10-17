# -*- coding: utf-8 -*-
"""Real-image fractional-offset benchmark on TotalCapture (60 fps, 8 genlocked cameras).

Same protocol as ap3d_benchmark.py, at a lower native rate: decimating the 60-fps streams by
S = 2 gives 30-fps streams, camera A keeps frames s0 + 2m and camera B frames s0 + 2j + k + 2m,
so B lags A by exactly delta = j + k/2 frames at 30 fps -- real images, real detector, no
interpolation. Sequences are minutes long, so fixed-length clips are drawn from each sequence
(start positions fixed by --seed); every camera pair of every clip is evaluated at every offset.
Statistical unit: the sequence (15 test sequences), and per subject.

Besides the estimate, each row stores the profile peak of CC and CSF (observability score) and
the clip's median joint speed, so observability-vs-viewpoint and motion-regime analyses reuse
the same rows.

Inputs: data/tc/keypoints/S<subj>_<seq>_cam<c>.npz  (xy (T,19,2) at 60 fps; Colab notebook).
TotalCapture is research-only, no redistribution: keep keypoints and rows private.

    python scripts/tc_benchmark.py --out docs/paper/clips/tc_rows_L60.json
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
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts", "realdata")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from vsynclock.benchmark.drift import peak_in_range
from vsynclock.benchmark.observability import profile_scores
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import cc_profile, csf_predictor, csf_profile

FPS_OUT, S = 30.0, 2
NAME = re.compile(r"S(?P<subj>\d)_(?P<seq>[a-z]+\d)_cam(?P<cam>\d+)\.npz$")


def load_sequences(kp_dir):
    """{(subj, seq): {cam: xy (T,19,2) float64 with NaNs}}"""
    seqs = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(kp_dir, "*.npz"))):
        m = NAME.search(os.path.basename(f))
        if m:
            seqs[(f"S{m['subj']}", m["seq"])][int(m["cam"])] = np.load(f)["xy"].astype(np.float64)
    return dict(seqs)


def clip_starts(T60, n_out, n_clips, margin, rng):
    """Start frames (60-fps index) of non-overlapping clips of n_out 30-fps frames."""
    span = S * n_out + margin
    slots = np.arange(margin, T60 - span, span)
    if len(slots) <= n_clips:
        return slots
    return np.sort(rng.choice(slots, n_clips, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/tc/keypoints")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--clip-len", type=int, default=60, help="frames at 30 fps (60 = 2 s)")
    ap.add_argument("--clips", type=int, default=8, help="clips per sequence")
    ap.add_argument("--int-offsets", type=int, nargs="+", default=[-1, 0, 1])
    ap.add_argument("--min-valid", type=float, default=0.5, help="min joint detection rate per clip and camera")
    ap.add_argument("--methods", nargs="+", default=["CC", "DTW", "Caspi-style", "CSF"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    seqs = load_sequences(a.kp_dir)
    print(f"{len(seqs)} sequences: " + ", ".join(f"{s}_{q}({len(c)} cams)" for (s, q), c in sorted(seqs.items())))

    cc = CrossCorrelation(max_lag=8, features=a.features)
    dtw = DTW(oversample_factor=4, max_warp_frames=8.0, max_frames=None, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.98, 1.02), alpha_steps=21, max_lag=8, features=a.features)
    est_fn = {
        "CC": lambda x, y: cc.predict(x, y).dt_frames,
        "DTW": lambda x, y: dtw.predict(x, y).dt_frames,
        "Caspi-style": lambda x, y: (lambda e: e.beta + (e.alpha - 1) * (x.T - 1) / 2)(caspi.estimate(x, y)),
        "CSF": csf_predictor(a.checkpoint) if "CSF" in a.methods else None,
    }
    prof = {"CC": cc_profile(8, a.features)}
    if "CSF" in a.methods:
        prof["CSF"] = csf_profile(a.checkpoint)
    rng = np.random.default_rng(a.seed)
    n, margin = a.clip_len, S * max(abs(j) for j in a.int_offsets) + S
    rows, skipped = [], collections.Counter()
    for (subj, seq), cams in sorted(seqs.items()):
        T60 = min(len(v) for v in cams.values())
        for c0 in clip_starts(T60, n, a.clips, margin, rng):
            for ca, cb in itertools.combinations(sorted(cams), 2):
                for j in a.int_offsets:
                    for k in range(S):
                        sa_raw = cams[ca][c0:c0 + S * n:S]
                        sb_raw = cams[cb][c0 + S * j + k:c0 + S * j + k + S * n:S]
                        va, vb = np.isfinite(sa_raw).all(-1).mean(), np.isfinite(sb_raw).all(-1).mean()
                        if min(va, vb) < a.min_valid or len(sb_raw) < n:
                            skipped["low detection" if len(sb_raw) == n else "short"] += 1
                            continue
                        A, B = _fill_gaps(sa_raw)[0], _fill_gaps(sb_raw)[0]
                        speed = float(np.nanmedian(np.linalg.norm(np.diff(A, axis=0), axis=-1)))
                        sa, sb = KeypointSequence(A, FPS_OUT), KeypointSequence(B, FPS_OUT)
                        truth = j + k / S
                        peaks = {m: profile_scores(*p(sa, sb), n=n)["peak"] for m, p in prof.items()}
                        for m in a.methods:
                            try:
                                est = float(est_fn[m](sa, sb))
                            except Exception:
                                est = float("nan")
                            rows.append(dict(subj=subj, seq=seq, c0=int(c0), pair=f"{ca}-{cb}", j=j, k=k,
                                             truth=truth, n=n, speed=speed, valid=float(min(va, vb)),
                                             method=m, est=est, err=abs(est - truth),
                                             peak=peaks.get(m, peaks["CC"])))
        print(f"  {subj}_{seq}: {len(rows)} rows so far", flush=True)
    print("skipped:", dict(skipped))
    json.dump(rows, open(a.out, "w", encoding="utf-8"))
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
