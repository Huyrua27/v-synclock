# -*- coding: utf-8 -*-
"""Rig-level runtime of the V-SyncLock pipeline on TotalCapture keypoints, as a function of the number
of cameras C (one minute of 30-fps video).

Stages per recording: coarse alignment and pass 1 (long-window CC) on every pair, the
self-calibrated null gate on every pair, pass 2 (CC-c2f or CSF-c2f) on the pair set, placement
(weighted least squares), resampling of every non-reference stream and DLT triangulation.
Pair set: all C(C-1)/2 pairs, or pruned after pass 1 to the k strongest edges of each camera
(union, <= kC edges), so the expensive pass 2 grows linearly in C. Pose estimation (MediaPipe,
measured separately: benchmark_runtime.py) is per camera and reported for scale.

Sequential times are on one CPU thread; "parallel" runs the per-pair pass 2 in a process pool.

    python scripts/rig_profile.py --out docs/paper/clips/rig_profile.json
"""
import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch

from vsynclock.benchmark.drift import fit_drift_c2f, peak_in_range, sliding_offsets
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.benchmark.surrey_calib import read_surrey
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from vsynclock.panoptic.triangulation import projection_matrix, triangulate_masked
from realdata_eval import cc_profile, coarse_offset, csf_profile, motion_signal
from tc_benchmark import load_sequences
from tc_camera_graph import ls_place

FPS, S = 30.0, 2
ORDER = [1, 5, 3, 7, 2, 6, 4, 8]          # subsets spread around the volume
MEDIAPIPE_S_PER_MIN = 59.0                # measured, one thread, 720p (Sec. 3.3)
_W = {}


def _init(ckpt, features="speed"):
    torch.set_num_threads(1)
    _W["cc"], _W["csf"] = cc_profile(8, features), csf_profile(ckpt)


def _pass2(job):
    est, a, b = job
    line, _, _ = fit_drift_c2f(_W[est], a, b, FPS, 20, 6, radius=3.0, prior="consensus")
    return float(np.median(line))


def prune(nodes, w, k):
    keep = set()
    for c in nodes:
        inc = sorted((e for e in w if c in e), key=lambda e: -w[e])
        keep |= set(inc[:k])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--cams", type=int, nargs="+", default=[3, 4, 6, 8])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--null-n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--features", default="speed", help="CC input (speed, speed+vy, speed+vyb)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    _init(a.checkpoint, a.features)
    n = int(a.seconds * FPS)
    seqs = load_sequences(a.kp_dir)
    key = next(k for k, cams in sorted(seqs.items(), key=lambda kv: -min(len(v) for v in kv[1].values()))
               if len(cams) == 8)
    cams60 = seqs[key]
    T60 = min(len(v) for v in cams60.values())
    s0 = max(40, (T60 - S * n) // 2)                 # middle of the performance
    X = {c: _fill_gaps(cams60[c][s0:s0 + S * n:S])[0] for c in ORDER}
    V = {c: np.isfinite(cams60[c][s0:s0 + S * n:S]).all(-1) for c in ORDER}
    assert all(len(x) == n for x in X.values()), "sequence too short"
    calib = read_surrey(os.path.join(a.kp_dir, "calibration.cal"))
    print(f"sequence {key}, {a.seconds:.0f}s at 30 fps ({n} frames); pass 2 window 20 / stride 6")
    rng = np.random.default_rng(0)
    pool = None
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.checkpoint, a.features))
        pool.map(_pass2, [("cc", X[1][:200], X[5][:200])] * a.workers)            # warm-up

    out = {"sequence": "_".join(key), "seconds": a.seconds, "k": a.k, "workers": a.workers,
           "mediapipe_s_per_min_per_cam": MEDIAPIPE_S_PER_MIN, "rows": []}
    for C in a.cams:
        nodes = ORDER[:C]
        pairs = list(itertools.combinations(nodes, 2))
        T = {}
        t0 = time.perf_counter()
        sig = {c: motion_signal(X[c]) for c in nodes}
        for u, v in pairs:
            coarse_offset(sig[u], sig[v], 400)
        T["coarse"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        w = {}
        cc = _W["cc"]
        for u, v in pairs:
            peaks = []
            for st in range(0, n - 80 + 1, 20):
                try:
                    l, sc = cc(KeypointSequence(X[u][st:st + 80], FPS), KeypointSequence(X[v][st:st + 80], FPS))
                except ValueError:                    # no motion in the window
                    continue
                peaks.append(np.max(sc))
            w[(u, v)] = float(np.mean(peaks)) if peaks else 0.0
        T["pass1"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        thr = {}
        for u, v in pairs:
            null = []
            while len(null) < a.null_n:
                t1, t2 = rng.integers(0, n - 80, 2)
                if abs(t1 - t2) < 3 * FPS:
                    continue
                try:
                    l, sc = cc(KeypointSequence(X[u][t1:t1 + 80], FPS), KeypointSequence(X[v][t2:t2 + 80], FPS))
                except ValueError:
                    continue
                null.append(np.max(sc))
            thr[(u, v)] = float(np.quantile(null, 0.95))
        T["null"] = time.perf_counter() - t0

        for est in ("cc", "csf"):
            for pset_name, pset in (("all", pairs), (f"k{a.k}", sorted(prune(nodes, w, a.k)))):
                jobs = [(est, X[u], X[v]) for u, v in pset]
                t0 = time.perf_counter()
                rel = [_pass2(j) for j in jobs]
                T[f"pass2_{est}_{pset_name}"] = time.perf_counter() - t0
                if pool is not None:
                    t0 = time.perf_counter()
                    pool.map(_pass2, jobs, chunksize=1)
                    T[f"pass2_{est}_{pset_name}_par"] = time.perf_counter() - t0
                T[f"edges_{pset_name}"] = len(pset)

        t0 = time.perf_counter()
        good = [e for e in pairs if w[e] >= thr[e]] or pairs
        relv = {e: 0.0 for e in pairs}
        ls_place(nodes, good, lambda u, v: relv[(u, v)], w, root=nodes[0])
        T["placement"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        q = np.arange(n, dtype=np.float64) - 0.37
        views = [X[nodes[0]]] + [cubic_sample(X[c], q, extrapolate=True) for c in nodes[1:]]
        T["resample"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        mats = [projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in nodes]
        pts = np.stack(views, axis=2).reshape(-1, C, 2)
        val = np.stack([V[c] for c in nodes], axis=2).reshape(-1, C)
        triangulate_masked(pts, mats, val)
        T["triangulation"] = time.perf_counter() - t0

        scale = 60.0 / a.seconds
        T = {k_: (v * scale if not k_.startswith("edges") else v) for k_, v in T.items()}   # per minute of video
        T["C"], T["pairs"] = C, len(pairs)
        out["rows"].append(T)
        shared = T["coarse"] + T["pass1"] + T["null"] + T["placement"] + T["resample"] + T["triangulation"]
        print(f"\nC={C}: {len(pairs)} pairs, pruned k={a.k}: {T[f'edges_k{a.k}']} edges  (s per minute of video)")
        for k_ in ("coarse", "pass1", "null", "placement", "resample", "triangulation"):
            print(f"  {k_:14s} {T[k_]:8.2f}")
        for est in ("cc", "csf"):
            for pn in ("all", f"k{a.k}"):
                p2 = T[f"pass2_{est}_{pn}"]
                par = T.get(f"pass2_{est}_{pn}_par")
                print(f"  pass2 {est.upper():3s} {pn:4s} {p2:8.2f}" + (f"  ({a.workers} proc: {par:.2f})" if par else "")
                      + f"   total sync {shared + p2:8.2f}")
        print(f"  MediaPipe, {C} cameras (1 thread each): {MEDIAPIPE_S_PER_MIN * C:.0f}")
    if pool is not None:
        pool.close()
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
