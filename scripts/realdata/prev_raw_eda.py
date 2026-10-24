# -*- coding: utf-8 -*-
"""Previous 5-camera capture (data_raw): V-SyncLock vs the hand-made clap synchronization.

The 'synced' videos are stream-copy trims of the raw ones; sync frame 0 is raw frame TRIM[c] of camera c
(found by exact image matching). The clap reference lag between cameras u, v on 30-Hz grids of their own
clocks is 30 * (t_v[TRIM_v] - t_u[TRIM_u]) (B[t + L] ~ A[t]); it was set by eye, so it is good to about
one frame.

    python scripts/realdata/prev_raw_eda.py
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path[:0] = [ROOT, os.path.dirname(__file__)]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from realdata_eval import cc_profile, coarse_offset, csf_profile, motion_signal  # noqa: E402
from vsynclock.benchmark.drift import fit_drift_c2f  # noqa: E402
from vsynclock.benchmark.interpolation import cubic_sample  # noqa: E402
from vsynclock.model.panoptic_dataset import _fill_gaps  # noqa: E402

FPS = 30.0
TRIM = {"cam_1": 553, "cam_2": 33, "cam_3": 222, "cam_4": 102, "cam_5": 229}


def load(cam, by_index=False):
    xy = np.load(f"data/prev_raw/pose/{cam}.npz")["xy"].astype(np.float64)
    t = np.load(f"data/prev_raw/data_raw/{cam}_pts.npy")[: len(xy)]
    filled = _fill_gaps(xy)[0]
    if by_index:
        return filled, np.arange(len(xy)) / FPS
    grid = np.arange(0, np.floor(t[-1] * FPS)) / FPS
    return cubic_sample(filled, np.interp(grid, t, np.arange(len(t))), extrapolate=True), t


def main():
    os.chdir(ROOT)
    cams = sorted(TRIM)
    prof = {"CC": cc_profile(8, "speed+vyb"), "CSF": csf_profile("checkpoints/csf_b1_t20_vy.pt")}
    out = {}
    for mode in ("timestamps", "frame index"):
        data = {c: load(c, by_index=(mode == "frame index")) for c in cams}
        print(f"\n== reading by {mode}")
        for u, v in itertools.combinations(cams, 2):
            (A, ta), (B, tb) = data[u], data[v]
            ref = FPS * (tb[TRIM[v]] - ta[TRIM[u]])
            L = coarse_offset(motion_signal(A), motion_signal(B), 600, 0, None)
            s0 = max(0, -L); e = min(len(A), len(B) - L - 9)
            r = dict(clap=float(ref), coarse=int(L))
            for m, p in prof.items():
                line = L - np.asarray(fit_drift_c2f(p, A[s0:e], B[s0 + L:e + L], FPS, 20, 6, prior="consensus")[0])
                r[m] = dict(median=float(np.median(line)), at_clap=float(line[min(max(0, int(TRIM[u] * (1 if mode == 'frame index' else 1)) - s0), len(line) - 1)]),
                            slope_fpm=float((line[-1] - line[0]) / (len(line) / FPS / 60)))
            out[f"{mode}|{u}-{v}"] = r
            print(f"{u}-{v}: clap {ref:+7.2f}  coarse {L:+5d}  CC-c2f {r['CC']['median']:+7.2f} ({r['CC']['slope_fpm']:+.2f} fr/min)  "
                  f"CSF-c2f {r['CSF']['median']:+7.2f} ({r['CSF']['slope_fpm']:+.2f})  |CC-clap| {abs(r['CC']['median'] - ref):.2f}", flush=True)
    json.dump(out, open("data/prev_raw/eda.json", "w"), indent=1)


if __name__ == "__main__":
    main()
