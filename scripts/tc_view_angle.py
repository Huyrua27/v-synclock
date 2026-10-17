# -*- coding: utf-8 -*-
"""Subject-centred viewpoint difference on TotalCapture, from Vicon.

The angle between optical axes ignores where the subject is; in TotalCapture the subject moves
around an 8 x 4 m volume, so the angle under which two cameras see the body changes per clip.
For every clip of a rows file (tc_benchmark.py) this computes the mean angle between the rays
from the subject's hips (Vicon, inches -> m; Vicon frame = video frame + VICON_OFFSET, measured
by reprojection) to the two camera centres, and tabulates failure rates against it.

    python scripts/tc_view_angle.py --rows docs/paper/clips/tc_rows_L300.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.observability import auc
from vsynclock.benchmark.surrey_calib import pair_angles, read_surrey

S, INCH, VICON_OFFSET = 2, 0.0254, -2
BINS = [(0, 30), (30, 60), (60, 75), (75, 90), (90, 105), (105, 120), (120, 150), (150, 181)]


def clip_view_angle(hips, centres, c0, n, u, v):
    idx = np.clip(np.arange(c0, c0 + S * n, S) + VICON_OFFSET, 0, len(hips) - 1)
    du, dv = centres[u] - hips[idx], centres[v] - hips[idx]
    cos = (du * dv).sum(-1) / np.linalg.norm(du, axis=-1) / np.linalg.norm(dv, axis=-1)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_rows_L300.json")
    ap.add_argument("--calib", default="data/tc/tc_keypoints/calibration.cal")
    ap.add_argument("--vicon-dir", default="data/tc/vicon")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    cams = read_surrey(a.calib)
    centres = {c: v["center"] for c, v in cams.items()}
    axis = {p: v["axis_deg"] for p, v in pair_angles(cams).items()}
    rows = json.load(open(a.rows, encoding="utf-8"))
    hips = {}
    cache = {}
    for r in rows:
        k = (r["subj"], r["seq"])
        if k not in hips:
            hips[k] = np.load(os.path.join(a.vicon_dir, f"{k[0]}_{k[1]}.npz"))["pos"][:, 0].astype(np.float64) * INCH
        ck = (k, r["c0"], r["pair"])
        if ck not in cache:
            u, v = map(int, r["pair"].split("-"))
            cache[ck] = clip_view_angle(hips[k], centres, r["c0"], r["n"], u, v)
        r["view"] = cache[ck]; r["axis"] = axis[r["pair"]]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    view = np.array([r["view"] for r in rows if r["method"] == methods[0]])
    ax = np.array([r["axis"] for r in rows if r["method"] == methods[0]])
    print(f"{a.rows}: {len(cache)} clip-pairs; |view - axis| median {np.median(np.abs(view - ax)):.1f} deg, "
          f"90th pct {np.percentile(np.abs(view - ax), 90):.1f} deg")
    out = {"bins": {}, "auc": {}}
    print(f"\nfailure rate (>1 frame) by subject-centred view angle")
    print(f"{'view deg':>9s} {'clip-pairs':>10s} " + " ".join(f"{m:>12s}" for m in methods))
    for b0, b1 in BINS:
        line, d = [], {}
        for m in methods:
            e = np.array([r["err"] for r in rows if r["method"] == m and b0 <= r["view"] < b1])
            d[m] = dict(n=len(e), gt1=float(np.mean(e > 1)) if len(e) else None,
                        mean=float(e.mean()) if len(e) else None)
            line.append(f"{np.mean(e > 1):5.0%} ({e.mean():4.2f})" if len(e) else "      --    ")
        out["bins"][f"{b0}-{b1}"] = d
        n_cp = len({(r['subj'], r['seq'], r['c0'], r['pair']) for r in rows
                    if r['method'] == methods[0] and b0 <= r['view'] < b1})
        print(f"{b0:4d}-{b1:<4d} {n_cp:10d} " + " ".join(line))
    print("\nhow well does each geometric quantity predict success (|err|<=1)? AUC")
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        ok = np.array([r["err"] <= 1 for r in R])
        sc = {"-view": -np.array([r["view"] for r in R]), "-axis": -np.array([r["axis"] for r in R]),
              "|cos view|": np.abs(np.cos(np.radians([r["view"] for r in R]))),
              "|cos axis|": np.abs(np.cos(np.radians([r["axis"] for r in R]))),
              "peak": np.array([r["peak"] for r in R])}
        out["auc"][m] = {k: auc(v, ok) for k, v in sc.items()}
        print(f"  {m:12s} " + "  ".join(f"{k} {v:.3f}" for k, v in out["auc"][m].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
