# -*- coding: utf-8 -*-
"""Camera-roll robustness of the vertical-velocity input (protocol: docs/paper/DEV_PROTOCOL.md,
"Camera-roll robustness"). Camera B of every sample is rotated in the image plane by theta;
variants: speed, speed + image vy, speed + body vy (vertical = per-clip median hip->neck axis).

    python scripts/roll_robustness.py --dataset tc --out docs/paper/clips/vy/roll_tc.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines.cross_correlation import lag_scores, refined_peak
from signal_baselines import samples_ap3d, samples_sp, samples_tc, z

ANGLES = (0, 5, 10, 15, 20, 30, 45, 60, 90)
VARIANTS = ("speed", "speed+vy", "speed+vy_body")
NECK, HIP = 0, 2


def rotate(xy, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return xy @ R.T


def features(xy, kind):
    xy = np.nan_to_num(xy)
    v = np.gradient(xy, axis=0)
    sp = np.linalg.norm(v, axis=-1)
    if kind == "speed":
        return z(sp)
    if kind == "speed+vy":
        d = np.array([0.0, 1.0])
    else:                                                       # body axis, pointing down (hip - neck)
        a = np.median(xy[:, HIP] - xy[:, NECK], axis=0)
        d = a / (np.linalg.norm(a) + 1e-9)
    return z(np.concatenate([sp, v @ d], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("tc", "ap3d", "sp"), required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    gen = {"tc": samples_tc, "ap3d": samples_ap3d, "sp": samples_sp}[a.dataset](a)
    unit, err = [], collections.defaultdict(list)
    for u, g, truth, A, B in gen:
        unit.append(u)
        for th in ANGLES:
            Br = rotate(B, th)
            for k in VARIANTS:
                try:
                    lags, sc = lag_scores(features(A, k), features(Br, k), 8)
                    e = abs(refined_peak(lags, sc)[0] - truth)
                except ValueError:
                    e = np.nan
                err[(k, th)].append(e)
    unit = np.array(unit); U = sorted(set(unit))
    res = {}
    print(f"{a.dataset}: {len(unit)} samples, {len(U)} units; mean error (frames, per unit) vs roll of camera B")
    print(f"{'deg':>4s} " + " ".join(f"{k:>14s}" for k in VARIANTS))
    for th in ANGLES:
        row = []
        for k in VARIANTS:
            e = np.array(err[(k, th)])
            m = float(np.mean([np.nanmean(e[unit == x]) for x in U]))
            res[f"{k}|{th}"] = dict(mean=m, gt1=float(np.nanmean(e > 1)))
            row.append(f"{m:7.3f} ({np.nanmean(e > 1):3.0%})")
        print(f"{th:4d} " + " ".join(f"{c:>14s}" for c in row))
    if a.out:
        json.dump({"dataset": a.dataset, "angles": ANGLES, "results": res}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
