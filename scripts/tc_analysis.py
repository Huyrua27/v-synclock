# -*- coding: utf-8 -*-
"""Analyse TotalCapture rows (tc_benchmark.py): estimators, viewpoint, observability, activity.

1. Estimators: mean error with 95% bootstrap CI over sequences, <=0.25 and >1-frame rates.
2. Viewpoint: error and failure rate per bin of the angle between the cameras' optical axes;
   per-pair failure rate vs angle with a logistic fit -> angle at which half the estimates fail.
3. Observability transfer: the AthletePose3D whole-clip peak threshold (frozen, not re-tuned)
   applied to TotalCapture -- share kept, precision, failures caught, pair-level AUC vs angle.
4. Activity: walking / acting / freestyle.

    python scripts/tc_analysis.py --rows docs/paper/clips/tc_rows_L60.json \
        --calib data/tc/tc_keypoints/calibration.cal --out docs/paper/clips/tc_summary_L60.json
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

BINS = [(0, 30), (30, 60), (60, 75), (75, 90), (90, 105), (105, 180)]


def boot_ci(per_seq, n_boot, rng):
    v = np.array(per_seq)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def logistic_half(angle, fail):
    """Fit P(fail) = 1 / (1 + exp(-(angle - a50) / s)) by grid search on the log-likelihood."""
    best = (-np.inf, None, None)
    for a50 in np.arange(30, 150.5, 0.5):
        for s in (2, 4, 6, 8, 10, 15, 20, 30):
            p = np.clip(1 / (1 + np.exp(-(angle - a50) / s)), 1e-6, 1 - 1e-6)
            ll = np.sum(fail * np.log(p) + (1 - fail) * np.log(1 - p))
            if ll > best[0]:
                best = (ll, a50, s)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_rows_L60.json")
    ap.add_argument("--calib", default="data/tc/tc_keypoints/calibration.cal")
    ap.add_argument("--choice", default="docs/paper/clips/observability_choice.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    rows = json.load(open(a.rows, encoding="utf-8"))
    ang = pair_angles(read_surrey(a.calib))
    for r in rows:
        r["angle"] = ang[r["pair"]]["axis_deg"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    rng = np.random.default_rng(0)
    out = {"n_rows": len(rows), "pair_angles": ang}

    print(f"{len(rows) // len(methods)} pair-offset samples, {len({(r['subj'], r['seq']) for r in rows})} sequences")
    print(f"\n{'method':12s} {'mean [95% CI over seqs]':>28s} {'<=.25':>6s} {'>1':>5s} | angle<=60: mean  >1 | angle>=105: mean  >1")
    out["methods"] = {}
    for m in methods:
        R = [r for r in rows if r["method"] == m and np.isfinite(r["err"])]
        per_seq = collections.defaultdict(list)
        for r in R:
            per_seq[(r["subj"], r["seq"])].append(r["err"])
        mean, lo, hi = boot_ci([np.mean(v) for v in per_seq.values()], a.n_boot, rng)
        e = np.array([r["err"] for r in R]); A = np.array([r["angle"] for r in R])
        nar, wid = e[A <= 60], e[A >= 105]
        print(f"{m:12s} {mean:8.3f} [{lo:6.3f}, {hi:6.3f}]     {np.mean(e <= .25):6.0%} {np.mean(e > 1):5.0%} | "
              f"{nar.mean():9.3f} {np.mean(nar > 1):4.0%} | {wid.mean():10.3f} {np.mean(wid > 1):4.0%}")
        d = dict(mean=mean, ci=[lo, hi], le025=float(np.mean(e <= .25)), gt1=float(np.mean(e > 1)), bins={})
        for b0, b1 in BINS:
            sel = (A >= b0) & (A < b1)
            per_seq_b = collections.defaultdict(list)
            for r, s in zip(R, sel):
                if s:
                    per_seq_b[(r["subj"], r["seq"])].append(r["err"])
            mb, lb, hb = boot_ci([np.mean(v) for v in per_seq_b.values()], a.n_boot, rng)
            d["bins"][f"{b0}-{b1}"] = dict(n_pairs=len({r["pair"] for r, s in zip(R, sel) if s}), mean=mb, ci=[lb, hb],
                                           le025=float(np.mean(e[sel] <= .25)), gt1=float(np.mean(e[sel] > 1)))
        out["methods"][m] = d

    print("\nby viewpoint angle (optical axes): mean error / share >1 frame")
    print(f"{'bin':>9s} {'pairs':>5s} " + " ".join(f"{m:>16s}" for m in methods))
    for b0, b1 in BINS:
        key = f"{b0}-{b1}"
        print(f"{key:>9s} {out['methods'][methods[0]]['bins'][key]['n_pairs']:5d} " +
              " ".join(f"{out['methods'][m]['bins'][key]['mean']:7.2f} / {out['methods'][m]['bins'][key]['gt1']:5.0%}"
                       for m in methods))

    print("\nper pair (sorted by angle): failure rate >1 frame per method; logistic 50% angle")
    out["pairs"] = {}
    for p in sorted({r["pair"] for r in rows}, key=lambda p: ang[p]["axis_deg"]):
        line = {m: float(np.mean([r["err"] > 1 for r in rows if r["pair"] == p and r["method"] == m])) for m in methods}
        out["pairs"][p] = dict(angle=ang[p]["axis_deg"], fail=line,
                               peak_cc=float(np.mean([r["peak"] for r in rows if r["pair"] == p and r["method"] == "CC"])))
        print(f"  {p:4s} {ang[p]['axis_deg']:5.0f}deg  peak {out['pairs'][p]['peak_cc']:.2f}  " +
              "  ".join(f"{m}:{v:4.0%}" for m, v in line.items()))
    out["half_fail_angle"] = {}
    for m in methods:
        R = [r for r in rows if r["method"] == m]
        a50, s = logistic_half(np.array([r["angle"] for r in R]), np.array([r["err"] > 1 for r in R], float))
        out["half_fail_angle"][m] = dict(a50=a50, scale=s)
        print(f"  {m:12s} 50% failure at {a50:.1f} deg (scale {s})")

    choice = json.load(open(a.choice))
    print("\nobservability transfer: AthletePose3D whole-clip threshold, frozen")
    out["observability"] = {}
    for m in ("CC", "CSF"):
        R = [r for r in rows if r["method"] == m]
        sc, err = np.array([r["peak"] for r in R]), np.array([r["err"] for r in R])
        thr = choice[m]["threshold"]
        succ, keep = err <= 1, sc >= thr
        pair_sc = collections.defaultdict(list)
        for r in R:
            pair_sc[r["pair"]].append(r["peak"])
        ps = np.array([np.mean(v) for v in pair_sc.values()])
        pair_ok = np.array([out["pairs"][p]["fail"][m] < 0.5 for p in pair_sc])
        d = dict(threshold=thr, auc=auc(sc, succ), kept=float(keep.mean()), precision=float(succ[keep].mean()),
                 failures_caught=float((~succ & ~keep).sum() / max(1, (~succ).sum())),
                 err_kept=float(err[keep].mean()), err_all=float(err.mean()), pair_auc=auc(ps, pair_ok))
        out["observability"][m] = d
        print(f"  {m}: peak>={thr:.3f}  AUC {d['auc']:.3f}  kept {d['kept']:.0%}  precision {d['precision']:.0%}  "
              f"failures caught {d['failures_caught']:.0%}  err kept {d['err_kept']:.2f} / all {d['err_all']:.2f}  "
              f"pair-level AUC (pair works) {d['pair_auc']:.3f}")

    print("\nby activity: mean error (share >1) on pairs <= 60 deg")
    out["activity"] = {}
    for act in ("walking", "acting", "freestyle"):
        out["activity"][act] = {}
        line = []
        for m in methods:
            e = np.array([r["err"] for r in rows if r["method"] == m and r["seq"].startswith(act) and r["angle"] <= 60])
            out["activity"][act][m] = dict(mean=float(e.mean()), gt1=float(np.mean(e > 1)))
            line.append(f"{m}:{e.mean():.2f}({np.mean(e > 1):.0%})")
        print(f"  {act:9s} " + "  ".join(line))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
