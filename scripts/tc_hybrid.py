# -*- coding: utf-8 -*-
"""Hybrid local estimator: classical CC by default, the learned CSF only for rhythmic clips.

Periodicity of a clip = maximum autocorrelation of the reference view's mean standardized joint
speed at lags 8-45 frames (0.27-1.5 s at 30 fps), i.e. how strongly the motion repeats itself.
If it exceeds tau, the clip is routed to CSF (about 10x the cost of CC per window), otherwise
to CC. tau = 0.5 is fixed a priori (no TotalCapture development data); the full sweep is
reported for transparency. Evaluated on the 10-s TotalCapture rows of tc_benchmark.py.

    python scripts/tc_hybrid.py --out docs/paper/clips/tc_hybrid_L300.json
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

from vsynclock.benchmark.observability import auc
from vsynclock.model.panoptic_dataset import _fill_gaps
from tc_benchmark import load_sequences

S, COST = 2, {"CC": 0.7, "CSF": 7.1}            # ms per window, one CPU thread (Sec. 3, cost)


def periodicity(xy, lo=8, hi=45):
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=-1)               # (T-1, J)
    sp = (sp - sp.mean(0)) / (sp.std(0) + 1e-9)
    x = sp.mean(1); x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x @ x + 1e-9)
    return float(ac[lo:hi + 1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_rows_L300.json")
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    seqs = load_sequences(a.kp_dir)
    rows = json.load(open(a.rows, encoding="utf-8"))
    by = collections.defaultdict(dict)
    for r in rows:
        if r["method"] in ("CC", "CSF"):
            by[(r["subj"], r["seq"], r["c0"], r["pair"], r["j"], r["k"])][r["method"]] = r
    per = {}
    for (subj, seq, c0, pair, j, k) in by:
        key = (subj, seq, c0, pair.split("-")[0])
        if key not in per:
            n = by[(subj, seq, c0, pair, j, k)]["CC"]["n"]
            per[key] = periodicity(_fill_gaps(seqs[(subj, seq)][int(key[3])][c0:c0 + S * n:S])[0])
    keys = [k for k, v in by.items() if "CC" in v and "CSF" in v]
    P = np.array([per[(k[0], k[1], k[2], k[3].split("-")[0])] for k in keys])
    eCC = np.array([by[k]["CC"]["err"] for k in keys]); eCSF = np.array([by[k]["CSF"]["err"] for k in keys])
    act = np.array([k[1].rstrip("0123456789") for k in keys])
    seq = np.array([f"{k[0]}_{k[1]}" for k in keys])
    print(f"{len(keys)} samples; periodicity AUC for walking vs other: {auc(P, act == 'walking'):.3f}")
    for g in ("walking", "acting", "freestyle"):
        print(f"  {g:9s} periodicity median {np.median(P[act == g]):.2f}")

    def summary(err, use_csf):
        cost = np.where(use_csf, COST["CSF"], COST["CC"]).mean() / COST["CC"]
        per_seq = [err[seq == s].mean() for s in sorted(set(seq))]
        return dict(mean=float(err.mean()), seq_mean=float(np.mean(per_seq)), gt1=float(np.mean(err > 1)),
                    csf_share=float(np.mean(use_csf)), cost_vs_cc=float(cost))

    out = {"tau": a.tau, "sweep": {}}
    print(f"\n{'rule':>18s} {'mean':>6s} {'>1':>5s} {'CSF share':>9s} {'cost/CC':>7s}")
    for name, use in (("CC only", np.zeros(len(P), bool)), ("CSF only", np.ones(len(P), bool)),
                      ("oracle (best)", eCSF < eCC)):
        err = np.where(use, eCSF, eCC); d = summary(err, use); out[name] = d
        print(f"{name:>18s} {d['mean']:6.3f} {d['gt1']:5.0%} {d['csf_share']:9.0%} {d['cost_vs_cc']:7.1f}")
    for tau in np.round(np.arange(0.2, 0.81, 0.05), 2):
        use = P > tau; err = np.where(use, eCSF, eCC); d = summary(err, use); out["sweep"][str(tau)] = d
        mark = "  <- a priori" if abs(tau - a.tau) < 1e-9 else ""
        print(f"{'hybrid tau=' + str(tau):>18s} {d['mean']:6.3f} {d['gt1']:5.0%} {d['csf_share']:9.0%} {d['cost_vs_cc']:7.1f}{mark}")
    use = P > a.tau; err = np.where(use, eCSF, eCC)
    rng = np.random.default_rng(0); S_ = sorted(set(seq))
    for ref_name, ref in (("CC", eCC), ("CSF", eCSF)):
        diff = np.array([err[seq == s].mean() - ref[seq == s].mean() for s in S_])
        b = diff[rng.integers(0, len(diff), (a.n_boot, len(diff)))].mean(1)
        out[f"hybrid_minus_{ref_name}"] = [float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        print(f"hybrid(tau={a.tau}) - {ref_name}: {diff.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] per sequence")
    out["csf_share_by_activity"] = {g: float(np.mean(use[act == g])) for g in ("walking", "acting", "freestyle")}
    print("CSF share by activity:", {g: f"{v:.0%}" for g, v in out["csf_share_by_activity"].items()})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
