# -*- coding: utf-8 -*-
"""Ground-truth-free observability gating on TotalCapture, per estimate.

For every sequence and camera pair, the null distribution of the profile peak is sampled from
windows of the same length taken >= --min-gap s apart (no shared timing); an estimate is kept
if its peak exceeds the pair's q-quantile. Evaluated on the rows of tc_benchmark.py.

    python scripts/tc_null_rows.py --rows docs/paper/clips/tc_rows_L300.json \
        --out docs/paper/clips/tc_null_L300.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.observability import auc, profile_scores
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import cc_profile, csf_profile
from tc_benchmark import load_sequences

S, FPS = 2, 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_rows_L300.json")
    ap.add_argument("--kp-dir", default="data/tc/MIST_TC_keypoints")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    ap.add_argument("--null-n", type=int, default=60)
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--methods", nargs="+", default=["CC", "CSF"])
    ap.add_argument("--phase-fixed-m", type=float, default=None,
                    help="also gate with a phase-fixed null: lag-0 correlation of windows >= min-gap apart, "
                         "threshold at quantile q**(1/M)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    rows = json.load(open(a.rows, encoding="utf-8"))
    n = rows[0]["n"]
    seqs = load_sequences(a.kp_dir)
    prof = {"CC": cc_profile(8, a.features), "CSF": csf_profile(a.checkpoint)}
    prof = {m: p for m, p in prof.items() if m in a.methods}
    rng = np.random.default_rng(a.seed)
    null, null0 = {}, {}
    for (subj, seq), cams in sorted(seqs.items()):
        T = min(len(v) for v in cams.values()) - S * n
        for u, v in itertools.combinations(sorted(cams), 2):
            vals = {m: [] for m in prof}
            vals0 = {m: [] for m in prof}
            for _ in range(a.null_n * 5):
                if len(vals["CC"]) >= a.null_n:
                    break
                t1, t2 = rng.integers(0, T, 2)
                if abs(t1 - t2) < a.min_gap * 60:
                    continue
                x1, x2 = cams[u][t1:t1 + S * n:S], cams[v][t2:t2 + S * n:S]
                if min(np.isfinite(x1).all(-1).mean(), np.isfinite(x2).all(-1).mean()) < a.min_valid:
                    continue
                s1, s2 = KeypointSequence(_fill_gaps(x1)[0], FPS), KeypointSequence(_fill_gaps(x2)[0], FPS)
                for m, p in prof.items():
                    lg, sc = p(s1, s2)
                    vals[m].append(profile_scores(lg, sc, n=n)["peak"])
                    vals0[m].append(float(np.interp(0.0, np.asarray(lg, dtype=np.float64), np.asarray(sc, dtype=np.float64))))
            null[(subj, seq, f"{u}-{v}")] = vals
            null0[(subj, seq, f"{u}-{v}")] = vals0
        print(f"  null {subj}_{seq}", flush=True)
    out = {"n": n, "null_n": a.null_n}
    for m in prof:
        R = [r for r in rows if r["method"] == m]
        err = np.array([r["err"] for r in R]); pk = np.array([r["peak"] for r in R])
        succ = err <= 1
        out[m] = {"none": dict(kept=1.0, precision=float(succ.mean()), mean_err=float(err.mean()))}
        print(f"\n{m} (window {n}): no gate  precision {succ.mean():.0%}  mean err {err.mean():.2f}")
        variants = [("", q) for q in a.quantiles]
        if a.phase_fixed_m:
            variants += [("fixed_", q) for q in a.quantiles]
        for kind, q in variants:
            src = null0 if kind else null
            qq = q ** (1.0 / a.phase_fixed_m) if kind else q
            thr = np.array([np.quantile(src[(r['subj'], r['seq'], r['pair'])][m], qq)
                            if len(src[(r['subj'], r['seq'], r['pair'])][m]) >= 20 else np.inf for r in R])
            keep = pk >= thr
            d = dict(kept=float(keep.mean()), precision=float(succ[keep].mean()) if keep.any() else None,
                     failures_caught=float((~succ & ~keep).sum() / max(1, (~succ).sum())),
                     mean_err_kept=float(err[keep].mean()) if keep.any() else None,
                     pairs_with_any_kept=len({r["pair"] for r, k in zip(R, keep) if k}))
            # per activity: does a sustained rhythm (walking) inflate the null and reject good pairs?
            act = np.array([r["seq"].rstrip("0123456789") for r in R])
            d["by_activity"] = {}
            for g in sorted(set(act)):
                s_ = act == g
                d["by_activity"][g] = dict(
                    kept=float(keep[s_].mean()), success_rate=float(succ[s_].mean()),
                    precision=float(succ[s_ & keep].mean()) if (s_ & keep).any() else None,
                    successes_kept=float(keep[s_ & succ].mean()) if (s_ & succ).any() else None,
                    median_threshold=float(np.median(thr[s_][np.isfinite(thr[s_])])))
            out[m][f"{kind}q{q}"] = d
            print(f"  {kind}null q{q}: kept {d['kept']:.0%}  precision {d['precision']:.0%}  failures caught "
                  f"{d['failures_caught']:.0%}  mean err kept {d['mean_err_kept']:.2f}")
            for g, v in d["by_activity"].items():
                print(f"      {g:9s} success {v['success_rate']:.0%}  kept {v['kept']:.0%}  precision "
                      f"{v['precision']:.0%}  successes kept {v['successes_kept']:.0%}  median thr {v['median_threshold']:.3f}")
        z = pk - np.array([np.median(null[(r['subj'], r['seq'], r['pair'])][m] or [np.nan]) for r in R])
        out[m]["auc_peak"] = auc(pk, succ); out[m]["auc_peak_minus_null_median"] = auc(z, succ)
        print(f"  AUC raw peak {out[m]['auc_peak']:.3f}; peak minus pair null median {out[m]['auc_peak_minus_null_median']:.3f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
