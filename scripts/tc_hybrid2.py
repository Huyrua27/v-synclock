# -*- coding: utf-8 -*-
"""Two-stage (post-CC) routing: run CC on every sample, call CSF only when the CC lag profile is
ambiguous (margin p1 - p2 below mu), optionally only for rhythmic clips (periodicity > 0.40).
Selection rule and grid: docs/paper/DEV_PROTOCOL.md ("Two-stage routing", written before results).

    python scripts/tc_hybrid2.py --rows docs/paper/clips/tc_dev_rows_L300.json \
        --kp-dir data/tc/tc_keypoints_dev --out docs/paper/clips/tc_dev_hybrid2_L300.json
    python scripts/tc_hybrid2.py --select M 0.10 --out docs/paper/clips/tc_hybrid2_L300.json   # test, once
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.drift import peak_in_range
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import cc_profile
from tc_benchmark import load_sequences
from tc_hybrid import COST, S, periodicity

TAU = 0.40                                   # frozen periodicity threshold (rule 2)
CLIP_MS = {"CC": 1.19, "Q": 3.60, "CSF": 45.4}  # 10-s clip, one thread (round 2)


def margin(lags, sc):
    i = int(np.argmax(sc))
    loc = [q for q in range(1, len(sc) - 1) if sc[q] >= sc[q - 1] and sc[q] >= sc[q + 1] and abs(q - i) >= 2]
    for q in (0, len(sc) - 1):                                   # edge maxima count too
        nb = q + 1 if q == 0 else q - 1
        if sc[q] >= sc[nb] and abs(q - i) >= 2:
            loc.append(q)
    p2 = max((sc[q] for q in loc), default=-1.0)
    return float(sc[i] - p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_rows_L300.json")
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--select", nargs=2, default=None, metavar=("VARIANT", "MU"),
                    help="evaluate only this rule (test); otherwise sweep and select (dev)")
    ap.add_argument("--features", default="speed", help="CC input for the routing signal (match the rows)")
    ap.add_argument("--round", type=int, default=1, help="1: margin rules (M, M+P); 2: temporal consistency (D)")
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
    keys = sorted(k for k, v in by.items() if "CC" in v and "CSF" in v)
    prof = cc_profile(8, a.features)
    per, M, D = {}, [], []
    for (subj, seq, c0, pair, j, k) in keys:
        n = by[(subj, seq, c0, pair, j, k)]["CC"]["n"]
        ca, cb = (int(x) for x in pair.split("-"))
        cams = seqs[(subj, seq)]
        if (subj, seq, c0, ca) not in per:
            per[(subj, seq, c0, ca)] = periodicity(_fill_gaps(cams[ca][c0:c0 + S * n:S])[0])
        A = _fill_gaps(cams[ca][c0:c0 + S * n:S])[0]
        B = _fill_gaps(cams[cb][c0 + S * j + k:c0 + S * j + k + S * n:S])[0]
        try:
            l, sc = prof(KeypointSequence(A, 30.0), KeypointSequence(B, 30.0))
            M.append(margin(l, sc)); full = peak_in_range(l, sc)
        except ValueError:
            M.append(-np.inf); full = np.nan                    # no CC evidence: always ambiguous
        q, qs = n // 4, []
        for i in range(4):
            try:
                qs.append(peak_in_range(*prof(KeypointSequence(A[i * q:(i + 1) * q], 30.0),
                                              KeypointSequence(B[i * q:(i + 1) * q], 30.0))))
            except ValueError:
                pass
        d = float(np.median(np.abs(np.array(qs) - full))) if qs else np.nan
        D.append(d if np.isfinite(d) else np.inf)
    M, D = np.array(M), np.array(D)
    P = np.array([per[(k[0], k[1], k[2], int(k[3].split("-")[0]))] for k in keys])
    eCC = np.array([by[k]["CC"]["err"] for k in keys]); eCSF = np.array([by[k]["CSF"]["err"] for k in keys])
    seq = np.array([f"{k[0]}_{k[1]}" for k in keys]); act = np.array([k[1].rstrip("0123456789") for k in keys])
    S_ = sorted(set(seq))
    seq_mean = lambda e: float(np.mean([e[seq == s].mean() for s in S_]))
    print(f"{len(keys)} samples, {len(S_)} sequences; CC {seq_mean(eCC):.3f}, CSF {seq_mean(eCSF):.3f} (per-sequence mean)")

    def rule(variant, mu):
        if variant == "D":                                      # round 2: temporal consistency
            return D >= mu
        use = M < mu
        if variant == "M+P":
            use &= P > TAU
        return use

    def summary(use, two_stage=True, quarters=False):
        err = np.where(use, eCSF, eCC)
        cost = ((COST["CC"] * (4 if quarters else 1) if two_stage else 0) +
                np.where(use, COST["CSF"], 0 if two_stage else COST["CC"])).mean() / COST["CC"]
        d = dict(seq_mean=seq_mean(err), gt1=float(np.mean(err > 1)), csf_share=float(np.mean(use)), cost_vs_cc=float(cost))
        if quarters:
            d["cost_vs_cc_clip"] = float((CLIP_MS["CC"] + CLIP_MS["Q"] + np.mean(use) * CLIP_MS["CSF"]) / CLIP_MS["CC"])
        return d

    out = {"CC": summary(np.zeros(len(M), bool)), "CSF": summary(np.ones(len(M), bool), two_stage=False),
           "periodicity_hybrid_tau040": summary(P > TAU, two_stage=False), "sweep": {}}
    out["CC"]["cost_vs_cc"] = 1.0
    for nm in ("CC", "CSF", "periodicity_hybrid_tau040"):
        d = out[nm]; print(f"  {nm:28s} {d['seq_mean']:.3f}  >1 {d['gt1']:.0%}  CSF {d['csf_share']:.0%}  cost {d['cost_vs_cc']:.1f}x")
    if a.select is None:
        cands = []
        grid = ([(v, mu) for v in ("M", "M+P") for mu in np.round(np.arange(0, 0.401, 0.02), 2)] if a.round == 1
                else [("D", mu) for mu in np.round(np.arange(0.25, 3.001, 0.25), 2)])
        for variant, mu in grid:
                d = summary(rule(variant, mu), quarters=(variant == "D")); out["sweep"][f"{variant}_{mu}"] = d
                ok = d["seq_mean"] <= out["CSF"]["seq_mean"] + 0.05
                print(f"  {variant:3s} mu={mu:.2f}  {d['seq_mean']:.3f}  >1 {d['gt1']:.0%}  CSF {d['csf_share']:.0%}  "
                      f"cost {d['cost_vs_cc']:.1f}x" + (f" (clip {d['cost_vs_cc_clip']:.1f}x)" if "cost_vs_cc_clip" in d else "")
                      + ("  qualifies" if ok else ""))
                if ok:
                    cands.append((d["cost_vs_cc"], variant, mu))
        if cands:
            best = min(c for c, _, _ in cands)
            tied = [(v != "M", mu, v) for c, v, mu in cands if c <= best + 0.1]
            _, mu, v = min(tied)
            if a.round == 2:                                      # round 2: largest qualifying delta
                _, v, mu = max(cands, key=lambda x: x[2])
            out["selected"] = [v, float(mu)]
            print(f"\nselected on this split: {v} mu={mu:.2f}")
        else:
            out["selected"] = None
            print("\nno rule within 0.05 frames of CSF-only: two-stage routing not useful")
    else:
        v, mu = a.select[0], float(a.select[1])
        use = rule(v, mu); err = np.where(use, eCSF, eCC)
        out["selected"] = [v, mu]; out["result"] = summary(use, quarters=(v == "D"))
        d = out["result"]
        print(f"\n  two-stage {v} mu={mu:.2f}: {d['seq_mean']:.3f}  >1 {d['gt1']:.0%}  CSF {d['csf_share']:.0%}  cost {d['cost_vs_cc']:.1f}x"
              + (f" (clip {d['cost_vs_cc_clip']:.1f}x)" if "cost_vs_cc_clip" in d else ""))
        rng = np.random.default_rng(0)
        hyb = np.where(P > TAU, eCSF, eCC)
        for ref_name, ref in (("CC", eCC), ("CSF", eCSF), ("periodicity_hybrid", hyb)):
            diff = np.array([err[seq == s].mean() - ref[seq == s].mean() for s in S_])
            b = diff[rng.integers(0, len(diff), (a.n_boot, len(diff)))].mean(1)
            out[f"minus_{ref_name}"] = [float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            print(f"  two-stage - {ref_name}: {diff.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
        out["by_activity"] = {g: dict(err=float(err[act == g].mean()), cc=float(eCC[act == g].mean()),
                                      csf=float(eCSF[act == g].mean()), csf_share=float(use[act == g].mean()))
                              for g in ("walking", "acting", "freestyle")}
        for g, d in out["by_activity"].items():
            print(f"    {g:9s} two-stage {d['err']:.3f}  CC {d['cc']:.3f}  CSF {d['csf']:.3f}  CSF share {d['csf_share']:.0%}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
