# -*- coding: utf-8 -*-
"""Two-stage router (fixed on TotalCapture) applied to the short clips of AthletePose3D and SportsPose.

D = median over the clip's four quarters of |CC quarter peak - CC clip peak| (as tc_hybrid2.py); CSF iff
D >= delta, for the cost point delta = 3 and the accuracy point delta = 0.75. Nothing is tuned here.
Pre-registration: docs/paper/DEV_PROTOCOL.md ("router transfer to short clips").

    python scripts/short_router.py --dataset ap3d
    python scripts/short_router.py --dataset sp
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
from realdata_eval import cc_profile

DELTAS = (3.0, 0.75)


def consistency(prof, A, B, n):
    try:
        full = peak_in_range(*prof(KeypointSequence(A, 30.0), KeypointSequence(B, 30.0)))
    except ValueError:
        return np.inf
    q, qs = n // 4, []
    for i in range(4):
        try:
            qs.append(peak_in_range(*prof(KeypointSequence(A[i * q:(i + 1) * q], 30.0),
                                          KeypointSequence(B[i * q:(i + 1) * q], 30.0))))
        except ValueError:
            pass
    d = float(np.median(np.abs(np.array(qs) - full))) if qs else np.nan
    return d if np.isfinite(d) else np.inf


def ap3d_samples(rows):
    from ap3d_benchmark import decimated, load_trials
    trials, _ = load_trials("data/ap3d/keypoints", 0.5)
    S, s0 = 4, 4
    for r in rows:
        cams = trials[(r["subj"], r["split"], r["trial"])]
        ca, cb = (int(x) for x in r["pair"].split("-"))
        n = r["n"]
        yield ((r["subj"], r["split"], r["trial"]), decimated(cams[ca][0], s0, S, n),
               decimated(cams[cb][0], s0 + S * r["j"] + r["k"], S, n), n)


def sp_samples(rows):
    import sp_benchmark as spb
    cache = {}
    for r in rows:
        if r["subj"] not in cache:
            cache = {r["subj"]: spb.load_subject("data/sportpose", r["subj"])[1]}
        tr = cache[r["subj"]][r["trial"]]
        u, v = (int(x) for x in r["pair"].split("-"))
        S, s0, n = spb.S, spb.S, r["n"]
        yield ((r["subj"], r["trial"]), spb.stream(tr["sync_up"][u], s0, n)[0],
               spb.stream(tr["sync_up"][v], s0 + S * r["j"] + r["k"], n)[0], n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ap3d", "sp"), required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    os.chdir(ROOT)
    src = {"ap3d": "data/final_rows/ap3d_rows.json", "sp": "data/final_rows/sp_rows_test.json"}[a.dataset]
    rows = json.load(open(src, encoding="utf-8"))
    by = collections.defaultdict(dict)
    for r in rows:
        key = tuple(r[f] for f in ("subj", "split", "trial", "pair", "j", "k") if f in r)
        by[key][r["method"]] = r
    keys = sorted(k for k, v in by.items() if "CC" in v and "CSF" in v)
    cc_rows = [by[k]["CC"] for k in keys]
    prof = cc_profile(8, "speed+vyb")
    gen = (ap3d_samples if a.dataset == "ap3d" else sp_samples)(cc_rows)
    D, trial, lens = [], [], []
    for i, (tk, A, B, n) in enumerate(gen):
        D.append(consistency(prof, A, B, n)); trial.append(tk); lens.append(n)
        if i % 2000 == 0:
            print(f"  {i}/{len(keys)}", flush=True)
    D = np.array(D)
    eCC = np.array([by[k]["CC"]["err"] for k in keys], dtype=float)
    eCSF = np.array([by[k]["CSF"]["err"] for k in keys], dtype=float)
    T = sorted(set(trial)); tid = np.array([T.index(t) for t in trial])

    def tmean(e):
        return np.array([np.nanmean(e[tid == t]) for t in range(len(T))])

    rng = np.random.default_rng(0)
    out = dict(n_samples=len(keys), n_trials=len(T), clip_frames=[int(np.min(lens)), int(np.median(lens)), int(np.max(lens))],
               D_inf_share=float(np.mean(~np.isfinite(D))))
    print(f"{len(keys)} samples, {len(T)} trials; clip frames min/median/max {out['clip_frames']}; "
          f"D undefined {out['D_inf_share']:.0%}")
    for name, err, use in [("CC", eCC, np.zeros(len(D), bool)), ("CSF", eCSF, np.ones(len(D), bool))] + \
            [(f"two-stage d={d}", np.where(D >= d, eCSF, eCC), D >= d) for d in DELTAS]:
        tm = tmean(err)
        d = dict(mean=float(tm.mean()), gt1=float(np.nanmean(err > 1)), csf_share=float(use.mean()))
        if name.startswith("two"):
            for ref_name, ref in (("CC", eCC), ("CSF", eCSF)):
                diff = tm - tmean(ref)
                b = diff[rng.integers(0, len(diff), (a.n_boot, len(diff)))].mean(1)
                d[f"minus_{ref_name}"] = [float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out[name] = d
        print(f"  {name:18s} {d['mean']:.3f}  >1 {d['gt1']:.0%}  CSF {d['csf_share']:.0%}" +
              "".join(f"  vs {r} {d[f'minus_{r}'][0]:+.3f} [{d[f'minus_{r}'][1]:+.3f}, {d[f'minus_{r}'][2]:+.3f}]"
                      for r in ("CC", "CSF") if f"minus_{r}" in d))
    os.makedirs("data/refine", exist_ok=True)
    json.dump(out, open(f"data/refine/router_{a.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
