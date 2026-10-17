# -*- coding: utf-8 -*-
"""Rig-level synchronization on TotalCapture (8 cameras around the volume): star vs spanning
tree vs observability-gated tree, as ap3d_camera_graph.py but with intermediate viewpoints that
can bridge camera groups.

Every camera of a clip gets its own known phase (0..3 frames at 60 fps = 0..1.5 frames at
30 fps); all pairs are estimated with CC over the clip and cameras are placed on camera 1's
timeline. Gate thresholds, neither tuned on TotalCapture:
  ap3d : the AthletePose3D whole-clip CC threshold (frozen)
  null : ground-truth-free, the q-quantile of CC peaks of the same camera pair at two moments
         >= --min-gap s apart in the same sequence (observability_null.py, within-take variant)

    python scripts/tc_camera_graph.py --clip-len 60 --out docs/paper/clips/tc_graph_L60.json
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

from ap3d_camera_graph import max_spanning_tree
from vsynclock.benchmark.drift import peak_in_range
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import cc_profile
from tc_benchmark import clip_starts, load_sequences

FPS, S = 30.0, 2


def component(nodes, good, root=1):
    comp, frontier = {root}, [root]
    while frontier:
        u = frontier.pop()
        for v in nodes:
            if v not in comp and tuple(sorted((u, v))) in good:
                comp.add(v); frontier.append(v)
    return comp


def ls_place(nodes, edges, rel, w, root=1, huber=0.5, iters=10):
    """Cycle-consistent placement: weighted least squares over all given edges,
    o_v - o_u = rel(u, v), with o_root = 0, re-weighted with a Huber loss so that edges
    inconsistent with the cycles they close are down-weighted (IRLS)."""
    idx = {c: i for i, c in enumerate(c for c in nodes if c != root)}
    E = [e for e in edges if e[0] in nodes and e[1] in nodes]
    if not E or not idx:
        return {root: 0.0}
    A = np.zeros((len(E), len(idx))); b = np.array([rel(u, v) for u, v in E])
    for k, (u, v) in enumerate(E):
        if v in idx: A[k, idx[v]] += 1
        if u in idx: A[k, idx[u]] -= 1
    base = np.array([max(w[e], 1e-3) for e in E]); rw = np.ones(len(E))
    for _ in range(iters):
        W = np.sqrt(base * rw)
        x = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
        r = np.abs(A @ x - b)
        rw = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
    out = {root: 0.0}
    out.update({c: float(x[i]) for c, i in idx.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--clip-len", type=int, default=60)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--choice", default="docs/paper/clips/observability_choice.json")
    ap.add_argument("--null-q", type=float, default=0.95)
    ap.add_argument("--null-n", type=int, default=200, help="null samples per camera pair and sequence")
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--prune-k", type=int, nargs="*", default=[],
                    help="also: keep only the k strongest edges of each camera before gating (rig_profile.py)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    seqs = load_sequences(a.kp_dir)
    prof = cc_profile(8, a.features)
    rng = np.random.default_rng(a.seed)
    n = a.clip_len
    thr_ap3d = json.load(open(a.choice))["CC"]["threshold"]
    pruned = tuple(f"gated_null_ls_k{k}" for k in a.prune_k)
    strategies = ("star", "tree", "ls_all", "gated_ap3d", "gated_null", "gated_null_ls") + pruned
    res = {m: [] for m in strategies}
    flagged = {m: [] for m in ("gated_ap3d", "gated_null", "gated_null_ls") + pruned}
    by_hops = {"tree": {}, "ls": {}}
    hops = []

    def stream(xy, start):
        x = xy[start:start + S * n:S]
        return x, np.isfinite(x).all(-1).mean()

    for (subj, seq), cams in sorted(seqs.items()):
        T60 = min(len(v) for v in cams.values())
        # ground-truth-free null for this sequence: same pair, unrelated moments
        thr_null = {}
        for u, v in itertools.combinations(sorted(cams), 2):
            null = []
            for _ in range(a.null_n * 3):
                if len(null) >= a.null_n:
                    break
                t1, t2 = rng.integers(0, T60 - S * n, 2)
                if abs(t1 - t2) < a.min_gap * 60:
                    continue
                (x1, v1), (x2, v2) = stream(cams[u], t1), stream(cams[v], t2)
                if min(v1, v2) < a.min_valid:
                    continue
                l, sc = prof(KeypointSequence(_fill_gaps(x1)[0], FPS), KeypointSequence(_fill_gaps(x2)[0], FPS))
                null.append(float(np.max(sc)))
            thr_null[(u, v)] = float(np.quantile(null, a.null_q)) if len(null) >= 20 else np.inf
        for c0 in clip_starts(T60, n, a.clips, 8, rng):
            phase = {c: int(rng.integers(0, 4)) for c in cams}
            st = {c: stream(cams[c], c0 + phase[c]) for c in cams}
            nodes = sorted(c for c in cams if st[c][1] >= a.min_valid)
            if 1 not in nodes or len(nodes) < 3:
                continue
            xs = {c: KeypointSequence(_fill_gaps(st[c][0])[0], FPS) for c in nodes}
            est, w = {}, {}
            for u, v in itertools.combinations(nodes, 2):
                lags, sc = prof(xs[u], xs[v])
                est[(u, v)] = peak_in_range(lags, sc); w[(u, v)] = float(np.max(sc))
            rel = lambda u, v: est[(u, v)] if (u, v) in est else -est[(v, u)]
            truth = {c: (phase[c] - phase[1]) / S for c in nodes}

            def place(parent):
                off = {1: 0.0}
                def get(c):
                    if c not in off:
                        off[c] = get(parent[c]) + rel(parent[c], c)
                    return off[c]
                for c in parent:
                    get(c)
                return off

            star = {c: (0.0 if c == 1 else rel(1, c)) for c in nodes}
            tree = place(max_spanning_tree(nodes, w, 1))
            lsall = ls_place(nodes, list(w), rel, w)
            for c in nodes[1:]:
                res["star"].append(abs(star[c] - truth[c])); res["tree"].append(abs(tree[c] - truth[c]))
                res["ls_all"].append(abs(lsall[c] - truth[c]))
            for name, good in (("gated_ap3d", {e for e, x in w.items() if x >= thr_ap3d}),
                               ("gated_null", {e for e, x in w.items() if x >= thr_null[e]})):
                comp = component(nodes, good)
                wg = {e: (x if e in good else -1e9) for e, x in w.items() if e[0] in comp and e[1] in comp}
                parent = max_spanning_tree(sorted(comp), wg, 1) if len(comp) > 1 else {1: None}
                off = place(parent) if len(comp) > 1 else {1: 0.0}
                for c in nodes[1:]:
                    flagged[name].append(c not in comp)
                    if c in comp:
                        res[name].append(abs(off[c] - truth[c]))
                        if name == "gated_null":
                            h, x = 0, c
                            while parent[x] is not None:
                                x = parent[x]; h += 1
                            hops.append(h)
                            by_hops["tree"].setdefault(min(h, 3), []).append(abs(off[c] - truth[c]))
                if name == "gated_null":
                    lsg = ls_place(sorted(comp), [e for e in good if e[0] in comp and e[1] in comp], rel, w)
                    for c in nodes[1:]:
                        flagged["gated_null_ls"].append(c not in comp)
                        if c in comp:
                            res["gated_null_ls"].append(abs(lsg[c] - truth[c]))
                            h, x = 0, c
                            while parent[x] is not None:
                                x = parent[x]; h += 1
                            by_hops["ls"].setdefault(min(h, 3), []).append(abs(lsg[c] - truth[c]))
                    for k in a.prune_k:
                        keep = set()
                        for c in nodes:
                            keep |= set(sorted((e for e in w if c in e), key=lambda e: -w[e])[:k])
                        gk = {e for e in good if e in keep}
                        ck = component(nodes, gk)
                        lsk = ls_place(sorted(ck), [e for e in gk if e[0] in ck and e[1] in ck], rel, w)
                        for c in nodes[1:]:
                            flagged[f"gated_null_ls_k{k}"].append(c not in ck)
                            if c in ck:
                                res[f"gated_null_ls_k{k}"].append(abs(lsk[c] - truth[c]))
        print(f"  {subj}_{seq} done", flush=True)

    out = {"clip_len": n, "thr_ap3d": thr_ap3d, "null_q": a.null_q}
    print(f"\nclip {n} frames @30fps; AP3D CC threshold {thr_ap3d:.3f}; null q{a.null_q}")
    for m in strategies:
        e = np.array(res[m])
        d = dict(placed=len(e), mean=float(e.mean()), median=float(np.median(e)),
                 le025=float(np.mean(e <= .25)), gt1=float(np.mean(e > 1)))
        if m in flagged:
            d["flagged"] = float(np.mean(flagged[m]))
        out[m] = d
        print(f"  {m:11s} placed {len(e):5d}" + (f" ({1 - d['flagged']:.0%})" if m in flagged else "       ") +
              f"  mean {d['mean']:.2f}  median {d['median']:.2f}  <=0.25 {d['le025']:.0%}  >1 {d['gt1']:.0%}")
    if hops:
        hc = np.bincount(hops)
        out["gated_null"]["hops"] = hc.tolist()
        print("  gated_null placed cameras by path length (edges):", {i: int(v) for i, v in enumerate(hc) if v})
    print("  gated (null) placed cameras, mean error by tree depth (3 = 3+): tree vs least squares")
    out["by_hops"] = {}
    for h in sorted(by_hops["tree"]):
        t_, l_ = np.array(by_hops["tree"][h]), np.array(by_hops["ls"].get(h, []))
        out["by_hops"][str(h)] = dict(n=len(t_), tree=float(t_.mean()), ls=float(l_.mean()),
                                      tree_gt1=float(np.mean(t_ > 1)), ls_gt1=float(np.mean(l_ > 1)))
        print(f"    depth {h}: n={len(t_):4d}  tree {t_.mean():.2f} (>1 {np.mean(t_ > 1):.0%})  LS {l_.mean():.2f} (>1 {np.mean(l_ > 1):.0%})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
