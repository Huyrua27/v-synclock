# -*- coding: utf-8 -*-
"""Rig-level synchronization on AthletePose3D: which camera pairs to trust.

Each camera of a trial receives its own known phase (0..7 frames at 120 fps -> 0..1.75 frames
at 30 fps). All camera pairs are estimated with whole-clip CC; every camera must then be
placed on the timeline of camera 1:
  star  : every camera paired directly with camera 1
  tree  : maximum-observability spanning tree from camera 1 (profile peak as edge weight)
  gated : same tree, but edges below the observability threshold chosen on the dev athletes
          are removed; cameras no longer connected to camera 1 are reported as
          'not synchronizable' instead of receiving an estimate.

    python scripts/ap3d_camera_graph.py --subjects S2
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

from ap3d_benchmark import decimated, load_trials
from vsynclock.benchmark.drift import peak_in_range
from vsynclock.core.types import KeypointSequence
from realdata_eval import cc_profile

FPS, S = 30.0, 4


def max_spanning_tree(nodes, w, root):
    """Prim: returns parent map maximizing total edge weight w[(u, v)]."""
    parent, inside = {root: None}, {root}
    while len(inside) < len(nodes):
        best = max(((w[tuple(sorted((u, v)))], u, v) for u in inside for v in nodes if v not in inside))
        parent[best[2]] = best[1]; inside.add(best[2])
    return parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/ap3d/keypoints")
    ap.add_argument("--subjects", nargs="+", default=["S2"])
    ap.add_argument("--choice", default="docs/paper/clips/observability_choice.json",
                    help="whole-clip observability threshold selected on the dev athletes")
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--threshold", type=float, default=None, help="override the gate threshold")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    thr = a.threshold if a.threshold is not None else json.load(open(a.choice))["CC"]["threshold"]
    trials, _ = load_trials(a.kp_dir, 0.5)
    prof = cc_profile(8, a.features)
    rng = np.random.default_rng(a.seed)
    res = {m: [] for m in ("star", "tree", "ls_all", "gated", "gated_ls")}
    flagged, comp_ok = [], []
    for key in sorted(k for k in trials if k[0] in a.subjects):
        cams = trials[key]
        nodes = sorted(cams)
        if 1 not in cams or len(nodes) < 2:
            continue
        phase = {c: int(rng.integers(0, 8)) for c in nodes}           # 120-fps frames
        T = min(len(v[0]) for v in cams.values())
        n = (T - 8) // S
        stream = {c: decimated(cams[c][0], phase[c], S, n) for c in nodes}
        est, w = {}, {}
        for u, v in itertools.combinations(nodes, 2):
            lags, sc = prof(KeypointSequence(stream[u], FPS), KeypointSequence(stream[v], FPS))
            est[(u, v)] = peak_in_range(lags, sc)                      # offset of v relative to u
            w[(u, v)] = float(np.max(sc))
        rel = lambda u, v: est[(u, v)] if (u, v) in est else -est[(v, u)]
        truth = {c: (phase[c] - phase[1]) / S for c in nodes}

        def place(parent):
            off = {1: 0.0}
            def get(c):
                if c not in off:
                    off[c] = get(parent[c]) + rel(parent[c], c)
                return off[c]
            for c in parent: get(c)
            return off

        star = {c: (0.0 if c == 1 else rel(1, c)) for c in nodes}
        tree = place(max_spanning_tree(nodes, w, 1))
        # gated: keep confident edges only, then tree within camera 1's component
        good = {e for e, x in w.items() if x >= thr}
        comp, frontier = {1}, [1]
        while frontier:
            u = frontier.pop()
            for v in nodes:
                if v not in comp and tuple(sorted((u, v))) in good:
                    comp.add(v); frontier.append(v)
        wg = {e: (x if e in good else -1e9) for e, x in w.items()}
        gtree = place(max_spanning_tree(sorted(comp), wg, 1)) if len(comp) > 1 else {1: 0.0}
        from tc_camera_graph import ls_place
        lsall = ls_place(nodes, list(w), rel, w)
        lsg = ls_place(sorted(comp), [e for e in good if e[0] in comp and e[1] in comp], rel, w) if len(comp) > 1 else {1: 0.0}
        for c in nodes[1:]:
            res["ls_all"].append(abs(lsall[c] - truth[c]))
            if c in comp:
                res["gated_ls"].append(abs(lsg[c] - truth[c]))
            res["star"].append(abs(star[c] - truth[c]))
            res["tree"].append(abs(tree[c] - truth[c]))
            if c in comp:
                res["gated"].append(abs(gtree[c] - truth[c]))
            flagged.append(c not in comp)
        # cameras outside camera 1's component: are they consistent among themselves?
        rest = [c for c in nodes if c not in comp]
        for u, v in itertools.combinations(rest, 2):
            if tuple(sorted((u, v))) in good:
                comp_ok.append(abs(rel(u, v) - (truth[v] - truth[u])))
    print(f"subjects {a.subjects}; threshold (dev) peak >= {thr:.3f}; {len(flagged)} non-reference cameras")
    for m, e in res.items():
        e = np.array(e)
        print(f"  {m:6s} cameras placed {len(e):4d}  mean err {e.mean():.2f}  median {np.median(e):.2f}  "
              f"<=0.25 {np.mean(e <= .25):4.0%}  fail>1 {np.mean(e > 1):4.0%}")
    print(f"  gated: {np.mean(flagged):.0%} of cameras reported as not synchronizable to camera 1")
    if comp_ok:
        e = np.array(comp_ok)
        print(f"  confident edges inside the unreachable group: {len(e)}, mean err {e.mean():.2f}, <=0.25 {np.mean(e <= .25):.0%}")
    if a.out:
        json.dump({"threshold": thr, "features": a.features, "flagged": float(np.mean(flagged)),
                   **{m: dict(placed=len(e), mean=float(np.mean(e)), gt1=float(np.mean(np.array(e) > 1)))
                      for m, e in res.items()}}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
