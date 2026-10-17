# -*- coding: utf-8 -*-
"""SportsPose replication: estimators, viewpoint and camera graph on real 90-fps sports video.

Plan fixed in docs/paper/DEV_PROTOCOL.md (SportsPose section) before running.
  * Synchronized streams: camera v at pose frame i is video frame video_index[v, i]; the first
    and last two pose frames are dropped.
  * Decimation 90 -> 30 fps: A keeps s0 + 3m, B keeps s0 + 3j + k + 3m -> B lags A by j + k/3.
  * All 21 camera pairs, whole trial, CC / DTW / Caspi-style / CSF; profile peak stored.
  * Subject-centred view angle from the SportsPose 3D hips and camera centres -R^T T.
  * Camera graph (--graph): random per-camera phases; star / tree / least squares / gated tree /
    gated least squares, gate = q-quantile of a cross-trial null of the same camera pair.

    python scripts/sp_benchmark.py --subjects S06 S12 S19 --out docs/paper/clips/sp_rows.json
    python scripts/sp_benchmark.py --subjects S06 S12 S19 --graph --out docs/paper/clips/sp_graph.json
"""
import argparse
import collections
import glob
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
from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation, DTW
from vsynclock.benchmark.drift import peak_in_range
from vsynclock.benchmark.observability import auc, profile_scores
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from realdata_eval import cc_profile, csf_predictor, csf_profile
from sp_extract_pose import load_pkl
from tc_camera_graph import component, ls_place

FPS, S, EDGE = 30.0, 3, 2
BINS = [(0, 30), (30, 60), (60, 75), (75, 90), (90, 105), (105, 120), (120, 150), (150, 181)]


def load_subject(root, subj):
    """-> calib list, {trial_id: dict(act, sync (7, T, 19, 2), hips (T, 3))}"""
    cpk = load_pkl(os.path.join(root, "ex", "data", "indoor", subj, "calib.pkl"))
    cal, n_rot = cpk["calibration"], [int(r) for r in cpk["numtimesrot90clockwise"]]
    trials = {}
    for f in sorted(glob.glob(os.path.join(root, "ex", "data", "indoor", subj, "*", "*.npy"))):
        t = load_pkl(f.replace(".npy", "_timing.pkl"))
        rec = str(t["video_path"][-1]); vi = np.asarray(t["video_index"])
        kp = []
        for c in range(7):
            p = os.path.join(root, "keypoints", subj, f"{rec}_CAM{c}.npz")
            if not os.path.exists(p):
                kp = None; break
            kp.append(np.load(p)["xy"].astype(np.float64)[vi[c]])
        if kp is None:
            continue
        X = np.load(f, allow_pickle=False)
        sl = slice(EDGE, X.shape[0] - EDGE)
        sync = np.stack(kp)[:, sl]
        trials[os.path.basename(f)[:-4]] = dict(act=os.path.basename(os.path.dirname(f)), sync=sync,
                                                sync_up=np.stack([upright(sync[c], n_rot[c]) for c in range(7)]),
                                                hips=X[sl][:, [11, 12]].mean(1))
    return cal, trials


def upright(xy, n):
    """Keypoints in display-upright axes (sensor image turned n x 90 deg clockwise), up to a constant
    offset; only velocities matter. Needed for features that use the image vertical (CC-vy)."""
    x, y = xy[..., 0], xy[..., 1]
    return np.stack({0: (x, y), 1: (-y, x), 2: (-x, -y), 3: (y, -x)}[n % 4], axis=-1)


def geometry(cal):
    centres = {c: -np.asarray(k["R"]).T @ np.asarray(k["T"]) for c, k in enumerate(cal)}
    axes = {c: np.asarray(k["R"])[2] for c, k in enumerate(cal)}
    return centres, axes


def view_angle(hips, centres, u, v, frames):
    h = hips[np.clip(frames, 0, len(hips) - 1)]
    du, dv = centres[u] - h, centres[v] - h
    cos = (du * dv).sum(-1) / np.linalg.norm(du, axis=-1) / np.linalg.norm(dv, axis=-1)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def stream(x, start, n):
    s = x[start:start + S * n:S]
    return _fill_gaps(s)[0], float(np.isfinite(s).all(-1).mean())


def benchmark(a, subjects):
    cc = CrossCorrelation(max_lag=8, features=a.features)
    dtw = DTW(oversample_factor=4, max_warp_frames=8.0, max_frames=None, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.98, 1.02), alpha_steps=21, max_lag=8, features=a.features)
    est = {"CC": lambda x, y: cc.predict(x, y).dt_frames, "DTW": lambda x, y: dtw.predict(x, y).dt_frames,
           "Caspi-style": lambda x, y: (lambda e: e.beta + (e.alpha - 1) * (x.T - 1) / 2)(caspi.estimate(x, y)),
           "CC-vy": (lambda c: lambda x, y: c.predict(x, y).dt_frames)(CrossCorrelation(max_lag=8, features="speed+vy"))}
    if "CSF" in a.methods:
        est["CSF"] = csf_predictor(a.checkpoint)
    est = {m: est[m] for m in a.methods}
    prof = {"CC": cc_profile(8, a.features), "CC-vy": cc_profile(8, "speed+vy")}
    if "CSF" in a.methods:
        prof["CSF"] = csf_profile(a.checkpoint)
    rows = []
    for subj in subjects:
        cal, trials = load_subject(a.root, subj)
        centres, axes = geometry(cal)
        for tid, tr in trials.items():
            T = tr["sync"].shape[1]; s0 = S
            for u, v in itertools.combinations(range(7), 2):
                axis = float(np.degrees(np.arccos(np.clip(axes[u] @ axes[v], -1, 1))))
                for j in (-1, 0, 1):
                    for k in range(S):
                        sh = S * j + k
                        n = (T - max(s0, s0 + sh)) // S
                        A, va = stream(tr["sync"][u], s0, n); B, vb = stream(tr["sync"][v], s0 + sh, n)
                        if min(va, vb) < a.min_valid:
                            continue
                        sa, sb = KeypointSequence(A, FPS), KeypointSequence(B, FPS)
                        up = (KeypointSequence(stream(tr["sync_up"][u], s0, n)[0], FPS),
                              KeypointSequence(stream(tr["sync_up"][v], s0 + sh, n)[0], FPS))
                        inp = lambda m: up if (m == "CC-vy" or a.features != "speed") else (sa, sb)
                        peaks = {m: profile_scores(*p(*inp(m)), n=n)["peak"] for m, p in prof.items()}
                        ang = view_angle(tr["hips"], centres, u, v, s0 + S * np.arange(n))
                        for m, f in est.items():
                            try:
                                e = float(f(*inp(m)))
                            except Exception:
                                e = float("nan")
                            rows.append(dict(subj=subj, trial=tid, act=tr["act"], pair=f"{u}-{v}", j=j, k=k,
                                             truth=j + k / S, n=n, valid=min(va, vb), method=m, est=e,
                                             err=abs(e - (j + k / S)), peak=peaks.get(m, peaks["CC"]),
                                             view=ang, axis=axis))
            print(f"  {subj} {tid}: {len(rows)} rows", flush=True)
    return rows


def summarize(rows, n_boot=10000):
    rng = np.random.default_rng(0)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = {"methods": {}, "bins": {}, "auc": {}, "activity": {}}
    print(f"\n{len(rows) // len(methods)} pair-offset samples, {len({(r['subj'], r['trial']) for r in rows})} trials")
    print(f"{'method':12s} {'mean [95% CI over trials]':>28s} {'<=.25':>6s} {'>1':>5s}  per subject")
    for m in methods:
        R = [r for r in rows if r["method"] == m and np.isfinite(r["err"])]
        per = collections.defaultdict(list)
        for r in R:
            per[(r["subj"], r["trial"])].append(r["err"])
        tm = np.array([np.mean(v) for v in per.values()])
        b = tm[rng.integers(0, len(tm), (n_boot, len(tm)))].mean(1)
        e = np.array([r["err"] for r in R])
        subj = {s: float(np.mean([np.mean(v) for k_, v in per.items() if k_[0] == s])) for s in sorted({k_[0] for k_ in per})}
        out["methods"][m] = dict(mean=float(tm.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                 le025=float(np.mean(e <= .25)), gt1=float(np.mean(e > 1)), per_subject=subj)
        print(f"{m:12s} {tm.mean():8.3f} [{np.percentile(b, 2.5):6.3f}, {np.percentile(b, 97.5):6.3f}]     "
              f"{np.mean(e <= .25):6.0%} {np.mean(e > 1):5.0%}  " + " ".join(f"{s}:{v:.3f}" for s, v in subj.items()))
    # paired differences between estimators (trial level)
    out["paired"] = {}
    per = {m: collections.defaultdict(list) for m in methods}
    for r in rows:
        if np.isfinite(r["err"]):
            per[r["method"]][(r["subj"], r["trial"])].append(r["err"])
    keys = sorted(set.intersection(*[set(p) for p in per.values()]))
    for a_, b_ in (("CSF", "CC"), ("CSF", "DTW"), ("DTW", "CC"), ("Caspi-style", "CC")):
        if a_ not in per or b_ not in per:
            continue
        d = np.array([np.mean(per[a_][k]) - np.mean(per[b_][k]) for k in keys])
        bb = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
        out["paired"][f"{a_}-{b_}"] = [float(d.mean()), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
        print(f"  paired {a_} - {b_}: {d.mean():+.3f} [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}]")
    print("\nby activity: mean error (share >1)")
    for act in sorted({r["act"] for r in rows}):
        out["activity"][act] = {}
        line = []
        for m in methods:
            e = np.array([r["err"] for r in rows if r["method"] == m and r["act"] == act and np.isfinite(r["err"])])
            out["activity"][act][m] = dict(mean=float(e.mean()), gt1=float(np.mean(e > 1)))
            line.append(f"{m}:{e.mean():.2f}({np.mean(e > 1):.0%})")
        print(f"  {act:15s} " + "  ".join(line))
    print("\nfailure rate (>1 frame) by subject-centred view angle")
    for b0, b1 in BINS:
        d = {}
        for m in methods:
            e = np.array([r["err"] for r in rows if r["method"] == m and b0 <= r["view"] < b1 and np.isfinite(r["err"])])
            d[m] = dict(n=len(e), gt1=float(np.mean(e > 1)) if len(e) else None, mean=float(e.mean()) if len(e) else None)
        out["bins"][f"{b0}-{b1}"] = d
        if d[methods[0]]["n"]:
            print(f"  {b0:4d}-{b1:<4d} n={d[methods[0]]['n']:5d} " + "  ".join(f"{m}:{d[m]['gt1']:.0%}({d[m]['mean']:.2f})" for m in methods))
    print("\nAUC for success (|err|<=1): angle / |cos angle| / peak")
    for m in methods:
        R = [r for r in rows if r["method"] == m and np.isfinite(r["err"])]
        ok = np.array([r["err"] <= 1 for r in R])
        sc = {"-view": -np.array([r["view"] for r in R]), "|cos view|": np.abs(np.cos(np.radians([r["view"] for r in R]))),
              "peak": np.array([r["peak"] for r in R])}
        out["auc"][m] = {k: auc(v, ok) for k, v in sc.items()}
        print(f"  {m:12s} " + "  ".join(f"{k} {v:.3f}" for k, v in out["auc"][m].items()))
    return out


def graph(a, subjects):
    prof = cc_profile(8, a.features); rng = np.random.default_rng(a.seed)
    key = "sync_up" if a.features != "speed" else "sync"
    strategies = ("star", "tree", "ls_all", "gated", "gated_ls")
    res = {m: [] for m in strategies}; flagged = []; depth = {"tree": collections.defaultdict(list), "ls": collections.defaultdict(list)}
    for subj in subjects:
        cal, trials = load_subject(a.root, subj)
        tids = sorted(trials); T = min(trials[t]["sync"].shape[1] for t in tids); n = (T - 6) // S
        # cross-trial null per camera pair: same pair, two different trials -> no shared timing
        thr = {}
        for u, v in itertools.combinations(range(7), 2):
            vals = []
            for _ in range(a.null_n):
                t1, t2 = rng.choice(len(tids), 2, replace=False)
                A, va = stream(trials[tids[t1]][key][u], S, n); B, vb = stream(trials[tids[t2]][key][v], S, n)
                if min(va, vb) >= a.min_valid:
                    vals.append(float(np.max(prof(KeypointSequence(A, FPS), KeypointSequence(B, FPS))[1])))
            thr[(u, v)] = float(np.quantile(vals, a.null_q)) if len(vals) >= 20 else np.inf
        for tid in tids:
            sync = trials[tid][key]
            phase = {c: int(rng.integers(0, 2 * S)) for c in range(7)}
            st = {c: stream(sync[c], phase[c], n) for c in range(7)}
            nodes = sorted(c for c in range(7) if st[c][1] >= a.min_valid)
            if 0 not in nodes or len(nodes) < 3:
                continue
            xs = {c: KeypointSequence(st[c][0], FPS) for c in nodes}
            est, w = {}, {}
            for u, v in itertools.combinations(nodes, 2):
                l, sc = prof(xs[u], xs[v]); est[(u, v)] = peak_in_range(l, sc); w[(u, v)] = float(np.max(sc))
            rel = lambda u, v: est[(u, v)] if (u, v) in est else -est[(v, u)]
            truth = {c: (phase[c] - phase[0]) / S for c in nodes}

            def place(parent):
                off = {0: 0.0}
                def get(c):
                    if c not in off:
                        off[c] = get(parent[c]) + rel(parent[c], c)
                    return off[c]
                for c in parent:
                    get(c)
                return off
            star = {c: (0.0 if c == 0 else rel(0, c)) for c in nodes}
            tree = place(max_spanning_tree(nodes, w, 0))
            lsall = ls_place(nodes, list(w), rel, w, root=0)
            good = {e for e, x in w.items() if x >= thr[e]}
            comp = component(nodes, good, root=0)
            wg = {e: (x if e in good else -1e9) for e, x in w.items() if e[0] in comp and e[1] in comp}
            parent = max_spanning_tree(sorted(comp), wg, 0) if len(comp) > 1 else {0: None}
            gt = place(parent) if len(comp) > 1 else {0: 0.0}
            gl = ls_place(sorted(comp), [e for e in good if e[0] in comp and e[1] in comp], rel, w, root=0) if len(comp) > 1 else {0: 0.0}
            for c in nodes[1:]:
                res["star"].append(abs(star[c] - truth[c])); res["tree"].append(abs(tree[c] - truth[c]))
                res["ls_all"].append(abs(lsall[c] - truth[c])); flagged.append(c not in comp)
                if c in comp:
                    res["gated"].append(abs(gt[c] - truth[c])); res["gated_ls"].append(abs(gl[c] - truth[c]))
                    h, x = 0, c
                    while parent[x] is not None:
                        x = parent[x]; h += 1
                    depth["tree"][min(h, 3)].append(abs(gt[c] - truth[c])); depth["ls"][min(h, 3)].append(abs(gl[c] - truth[c]))
        print(f"  graph {subj} done", flush=True)
    out = {"null_q": a.null_q}
    print(f"\ncamera graph, {len(flagged)} non-reference cameras; gated places {1 - np.mean(flagged):.0%}")
    for m in strategies:
        e = np.array(res[m])
        out[m] = dict(placed=len(e), mean=float(e.mean()), median=float(np.median(e)), gt1=float(np.mean(e > 1)))
        print(f"  {m:9s} placed {len(e):4d}  mean {e.mean():.2f}  median {np.median(e):.2f}  >1 {np.mean(e > 1):.0%}")
    out["by_depth"] = {}
    for h in sorted(depth["tree"]):
        t_, l_ = np.array(depth["tree"][h]), np.array(depth["ls"][h])
        out["by_depth"][str(h)] = dict(n=len(t_), tree=float(t_.mean()), ls=float(l_.mean()))
        print(f"    depth {h}: n={len(t_):4d} tree {t_.mean():.2f} (>1 {np.mean(t_ > 1):.0%})  LS {l_.mean():.2f} (>1 {np.mean(l_ > 1):.0%})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sportpose")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--null-n", type=int, default=200)
    ap.add_argument("--null-q", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--methods", nargs="+", default=["CC", "DTW", "Caspi-style", "CSF"],
                    help="subset of CC, CC-vy (speed + signed vertical velocity), DTW, Caspi-style, CSF")
    ap.add_argument("--rows", default=None, help="summarize an existing rows file instead of recomputing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.graph:
        json.dump(graph(a, a.subjects), open(a.out, "w"), indent=1); print("wrote", a.out); return
    rows = json.load(open(a.rows)) if a.rows else benchmark(a, a.subjects)
    if not a.rows:
        json.dump(rows, open(a.out, "w"))
    summ = summarize(rows)
    json.dump(summ, open(a.out.replace(".json", "_summary.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
