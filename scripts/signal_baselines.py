# -*- coding: utf-8 -*-
"""Signal formulations for keypoint cross-correlation sync, on identical real-image samples.

Prior tools already correlate vertical keypoint motion; this compares their formulations with
V-SyncLock's per-joint input under the same +-8-lag cross-correlation (protocol fixed in advance:
docs/paper/DEV_PROTOCOL.md, "Signal-formulation baselines"):
  pose2sim  sum_j |vy_j|, zero-phase Butterworth 4th order 6 Hz (Pose2Sim synchronization.py)
  opencap   mean of signed vy over the two ankles (OpenCap utilsSync.py, general mode)
  abs_vy    per-joint |vy_j|
  speed     per-joint image speed (previous V-SyncLock input)
  vy        per-joint signed vy_j
  speed+vy  per-joint speed and signed vy (V-SyncLock input)
Each with an integer-lag readout (argmax, as the tools) and parabolic sub-frame refinement.

    python scripts/signal_baselines.py --dataset tc   --out docs/paper/clips/vy/signal_baselines_tc.json
    python scripts/signal_baselines.py --dataset ap3d --out docs/paper/clips/vy/signal_baselines_ap3d.json
    python scripts/signal_baselines.py --dataset sp   --out docs/paper/clips/vy/signal_baselines_sp.json
"""
import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines.cross_correlation import lag_scores, refined_peak
from vsynclock.model.panoptic_dataset import _fill_gaps

FPS = 30.0
ANKLES = (8, 14)
KINDS = ("pose2sim", "opencap", "abs_vy", "speed", "vy", "speed+vy")
_B, _A = butter(4, 6.0 / (FPS / 2))


def z(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    x = x - x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return x / np.where(sd > 1e-8, sd, 1.0)


def features(xy, kind):
    xy = np.nan_to_num(xy)          # joints missing in the whole clip -> constant (as motion_features)
    if kind == "pose2sim":
        y = filtfilt(_B, _A, xy[..., 1], axis=0)
        s = np.abs(np.gradient(y, axis=0)).sum(1)
        return z(filtfilt(_B, _A, s))
    v = np.gradient(xy, axis=0)
    sp, vy = np.linalg.norm(v, axis=-1), v[..., 1]
    return z({"opencap": vy[:, list(ANKLES)].mean(1), "abs_vy": np.abs(vy), "speed": sp, "vy": vy,
              "speed+vy": np.concatenate([sp, vy], 1)}[kind])


def estimate(A, B):
    out = {}
    for k in KINDS:
        try:
            lags, sc = lag_scores(features(A, k), features(B, k), 8)
            out[(k, "int")] = float(lags[int(np.argmax(sc))])
            out[(k, "sub")] = refined_peak(lags, sc)[0]
        except ValueError:
            out[(k, "int")] = out[(k, "sub")] = np.nan
    return out


def samples_tc(a):
    from tc_benchmark import load_sequences
    S = 2
    seqs = load_sequences("data/tc/tc_keypoints")
    for r in json.load(open("docs/paper/clips/tc_rows_L300.json", encoding="utf-8")):
        if r["method"] != "CC":
            continue
        n, (ca, cb) = r["n"], (int(x) for x in r["pair"].split("-"))
        cams = seqs[(r["subj"], r["seq"])]
        o = r["c0"] + S * r["j"] + r["k"]
        yield (f"{r['subj']}_{r['seq']}", r["seq"].rstrip("0123456789"), r["truth"],
               _fill_gaps(cams[ca][r["c0"]:r["c0"] + S * n:S])[0], _fill_gaps(cams[cb][o:o + S * n:S])[0])


def samples_ap3d(a):
    from ap3d_benchmark import decimated, load_trials
    S = 4
    trials, _ = load_trials("data/ap3d/keypoints", 0.5)
    for (subj, split, trial), cams in sorted(trials.items()):
        T = min(len(v[0]) for v in cams.values())
        for ca, cb in itertools.combinations(sorted(cams), 2):
            for j in (-1, 0, 1):
                for k in range(S):
                    sh = S * j + k
                    n = (T - max(S, S + sh)) // S
                    if n < 24:
                        continue
                    yield (f"{subj}_{split}_{trial}", "narrow" if f"{ca}-{cb}" in ("2-3", "1-4") else "wide",
                           j + k / S, decimated(cams[ca][0], S, S, n), decimated(cams[cb][0], S + sh, S, n))


def samples_sp(a):
    from sp_benchmark import S, load_subject, stream
    for subj in ("S06", "S12", "S19"):
        _, trials = load_subject("data/sportpose", subj)
        for tid, tr in trials.items():
            T = tr["sync_up"].shape[1]
            for u, v in itertools.combinations(range(7), 2):
                for j in (-1, 0, 1):
                    for k in range(S):
                        sh = S * j + k
                        n = (T - max(S, S + sh)) // S
                        A, va = stream(tr["sync_up"][u], S, n); B, vb = stream(tr["sync_up"][v], S + sh, n)
                        if min(va, vb) < 0.5:
                            continue
                        yield f"{subj}_{tid}", tr["act"], j + k / S, A, B


TAKE2_SEGMENTS = [("standing", 155, 188), ("upper body", 188, 232), ("walking", 232, 263), ("vertical", 263, 297),
                  ("horizontal", 297, 354), ("non-upright", 354, 410), ("fast", 410, 453), ("two people", 453, 499),
                  ("free", 499, 531)]


def samples_take2(a):
    """Our take 2: the 10-s windows of scripts/realdata/take2_eval.py whose coarse stage succeeded."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "realdata"))
    from take2_eval import CAMS, load
    data = {c: load("captures/take2", "data/realdata/s2_t2/pose", c) for c in CAMS}
    rows = json.load(open("data/take2/rows_L300.json", encoding="utf-8"))
    for r in rows:
        if abs(r["truth"]) >= 2:
            continue
        seg = next((nm for nm, lo, hi in TAKE2_SEGMENTS if lo <= r["t_ref"] < hi), "other")
        ca, cb = r["pair"].split("-")
        n, s0, L = 300, r["s0"], r["coarse"]
        A, B = _fill_gaps(data[ca][0][s0:s0 + n])[0], _fill_gaps(data[cb][0][s0 + L:s0 + L + n])[0]
        yield r["pair"], seg, -r["truth"], A, B     # estimator convention (see take2_eval sign check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("tc", "ap3d", "sp", "take2"), required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    gen = {"tc": samples_tc, "ap3d": samples_ap3d, "sp": samples_sp, "take2": samples_take2}[a.dataset](a)
    unit, group, err = [], [], collections.defaultdict(list)
    for u, g, truth, A, B in gen:
        est = estimate(A, B)
        unit.append(u); group.append(g)
        for key, e in est.items():
            err[key].append(abs(e - truth))
    unit, group = np.array(unit), np.array(group)
    U = sorted(set(unit))
    E = {k: np.array(v) for k, v in err.items()}
    um = lambda e, m=None: np.array([np.nanmean(e[(unit == x) & (m if m is not None else True)])
                                     for x in U if ((unit == x) & (m if m is not None else True)).any()])
    out = {"dataset": a.dataset, "samples": int(len(unit)), "units": len(U), "variants": {}, "paired": {}}
    groups = sorted(set(group))
    print(f"{a.dataset}: {len(unit)} samples, {len(U)} units; mean error (frames, per unit) / share > 1 / failures")
    print(f"{'variant':>16s} {'all':>14s} " + " ".join(f"{g:>12s}" for g in groups))
    for (k, r), e in E.items():
        d = dict(mean=float(um(e).mean()), gt1=float(np.nanmean(e > 1)), nan=int(np.isnan(e).sum()),
                 by_group={g: float(np.nanmean(e[group == g])) for g in groups})
        out["variants"][f"{k}|{r}"] = d
        print(f"{k + '|' + r:>16s} {d['mean']:6.3f} ({d['gt1']:4.0%}) " + " ".join(f"{d['by_group'][g]:12.3f}" for g in groups)
              + (f"  nan {d['nan']}" if d["nan"] else ""))
    rng = np.random.default_rng(0)
    ref = E[("speed+vy", "sub")]
    print("\npaired, speed+vy|sub minus each variant (per unit):")
    for (k, r), e in E.items():
        if (k, r) == ("speed+vy", "sub"):
            continue
        dlt = um(ref) - um(e)
        b = dlt[rng.integers(0, len(dlt), (a.n_boot, len(dlt)))].mean(1)
        out["paired"][f"{k}|{r}"] = [float(dlt.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        print(f"  - {k + '|' + r:16s} {dlt.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
