# -*- coding: utf-8 -*-
"""Which motion features break the gait aliasing of CC? A training-free probe on TotalCapture.

CC is run on the same samples as tc_benchmark.py rows (clips, camera pairs, offsets j + k/2),
with different per-frame feature sets, each z-scored per channel over the clip:
  speed      per-joint speed |dx/dt| (the current CC and CSF input)
  vy         per-joint signed vertical image velocity (view-tolerant for upright cameras)
  lr_speed   left-right differences of speed for the 6 symmetric joint pairs
  lr_vy      left-right differences of vertical velocity
Used on the development split to choose the input of a retrained CSF; the test split is left
for the final comparison.

    python scripts/tc_feature_probe.py --rows docs/paper/clips/tc_dev_rows_L300.json \
        --kp-dir data/tc/tc_keypoints_dev
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

from vsynclock.benchmark.baselines.cross_correlation import lag_scores
from vsynclock.benchmark.drift import peak_in_range
from vsynclock.model.panoptic_dataset import _fill_gaps
from tc_benchmark import load_sequences

S = 2
LR = [(3, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14)]   # shoulder, elbow, wrist, hip, knee, ankle


def z(x):
    x = x - x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return x / np.where(sd > 1e-8, sd, 1.0)


def features(xy, kind):
    v = np.gradient(xy, axis=0)                                  # (T,J,2)
    sp, vy = np.linalg.norm(v, axis=-1), v[..., 1]
    L, R = [a for a, _ in LR], [b for _, b in LR]
    parts = {"speed": sp, "vy": vy, "lr_speed": sp[:, L] - sp[:, R], "lr_vy": vy[:, L] - vy[:, R]}
    return np.concatenate([z(parts[k]) for k in kind.split("+")], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="docs/paper/clips/tc_dev_rows_L300.json")
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints_dev")
    ap.add_argument("--kinds", nargs="+", default=["speed", "vy", "speed+vy", "speed+lr_speed", "speed+lr_vy",
                                                   "speed+vy+lr_vy"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    seqs = load_sequences(a.kp_dir)
    rows = [r for r in json.load(open(a.rows, encoding="utf-8")) if r["method"] == "CC"]
    err = collections.defaultdict(list)
    seq = []
    for r in rows:
        n, (ca, cb) = r["n"], (int(x) for x in r["pair"].split("-"))
        cams = seqs[(r["subj"], r["seq"])]
        A = _fill_gaps(cams[ca][r["c0"]:r["c0"] + S * n:S])[0]
        o = r["c0"] + S * r["j"] + r["k"]
        B = _fill_gaps(cams[cb][o:o + S * n:S])[0]
        for kind in a.kinds:
            try:
                est = peak_in_range(*lag_scores(features(A, kind), features(B, kind), 8))
            except ValueError:
                est = np.nan
            err[kind].append(abs(est - r["truth"]))
        seq.append((f"{r['subj']}_{r['seq']}", r["seq"].rstrip("0123456789")))
    sq = np.array([s for s, _ in seq]); act = np.array([g for _, g in seq])
    S_ = sorted(set(sq))
    out = {}
    print(f"{len(rows)} samples, {len(S_)} sequences; mean error (frames, per-sequence mean) / share > 1 frame")
    print(f"{'features':>18s} {'all':>12s} {'walking':>12s} {'acting':>12s} {'freestyle':>12s}")
    for kind in a.kinds:
        e = np.nan_to_num(np.array(err[kind]), nan=8.0)
        cell = lambda m: f"{np.mean([e[(sq == s) & m].mean() for s in S_ if ((sq == s) & m).any()]):.3f} ({np.mean(e[m] > 1):.0%})"
        out[kind] = {g: float(np.mean([e[(sq == s) & m].mean() for s in S_ if ((sq == s) & m).any()]))
                     for g, m in (("all", np.ones(len(e), bool)), ("walking", act == "walking"),
                                  ("acting", act == "acting"), ("freestyle", act == "freestyle"))}
        print(f"{kind:>18s} {cell(np.ones(len(e), bool)):>12s} {cell(act == 'walking'):>12s} "
              f"{cell(act == 'acting'):>12s} {cell(act == 'freestyle'):>12s}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
