# -*- coding: utf-8 -*-
"""Evaluate a trained ContinuSyncFormer against classical baselines on B1 pairs.

    python scripts/eval_b1.py --checkpoint checkpoints/csf_b1.pt --split validation
    python scripts/eval_b1.py --checkpoint checkpoints/csf_b1.pt --split test

All methods see the *identical* (kp_a, kp_b, Δt) pairs, so the comparison is
apples-to-apples. Task #16 (WS2).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from vsynclock.benchmark import metrics
from vsynclock.benchmark.baselines import CrossCorrelation, DTW
from vsynclock.core.types import KeypointSequence
from vsynclock.model.continusyncformer import ContinuSyncFormer
from vsynclock.model.panoptic_dataset import FPS, PanopticPairDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1.pt")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--root", default="data/panoptic_raw")
    ap.add_argument("--occlusion-p", type=float, default=0.0,
                    help="occlude each keypoint with this probability (task #18)")
    ap.add_argument("--occlusion-aware", action="store_true",
                    help="checkpoint was trained with occlusion_aware=True")
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--outlier-p", type=float, default=0.0)
    args = ap.parse_args()

    ds = PanopticPairDataset(root=args.root, split=args.split, n=args.n,
                             seed=args.seed, occlusion_p=args.occlusion_p,
                             noise_px=args.noise_px, outlier_p=args.outlier_p)
    print(f"split={args.split}  clips={len(ds.clips)}  samples={args.n}"
          f"  occlusion_p={args.occlusion_p}")
    kas, kbs, gts, vas, vbs = [], [], [], [], []
    for i in range(args.n):
        ka, kb, dt, va, vb = ds[i]
        kas.append(ka); kbs.append(kb); gts.append(float(dt))
        vas.append(va); vbs.append(vb)
    gts = np.asarray(gts)
    has_vis = args.occlusion_p > 0

    from vsynclock.model.continusyncformer import load_csf
    model = load_csf(args.checkpoint, occlusion_aware=args.occlusion_aware)

    preds = {"ContinuSyncFormer": [], "CC+parabolic": [], "DTW": []}
    with torch.no_grad():
        ka_t, kb_t = torch.stack(kas), torch.stack(kbs)
        va_t = torch.stack(vas) if has_vis else None
        vb_t = torch.stack(vbs) if has_vis else None
        for s in range(0, args.n, 256):
            sl = slice(s, s + 256)
            out = model(ka_t[sl], kb_t[sl],
                        va_t[sl] if has_vis else None,
                        vb_t[sl] if has_vis else None)
            preds["ContinuSyncFormer"] += out["dt_hard"].tolist()
    cc, dtw = CrossCorrelation(features=args.features), DTW()
    for ka, kb, va, vb in zip(kas, kbs, vas, vbs):
        xa = ka.numpy().astype(np.float64)
        xb = kb.numpy().astype(np.float64)
        if has_vis:  # classical methods see occlusion as missing values (NaN)
            xa[va.numpy() == 0] = np.nan
            xb[vb.numpy() == 0] = np.nan
        a = KeypointSequence(xa, FPS, name="a")
        b = KeypointSequence(xb, FPS, name="b")
        for name, method in (("CC+parabolic", cc), ("DTW", dtw)):
            try:
                preds[name].append(method.predict(a, b).dt_frames)
            except Exception:
                preds[name].append(0.0)

    print(f"\n{'Method':>18} {'Frm.err':>9} {'Accin@0.1':>10} "
          f"{'Accin@0.25':>11} {'MAE_ms':>8}")
    for name, p in preds.items():
        p = np.asarray(p)
        print(f"{name:>18} {metrics.frm_err(p, gts):>9.4f} "
              f"{metrics.accin(p, gts, 0.1):>10.3f} "
              f"{metrics.accin(p, gts, 0.25):>11.3f} "
              f"{metrics.mae_ms(p, gts, FPS):>8.2f}")


if __name__ == "__main__":
    main()
