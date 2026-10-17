# -*- coding: utf-8 -*-
"""Train ContinuSyncFormer on the synthetic benchmark or Panoptic B1.

    python scripts/train_model.py --epochs 10
    python scripts/train_model.py --epochs 10 --occlusion 0.2   # occlusion-robust variant
    python scripts/train_model.py --data panoptic --epochs 10   # B1, task #16
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsynclock.model.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--occlusion", type=float, default=0.0, help="p occlude keypoint")
    ap.add_argument("--out", default="checkpoints/csf.pt")
    ap.add_argument("--data", choices=("synthetic", "panoptic"), default="synthetic",
                    help="panoptic = B1: real projected trajectories (task #16)")
    ap.add_argument("--root", default="data/panoptic_raw", help="Panoptic raw data root")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--clip-len", type=int, default=72,
                    help="Panoptic clip length T (short clips favour the learned model)")
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="2D detector jitter std in pixels (realistic HRNet/ViTPose noise)")
    ap.add_argument("--outlier-p", type=float, default=0.0,
                    help="per-joint gross detector-failure probability")
    ap.add_argument("--select", choices=("accin", "frmerr"), default="accin",
                    help="checkpoint selection metric (frmerr for drift line-fitting)")
    ap.add_argument("--seed", type=int, default=1,
                    help="training seed (weights + data sampling) for multi-seed error bars")
    ap.add_argument("--mediapipe", action="store_true",
                    help="simulate MediaPipe-like detector failures on both views (real-data transfer)")
    ap.add_argument("--max-pair-angle", type=float, default=60.0,
                    help="widest camera pair (deg) sampled for training/validation")
    ap.add_argument("--aist", action="store_true",
                    help="mix in AIST++ dance pairs (half of train/val) for periodic motion")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--motion", action="store_true",
                    help="motion (per-joint speed) input — recommended for --data panoptic")
    ap.add_argument("--features", choices=("speed", "speed+vy"), default="speed",
                    help="motion input: speed, or speed + signed vertical velocity")
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    kw = {"occlusion_aware": a.occlusion > 0, "motion_input": (True if a.features == "speed" else a.features) if a.motion else False}
    train_ds = val_ds = None
    if a.data == "panoptic":
        from vsynclock.model.panoptic_dataset import PanopticPairDataset
        kw["n_joints"] = 19
        nz = dict(noise_px=a.noise_px, outlier_p=a.outlier_p, mediapipe=a.mediapipe,
                  max_pair_angle=a.max_pair_angle)
        train_ds = PanopticPairDataset(root=a.root, split="train", n=a.n_train,
                                       seed=a.seed, occlusion_p=a.occlusion, stride=6,
                                       T=a.clip_len, **nz)
        val_ds = PanopticPairDataset(root=a.root, split="validation", n=a.n_val,
                                     seed=999, occlusion_p=a.occlusion, T=a.clip_len, **nz)
        if a.aist:
            from vsynclock.model.aistpp_dataset import AISTPairDataset
            from vsynclock.model.dataset import MixedPairDataset
            at = AISTPairDataset(split="train", n=a.n_train, seed=a.seed, T=a.clip_len,
                                 occlusion_p=a.occlusion, stride=6, **nz)
            av = AISTPairDataset(split="validation", n=a.n_val, seed=999, T=a.clip_len,
                                 occlusion_p=a.occlusion, **nz)
            train_ds = MixedPairDataset([train_ds, at], a.n_train)
            val_ds = MixedPairDataset([val_ds, av], a.n_val)
    best = train(epochs=a.epochs, batch=a.batch, lr=a.lr, out=a.out,
                 model_kwargs=kw, occlusion_p=a.occlusion,
                 n_train=a.n_train, n_val=a.n_val, train_ds=train_ds, val_ds=val_ds,
                 select=a.select, init=a.init)
    print(f"\nBest {a.select} score = {best:.3f}")


if __name__ == "__main__":
    main()
