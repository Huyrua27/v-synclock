# -*- coding: utf-8 -*-
"""Extract COCO-19 2D keypoint trajectories from each camera's video (MediaPipe).

    python scripts/realdata/extract_pose2d.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 --out-dir data/realdata/session1/pose
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.realworld.pose2d import extract_pose2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        xy, fps = extract_pose2d(v, max_frames=args.max_frames)
        valid = np.isfinite(xy).all(axis=(1, 2)).mean()
        out = os.path.join(args.out_dir, f"{name}.npz")
        np.savez(out, xy=xy, fps=fps, name=name)
        print(f"{name:>14}: {xy.shape[0]} frames, fps {fps}, "
              f"{100*valid:.0f}% full-body frames -> {out}")


if __name__ == "__main__":
    main()
