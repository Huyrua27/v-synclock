# -*- coding: utf-8 -*-
"""Extrinsics for a camera that never sees the static board well (take 2: iphone1).

The waving ChArUco segment is seen by every camera. With the flash ground truth we know which
frame of camera X shows the same instant as a frame of the reference camera, so each co-visible
instant gives the relative pose  ref -> X  (two PnP solutions on the same board pose). The
world pose of the reference follows from X's static-board extrinsics; we take a robust average
over instants and over the calibrated cameras.

    python scripts/realdata/take2_calib_ref.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
import cv2  # noqa: E402

from calibrate_charuco import board_points, detect, make_board  # noqa: E402


def clock(take, pose_root, cam):
    p = os.path.join(take, f"{cam}_timestamps.csv")
    if os.path.exists(p):
        return np.array([float(r["t_rel_s"]) for r in csv.DictReader(open(p, encoding="utf-8-sig"))])
    return np.load(os.path.join(pose_root, f"pts_{cam}.npy"))


def detections(video, frames, board, ad, det, min_corners):
    """Board detections at the requested (sorted) frame indices, read sequentially."""
    want, out, cap, i = set(frames), {}, cv2.VideoCapture(video), 0
    last = max(frames)
    while i <= last:
        if i in want:
            ok, bgr = cap.read()
            if not ok:
                break
            c, ids = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), board, ad, det)
            if ids is not None and len(ids) >= min_corners:
                out[i] = (ids.ravel().astype(int), c.reshape(-1, 2).astype(np.float64))
        elif not cap.grab():
            break
        i += 1
    return out


def pnp(obj, ids, xy, K, dist):
    ok, rv, tv = cv2.solvePnP(obj[ids], xy, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj[ids], xy, K, dist, rv, tv)
    pr, _ = cv2.projectPoints(obj[ids], rv, tv, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - xy) ** 2, 1))))
    return cv2.Rodrigues(rv)[0], tv.ravel(), rms


def mean_rotation(Rs):
    q = np.array([cv2.Rodrigues(R)[0].ravel() for R in Rs])          # rotation vectors near each other
    M = sum(R for R in Rs)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt     # chordal L2 mean
    return R, q


def angle(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take-dir", default="captures/take2")
    ap.add_argument("--root", default="data/realdata/s2_t2")
    ap.add_argument("--calib", default="data/realdata/s2_t2/calib.json")
    ap.add_argument("--intrinsics", default="data/realdata/s2_t2/calib_iphone1.json")
    ap.add_argument("--gt", default="data/take2/flash_gt.json")
    ap.add_argument("--ref", default="iphone1")
    ap.add_argument("--t0", type=float, default=40.0)
    ap.add_argument("--t1", type=float, default=115.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-corners", type=int, default=10)
    ap.add_argument("--max-rms", type=float, default=2.0)
    ap.add_argument("--out", default="data/realdata/s2_t2/calib_full.json")
    a = ap.parse_args()
    calib = json.load(open(a.calib, encoding="utf-8"))
    bd = calib["board"]
    board, ad, det = make_board(argparse.Namespace(dict=bd["dict"], squares_x=bd["squares_x"], squares_y=bd["squares_y"],
                                                   square_len=bd["square_len"], marker_len=bd["marker_len"], legacy=False))
    obj = board_points(board)
    gt = json.load(open(a.gt, encoding="utf-8"))
    ref_cam = json.load(open(a.intrinsics, encoding="utf-8"))["cameras"][a.ref]
    Kr, dr = np.array(ref_cam["K"]), np.array(ref_cam["dist"])
    tr = clock(a.take_dir, a.root, a.ref)
    fr = [i for i in range(len(tr)) if a.t0 <= tr[i] <= a.t1][::a.stride]
    Dr = detections(os.path.join(a.take_dir, f"{a.ref}.mp4"), fr, board, ad, det, a.min_corners)
    print(f"{a.ref}: board in {len(Dr)}/{len(fr)} sampled frames")
    Pr = {f: p for f, (ids, xy) in Dr.items() if (p := pnp(obj, ids, xy, Kr, dr)) and p[2] < a.max_rms}

    world = []          # per-instant estimates of (R_ref, t_ref) in the world frame
    for cam, c in calib["cameras"].items():
        g = gt[cam]
        tx = clock(a.take_dir, a.root, cam)
        o = lambda t: g["offset_mid_s"] + g["slope_ms_per_min"] / 6e4 * (t - g["t_mid"])
        match = {}
        for f in Pr:
            t = tr[f] + o(tr[f])
            j = int(np.argmin(np.abs(tx - t)))
            if abs(tx[j] - t) < 0.006:                     # same instant within 6 ms
                match[j] = f
        if not match:
            continue
        Dx = detections(os.path.join(a.take_dir, f"{cam}.mp4"), sorted(match), board, ad, det, a.min_corners)
        Kx, dx = np.array(c["K"]), np.array(c["dist"])
        Rw, tw = np.array(c["R"]), np.array(c["t"])
        n = 0
        for j, (ids, xy) in Dx.items():
            p = pnp(obj, ids, xy, Kx, dx)
            if not p or p[2] > a.max_rms:
                continue
            Rb_x, tb_x, _ = p
            Rb_r, tb_r, _ = Pr[match[j]]
            R_rx = Rb_x @ Rb_r.T                            # ref -> X
            t_rx = tb_x - R_rx @ tb_r
            R_ref = R_rx.T @ Rw
            t_ref = R_rx.T @ (tw - t_rx)
            world.append((cam, R_ref, t_ref)); n += 1
        print(f"  via {cam}: {len(match)} matched instants, {n} usable")
    Rs = [w[1] for w in world]
    R0, _ = mean_rotation(Rs)
    keep = [w for w in world if angle(w[1], R0) < 3.0]
    R, _ = mean_rotation([w[1] for w in keep])
    centers = np.array([-w[1].T @ w[2] for w in keep])
    C = np.median(centers, 0)
    t = -R @ C
    spread = np.median(np.linalg.norm(centers - C, axis=1))
    print(f"{a.ref}: {len(keep)}/{len(world)} instants kept, centre {np.round(C, 2)} m, "
          f"median centre spread {spread * 100:.1f} cm, median rot dev {np.median([angle(w[1], R) for w in keep]):.2f} deg")
    for cam in calib["cameras"]:
        cs = [(-w[1].T @ w[2]) for w in keep if w[0] == cam]
        if cs:
            print(f"   per-camera centre via {cam}: {np.round(np.median(cs, 0), 2)} (n={len(cs)})")
    out = dict(calib)
    out["cameras"] = dict(calib["cameras"])
    out["cameras"][a.ref] = dict(ref_cam, R=R.tolist(), t=t.tolist(), center=C.tolist(),
                                 extrinsics="via flash-synchronized waving board", instants=len(keep),
                                 centre_spread_m=float(spread))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print("angles to", a.ref, {c: round(np.degrees(np.arccos(np.clip(R[2] @ np.array(v["R"])[2], -1, 1))))
                               for c, v in calib["cameras"].items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
