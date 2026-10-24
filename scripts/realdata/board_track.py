# -*- coding: utf-8 -*-
"""Track a ChArUco board in every frame of a video: corners, rotation and translation (camera frame).

For static cameras the board's angular speed |omega(t)| is the same in every view, so it is a timing
signal independent of human keypoints (used as a cross-check where no optical clock exists).

    python scripts/realdata/board_track.py --video V.mov --calib calib_info.json --out V_board.npz
"""
import argparse
import json

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True, help="list of cameras with K (full resolution); the median K is used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--squares", type=int, nargs=2, default=[6, 5])
    ap.add_argument("--square-len", type=float, default=0.15)
    ap.add_argument("--marker-len", type=float, default=0.11)
    ap.add_argument("--scale", type=float, default=0.5, help="detect on a downscaled frame")
    ap.add_argument("--min-corners", type=int, default=6)
    a = ap.parse_args()

    cams = json.load(open(a.calib, encoding="utf-8"))
    K = np.median(np.array([np.array(c["K"], float) for c in cams]), axis=0)
    K[:2] *= a.scale
    dist = np.zeros(5)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(tuple(a.squares), a.square_len, a.marker_len, d)
    det = cv2.aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()

    cap = cv2.VideoCapture(a.video)
    n_c, rv, tv, ctr = [], [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(bgr, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = det.detectBoard(g)
        if ci is not None and len(ci) >= a.min_corners:
            obj = obj_all[ci.ravel()].astype(np.float64)
            ok2, r, t = cv2.solvePnP(obj, cc.reshape(-1, 2).astype(np.float64), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                n_c.append(len(ci)); rv.append(r.ravel()); tv.append(t.ravel()); ctr.append(cc.reshape(-1, 2).mean(0) / a.scale)
                continue
        n_c.append(0 if ci is None else len(ci)); rv.append([np.nan] * 3); tv.append([np.nan] * 3); ctr.append([np.nan] * 2)
    cap.release()
    np.savez(a.out, n_corners=np.array(n_c), rvec=np.array(rv), tvec=np.array(tv), center=np.array(ctr))
    n = np.array(n_c)
    print(f"{a.video}: {len(n)} frames, board posed in {np.mean(n >= a.min_corners):.0%}")


if __name__ == "__main__":
    main()
