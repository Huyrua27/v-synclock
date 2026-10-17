# -*- coding: utf-8 -*-
"""ChArUco calibration for the real multi-camera rig (optional, for 3D).

Intrinsics come from the waving-board segment (many views per camera); extrinsics
come from a moment where all cameras see the SAME STATIC board (the board is the
shared world origin, so camera desync does not matter while it is still).
Outputs one projection matrix P = K[R|t] per camera.  Task #24 (real data).

Because the cameras are not synchronized, the static moment falls at a different
frame index in each video. By default each camera's static window is found
automatically (longest run where the detected corners do not move), and corners are
pooled (per-id median) over that window. Cameras that see the board far/obliquely are
re-detected on an upscaled crop around the board to recover more corners.

Board size convention is OpenCV's (squares_x = columns, squares_y = rows); if nothing
is detected, try swapping them.

    python scripts/realdata/calibrate_charuco.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 \
        --squares-x 6 --squares-y 5 --square-len 0.15 --marker-len 0.11 \
        --out data/realdata/session1/calib.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import cv2


def make_board(a):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("need opencv-contrib-python for cv2.aruco")
    ad = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
    try:                                          # OpenCV >= 4.7 API
        board = cv2.aruco.CharucoBoard((a.squares_x, a.squares_y),
                                       a.square_len, a.marker_len, ad)
        board.setLegacyPattern(a.legacy)
        detector = cv2.aruco.CharucoDetector(board)
        return board, ad, detector
    except Exception:                             # older API
        board = cv2.aruco.CharucoBoard_create(a.squares_x, a.squares_y,
                                               a.square_len, a.marker_len, ad)
        return board, ad, None


def detect(gray, board, ad, detector):
    if detector is not None:
        corners, ids, _, _ = detector.detectBoard(gray)
        return corners, ids
    mc, mids, _ = cv2.aruco.detectMarkers(gray, ad)
    if mids is None or len(mids) == 0:
        return None, None
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)
    return corners, ids


def board_points(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)


def scan(video, board, ad, detector, stride, max_frames=None):
    """Detections every `stride` frames -> list of (frame, ids(N,), xy(N,2)), size."""
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dets, i = [], 0
    while max_frames is None or i < max_frames:
        if i % stride == 0:
            ok, bgr = cap.read()
            if not ok:
                break
            c, ids = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), board, ad, detector)
            if ids is not None and len(ids) > 0:
                dets.append((i, ids.ravel().astype(int), c.reshape(-1, 2).astype(np.float64)))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return dets, (W, H)


def find_static_window(dets, stride, move_px, min_sec, fps=30.0):
    """Longest run of consecutive detections whose shared corners move < move_px."""
    runs, cur = [], []
    for (fa, ia, xa), (fb, ib, xb) in zip(dets, dets[1:]):
        if fb - fa > 4 * stride:                  # tolerate a few missed detections
            runs.append(cur); cur = []
            continue
        da, db = dict(zip(ia, xa)), dict(zip(ib, xb))
        com = set(da) & set(db)
        m = np.median([np.linalg.norm(da[k] - db[k]) for k in com]) if len(com) >= 2 else np.inf
        if m < move_px:
            cur = cur or [fa]
            cur.append(fb)
        else:
            runs.append(cur); cur = []
    runs.append(cur)
    runs = [r for r in runs if r and (r[-1] - r[0]) >= min_sec * fps]
    if not runs:
        return None
    # merge runs separated by short interruptions (people walking past), keep the longest
    best = max(runs, key=lambda r: r[-1] - r[0])
    return best[0], best[-1]


def pool_static(video, window, dets, board, ad, detector, upscale, step):
    """Per-id median corner position over the static window (+ upscaled re-detection)."""
    lo, hi = window
    obs = {}
    for f, ids, xy in dets:
        if lo <= f <= hi:
            for k, p in zip(ids, xy):
                obs.setdefault(int(k), []).append(p)
    if upscale > 1 and obs:
        allp = np.concatenate([np.asarray(v) for v in obs.values()])
        x0, y0 = allp.min(0); x1, y1 = allp.max(0)
        pad = 0.6 * max(x1 - x0, y1 - y0) + 40
        cap = cv2.VideoCapture(video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        X1, Y1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
        for f in range(lo, hi + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            if not ok:
                continue
            crop = cv2.cvtColor(bgr[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
            big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            c, ids = detect(big, board, ad, detector)
            if ids is None:
                continue
            for k, p in zip(ids.ravel(), c.reshape(-1, 2)):
                obs.setdefault(int(k), []).append(p / upscale + (X0, Y0))
        cap.release()
    ids = sorted(k for k, v in obs.items() if len(v) >= 2)
    xy = np.array([np.median(np.asarray(obs[k]), axis=0) for k in ids])
    return np.array(ids, dtype=int), xy


def pick_intrinsic_views(dets, window, min_corners, max_views):
    cand = [d for d in dets if len(d[1]) >= min_corners
            and not (window and window[0] <= d[0] <= window[1])]
    if len(cand) > max_views:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, max_views).round().astype(int)]
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-len", type=float, required=True, help="metres")
    ap.add_argument("--marker-len", type=float, required=True, help="metres")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--legacy", action="store_true", help="pre-4.6 OpenCV board layout")
    ap.add_argument("--static-frame", type=int, default=None,
                    help="(legacy) one frame index used for ALL cameras; default: auto per camera")
    ap.add_argument("--stride", type=int, default=5, help="detection stride (frames)")
    ap.add_argument("--move-px", type=float, default=1.0, help="static if corners move < this")
    ap.add_argument("--min-static-sec", type=float, default=2.0)
    ap.add_argument("--upscale", type=float, default=2.0, help="re-detect static board on xN crop")
    ap.add_argument("--max-views", type=int, default=60, help="intrinsic views per camera")
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None, help="only scan the first N frames")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    board, ad, detector = make_board(a)
    obj = board_points(board)

    out = {"board": {"squares_x": a.squares_x, "squares_y": a.squares_y,
                     "square_len": a.square_len, "marker_len": a.marker_len, "dict": a.dict},
           "cameras": {}}
    for v in a.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        dets, (W, H) = scan(v, board, ad, detector, a.stride, a.max_frames)
        if a.static_frame is not None:
            window = (a.static_frame, a.static_frame)
        else:
            window = find_static_window(dets, a.stride, a.move_px, a.min_static_sec)
        views = pick_intrinsic_views(dets, window, a.min_corners, a.max_views)
        if len(views) < 5 or window is None:
            print(f"{name}: insufficient board views ({len(views)}) or no static window -> skip")
            continue
        # cv2.aruco.calibrateCameraCharuco was removed in OpenCV 4.8+; use the generic call.
        ret, K, dist, *_ = cv2.calibrateCamera(
            [obj[i].astype(np.float32) for _, i, _ in views],
            [x.astype(np.float32) for _, _, x in views],
            (W, H), None, None, flags=cv2.CALIB_FIX_K3)
        sids, sxy = pool_static(v, window, dets, board, ad, detector, a.upscale, a.stride)
        if len(sids) < 4:
            print(f"{name}: only {len(sids)} static corners -> skip")
            continue
        ok, rvec, tvec = cv2.solvePnP(obj[sids], sxy, K, dist, flags=cv2.SOLVEPNP_IPPE)
        rvec, tvec = cv2.solvePnPRefineLM(obj[sids], sxy, K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj[sids], rvec, tvec, K, dist)
        pnp_rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - sxy) ** 2, 1))))
        R, _ = cv2.Rodrigues(rvec)
        center = (-R.T @ tvec).ravel()
        out["cameras"][name] = {"K": K.tolist(), "dist": dist.ravel().tolist(),
                                "R": R.tolist(), "t": tvec.ravel().tolist(),
                                "reproj_rms": float(ret), "width": W, "height": H,
                                "static_window": [int(window[0]), int(window[1])],
                                "static_corners": int(len(sids)), "pnp_rms": pnp_rms,
                                "center": center.tolist()}
        print(f"{name}: intrinsics rms {ret:.3f}px on {len(views)} views, fx {K[0,0]:.0f}; "
              f"static frames {window[0]}-{window[1]} ({len(sids)} corners), "
              f"extrinsic reproj {pnp_rms:.2f}px, centre {np.round(center, 2)} m")

    cams = list(out["cameras"])
    if len(cams) > 1:
        print("\npairwise angle between optical axes (deg):")
        for i, c1 in enumerate(cams):
            for c2 in cams[i + 1:]:
                z1 = np.asarray(out["cameras"][c1]["R"])[2]
                z2 = np.asarray(out["cameras"][c2]["R"])[2]
                ang = np.degrees(np.arccos(np.clip(z1 @ z2, -1, 1)))
                print(f"  {c1} - {c2}: {ang:.0f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote calibration ({len(out['cameras'])} cams) -> {a.out}")


if __name__ == "__main__":
    main()
