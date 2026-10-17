# -*- coding: utf-8 -*-
"""SportsPose: run the 2D pose detector on every frame of every camera, one recording at a time.

The release ships each subject's videos as one zip (7 MJPG AVIs per recording, 270 frames at
90 fps). A recording's AVIs are extracted to a temporary folder, processed in parallel (one
process per camera) and deleted, so disk use stays at one recording. Cameras are mounted
rotated (calib.pkl: numtimesrot90clockwise); frames are rotated upright before detection and
keypoints are mapped back to the ORIGINAL image coordinates, the frame of the calibration.
Detector: MediaPipe Pose, model_complexity=1 (the same BlazePose-full model as the Tasks API used
for AthletePose3D / TotalCapture), COCO-19 layout (vsynclock/realworld/pose2d.py).

SportsPose is academic-use only, no redistribution: keypoints stay in data/ (git-ignored).

    python scripts/sp_extract_pose.py --subjects S00 S06 S19 --workers 7
"""
import argparse
import os
import pickle
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype")}


class SafeUnpickler(pickle.Unpickler):
    """The release pickles contain only numpy arrays; refuse anything else."""
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        import numpy.core.multiarray as ma
        return getattr(ma, name) if name == "_reconstruct" else getattr(np, name)


def load_pkl(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


def to_original(xy_rot, n_rot, W, H):
    """Map (x, y) detected on the image rotated n_rot x 90 deg clockwise back to the original."""
    # Clockwise rotation k times maps original (x, y) -> rotated; invert one turn at a time.
    pts = np.asarray(xy_rot, dtype=np.float64)
    w, h = W, H
    sizes = []
    for _ in range(n_rot % 4):          # sizes of the image before each clockwise turn
        sizes.append((w, h)); w, h = h, w
    for (w0, h0) in reversed(sizes):    # rotated (xr, yr) of an image of size (h0, w0) -> (x, y) of (w0, h0)
        xr, yr = pts[..., 0], pts[..., 1]
        pts = np.stack([yr, h0 - 1 - xr], -1)
    return pts


def run_camera(job):
    path, out, n_rot, min_vis = job
    if os.path.exists(out):
        return out, "skip"
    import cv2
    import mediapipe as mp
    from vsynclock.realworld.pose2d import mediapipe_to_coco19
    cap = cv2.VideoCapture(path)
    W, H = int(cap.get(3)), int(cap.get(4))
    rot = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(n_rot % 4)
    pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    xy, vis = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        img = cv2.rotate(bgr, rot) if rot is not None else bgr
        h, w = img.shape[:2]
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lm = np.full((33, 2), np.nan); v = np.zeros(33)
        if res.pose_landmarks:
            for k, p in enumerate(res.pose_landmarks.landmark):
                v[k] = p.visibility
                if p.visibility >= min_vis:
                    lm[k] = (p.x * w, p.y * h)
        xy.append(to_original(mediapipe_to_coco19(lm), n_rot, W, H)); vis.append(v)
    cap.release(); pose.close()
    tmp = out[:-4] + ".part.npz"
    np.savez(tmp, xy=np.asarray(xy, np.float32), vis=np.asarray(vis, np.float32), W=W, H=H, n_rot=n_rot)
    os.replace(tmp, out)
    return out, f"{len(xy)} frames {W}x{H} rot{n_rot}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sportpose")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--recordings", nargs="*", default=None, help="limit to these recording names (pilot)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--min-vis", type=float, default=0.3)
    a = ap.parse_args()
    os.chdir(ROOT)
    for subj in a.subjects:
        calib = load_pkl(os.path.join(a.root, "ex", "data", "indoor", subj, "calib.pkl"))
        n_rot = [int(r) for r in calib["numtimesrot90clockwise"]]
        out_dir = os.path.join(a.root, "keypoints", subj); os.makedirs(out_dir, exist_ok=True)
        zf = zipfile.ZipFile(os.path.join(a.root, "video", "indoor", f"{subj}.zip"))
        recs = sorted({n.split("/")[1] for n in zf.namelist() if n.endswith(".avi")})
        if a.recordings:
            recs = [r for r in recs if r in a.recordings]
        t0 = time.time()
        for ri, rec in enumerate(recs):
            outs = {c: os.path.join(out_dir, f"{rec}_CAM{c}.npz") for c in range(7)}
            if all(os.path.exists(o) for o in outs.values()):
                continue
            tmp = os.path.join(a.root, "tmp", subj, rec)
            for c in range(7):
                name = f"{subj}/{rec}/CAM{c}.avi"
                if not os.path.exists(os.path.join(tmp, name)):
                    zf.extract(name, tmp)
            jobs = [(os.path.join(tmp, f"{subj}/{rec}/CAM{c}.avi"), outs[c], n_rot[c], a.min_vis) for c in range(7)]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                msgs = [m for _, m in ex.map(run_camera, jobs)]
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{subj} {ri + 1}/{len(recs)} {rec}: {msgs[0]}  ({time.time() - t0:.0f}s)", flush=True)
        shutil.rmtree(os.path.join(a.root, "tmp", subj), ignore_errors=True)


if __name__ == "__main__":
    main()
