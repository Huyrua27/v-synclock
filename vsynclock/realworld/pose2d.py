# -*- coding: utf-8 -*-
"""Extract 2D keypoint trajectories from real video as COCO-19 (Panoptic order).

Uses MediaPipe Pose (33 BlazePose landmarks) and remaps to the 19-joint layout the
benchmark and ContinuSyncFormer expect, so real footage plugs into the same harness
as the Panoptic-projected trajectories.  Task #24 (real data).
"""
from __future__ import annotations

import numpy as np

# COCO-19 (Panoptic order) <- MediaPipe landmark index, or a midpoint of two.
# 0 Neck,1 Nose,2 BodyCenter,3 lSho,4 lElb,5 lWri,6 lHip,7 lKnee,8 lAnk,
# 9 rSho,10 rElb,11 rWri,12 rHip,13 rKnee,14 rAnk,15 lEye,16 lEar,17 rEye,18 rEar
_MID = {0: (11, 12), 2: (23, 24)}                      # neck, body-center = midpoints
_DIRECT = {1: 0, 3: 11, 4: 13, 5: 15, 6: 23, 7: 25, 8: 27,
           9: 12, 10: 14, 11: 16, 12: 24, 13: 26, 14: 28,
           15: 2, 16: 7, 17: 5, 18: 8}


def mediapipe_to_coco19(landmarks_xy: np.ndarray) -> np.ndarray:
    """landmarks_xy: (33, 2) pixels -> (19, 2) COCO-19."""
    out = np.full((19, 2), np.nan, dtype=np.float64)
    for j, src in _DIRECT.items():
        out[j] = landmarks_xy[src]
    for j, (a, b) in _MID.items():
        out[j] = 0.5 * (landmarks_xy[a] + landmarks_xy[b])
    return out


def extract_pose2d(video_path: str, max_frames: int | None = None,
                   min_visibility: float = 0.3) -> tuple[np.ndarray, float]:
    """Run MediaPipe Pose on a video -> (xy (T,19,2) pixels, fps).

    Missing / low-confidence joints are NaN; the harness fills short gaps.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install mediapipe  (see requirements-realdata.txt)") from exc
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        lm = np.full((33, 2), np.nan, dtype=np.float64)
        if res.pose_landmarks:
            for k, p in enumerate(res.pose_landmarks.landmark):
                if p.visibility >= min_visibility:
                    lm[k] = (p.x * W, p.y * H)
        frames.append(mediapipe_to_coco19(lm))
        i += 1
    cap.release()
    pose.close()
    return np.asarray(frames), round(float(fps), 4)
