# -*- coding: utf-8 -*-
"""Take 2, whole rig: offline and streaming V-SyncLock against flash ground truth, and 3D.

Every camera is first coarse-aligned to the reference (iphone1) with the whole-take motion
correlation (integer frames), exactly as the pipeline does. The rig stage (vsynclock.online:
coarse-to-fine pair lines, self-calibrated null gate, Huber least-squares placement) then
estimates delta_c(t) for every camera; we score it against the flash line.

  offline   : all frames pushed, one update (the offline pipeline)
  streaming : frames pushed in 1-s chunks, refit every 5 s, causal estimate scored per chunk
  3D        : leave-one-camera-out reprojection error of triangulated keypoints with timing
              from (coarse only | V-SyncLock offline | flash ground truth)

    python scripts/realdata/take2_rig.py --out data/take2/rig.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from realdata_eval import coarse_offset, evaluate, motion_signal, undistort  # noqa: E402
from take2_eval import CAMS, FPS, gt_offset, load  # noqa: E402

from vsynclock.benchmark.interpolation import cubic_sample  # noqa: E402
from vsynclock.online import StreamingSynchronizer, cc_profile  # noqa: E402
from vsynclock.panoptic.triangulation import projection_matrix  # noqa: E402

REF = "iphone1"


def csf_profile_fn(ckpt):
    from realdata_eval import csf_profile
    return csf_profile(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take-dir", default="captures/take2")
    ap.add_argument("--pose-dir", default="data/realdata/s2_t2/pose")
    ap.add_argument("--gt", default="data/take2/flash_gt.json")
    ap.add_argument("--calib", default="data/realdata/s2_t2/calib_full.json")
    ap.add_argument("--start-sec", type=float, default=30.0)
    ap.add_argument("--chunk-sec", type=float, default=1.0)
    ap.add_argument("--skip-stream", action="store_true")
    ap.add_argument("--out", default="data/take2/rig.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    gt = json.load(open(a.gt, encoding="utf-8"))
    data = {c: load(a.take_dir, a.pose_dir, c) for c in CAMS}
    k0 = {c: data[c][2] for c in CAMS}
    T = len(data[REF][0])
    s0 = int(a.start_sec * FPS)

    # ---- coarse stage (star from the reference) --------------------------------------------
    sig = {c: motion_signal(data[c][0]) for c in CAMS}
    L = {REF: 0}
    for c in CAMS[1:]:
        L[c] = coarse_offset(sig[REF], sig[c], 400, s0, None)

    def lag_true(c, f):          # frame of c's grid showing the instant of ref grid frame f
        tr = (np.asarray(f, dtype=np.float64) + k0[REF]) / FPS
        return FPS * gt_offset(gt, c, tr) + k0[REF] - k0[c]

    t = np.arange(T, dtype=np.float64)
    streams, vis = {}, {}
    for c in CAMS:                      # stream_c[f] = c's grid frame f + L_c (NaN outside)
        xy, v, _ = data[c]
        idx = np.arange(T) + L[c]
        ok = (idx >= 0) & (idx < len(xy))
        s = np.full((T,) + xy.shape[1:], np.nan); s[ok] = xy[idx[ok]]
        w = np.zeros((T, xy.shape[1]), bool); w[ok] = v[idx[ok]]
        streams[c], vis[c] = s, w
    # residual after coarse: stream_c[f + o_c(f)] shows the instant of ref frame f
    o_true = {c: lag_true(c, t) - L[c] for c in CAMS}
    print("coarse L:", L, " residual o_true at mid:", {c: round(float(o_true[c][T // 2]), 2) for c in CAMS})

    def score(delta, name):
        """delta_c(f) = -o_c(f) in the online module's convention; error on the frames each camera covers."""
        out = {}
        for c in CAMS[1:]:
            cover = np.isfinite(streams[c][:, 0, 0]) & (t >= s0)
            d = np.asarray(delta[c], dtype=np.float64)
            e = np.abs(d + o_true[c])[cover]
            out[c] = dict(mae=float(np.nanmean(e)), p_gt1=float(np.nanmean(e > 1)), n=int(cover.sum()),
                          placed=bool(np.isfinite(d[cover]).any()))
        m = np.mean([v["mae"] for v in out.values() if np.isfinite(v["mae"])])
        print(f"{name:>22}: " + "  ".join(f"{c} {v['mae']:.3f}" for c, v in out.items()) + f" | mean {m:.3f}")
        return dict(per_cam=out, mean=float(m))

    res = {"coarse": L, "offline": {}, "stream": {}}
    res["offline"]["coarse only"] = score({c: np.zeros(T) for c in CAMS}, "coarse only")
    profiles = {"CC": cc_profile("speed+vyb"), "CSF": csf_profile_fn("checkpoints/csf_b1_t20_vy.pt")}
    lines = {}
    for pname, prof in profiles.items():
        for prior in (None, 2.0):
            tag = f"{pname} offline" + (" +prior" if prior else "")
            sync = StreamingSynchronizer(CAMS, REF, FPS, profile=prof, prior=prior)
            for c in CAMS:
                sync.push(c, streams[c])
            t1 = time.perf_counter(); sync.update(); dt = time.perf_counter() - t1
            delta = {c: sync.offset(c, t) for c in CAMS}
            res["offline"][tag] = score(delta, tag)
            res["offline"][tag]["seconds"] = dt
            res["offline"][tag]["lines"] = {c: [float(v) for v in sync.estimate.get(c, (np.nan, np.nan))] for c in CAMS}
            lines[tag] = delta
    # true slope for reference (frames/min)
    res["true_slope_fpm"] = {c: float((o_true[c][-1] - o_true[c][0]) / (T / FPS / 60)) for c in CAMS[1:]}
    print("true drift (frames/min):", {c: round(v, 3) for c, v in res["true_slope_fpm"].items()})
    for tag, d in lines.items():
        print(f"  est drift {tag:>18}:", {c: round(float((d[c][-1] - d[c][0]) / (T / FPS / 60)), 3) for c in CAMS[1:]})

    # ---- streaming ---------------------------------------------------------------------------
    if not a.skip_stream:
        for pname, prof in profiles.items():
            for prior in (None, 2.0):
                tag = f"{pname} stream" + (" +prior" if prior else "")
                sync = StreamingSynchronizer(CAMS, REF, FPS, profile=prof, prior=prior)
                ch = int(a.chunk_sec * FPS)
                delta = {c: np.full(T, np.nan) for c in CAMS}
                for f0 in range(0, T, ch):
                    f1 = min(T, f0 + ch)
                    for c in CAMS[1:]:
                        delta[c][f0:f1] = sync.offset(c, t[f0:f1])          # causal: before seeing the chunk
                    for c in CAMS:
                        sync.push(c, streams[c][f0:f1])
                    sync.maybe_update()
                r = score(delta, tag)
                r["refit_seconds_mean"] = float(np.mean(sync.timing)) if sync.timing else None
                # error after the first minute
                late = {}
                for c in CAMS[1:]:
                    cover = np.isfinite(streams[c][:, 0, 0]) & (t >= 60 * FPS)
                    late[c] = float(np.nanmean(np.abs(delta[c] + o_true[c])[cover]))
                r["mae_after_60s"] = late
                print(f"{'':>22}  after 60 s: " + "  ".join(f"{c} {v:.3f}" for c, v in late.items())
                      + f" | refit {r['refit_seconds_mean']:.2f}s")
                res["stream"][tag] = r

    # ---- 3D: leave-one-camera-out reprojection -------------------------------------------------
    calib = json.load(open(a.calib, encoding="utf-8"))["cameras"]
    cams3 = [c for c in CAMS if c in calib]
    mats = [projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in cams3]
    seg = np.arange(s0, min(T, int(300 * FPS) - 10))        # all five cameras recorded
    ref = streams[REF][seg]
    speed = np.nanmedian(np.linalg.norm(np.diff(ref, axis=0, prepend=ref[:1]), axis=-1), axis=1)
    fast = speed >= np.nanpercentile(speed, 75)

    def views_for(delta):
        vs, vv = [], []
        for c in cams3:
            if c == REF:
                vs.append(undistort(ref, calib[c])); vv.append(vis[c][seg]); continue
            src = seg - delta[c][seg] if delta is not None else seg.astype(np.float64)   # o = -delta
            filled = np.nan_to_num(streams[c])
            vs.append(undistort(cubic_sample(filled, src, extrapolate=True), calib[c]))
            lo = np.clip(np.floor(src).astype(int), 0, T - 1); hi = np.clip(np.ceil(src).astype(int), 0, T - 1)
            vv.append(vis[c][lo] & vis[c][hi])
        return np.stack(vs, axis=2), np.stack(vv, axis=2)

    res["3d"] = {}
    for tag, delta in [("coarse only", None), ("CC offline", lines["CC offline"]), ("CSF offline", lines["CSF offline"]),
                       ("flash GT", {c: -o_true[c] for c in CAMS})]:
        V, W = views_for(delta)
        r = evaluate(V, W, mats, fast)
        res["3d"][tag] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        print(f"3D {tag:>12}: reproj {r['reproj']:.2f}px (fast {r['reproj_fast']:.2f}) | LOO {r['loo']:.2f} "
              f"(fast {r['loo_fast']:.2f}) | LOO median {r['loo_med']:.2f} (fast {r['loo_fast_med']:.2f})")
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
