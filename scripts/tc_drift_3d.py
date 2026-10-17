# -*- coding: utf-8 -*-
"""Drift recovery and downstream 3D on TotalCapture: real images, real detector, Vicon 3D.

Semi-real drift, as the Panoptic protocol but on MediaPipe keypoints of genlocked 60-fps video:
the reference camera keeps frames s0 + 2t (30 fps); every other camera k is observed at 60-fps
time s0 + 2(alpha*t + beta), alpha = 1 + d/N (cubic resampling of the real 60-fps detections),
so the true offset delta(t) = (alpha-1) t + beta is exact while detector noise and failures are
real. Each method estimates delta(t) for (ref, k); camera k is resampled back by the estimate
and all views are triangulated (DLT, calibrated cameras).

Reported per drift d (frames accumulated over the clip):
  E_drift   mean |delta_est - delta_true| over the clip interior (frames, per camera)
  3D vs sync  MPJPE (mm) to the triangulation of the genlocked, undrifted detections
              -- isolates the synchronization error from detector error
  3D vs Vicon MPJPE (mm) to Vicon joints (COCO -> Vicon joint map; marker-to-joint
              offsets add a constant bias, so compare methods, not absolute values)

    python scripts/tc_drift_3d.py --out docs/paper/clips/tc_drift3d.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from vsynclock.benchmark.baselines import CaspiIrani, CrossCorrelation
from vsynclock.benchmark.drift import fit_drift_c2f, shrink_slope, theilsen_line
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.benchmark.surrey_calib import read_surrey
from vsynclock.core.types import KeypointSequence
from vsynclock.model.panoptic_dataset import _fill_gaps
from vsynclock.panoptic.triangulation import mpjpe, projection_matrix, triangulate_masked
from realdata_eval import cc_profile, csf_profile
from tc_benchmark import load_sequences

FPS, INCH, VICON_OFFSET = 30.0, 0.0254, -2
# COCO-19 index -> Vicon joint name (MediaPipe "left" = subject's left)
COCO_TO_VICON = {0: "Neck", 2: "Hips", 3: "LeftArm", 4: "LeftForeArm", 5: "LeftHand", 6: "LeftUpLeg",
                 7: "LeftLeg", 8: "LeftFoot", 9: "RightArm", 10: "RightForeArm", 11: "RightHand",
                 12: "RightUpLeg", 13: "RightLeg", 14: "RightFoot"}


def tested_correction(line, centers, offsets, n, window, stride, rng, n_boot=500, alpha=0.05):
    """Apply only what the evidence supports. Bootstrap (non-overlapping windows) CIs for the
    Theil-Sen slope and for the median offset: slope CI containing 0 -> constant offset;
    constant CI also containing 0 -> no correction. Returns (curve, decision)."""
    t = np.arange(n, dtype=np.float64)
    if len(centers) < 4:
        return line, "line"
    step = max(1, int(np.ceil(window / stride)))
    c, o = np.asarray(centers)[::step], np.asarray(offsets)[::step]
    if len(c) < 4:
        return line, "line"
    sl, md = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(c), len(c))
        if len(np.unique(c[i])) < 2:
            continue
        sl.append(theilsen_line(c[i], o[i])[0]); md.append(np.median(o[i]))
    lo, hi = np.percentile(sl, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if lo > 0 or hi < 0:
        return line, "line"
    lo, hi = np.percentile(md, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if lo > 0 or hi < 0:
        return np.full(n, float(np.median(offsets))), "const"
    return np.zeros(n), "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp-dir", default="data/tc/tc_keypoints")
    ap.add_argument("--vicon-dir", default="data/tc/vicon")
    ap.add_argument("--checkpoint", default="checkpoints/csf_b1_t20.pt")
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 8, 4, 7], help="first = reference")
    ap.add_argument("--drifts", type=float, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--clip-len", type=int, default=300, help="frames at 30 fps")
    ap.add_argument("--clips", type=int, default=4, help="clips per sequence")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--min-valid", type=float, default=0.8)
    ap.add_argument("--features", choices=("speed", "speed+vy", "speed+vyb"), default="speed",
                    help="CC input: speed, or speed + signed vertical image velocity")
    ap.add_argument("--prior", type=float, default=None,
                    help="also report c2f with the drift slope shrunk by a N(0, prior^2) prior, frames/min")
    ap.add_argument("--no-test", action="store_true", help="skip the bootstrap-tested correction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    calib = read_surrey(os.path.join(a.kp_dir, "calibration.cal"))
    mats = [projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]) for c in a.cameras]
    seqs = load_sequences(a.kp_dir)
    cc = CrossCorrelation(max_lag=8, features=a.features)
    caspi = CaspiIrani(alpha_range=(0.985, 1.015), alpha_steps=31, max_lag=8)
    ccp, csfp = cc_profile(8, a.features), csf_profile(a.checkpoint)
    ref, others = a.cameras[0], a.cameras[1:]
    N, edge = a.clip_len, int(np.ceil(max(a.drifts) + abs(a.beta))) + 2
    margin = 2 * (edge + 2)                                      # 60-fps frames
    methods = ["naive", "CC-const", "Caspi-style", "CC-c2f", "CSF-c2f"]
    methods += [] if a.no_test else ["CC-c2f+test", "CSF-c2f+test"]
    methods += [] if a.prior is None else ["CC-c2f+prior", "CSF-c2f+prior"]
    methods += ["oracle"]
    trng = np.random.default_rng(7)
    decisions = {d: {"CC": [], "CSF": []} for d in a.drifts}
    res = {d: {m: {"edrift": [], "sync": [], "vicon": []} for m in methods} for d in a.drifts}
    clip_ids = []
    score = slice(edge, N - edge)
    t = np.arange(N, dtype=np.float64)
    for (subj, seq), cams in sorted(seqs.items()):
        if not all(c in cams for c in a.cameras):
            continue
        V = np.load(os.path.join(a.vicon_dir, f"{subj}_{seq}.npz"))
        names = list(V["names"]); vic = V["pos"].astype(np.float64) * INCH
        vj = [names.index(COCO_TO_VICON[j]) for j in sorted(COCO_TO_VICON)]
        cj = sorted(COCO_TO_VICON)
        T60 = min(min(len(cams[c]) for c in a.cameras), len(vic) - VICON_OFFSET)
        filled = {c: _fill_gaps(cams[c])[0] for c in a.cameras}
        valid60 = {c: np.isfinite(cams[c]).all(-1) for c in a.cameras}       # (T,19)
        span = 2 * N + 2 * margin
        cands = []
        for s0 in range(margin, T60 - span, 2 * N // 2):
            ok = all(valid60[c][s0:s0 + 2 * N].mean() >= a.min_valid for c in a.cameras)
            if ok:
                cands.append(s0)
        picks = [cands[i] for i in np.linspace(0, len(cands) - 1, min(a.clips, len(cands))).astype(int)] if cands else []
        # keep picks non-overlapping
        sel = []
        for s0 in picks:
            if all(abs(s0 - p) >= 2 * N for p in sel):
                sel.append(s0)
        for s0 in sel:
            clip_ids.append(f"{subj}_{seq}_{s0}")
            ref_s = filled[ref][s0:s0 + 2 * N:2]
            ref_v = valid60[ref][s0:s0 + 2 * N:2]
            sync_views = [ref_s] + [filled[c][s0:s0 + 2 * N:2] for c in others]
            sync_valid = [ref_v] + [valid60[c][s0:s0 + 2 * N:2] for c in others]

            def tri(views, valids):
                pts = np.stack(views, axis=2)                                   # (N,19,V,2)
                val = np.stack(valids, axis=2)
                X = triangulate_masked(pts.reshape(-1, len(views), 2), mats, val.reshape(-1, len(views)))
                return X.reshape(N, 19, 3)

            X_sync = tri(sync_views, sync_valid)
            vic_idx = np.clip(np.arange(s0, s0 + 2 * N, 2) + VICON_OFFSET, 0, len(vic) - 1)
            X_vic = vic[vic_idx][:, vj]
            for d in a.drifts:
                alpha = 1.0 + d / N
                truth = (alpha - 1.0) * t + a.beta
                obs, obs_valid = {}, {}
                for c in others:
                    idx60 = s0 + 2 * (alpha * t + a.beta)
                    obs[c] = cubic_sample(filled[c], idx60, extrapolate=True)
                    lo, hi = np.floor(idx60).astype(int), np.ceil(idx60).astype(int)
                    obs_valid[c] = valid60[c][np.clip(lo, 0, len(cams[c]) - 1)] & valid60[c][np.clip(hi, 0, len(cams[c]) - 1)]
                sa = KeypointSequence(ref_s, FPS)
                est = {m: {} for m in methods}
                for c in others:
                    sb = KeypointSequence(obs[c], FPS)
                    est["naive"][c] = np.zeros(N)
                    est["oracle"][c] = truth
                    est["CC-const"][c] = np.full(N, cc.predict(sa, sb).dt_frames)
                    e = caspi.estimate(sa, sb)
                    est["Caspi-style"][c] = (e.alpha - 1) * t + e.beta
                    for nm, pf in (("CC", ccp), ("CSF", csfp)):
                        line, cen, off = fit_drift_c2f(pf, ref_s, obs[c], FPS, a.window, a.stride, radius=3.0, prior="consensus")
                        est[f"{nm}-c2f"][c] = line
                        if not a.no_test:
                            est[f"{nm}-c2f+test"][c], dec = tested_correction(line, cen, off, N, a.window, a.stride, trng)
                            decisions[d][nm].append(dec)
                        if a.prior is not None:
                            sp, ip = shrink_slope(cen, off, a.prior / (60 * FPS), a.window, a.stride)
                            est[f"{nm}-c2f+prior"][c] = sp * t + ip
                for m in methods:
                    views, valids = [ref_s], [ref_v]
                    for c in others:
                        q = t - est[m][c]
                        views.append(cubic_sample(obs[c], q, extrapolate=True))
                        qi = np.clip(np.round(q).astype(int), 0, N - 1)
                        valids.append(obs_valid[c][qi])
                    X = tri(views, valids)
                    res[d][m]["edrift"].append(float(np.mean([np.mean(np.abs(est[m][c] - truth)[score]) for c in others])))
                    res[d][m]["sync"].append(1000 * mpjpe(X[score], X_sync[score]))
                    res[d][m]["vicon"].append(1000 * mpjpe(X[score][:, cj], X_vic[score]))
        print(f"  {subj}_{seq}: {len(sel)} clips", flush=True)

    print(f"\n{len(clip_ids)} clips of {N} frames; cameras {a.cameras} (ref {ref}); window {a.window}")
    for key, unit in (("edrift", "frames"), ("sync", "mm"), ("vicon", "mm")):
        print(f"\n{key} ({unit}), mean over clips")
        print(f"{'drift':>5} " + " ".join(f"{m:>12s}" for m in methods))
        for d in a.drifts:
            print(f"{d:5.1f} " + " ".join(f"{np.nanmean(res[d][m][key]):12.3f}" for m in methods))
    print("\ntested correction: share of (clip, camera) decisions line / const / none")
    for d in a.drifts:
        print(f"  d={d}: " + "  ".join(f"{nm}: " + " ".join(f"{k} {np.mean(np.array(v) == k):.0%}" for k in ("line", "const", "none"))
                                       for nm, v in decisions[d].items()))
    if a.out:
        json.dump({"decisions": {str(d): v for d, v in decisions.items()}, "clips": clip_ids, "cameras": a.cameras, "clip_len": N,
                   "results": {str(d): v for d, v in res.items()}}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
