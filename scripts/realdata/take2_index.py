# -*- coding: utf-8 -*-
"""Capture 2 read without timestamps (frame index at the nominal 30 fps): real drift, flash ground truth.

Pre-registration: docs/paper/DEV_PROTOCOL.md ("timestamp-free reading of capture 2"). Pairs (iphone1, c);
the true lag at reference frame i is k_c(i) - i, where k_c(i) is the frame of c whose clock time maps to
iphone1's frame i through the flash lines. Single lag vs CC-c2f vs CSF-c2f.

    python scripts/realdata/take2_index.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path[:0] = [ROOT, os.path.dirname(__file__)]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from realdata_eval import cc_profile, coarse_offset, csf_profile, load_host_times, motion_signal  # noqa: E402
from take2_eval import gt_offset  # noqa: E402
from vsynclock.benchmark.baselines import CrossCorrelation  # noqa: E402
from vsynclock.benchmark.drift import fit_drift_c2f  # noqa: E402
from vsynclock.core.types import KeypointSequence  # noqa: E402
from vsynclock.model.panoptic_dataset import _fill_gaps  # noqa: E402

TAKE, POSE = "captures/take2", "data/realdata/s2_t2/pose"
FPS, REF, START = 30.0, "iphone1", 900
OTHERS = ["iphone2", "iphone3", "webcam0", "webcam1"]


def clock(c):
    t = load_host_times(TAKE, c)
    if t is None:
        return np.load(f"data/realdata/s2_t2/pts_{c}.npy").astype(np.float64)
    t = np.asarray(t, np.float64)
    return t - t[0]


def main():
    os.chdir(ROOT)
    gt = json.load(open("data/take2/flash_gt.json", encoding="utf-8"))
    xy = {c: _fill_gaps(np.load(f"{POSE}/{c}.npz")["xy"].astype(np.float64))[0] for c in [REF] + OTHERS}
    tc = {c: clock(c) for c in [REF] + OTHERS}
    prof = {"CC-c2f": cc_profile(8, "speed+vyb"), "CSF-c2f": csf_profile("checkpoints/csf_b1_t20_vy.pt")}
    cc = CrossCorrelation(8, "speed+vyb")

    # sign convention of the c2f line, on a known shift: b[i + 2] = a[i]  ->  expect line = -2
    A0 = xy[REF][START:START + 1800]
    B0 = np.concatenate([np.repeat(A0[:1], 2, 0), A0])[:len(A0)]
    l0 = fit_drift_c2f(prof["CC-c2f"], A0, B0, FPS, 20, 6, prior="consensus")[0]
    sgn = -1.0 if np.median(l0) < 0 else 1.0
    print(f"sign check: known lag +2 -> line median {np.median(l0):+.2f} (lag = L {'-' if sgn < 0 else '+'} line)")

    out = {}
    A = xy[REF]
    for c in OTHERS:
        B = xy[c]
        i = np.arange(len(A), dtype=np.float64)
        t_ref = tc[REF][: len(A)]
        k = np.interp(t_ref + gt_offset(gt, c, t_ref), tc[c], np.arange(len(tc[c])), left=np.nan, right=np.nan)
        true = k - i
        L = coarse_offset(motion_signal(A), motion_signal(B), 400, START, None)
        e = min(len(A), len(B) - L - 9)
        s = START
        Aseg, Bseg = A[s:e], B[s + L:e + L]
        tr = true[s:e]
        ok = np.isfinite(tr)
        res = dict(coarse=int(L), frames=int(e - s), true_slope=float(np.polyfit(np.arange(e - s)[ok] / FPS / 60, tr[ok], 1)[0]),
                   true_range=[float(np.nanmin(tr)), float(np.nanmax(tr))])
        try:
            d = cc.predict(KeypointSequence(Aseg, FPS), KeypointSequence(Bseg, FPS)).dt_frames
            est = np.full(e - s, L + sgn * float(d))
        except Exception:
            est = np.full(e - s, float(L))
        ests = {"single lag": est}
        for m, p in prof.items():
            line = fit_drift_c2f(p, Aseg, Bseg, FPS, 20, 6, prior="consensus")[0]
            ests[m] = L + sgn * np.asarray(line)
        tm = np.arange(e - s) / FPS / 60
        for m, v in ests.items():
            err = np.abs(v - tr)[ok]
            res[m] = dict(E_drift=float(err.mean()), p95=float(np.percentile(err, 95)),
                          slope_err=float(np.polyfit(tm[ok], v[ok], 1)[0] - res["true_slope"]))
        out[c] = res
        print(f"{c}: coarse {L}, span {(e - s) / FPS:.0f} s, true slope {res['true_slope']:+.2f} fr/min, "
              f"true lag {res['true_range'][0]:+.1f}..{res['true_range'][1]:+.1f}")
        for m in ests:
            r = res[m]
            print(f"   {m:10s} E_drift {r['E_drift']:.3f}  p95 {r['p95']:.2f}  slope err {r['slope_err']:+.2f} fr/min")
    os.makedirs("data/refine", exist_ok=True)
    json.dump(out, open("data/refine/take2_index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
