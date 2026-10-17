# -*- coding: utf-8 -*-
"""Fig. 1 (v4): V-SyncLock system figure, minimal academic style.

One left-to-right pipeline of six numbered stages (the order the code runs), one visual per stage,
computed from the keypoints of our own five-device capture (take 1). A detail row underneath shows
the pairwise estimator (CC / CSF with the two-stage switch), the gated camera graph and the
synchronized 3D pose. Each device keeps one colour from input to output.

    python scripts/make_overview_v4.py --out docs/paper/mmsys27_v2/figures/fig_overview.pdf
"""
import argparse
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]

from vsynclock.benchmark.baselines.cross_correlation import body_down
from vsynclock.benchmark.drift import fit_drift_c2f
from vsynclock.benchmark.interpolation import cubic_sample
from vsynclock.core.types import KeypointSequence
from vsynclock.panoptic.triangulation import projection_matrix, triangulate_masked
from realdata_eval import cc_profile, coarse_offset, load_camera, load_host_times, motion_signal, undistort
from tc_camera_graph import ls_place

FPS = 30.0
TAKE, POSE, CALIB = "captures/take1", "data/realdata/s1_t1/pose", "data/realdata/s1_t1/calib.json"
REF = "iphone_cam1"
NAMES = {"iphone_cam1": "Phone A", "iphone_cam2": "Phone B", "iphone_cam3": "Phone C", "webcam0": "Webcam 0", "webcam1": "Webcam 1"}
DEV = {"iphone_cam1": "#3a3a3a", "iphone_cam2": "#1f6fb2", "iphone_cam3": "#2a9d55", "webcam0": "#D94841", "webcam1": "#7b5cb8"}
INK, GREY, LIGHT = "#1a1a1a", "#6e6e6e", "#bdbdbd"
T_DATA, T_CORE, T_OUT = "#DCEBFA", "#E7DFFA", "#DDF2E4"
BONES = [(0, 1), (0, 3), (3, 4), (4, 5), (0, 9), (9, 10), (10, 11), (0, 2), (2, 6), (6, 7), (7, 8),
         (2, 12), (12, 13), (13, 14)]
FS_STAGE, FS_LAB, FS_TICK, FS_NOTE = 7.4, 5.8, 5.2, 5.4


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=GREY, labelsize=FS_TICK, width=0.6, length=2, pad=1)
    ax.patch.set_alpha(0)


def webcam_src(name):
    t = load_host_times(TAKE, name); t = t - t[0]
    return np.interp(np.arange(0, t[-1], 1.0 / FPS), t, np.arange(len(t)))


def compute():
    calib = json.load(open(CALIB, encoding="utf-8"))["cameras"]
    cams = [REF] + sorted(c for c in calib if c != REF)
    data = {c: load_camera(POSE, TAKE, c)[0] for c in cams}
    raw = {c: np.load(os.path.join(POSE, f"{c}.npz"))["xy"].astype(np.float64) for c in cams}
    s0 = int(130 * FPS)
    sig = {c: motion_signal(data[c]) for c in cams}
    coarse = {REF: 0}
    for c in cams[1:]:
        coarse[c] = coarse_offset(sig[REF], sig[c], 400, s0, len(data[REF]))
    s1 = min(len(data[c]) - coarse[c] - 3 for c in cams)
    s0 = max([s0] + [-coarse[c] + 3 for c in cams])
    idx = np.arange(s0, s1, dtype=np.float64)
    S = {c: (data[c][s0:s1] if c == REF else cubic_sample(data[c], idx + coarse[c], extrapolate=True)) for c in cams}
    n = len(idx)
    R = S[REF]
    width = np.ptp(R[:, [5, 8, 11, 14], 0], axis=1) / (np.ptp(R[:, :, 1], axis=1) + 1e-9)
    t_star = int(np.argmax(np.where(np.isfinite(width), width, 0)[300:-300])) + 300
    prof = cc_profile(8, "speed+vyb")
    lines, rel, peak = {}, {}, {}
    for u, v in itertools.combinations(cams, 2):
        l2, cen, off = fit_drift_c2f(prof, S[u], S[v], FPS, 20, 6, prior="consensus")
        lines[(u, v)] = (l2, np.asarray(cen), np.asarray(off)); rel[(u, v)] = float(np.median(l2))
        pk = []
        for s in range(0, n - 80, 40):
            try:
                pk.append(np.max(prof(KeypointSequence(S[u][s:s + 80], FPS), KeypointSequence(S[v][s:s + 80], FPS))[1]))
            except ValueError:
                pass
        peak[(u, v)] = float(np.mean(pk))
    place = ls_place(cams, sorted(rel), lambda u, v: rel[(u, v)] if (u, v) in rel else -rel[(v, u)], peak, root=REF)
    t = np.arange(n, dtype=np.float64)
    views, mats = [], []
    for c in cams:
        d = 0.0 if c == REF else lines[(REF, c)][0]
        st = S[c] if c == REF else cubic_sample(S[c], t - d, extrapolate=True)
        views.append(undistort(st[t_star:t_star + 1], calib[c]))
        mats.append(projection_matrix(calib[c]["K"], calib[c]["R"], calib[c]["t"]))
    X = triangulate_masked(np.stack(views, 2).reshape(-1, len(cams), 2), mats, np.ones((19, len(cams)), bool)).reshape(19, 3)
    frames = {}
    for c in cams:
        r = s0 + t_star + coarse[c]
        if c.startswith("webcam"):
            r = int(round(webcam_src(c)[int(r)]))
        frames[c] = raw[c][int(r)]
    return dict(calib=calib, cams=cams, raw=raw, S=S, n=n, t_star=t_star, prof=prof, lines=lines, peak=peak,
                place=place, X=X, frames=frames)


# ------------------------------------------------------------------ drawing
class Canvas:
    def __init__(self, fig):
        self.fig = fig
        self.bg = fig.add_axes([0, 0, 1, 1]); self.bg.set_xlim(0, 1); self.bg.set_ylim(0, 1); self.bg.axis("off"); self.bg.set_zorder(-5)
        self.fg = fig.add_axes([0, 0, 1, 1]); self.fg.set_xlim(0, 1); self.fg.set_ylim(0, 1); self.fg.axis("off"); self.fg.set_zorder(20)
        self.fg.patch.set_alpha(0)

    def stage(self, x, y, w, h, num, title, fill):
        if fill:
            self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc=fill, ec="none"))
        else:
            self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec="#d6d6d6", lw=0.8))
        self.bg.add_patch(Circle((x + 0.016, y + h - 0.030), 0.0115, fc=INK, ec="none", transform=self.bg.transData))
        self.bg.text(x + 0.016, y + h - 0.031, num, fontsize=5.8, color="white", ha="center", va="center", fontweight="bold")
        self.bg.text(x + 0.032, y + h - 0.031, title, fontsize=FS_STAGE, color=INK, ha="left", va="center", fontweight="bold")

    def arrow(self, p, q, color=GREY, lw=1.4, ls="-", rad=0.0, head=True):
        self.fg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-", mutation_scale=8, color=color, lw=lw, ls=ls,
                                          connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def block(self, x, y, w, h, text, fc="#f2f2f2", ec="#9a9a9a", fs=5.4, bold=False, color=INK):
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", fc=fc, ec=ec, lw=0.7))
        self.bg.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", color=color, linespacing=1.1,
                     fontweight="bold" if bold else "normal")


def draw(D, out):
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(7.0, 3.55))
    K = Canvas(fig)
    cams, S, n = D["cams"], D["S"], D["n"]
    # stage geometry (top row): six columns
    top_y, top_h = 0.475, 0.515
    xs = [0.006, 0.172, 0.338, 0.504, 0.670, 0.836]
    w = 0.158
    stages = [("1", "Input", None), ("2", "Device timing", T_DATA), ("3", "Signed motion", T_DATA),
              ("4", "Local lag", T_CORE), ("5", "Drift fit", T_CORE), ("6", "Output", T_OUT)]
    for x, (num, ttl, fill) in zip(xs, stages):
        K.stage(x, top_y, w, top_h, num, ttl, fill)
    for k in range(5):
        K.arrow((xs[k] + w + 0.0005, top_y + top_h * 0.47), (xs[k + 1] - 0.0005, top_y + top_h * 0.47), color=INK, lw=1.3)

    # (1) skeletons, one colour per device
    pos = [(0.014, 0.770), (0.090, 0.770), (0.014, 0.655), (0.090, 0.655), (0.052, 0.540)]
    for (x, y), c in zip(pos, cams):
        ax = fig.add_axes([x, y, 0.066, 0.095])
        kp = D["frames"][c]
        q = (kp - np.nanmean(kp, 0)) / (np.nanmax(np.ptp(kp[np.isfinite(kp).all(1)], 0)) + 1e-9)
        for (i, j) in BONES:
            if np.isfinite(q[[i, j]]).all():
                ax.plot(q[[i, j], 0], -q[[i, j], 1], color=DEV[c], lw=1.5, solid_capstyle="round")
        ax.plot(q[1, 0], -q[1, 1], "o", ms=2.6, color=DEV[c])
        ax.set_aspect("equal"); ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.62, 0.55); ax.axis("off")
        ax.text(0.5, -0.04, NAMES[c], transform=ax.transAxes, fontsize=FS_LAB - 0.4, ha="center", va="top", color=DEV[c], fontweight="bold")

    # (2) frame-clock error over the take
    ax = fig.add_axes([xs[1] + 0.045, 0.625, 0.103, 0.265])
    for c in cams:
        if c.startswith("webcam"):
            ht = load_host_times(TAKE, c); ht = ht - ht[0]
            err, tm = (np.arange(len(ht)) / FPS - ht) * FPS, ht / 60
        else:
            fps_c = {"iphone_cam1": 30.001, "iphone_cam2": 29.986, "iphone_cam3": 30.016}[c]
            k = np.arange(len(D["raw"][c])); tm = k / fps_c / 60; err = (k / FPS - k / fps_c) * FPS
        ax.plot(tm, err, color=DEV[c], lw=1.6 if c == "webcam0" else 1.2)
    ax.annotate("Webcam 0:\n29.04 fps", xy=(2.6, -148), xytext=(2.5, -60), fontsize=FS_NOTE - 0.4, color=DEV["webcam0"],
                arrowprops=dict(arrowstyle="-", color=DEV["webcam0"], lw=0.6))
    ax.set_xticks([0, 2, 4]); ax.set_yticks([0, -100, -200])
    ax.set_xlabel("time (min)", fontsize=FS_LAB, color=GREY, labelpad=1)
    ax.set_ylabel("frame error", fontsize=FS_LAB - 0.3, color=GREY, labelpad=1)
    style(ax)
    K.bg.text(xs[1] + w / 2, 0.508, "resample on timestamps,\nfix integer lag", fontsize=FS_NOTE, ha="center", va="center", color=GREY)

    # (3) signed motion channels of two views
    t0 = D["t_star"] - 45
    for k, c in enumerate((REF, "iphone_cam2")):
        A = S[c][t0:t0 + 90]; v = np.gradient(A, axis=0)
        F = np.concatenate([np.linalg.norm(v, axis=-1), v @ body_down(A)], 1)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        ax = fig.add_axes([xs[2] + 0.024 + k * 0.067, 0.615, 0.060, 0.275])
        ax.imshow(F.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.axhline(18.5, color="white", lw=1.0)
        ax.set_xticks([]); ax.set_yticks([9, 28] if k == 0 else [])
        if k == 0:
            ax.set_yticklabels(["$s_j$", "$u_j$"], fontsize=FS_LAB)
        for sp in ax.spines.values():
            sp.set_color(DEV[c]); sp.set_linewidth(1.4)
        ax.tick_params(length=0, pad=1)
        ax.set_title(NAMES[c], fontsize=FS_LAB, color=DEV[c], pad=2, fontweight="bold")
    K.bg.text(xs[2] + w / 2, 0.548, "$s_j=\\|\\dot{\\mathbf{x}}_j\\|$,   $u_j=\\dot{\\mathbf{x}}_j\\cdot\\mathbf{d}$", fontsize=FS_NOTE + 0.4,
              ha="center", va="center", color=INK)
    K.bg.text(xs[2] + w / 2, 0.506, "$\\mathbf{d}$: body axis (neck$\\rightarrow$hip)", fontsize=FS_NOTE - 0.3, ha="center", va="center", color=GREY)

    # (4) lag profile of one window, with the search band
    l2, cen, off = D["lines"][(REF, "iphone_cam2")]
    mid = int(cen[len(cen) // 2]); s = mid - 10
    lags, sc = D["prof"](KeypointSequence(S[REF][s:s + 20], FPS), KeypointSequence(S["iphone_cam2"][s:s + 20], FPS))
    sc = (sc - sc.mean()) / (sc.std() + 1e-9); prior = l2[mid]
    ax = fig.add_axes([xs[3] + 0.016, 0.625, 0.128, 0.265])
    ax.axvspan(prior - 3, prior + 3, color="#c9b8ef", lw=0)
    ax.plot(lags, sc, color=INK, lw=1.4)
    kk = int(np.argmax(np.where(np.abs(lags - prior) <= 3, sc, -9)))
    ax.plot(lags[kk], sc[kk], "o", color="#D94841", ms=4.2, mec="white", mew=0.6, zorder=3)
    ax.set_xlim(-8, 8); ax.set_xticks([-8, 0, 8]); ax.set_yticks([])
    ax.set_xlabel("lag $k$ (frames)", fontsize=FS_LAB, color=GREY, labelpad=1); style(ax)
    ax.text(prior, 1.02, "$\\pm\\rho$", transform=ax.get_xaxis_transform(), fontsize=FS_LAB, ha="center", va="bottom", color="#5b3fa8")
    K.bg.text(xs[3] + w / 2, 0.508, "$\\hat\\delta_i$: sub-frame peak\nper 20-frame window", fontsize=FS_NOTE, ha="center", va="center", color=GREY)

    # (5) drift fit, Phone B vs Phone A
    ax = fig.add_axes([xs[4] + 0.044, 0.625, 0.104, 0.265])
    tm = cen / FPS
    b = np.arange(0, tm.max() + 5, 5.0)
    bc = [(x + 2.5, np.median(off[(tm >= x) & (tm < x + 5)])) for x in b if ((tm >= x) & (tm < x + 5)).sum() >= 3]
    tl = np.arange(n) / FPS
    ax.fill_between(tl, l2 - 3, l2 + 3, color="#c9b8ef", lw=0, alpha=0.7)
    ax.scatter([p[0] for p in bc], [p[1] for p in bc], s=6, color=GREY, lw=0, zorder=2)
    ax.plot(tl, l2, color=DEV["iphone_cam2"], lw=1.8, zorder=3)
    ax.set_ylim(np.median(l2) - 4, np.median(l2) + 4)
    ax.set_xlabel("time (s)", fontsize=FS_LAB, color=GREY, labelpad=1); ax.set_ylabel("$\\hat\\delta$ (frames)", fontsize=FS_LAB, color=GREY, labelpad=1)
    style(ax)
    slope = (l2[-1] - l2[0]) / (n / FPS / 60)
    K.bg.text(xs[4] + w / 2, 0.508, f"Theil–Sen line, coarse-to-fine\nPhone B: {slope:+.1f} frames/min", fontsize=FS_NOTE, ha="center", va="center", color=GREY)

    # (6) per-camera timing trajectories
    ax = fig.add_axes([xs[5] + 0.046, 0.625, 0.102, 0.265])
    tt = np.arange(n) / FPS / 60
    for c in cams[1:]:
        d = D["lines"][(REF, c)][0]
        ax.plot(tt, d - np.median(d) + D["place"][c], color=DEV[c], lw=1.5)
    ax.axhline(0, color=DEV[REF], lw=1.2, ls=(0, (3, 2)))
    ax.set_xlabel("time (min)", fontsize=FS_LAB, color=GREY, labelpad=1); ax.set_ylabel("$\\hat\\delta_c(t)$ (frames)", fontsize=FS_LAB, color=GREY, labelpad=1)
    style(ax)
    K.bg.text(xs[5] + w / 2, 0.508, "sub-frame offset and drift\nper camera, or abstain", fontsize=FS_NOTE, ha="center", va="center", color=GREY)

    # ------------------------------------------------------------------ detail row
    dy, dh = 0.012, 0.425
    # estimator detail under (3)-(4)
    ex, ew = 0.172, 0.490
    K.bg.add_patch(FancyBboxPatch((ex, dy), ew, dh, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec="#c9b8ef", lw=1.0))
    K.bg.text(ex + 0.010, dy + dh - 0.030, "Pairwise local estimator (stage 4)", fontsize=FS_STAGE - 0.6, color=INK, fontweight="bold", va="center")
    K.arrow((xs[3] + w / 2, top_y), (xs[3] + w / 2, dy + dh), color=LIGHT, lw=1.0, ls=(0, (3, 2)), head=False)
    yA, yB, bh = 0.255, 0.105, 0.075
    K.block(ex + 0.010, yA, 0.050, bh, "$\\mathbf{f}^u$\n$2J{\\times}W$", fc="#f5f5f5")
    K.block(ex + 0.010, yB, 0.050, bh, "$\\mathbf{f}^v$\n$2J{\\times}W$", fc="#f5f5f5")
    cx = [ex + 0.074, ex + 0.140, ex + 0.214]
    for yy in (yA, yB):
        K.block(cx[0], yy, 0.054, bh, "Linear\n$\\rightarrow$128", fs=5.0)
        K.block(cx[1], yy, 0.062, bh, "3$\\times$ temporal\nattention\n+ RoPE", fs=5.0)
        K.arrow((ex + 0.061, yy + bh / 2), (cx[0] - 0.001, yy + bh / 2), color=GREY, lw=0.9)
        K.arrow((cx[0] + 0.055, yy + bh / 2), (cx[1] - 0.001, yy + bh / 2), color=GREY, lw=0.9)
        K.arrow((cx[1] + 0.063, yy + bh / 2), (cx[2] - 0.001, yy + bh / 2 + (0.03 if yy == yB else -0.03)), color=GREY, lw=0.9)
    K.block(cx[2], yB + 0.02, 0.054, yA - yB + bh - 0.04, "cross-\nview\nattention", fc="#efe9fb", ec="#9b86d6", fs=5.0)
    K.block(cx[2] + 0.070, 0.165, 0.066, 0.080, "$\\phi_\\mathrm{CSF}(k)$\ncosine,\n17 lags", fc="#efe9fb", ec="#9b86d6", fs=5.0)
    K.arrow((cx[2] + 0.055, 0.205), (cx[2] + 0.069, 0.205), color=GREY, lw=0.9)
    K.bg.text((cx[0] + cx[2] + 0.05) / 2, 0.045, "CSF (0.8 M parameters, shared weights)", fontsize=FS_NOTE - 0.2, ha="center", color="#5b3fa8", style="italic")
    # CC path
    K.block(cx[2] + 0.070, 0.300, 0.066, 0.062, "$\\phi_\\mathrm{CC}(k)$\nmean $\\mathbf{f}^u{\\cdot}\\mathbf{f}^v$", fc="#f5f5f5", fs=5.0)
    K.arrow((ex + 0.035, yA + bh + 0.001), (cx[2] + 0.069, 0.345), color=GREY, lw=0.9, rad=-0.18)
    # switch and output
    swx = cx[2] + 0.158
    K.block(swx, 0.180, 0.084, 0.150, "two-stage\nswitch\n\nCSF if CC's\nquarter peaks\ndisagree:\n$D\\geq\\delta$", fc="#efe9fb", ec="#5b3fa8", fs=5.2)
    K.arrow((cx[2] + 0.137, 0.330), (swx - 0.001, 0.300), color=GREY, lw=0.9)
    K.arrow((cx[2] + 0.137, 0.205), (swx - 0.001, 0.215), color=GREY, lw=0.9)
    K.bg.text(swx + 0.042, 0.135, "4% of clips\nuse CSF", fontsize=FS_NOTE - 0.2, ha="center", va="center", color="#5b3fa8")

    # rig detail under (5)
    rx, rw = 0.670, 0.158
    K.bg.add_patch(FancyBboxPatch((rx, dy), rw, dh, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec="#c9b8ef", lw=1.0))
    K.bg.text(rx + 0.010, dy + dh - 0.030, "Gate + placement", fontsize=FS_STAGE - 0.6, color=INK, fontweight="bold", va="center")
    K.arrow((xs[4] + w / 2, top_y), (xs[4] + w / 2, dy + dh), color=LIGHT, lw=1.0, ls=(0, (3, 2)), head=False)
    ax = fig.add_axes([rx + 0.006, 0.075, rw - 0.012, 0.30])
    calib = D["calib"]
    xy2 = {c: np.array(calib[c]["center"])[:2] for c in cams}
    pk = np.array(list(D["peak"].values()))
    for (u, v), wgt in D["peak"].items():
        p, q = xy2[u], xy2[v]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#8f7bd0", lw=0.5 + 2.2 * (wgt - pk.min()) / (np.ptp(pk) + 1e-9), zorder=1)
    for c in cams:
        ax.plot(*xy2[c], "o", ms=8, color=DEV[c], mec="white", mew=0.8, zorder=3)
        dyy = -1.15 if xy2[c][1] < -1 else 0.95
        ax.text(xy2[c][0], xy2[c][1] + dyy, "ref" if c == REF else f"{D['place'][c]:+.2f}", fontsize=FS_NOTE - 0.2,
                ha="center", va="center", color=DEV[c], fontweight="bold")
    allp = np.array(list(xy2.values()))
    ax.set_xlim(allp[:, 0].min() - 1.2, allp[:, 0].max() + 1.2); ax.set_ylim(allp[:, 1].min() - 1.8, allp[:, 1].max() + 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    K.bg.text(rx + rw / 2, 0.048, "null-gated edges,\nleast-squares offsets", fontsize=FS_NOTE - 0.2, ha="center", va="center", color=GREY)

    # 3D detail under (6)
    ox, ow = 0.836, 0.158
    K.bg.add_patch(FancyBboxPatch((ox, dy), ow, dh, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec="#9fd4b1", lw=1.0))
    K.bg.text(ox + 0.010, dy + dh - 0.030, "Synchronized 3D", fontsize=FS_STAGE - 0.6, color=INK, fontweight="bold", va="center")
    K.arrow((xs[5] + w / 2, top_y), (xs[5] + w / 2, dy + dh), color=LIGHT, lw=1.0, ls=(0, (3, 2)), head=False)
    ax3 = fig.add_axes([ox + 0.005, 0.045, ow - 0.01, 0.33], projection="3d")
    P = D["X"] - D["X"][2]
    floor = -P[:, 2].max() - 0.05
    for (i, j) in BONES:
        ax3.plot(P[[i, j], 0], P[[i, j], 1], -P[[i, j], 2], color=INK, lw=1.8)
        ax3.plot(P[[i, j], 0], P[[i, j], 1], np.full(2, floor), color=LIGHT, lw=0.7)
    ax3.view_init(elev=14, azim=-35); ax3.set_box_aspect((1, 1, 1.7))
    lim = np.abs(P).max(); ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
    ax3.set_axis_off(); ax3.patch.set_alpha(0)
    K.bg.text(ox + ow / 2, 0.035, "DLT from 5 views", fontsize=FS_NOTE - 0.2, ha="center", va="center", color=GREY)

    # left of the detail row: clock model and device key
    K.bg.add_patch(FancyBboxPatch((0.006, dy), 0.158, dh, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec="#d6d6d6", lw=0.8))
    K.bg.text(0.016, dy + dh - 0.030, "Clock model", fontsize=FS_STAGE - 0.6, color=INK, fontweight="bold", va="center")
    K.bg.text(0.085, 0.300, "$t_c(n) = \\alpha_c\\, n + \\beta_c$\n\n$\\delta_c(t) = (\\alpha_c - 1)\\,t + \\beta_c$",
              fontsize=FS_LAB + 0.6, ha="center", va="center", color=INK, linespacing=1.2)
    for k, c in enumerate(cams):
        yy = 0.170 - 0.028 * k
        K.bg.plot([0.025, 0.045], [yy, yy], color=DEV[c], lw=2.0)
        K.bg.text(0.052, yy, NAMES[c] + (" (reference)" if c == REF else ""), fontsize=FS_NOTE - 0.2, va="center", color=INK)
    fig.savefig(out, dpi=300)
    print("wrote", out, "slope", round(slope, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    draw(compute(), a.out)


if __name__ == "__main__":
    main()
