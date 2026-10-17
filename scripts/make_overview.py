# -*- coding: utf-8 -*-
"""Fig. 1 (v6): V-SyncLock method figure; one saturated colour per module, natural arrow routing.

A. input and device timing (compact) -> B. pairwise temporal alignment, the contribution (largest,
the only tinted block: signed motion -> lag estimation -> drift correction) -> C. global placement
-> D. output. One accent colour for the method; other colours only identify devices, and each
device keeps its colour throughout. All panels are computed from our five-device capture.

    python scripts/make_overview.py --out docs/paper/mmsys27_v2/figures/fig_overview.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]

from make_overview_v4 import BONES, DEV, FPS, NAMES, REF, TAKE, compute
DEV = dict(DEV, iphone_cam1="#a0794b")                 # reference phone: ochre, so that only arrows are black
ARROW = "#000000"
LINE = "#8a8a8a"                                         # non-arrow neutral lines
from vsynclock.benchmark.baselines.cross_correlation import body_down
from vsynclock.core.types import KeypointSequence
from realdata_eval import load_host_times

INK, DARK, GREY, LIGHT, RULE = "#1b1b1b", "#3c3c3c", "#737373", "#b5b5b5", "#cfcfcf"
MOD = {"A": "#00796b", "B": "#1a4f9c", "C": "#8e2472", "D": "#d35400"}   # one saturated colour per module
ACC = MOD["B"]
ACC_L = MOD["B"]
WARM_LINE, WARM_BAND, WARM_DOT = "#e65100", "#f6b400", "#c62828"   # the two plots in B (band drawn at alpha 0.45)
CC_C, CSF_C = "#1e8e3e", "#c62828"
LW_PIPE, LW_BORDER, LW_DATA, LW_AUX = 0.7, 0.5, 0.85, 0.45
FS_H, FS_SUB, FS_LAB, FS_TICK = 7.6, 6.4, 5.6, 5.0


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LIGHT); ax.spines[s].set_linewidth(0.4)
    ax.tick_params(colors=GREY, labelsize=FS_TICK, width=0.4, length=1.6, pad=1)
    ax.patch.set_alpha(0)


class Canvas:
    def __init__(self, fig):
        self.bg = fig.add_axes([0, 0, 1, 1]); self.bg.set_xlim(0, 1); self.bg.set_ylim(0, 1); self.bg.axis("off"); self.bg.set_zorder(-5)
        self.fg = fig.add_axes([0, 0, 1, 1]); self.fg.set_xlim(0, 1); self.fg.set_ylim(0, 1); self.fg.axis("off"); self.fg.set_zorder(30)
        self.fg.patch.set_alpha(0)

    def module(self, x, y, w, h, letter, title, fill=None, edge=RULE, tag=None):
        col = MOD[letter]
        hb = 0.052                                                 # header band height
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                         fc="white", ec=col, lw=2.4))
        self.bg.add_patch(FancyBboxPatch((x, y + h - hb), w, hb, boxstyle="round,pad=0,rounding_size=0.006",
                                         fc=col, ec=col, lw=2.4))
        self.bg.add_patch(plt.Rectangle((x, y + h - hb), w, hb / 2, fc=col, ec="none"))   # square lower corners
        self.bg.text(x + 0.008, y + h - hb / 2, f"{letter}  {title}", fontsize=FS_H, color="white", fontweight="bold", va="center")
        if tag:
            self.bg.text(x + w - 0.008, y + h - hb / 2, tag, fontsize=FS_LAB, color="white", fontweight="semibold", va="center", ha="right",
                         style="italic")

    def sub(self, x, y, text, color=DARK):
        self.bg.text(x, y, text, fontsize=FS_SUB, color=color, fontweight="semibold", va="center")

    def note(self, x, y, text, ha="center", color=GREY, fs=FS_LAB):
        self.bg.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center", linespacing=1.15)

    def arrow(self, p, q, color=ARROW, lw=1.2, rad=0.0, ls="-", head=5.5, style=None):
        cs = style or f"arc3,rad={rad}"
        self.fg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                          connectionstyle=cs, shrinkA=0, shrinkB=0, joinstyle="round", capstyle="round"))

    def line(self, pts, color=ARROW, lw=1.2):
        xs, ys = zip(*pts)
        self.fg.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")

    def chip(self, x, y, w, h, text, fc="white", ec=LIGHT, color=INK, fs=FS_LAB, bold=False):
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.004", fc=fc, ec=ec, lw=0.45))
        self.bg.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", color=color, linespacing=1.1,
                     fontweight="semibold" if bold else "normal")


LINE_SCALE = 0.5            # global stroke scale applied to every line, border, arrow and tick


def scale_strokes(fig, k):
    """Multiply the stroke width of every artist in the figure by k (text and fills unchanged)."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for o in fig.findobj():
        if isinstance(o, Line2D):
            o.set_linewidth(o.get_linewidth() * k)
            o.set_markeredgewidth(o.get_markeredgewidth() * k)
        elif isinstance(o, (Patch, Collection)) and o is not fig.patch:
            lw = o.get_linewidth()
            o.set_linewidth(np.asarray(lw) * k if np.ndim(lw) else lw * k)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                for ln in (t.tick1line, t.tick2line):
                    ln.set_markeredgewidth(ln.get_markeredgewidth() * k)


def skeleton(ax, kp, color):
    q = (kp - np.nanmean(kp, 0)) / (np.nanmax(np.ptp(kp[np.isfinite(kp).all(1)], 0)) + 1e-9)
    for (i, j) in BONES:
        if np.isfinite(q[[i, j]]).all():
            ax.plot(q[[i, j], 0], -q[[i, j], 1], color=color, lw=0.8, solid_capstyle="round")
    ok = np.isfinite(q).all(1)
    ax.scatter(q[ok, 0], -q[ok, 1], s=1.0, color=color, lw=0, zorder=3)
    ax.plot(q[1, 0], -q[1, 1], "o", ms=1.9, color=color)
    ax.set_aspect("equal"); ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.55); ax.axis("off")


def draw(D, out):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(7.0, 3.15))
    K = Canvas(fig)
    cams, S, n = D["cams"], D["S"], D["n"]
    other = "iphone_cam2"
    Y0, H = 0.03, 0.94
    mid = Y0 + H * 0.47

    # ---------------- A. input and device timing
    ax_, aw = 0.005, 0.178
    K.module(ax_, Y0, aw, H, "A", "Input")
    spots = [(0.010, 0.745), (0.068, 0.745), (0.126, 0.745), (0.039, 0.600), (0.097, 0.600)]
    for (x, y), c in zip(spots, cams):
        a = fig.add_axes([x, y, 0.050, 0.135])
        skeleton(a, D["frames"][c], DEV[c])
        K.bg.text(x + 0.025, y - 0.010, NAMES[c], fontsize=FS_TICK, color=DEV[c], ha="center", va="center", fontweight="semibold")
    a = fig.add_axes([ax_ + 0.042, 0.165, 0.123, 0.30])
    for c in cams:
        if c.startswith("webcam"):
            ht = load_host_times(TAKE, c); ht = ht - ht[0]
            err, tm = (np.arange(len(ht)) / FPS - ht) * FPS, ht / 60
        else:
            fps_c = {"iphone_cam1": 30.001, "iphone_cam2": 29.986, "iphone_cam3": 30.016}[c]
            k = np.arange(len(D["raw"][c])); tm = k / fps_c / 60; err = (k / FPS - k / fps_c) * FPS
        a.plot(tm, err, color=DEV[c], lw=LW_DATA if c == "webcam0" else LW_AUX + 0.1)
    a.text(2.4, -70, "W0: 29.04 fps\n(file says 30)", fontsize=FS_TICK, color=DEV["webcam0"])
    a.set_xticks([0, 2, 4]); a.set_yticks([0, -100, -200])
    a.set_xlabel("time (min)", fontsize=FS_LAB, color=GREY, labelpad=1); a.set_ylabel("frame-clock error", fontsize=FS_LAB, color=GREY, labelpad=1)
    style(a)
    K.sub(ax_ + 0.008, 0.520, "Device timing", color=MOD["A"])
    K.note(ax_ + aw / 2, 0.065, "resample on timestamps,\nfix the integer lag")

    # ---------------- B. pairwise temporal alignment (the contribution)
    bx, bw = 0.208, 0.465
    K.module(bx, Y0, bw, H, "B", "Pairwise temporal alignment", tag="our contribution")
    # B1 signed motion
    K.sub(bx + 0.010, 0.875, "1  Signed motion", color=ACC)
    t0 = D["t_star"] - 45
    for k, c in enumerate((REF, other)):
        A = S[c][t0:t0 + 90]; v = np.gradient(A, axis=0)
        F = np.concatenate([np.linalg.norm(v, axis=-1), v @ body_down(A)], 1)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        a = fig.add_axes([bx + 0.030 + k * 0.066, 0.555, 0.058, 0.27])
        a.imshow(F.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        a.axhline(18.5, color="white", lw=0.6)
        a.set_xticks([]); a.set_yticks([9, 28] if k == 0 else [])
        if k == 0:
            a.set_yticklabels(["$s_j$", "$u_j$"], fontsize=FS_LAB)
        for sp in a.spines.values():
            sp.set_color(DEV[c]); sp.set_linewidth(0.7)
        a.tick_params(length=0, pad=1)
        a.set_title(NAMES[c], fontsize=FS_LAB, color=DEV[c], pad=2, fontweight="semibold")
    K.note(bx + 0.093, 0.515, "$s_j=\\|\\dot{\\mathbf{x}}_j\\|$,  $u_j=\\dot{\\mathbf{x}}_j\\cdot\\mathbf{d}$", color=INK)
    # B2 lag estimation
    x2 = bx + 0.175
    K.sub(x2, 0.875, "2  Lag estimation", color=ACC)
    K.chip(x2, 0.745, 0.085, 0.07, "CC  $\\phi_\\mathrm{CC}(k)$", fc=CC_C, ec=CC_C, color="white", bold=True)
    K.chip(x2, 0.595, 0.085, 0.085, "CSF  $\\phi_\\mathrm{CSF}(k)$\nattention, 0.8 M", fc=CSF_C, ec=CSF_C, color="white", bold=True)
    K.note(x2 + 0.0425, 0.540, "CSF only where CC's\nquarters disagree:\n$D\\geq\\delta$ (4% of clips)", color=ACC, fs=FS_TICK + 0.2)
    fx = x2 - 0.016                                          # fork point
    K.line([(bx + 0.157, 0.69), (fx, 0.69)])
    K.arrow((fx, 0.69), (x2 - 0.002, 0.78), style="angle,angleA=90,angleB=180,rad=0")
    K.arrow((fx, 0.69), (x2 - 0.002, 0.637), style="angle,angleA=-90,angleB=180,rad=0")
    # B3 correlation profile
    x3 = bx + 0.292
    l2, cen, off = D["lines"][(REF, other)]
    m = int(cen[len(cen) // 2]); s = m - 10
    lags, sc = D["prof"](KeypointSequence(S[REF][s:s + 20], FPS), KeypointSequence(S[other][s:s + 20], FPS))
    sc = (sc - sc.mean()) / (sc.std() + 1e-9); prior = l2[m]
    a = fig.add_axes([x3 + 0.012, 0.585, 0.150, 0.235])
    a.axvspan(prior - 3, prior + 3, color=WARM_BAND, lw=0, alpha=0.45)
    a.plot(lags, sc, color=WARM_LINE, lw=LW_DATA + 0.3)
    kk = int(np.argmax(np.where(np.abs(lags - prior) <= 3, sc, -9)))
    a.plot(lags[kk], sc[kk], "o", color=WARM_DOT, ms=3.2, mec="white", mew=0.4, zorder=3)
    a.set_xlim(-8, 8); a.set_xticks([-8, 0, 8]); a.set_yticks([]); style(a)
    a.set_xlabel("lag $k$ (frames)", fontsize=FS_LAB, color=GREY, labelpad=1)
    a.text(prior, 1.02, "$\\pm\\rho$ of prior", transform=a.get_xaxis_transform(), fontsize=FS_TICK, ha="center", va="bottom", color=WARM_LINE)
    K.sub(x3, 0.875, "3  Correlation profile", color=ACC)
    jx = x2 + 0.098                                          # join point
    K.line([(x2 + 0.085, 0.78), (jx, 0.78), (jx, 0.637), (x2 + 0.085, 0.637)])
    K.arrow((jx, 0.705), (x3 + 0.010, 0.705))
    # B4 drift correction (second tier, full width)
    K.bg.plot([bx + 0.010, x3 + 0.095], [0.475, 0.475], color=MOD["B"], lw=0.9, alpha=0.5)
    K.sub(bx + 0.010, 0.440, "4  Drift correction", color=ACC)
    a = fig.add_axes([bx + 0.060, 0.085, 0.250, 0.31])
    tm = cen / FPS
    bins = np.arange(0, tm.max() + 5, 5.0)
    bc = [(x + 2.5, np.median(off[(tm >= x) & (tm < x + 5)])) for x in bins if ((tm >= x) & (tm < x + 5)).sum() >= 3]
    tl = np.arange(n) / FPS
    a.fill_between(tl, l2 - 3, l2 + 3, color=WARM_BAND, lw=0, alpha=0.40)
    a.scatter([p[0] for p in bc], [p[1] for p in bc], s=4, color=WARM_LINE, lw=0, zorder=2, label="window estimates $\\hat\\delta_i$ (5-s medians)")
    a.plot(tl, l2, color=WARM_DOT, lw=LW_DATA + 0.3, zorder=3, label="coarse-to-fine Theil–Sen line")
    a.set_ylim(np.median(l2) - 4, np.median(l2) + 4)
    a.set_xlabel("time (s)", fontsize=FS_LAB, color=GREY, labelpad=1); a.set_ylabel("$\\hat\\delta$ (frames)", fontsize=FS_LAB, color=GREY, labelpad=1)
    a.legend(fontsize=FS_TICK, frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
    style(a)
    slope = (l2[-1] - l2[0]) / (n / FPS / 60)
    K.note(bx + 0.395, 0.225, f"Phone B vs. Phone A\n{slope:+.1f} frames/min\n\nband: pass-2 search\n$\\pm\\rho$ around the\nsupported prior", color=DARK, fs=FS_LAB)
    K.arrow((x3 + 0.140, 0.512), (bx + 0.318, 0.300), rad=-0.35)
    K.note(x3 + 0.118, 0.440, "$\\hat\\delta_i$ per window $t_i$", color=DARK, fs=FS_TICK, ha="right")

    # ---------------- C. global placement
    cx_, cw = 0.698, 0.135
    K.module(cx_, Y0, cw, H, "C", "Rig placement")
    a = fig.add_axes([cx_ + 0.004, 0.36, cw - 0.008, 0.48])
    calib = D["calib"]
    xy2 = {c: np.array(calib[c]["center"])[:2] for c in cams}
    pk = np.array(list(D["peak"].values()))
    for (u, v), wgt in D["peak"].items():
        p, q = xy2[u], xy2[v]
        a.plot([p[0], q[0]], [p[1], q[1]], color=MOD["C"], alpha=0.75, lw=0.3 + 1.1 * (wgt - pk.min()) / (np.ptp(pk) + 1e-9), zorder=1)
    for c in cams:
        a.plot(*xy2[c], "o", ms=6.0, color=DEV[c], mec="white", mew=0.6, zorder=3)
        dyy = -1.2 if xy2[c][1] < -1 else 1.0
        a.text(xy2[c][0], xy2[c][1] + dyy, "ref" if c == REF else f"{D['place'][c]:+.2f}", fontsize=FS_TICK + 0.3,
               ha="center", va="center", color=DEV[c], fontweight="semibold")
    allp = np.array(list(xy2.values()))
    a.set_xlim(allp[:, 0].min() - 1.3, allp[:, 0].max() + 1.3); a.set_ylim(allp[:, 1].min() - 1.9, allp[:, 1].max() + 1.6)
    a.set_aspect("equal"); a.axis("off")
    K.note(cx_ + cw / 2, 0.215, "edges kept if the lag-\nprofile peak beats a null\ncalibrated on the\nrecording ($q{=}0.95$);\noffsets by least squares\nover all cycles", color=DARK)
    K.note(cx_ + cw / 2, 0.085, "offsets to Phone A (frames)", color=GREY, fs=FS_TICK)

    # ---------------- D. output
    dx, dw = 0.858, 0.137
    K.module(dx, Y0, dw, H, "D", "Output")
    a = fig.add_axes([dx + 0.046, 0.585, 0.088, 0.25])
    tt = np.arange(n) / FPS / 60
    for c in cams[1:]:
        d = D["lines"][(REF, c)][0]
        a.plot(tt, d - np.median(d) + D["place"][c], color=DEV[c], lw=LW_DATA)
    a.axhline(0, color=DEV[REF], lw=LW_AUX, ls=(0, (3, 2)))
    a.set_xlabel("time (min)", fontsize=FS_LAB, color=GREY, labelpad=1); a.set_ylabel("$\\hat\\delta_c(t)$", fontsize=FS_LAB, color=GREY, labelpad=1)
    style(a)
    K.sub(dx + 0.008, 0.865, "Timing per camera", color=MOD["D"])
    a3 = fig.add_axes([dx + 0.002, 0.075, dw - 0.004, 0.40], projection="3d")
    X = D["X"]; hip = X[2]
    P = (X - hip) * np.array([1, 1, -1])                 # world z points down: flip for display
    for (i, j) in BONES:
        a3.plot(P[[i, j], 0], P[[i, j], 1], P[[i, j], 2], color="#5f5f5f", lw=0.8)
    a3.scatter(P[:15, 0], P[:15, 1], P[:15, 2], s=1.0, color="#5f5f5f", depthshade=False)
    for c in cams:
        C = (np.array(calib[c]["center"]) - hip) * np.array([1, 1, -1])
        C = C / np.linalg.norm(C) * 1.25
        a3.plot([C[0], 0], [C[1], 0], [C[2], 0], color=DEV[c], lw=0.4, alpha=0.6)
        a3.scatter([C[0]], [C[1]], [C[2]], s=12, marker="s", color=DEV[c], depthshade=False, edgecolors="white", linewidths=0.4)
    a3.view_init(elev=16, azim=-58)
    r = 1.1; a3.set_xlim(-r, r); a3.set_ylim(-r, r); a3.set_zlim(-r, r)
    a3.set_box_aspect((1, 1, 1)); a3.set_axis_off(); a3.patch.set_alpha(0)
    K.sub(dx + 0.008, 0.475, "Synchronized 3D", color=MOD["D"])
    K.note(dx + dw / 2, 0.065, "DLT from 5 views\n(cameras: direction only)", color=GREY, fs=FS_TICK)

    # ---------------- main pipeline arrows (one axis, one style)
    for x0, x1 in ((ax_ + aw, bx), (bx + bw, cx_), (cx_ + cw, dx)):
        K.arrow((x0 + 0.003, mid), (x1 - 0.003, mid), lw=1.6, head=7.0)
    scale_strokes(fig, LINE_SCALE)
    fig.savefig(out, dpi=300)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    draw(compute(), a.out)


if __name__ == "__main__":
    main()
