# -*- coding: utf-8 -*-
"""Figures of the v2 paper (signed vertical motion), from the frozen result files.

  fig_failure_modes.pdf  failure rate vs subject-centred view angle (TotalCapture, SportsPose) and
                         AthletePose3D narrow vs wide pairs: speed input vs V-SyncLock input
  fig_roll.pdf           error vs camera roll: speed, image-vertical vy, body-axis vy
  fig_drift3d.pdf        MPJPE vs drift, Panoptic and TotalCapture, V-SyncLock input (speed input dashed)
  fig_online.pdf         streaming V-SyncLock: offset error vs elapsed time

    python scripts/make_v2_figures.py --out docs/paper/mmsys27_v2/figures
"""
import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CL = "docs/paper/clips"
C = {"CSF": "#2a78d6", "CC": "#eb6834", "Caspi": "#1baf7a", "CCc2f": "#4a3aa7"}
INK, MUTED, GRID, NEUTRAL, FAINT = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985", "#c9c8c3"
COL1, COL2 = 3.4, 7.0


def style(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=6.5, width=0.6, length=2.5)
    if grid:
        ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def title(ax, t, fs=7):
    ax.set_title(t, fontsize=fs, color=INK, loc="left", pad=3)


def view_curve(ax, bins, method, **kw):
    keys = [k for k in bins if bins[k][method]["n"]]
    x = [np.mean([float(v) for v in k.split("-")]) for k in keys]
    ax.plot(x, [100 * bins[k][method]["gt1"] for k in keys], **kw)


def fig_failure_modes(out):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.9), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.9]))
    for ax, name, old, new in ((axes[0], "(a) TotalCapture, 10-s clips", f"{CL}/tc_view_L300.json", f"{CL}/final/tc_view_L300.json"),
                               (axes[1], "(b) SportsPose, 3-s trials", f"{CL}/sp_rows_test_summary.json",
                                "data/final_rows/sp_rows_test_summary.json")):
        ax.axvspan(75, 105, color="#f1f0ec", zorder=0)
        bo, bn = json.load(open(old))["bins"], json.load(open(new))["bins"]
        for m in ("CC", "CSF"):
            view_curve(ax, bo, m, color=C[m], ls="--", lw=1.0, marker="o", ms=3, mfc="white", mec=C[m], mew=0.7)
            view_curve(ax, bn, m, color=C[m], ls="-", lw=1.4, marker="o", ms=3.5, mec="white", mew=0.7)
        ax.set_xlim(0, 180); ax.set_ylim(0, 60)
        ax.set_xlabel("angle between the two views (deg)", fontsize=6.5, color=INK)
        style(ax); title(ax, name)
    axes[0].set_ylabel("estimates > 1 frame off (%)", fontsize=6.5, color=INK)
    h = [plt.Line2D([], [], color=C["CC"], lw=1.4, label="CC"), plt.Line2D([], [], color=C["CSF"], lw=1.4, label="CSF"),
         plt.Line2D([], [], color=MUTED, ls="--", lw=1.0, marker="o", ms=3, mfc="white", label="speed input"),
         plt.Line2D([], [], color=MUTED, ls="-", lw=1.4, marker="o", ms=3.5, mec="white", label="speed + signed vertical")]
    axes[0].legend(handles=h, fontsize=5.6, frameon=False, loc="upper right", ncol=2, columnspacing=0.8)
    ax = axes[2]
    groups = collections.defaultdict(list)
    for f, tag in ((f"{CL}/ap3d_rows.json", "speed"), ("data/final_rows/ap3d_rows.json", "final")):
        for r in json.load(open(f, encoding="utf-8")):
            if r["method"] == "CC":
                g = "narrow" if r["pair"] in ("2-3", "1-4") else "wide"
                groups[(tag, g)].append(r["err"] > 1)
    xs = np.arange(2); w = 0.36
    for i, (tag, col, lab) in enumerate((("speed", FAINT, "speed"), ("final", C["CC"], "speed + signed vert."))):
        y = [100 * np.mean(groups[(tag, g)]) for g in ("narrow", "wide")]
        ax.bar(xs + (i - 0.5) * w, y, w, color=col, label=lab)
        for x, v in zip(xs + (i - 0.5) * w, y):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(["pairs <=69 deg", ">=98 deg"], fontsize=6)
    ax.set_ylim(0, 95); style(ax); title(ax, "(c) AthletePose3D, CC")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.2, wspace=0.28)
    fig.savefig(os.path.join(out, "fig_failure_modes.pdf")); plt.close(fig)


def fig_roll(out):
    fig, axes = plt.subplots(1, 3, figsize=(COL1, 1.55), sharex=True)
    for ax, ds, name in zip(axes, ("tc", "ap3d", "sp"), ("TotalCapture", "AthletePose3D", "SportsPose")):
        R = json.load(open(f"{CL}/vy/roll_{ds}.json"))
        ang = R["angles"]
        for k, col, ls, lab in (("speed", FAINT, "-", "speed"), ("speed+vy", C["CC"], "--", "image vertical"),
                                ("speed+vy_body", C["CCc2f"], "-", "body vertical")):
            ax.plot(ang, [R["results"][f"{k}|{a}"]["mean"] for a in ang], color=col, ls=ls, lw=1.3, label=lab)
        ax.set_xticks([0, 45, 90]); ax.set_ylim(bottom=0)
        style(ax); title(ax, name, fs=6.5)
        ax.set_xlabel("roll (deg)", fontsize=6.3, color=INK)
    axes[0].set_ylabel("error (frames)", fontsize=6.3, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.36, wspace=0.38)
    fig.savefig(os.path.join(out, "fig_roll.pdf")); plt.close(fig)


DSTYLE = {"naive": dict(color=NEUTRAL, ls=":", marker="x", label="no correction"),
          "CC-const": dict(color=C["Caspi"], ls="-", marker="^", label="single lag (CC)"),
          "CC-c2f": dict(color=C["CC"], ls="-", marker="s", label="V-SyncLock, CC"),
          "CSF-c2f": dict(color=C["CSF"], ls="-", marker="o", label="V-SyncLock, CSF"),
          "oracle": dict(color=INK, ls="-", marker=None, label="true timing")}


def fig_drift3d(out):
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.95))
    P_new, P_old = json.load(open(f"{CL}/final/tri_clean_cons.json")), json.load(open(f"{CL}/tri_clean_cons.json"))
    T_new, T_old = json.load(open(f"{CL}/final/tc_drift3d.json")), json.load(open(f"{CL}/tc_drift3d.json"))

    def tc_mean(D, d, m):
        seq = [c.rsplit("_", 1)[0] for c in D["clips"]]
        g = collections.defaultdict(list)
        for s, x in zip(seq, D["results"][d][m]["sync"]):
            g[s].append(x)
        return float(np.mean([np.mean(v) for v in g.values()]))
    for ax, name in ((axes[0], "(a) Panoptic, projected"), (axes[1], "(b) TotalCapture, real det.")):
        if ax is axes[0]:
            ds = sorted(P_new, key=float)
            get = lambda D, d, m: float(np.mean(D[d][m]))
            new, old, xs = P_new, P_old, [float(d) for d in ds]
        else:
            ds = sorted(T_new["results"], key=float)
            get = lambda D, d, m: tc_mean(D, d, m)
            new, old, xs = T_new, T_old, [float(d) for d in ds]
        for m, st in DSTYLE.items():
            ax.plot(xs, [get(new, d, m) for d in ds], color=st["color"], ls=st["ls"], lw=1.3, marker=st["marker"],
                    ms=3.4, mec="white", mew=0.6, label=st["label"], zorder=3)
        ax.plot(xs, [get(old, d, "CC-c2f") for d in ds], color=C["CC"], ls="--", lw=0.9, alpha=0.6,
                label="CC-c2f, speed input")
        ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(bottom=0)
        ax.set_xlabel("drift over the clip (frames)", fontsize=6.3, color=INK)
        style(ax); title(ax, name, fs=6.6)
    axes[0].set_ylabel("MPJPE (mm)", fontsize=6.3, color=INK)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.38, wspace=0.3)
    fig.savefig(os.path.join(out, "fig_drift3d.pdf")); plt.close(fig)


def fig_online(out):
    O = json.load(open(f"{CL}/final/tc_online_rig.json"))
    xs = [5, 20, 45, 75]
    fig, ax = plt.subplots(figsize=(COL1, 1.6))
    for m, col, ls, lab in (("uncorrected", NEUTRAL, ":", "no correction"), ("causal_p0", C["Caspi"], "--", "streaming"),
                            ("causal_p2", C["CC"], "-", "streaming + clock prior"), ("offline_p0", INK, "-", "offline")):
        y = [np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan for v in O["results"][m]["bins"]]
        ax.plot(xs, y, color=col, ls=ls, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.6, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(["0-10", "10-30", "30-60", "60+"])
    ax.set_xlabel("elapsed time (s)", fontsize=6.3, color=INK); ax.set_ylabel("offset error (frames)", fontsize=6.3, color=INK)
    ax.set_ylim(0, 2.7); style(ax)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.25)
    fig.savefig(os.path.join(out, "fig_online.pdf")); plt.close(fig)


def fig_cliplen(out):
    old = {L: json.load(open(f"{CL}/tc_summary_L{L}.json"))["activity"] for L in (60, 150, 300)}
    new = {L: json.load(open(f"{CL}/final/tc_summary_L{L}.json"))["activity"] for L in (60, 150, 300)}
    secs = [2, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(COL1, 1.6), sharey=True)
    for ax, act in zip(axes, ("walking", "acting", "freestyle")):
        for m in ("CC", "CSF"):
            ax.plot(secs, [100 * old[L][act][m]["gt1"] for L in (60, 150, 300)], color=C[m], ls="--", lw=0.9,
                    marker="o", ms=2.8, mfc="white", mec=C[m], mew=0.6)
            ax.plot(secs, [100 * new[L][act][m]["gt1"] for L in (60, 150, 300)], color=C[m], ls="-", lw=1.3,
                    marker="o", ms=3.2, mec="white", mew=0.6, label=m)
        ax.set_xticks(secs); ax.set_xlim(1, 11); ax.set_ylim(0, 80)
        style(ax); title(ax, act, fs=6.6)
    axes[0].set_ylabel("> 1 frame off (%)", fontsize=6.2, color=INK)
    axes[1].set_xlabel("clip length (s)", fontsize=6.2, color=INK)
    h = [plt.Line2D([], [], color=C["CC"], lw=1.3, label="CC"), plt.Line2D([], [], color=C["CSF"], lw=1.3, label="CSF"),
         plt.Line2D([], [], color=MUTED, ls="--", lw=0.9, marker="o", ms=2.8, mfc="white", label="speed input"),
         plt.Line2D([], [], color=MUTED, ls="-", lw=1.3, marker="o", ms=3.2, mec="white", label="signed input")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.36, wspace=0.15)
    fig.savefig(os.path.join(out, "fig_cliplen.pdf")); plt.close(fig)


BONES = [(0, 1), (0, 3), (3, 4), (4, 5), (0, 9), (9, 10), (10, 11), (0, 2), (2, 6), (6, 7), (7, 8),
         (2, 12), (12, 13), (13, 14)]
LEFT = {(0, 3), (3, 4), (4, 5), (2, 6), (6, 7), (7, 8)}


def _pick(old_f, new_f, key, cond, rank):
    """Deterministic example: among CC samples meeting cond(old_err, new_err, row), the median by rank."""
    old = {tuple(r[k] for k in key): r for r in json.load(open(old_f, encoding="utf-8")) if r["method"] == "CC"}
    cands = []
    for r in json.load(open(new_f, encoding="utf-8")):
        if r["method"] != "CC":
            continue
        k = tuple(r[q] for q in key)
        if k in old and np.isfinite(r["err"]) and np.isfinite(old[k]["err"]) and cond(old[k]["err"], r["err"], r):
            cands.append((rank(old[k]["err"], r["err"], r), k, r))
    cands.sort(key=lambda x: x[0])
    return cands[len(cands) // 2]


def fig_examples(out):
    import sys
    sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "realdata")]
    from vsynclock.model.panoptic_dataset import _fill_gaps
    from realdata_eval import cc_profile
    from vsynclock.core.types import KeypointSequence
    ex = []
    # (1) AthletePose3D, wide pair: speed fails, signed succeeds
    from ap3d_benchmark import decimated, load_trials
    _, k, r = _pick(f"{CL}/ap3d_rows.json", "data/final_rows/ap3d_rows.json", ("subj", "split", "trial", "pair", "j", "k"),
                    lambda o, nw, r: r["pair"] not in ("2-3", "1-4") and o > 1.5 and nw < 0.25, lambda o, nw, r: -o)
    T, _ = load_trials("data/ap3d/keypoints", 0.5)
    cams = T[(r["subj"], r["split"], r["trial"])]
    ca, cb = (int(x) for x in r["pair"].split("-")); S = 4; sh = S * r["j"] + r["k"]
    ex.append(("AthletePose3D, wide pair", decimated(cams[ca][0], S, S, r["n"]), decimated(cams[cb][0], S + sh, S, r["n"]), r["truth"]))
    # (2) TotalCapture walking: speed aliases, signed succeeds
    from tc_benchmark import load_sequences
    seqs = load_sequences("data/tc/tc_keypoints")
    _, k, r = _pick(f"{CL}/tc_rows_L300.json", "data/final_rows/tc_rows_L300.json", ("subj", "seq", "c0", "pair", "j", "k"),
                    lambda o, nw, r: r["seq"].startswith("walking") and o > 1.5 and nw < 0.25, lambda o, nw, r: -o)
    c = seqs[(r["subj"], r["seq"])]; ca, cb = (int(x) for x in r["pair"].split("-")); o0 = r["c0"] + 2 * r["j"] + r["k"]
    ex.append(("TotalCapture, walking", _fill_gaps(c[ca][r["c0"]:r["c0"] + 2 * r["n"]:2])[0],
               _fill_gaps(c[cb][o0:o0 + 2 * r["n"]:2])[0], r["truth"]))
    # (3) SportsPose, near-orthogonal: speed fails, signed succeeds; (4) SportsPose: both fail
    from sp_benchmark import S as SPS, load_subject, stream
    for cond, title in ((lambda o, nw, r: 75 <= r["view"] <= 105 and o > 1.5 and nw < 0.25, "SportsPose, views {:.0f}$^\\circ$ apart"),
                        (lambda o, nw, r: 75 <= r["view"] <= 105 and o > 1.5 and nw > 1.5, "SportsPose, {:.0f}$^\\circ$: both fail")):
        _, k, r = _pick(f"{CL}/sp_rows_test.json", "data/final_rows/sp_rows_test.json", ("subj", "trial", "pair", "j", "k"),
                        cond, lambda o, nw, r: -o)
        _, trials = load_subject("data/sportpose", r["subj"])
        tr = trials[r["trial"]]; u, v = (int(x) for x in r["pair"].split("-")); sh = SPS * r["j"] + r["k"]
        ex.append((title.format(r["view"]), stream(tr["sync_up"][u], SPS, r["n"])[0],
                   stream(tr["sync_up"][v], SPS + sh, r["n"])[0], r["truth"]))
    p_old, p_new = cc_profile(8, "speed"), cc_profile(8, "speed+vyb")
    fig = plt.figure(figsize=(COL2, 2.25))
    for i, (ttl, A, B, truth) in enumerate(ex):
        x0 = 0.01 + i * 0.25
        sp = np.nanmedian(np.linalg.norm(np.diff(A, axis=0), axis=-1), axis=1)
        t = int(np.argmax(np.convolve(sp, np.ones(5) / 5, "same")[3:-3])) + 3
        for q, (X, lab) in enumerate(((A, "view 1"), (B, "view 2"))):
            ax = fig.add_axes([x0 + 0.01 + q * 0.105, 0.50, 0.10, 0.38])
            p = X[t]; p = (p - np.nanmean(p, 0)) / (np.nanmax(np.ptp(p, 0)) + 1e-9)
            for (a_, b_) in BONES:
                ax.plot(p[[a_, b_], 0], -p[[a_, b_], 1], color=C["CSF"] if (a_, b_) in LEFT else C["CC"], lw=1.1,
                        solid_capstyle="round")
            ax.plot(p[1, 0], -p[1, 1], "o", ms=2.4, color=INK)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6)
            ax.text(0.5, -0.02, lab, transform=ax.transAxes, fontsize=5, ha="center", va="top", color=MUTED)
        fig.text(x0 + 0.005, 0.93, f"({'abcd'[i]}) {ttl}", fontsize=6.1, color=INK, va="bottom")
        ax = fig.add_axes([x0 + 0.03, 0.12, 0.195, 0.27])
        sa, sb = KeypointSequence(A, 30.0), KeypointSequence(B, 30.0)
        for pf, col, lab, lw in ((p_old, FAINT, "speed", 1.2), (p_new, C["CSF"], "signed", 1.3)):
            l, s = pf(sa, sb); s = (s - s.mean()) / (s.std() + 1e-9)
            ax.plot(l, s, color=col, lw=lw, label=lab)
            kk = int(np.argmax(s)); ax.plot(l[kk], s[kk], "o", color=col, ms=3, mec="white", mew=0.5)
        ax.axvline(truth, color=INK, lw=0.8, ls="--", label="true lag")
        ax.set_xlim(-8, 8); ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([]); style(ax)
        ax.set_xlabel("lag (frames)", fontsize=5.6, color=INK, labelpad=1)
        if i == 0:
            ax.legend(fontsize=4.8, frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.0)
    fig.savefig(os.path.join(out, "fig_examples.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/mmsys27_v2/figures")
    a = ap.parse_args()
    os.chdir(ROOT)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    for fn in (fig_failure_modes, fig_roll, fig_drift3d, fig_online, fig_cliplen, fig_examples):
        fn(a.out); print("done", fn.__name__)


if __name__ == "__main__":
    main()
