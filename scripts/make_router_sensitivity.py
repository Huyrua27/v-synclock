# -*- coding: utf-8 -*-
"""Supplementary figure: two-stage router threshold sensitivity on the TotalCapture development split.

Error (sequence mean, frames) and share of clips sent to CSF for every delta of the pre-registered
grid (docs/paper/clips/final/tc_dev_hybrid2r3_L300.json, scripts/tc_hybrid2.py --round 2 on the final
rows), with CC and CSF alone; the selected delta and its single test result are marked.

    python scripts/make_router_sensitivity.py --out docs/paper/mmsys27_v2_supp/figures/fig_s_router.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="docs/paper/clips/final/tc_dev_hybrid2r3_L300.json")
ap.add_argument("--test", default="docs/paper/clips/final/tc_hybrid2r3_L300.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = json.load(open(a.dev))
sw = sorted(((float(k.split("_")[1]), v) for k, v in dev["sweep"].items() if k.startswith("D_")), key=lambda x: x[0])
d = [x for x, _ in sw]
err = [v["seq_mean"] for _, v in sw]
share = [100 * v["csf_share"] for _, v in sw]
cc, csf = dev["CC"]["seq_mean"], dev["CSF"]["seq_mean"]
try:
    test = json.load(open(a.test)).get("result", {}).get("seq_mean")
except FileNotFoundError:
    test = None

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.axhline(cc, color="#1e8e3e", lw=1, ls="--", label=f"CC alone ({cc:.3f})")
ax.axhline(csf, color="#c62828", lw=1, ls="--", label=f"CSF alone ({csf:.3f})")
ax.axhspan(csf - 0.05, csf + 0.05, color="#c62828", alpha=0.08, lw=0)
ax.plot(d, err, "o-", color="#1a4f9c", ms=3, lw=1.2, label="two-stage (dev)")
if test is not None:
    ax.plot([3.0], [test], "*", color="black", ms=7, label=f"selected, test ({test:.3f})")
ax.set_xlabel(r"threshold $\delta$ on quarter disagreement $D$ (frames)")
ax.set_ylabel("error (frames)")
ax2 = ax.twinx()
ax2.bar(d, share, width=0.15, color="#8a8a8a", alpha=0.35)
ax2.set_ylabel("clips sent to CSF (%)", color="#555555")
ax2.set_ylim(0, 100)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
for s in ("top",):
    ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax.legend(fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(a.out)
print("wrote", a.out, "test", test)
