# -*- coding: utf-8 -*-
"""Artifact smoke test: reproduce the capture-2 results of the paper from the data shipped here.

Runs in a few minutes on a CPU and needs no licensed dataset:
  1. scripts/realdata/take2_eval.py  -> 10-s windows of every camera pair, CC / CSF (both inputs) and router
  2. compares every window estimate with the shipped data/take2/rows_L300.json (exact match expected)
  3. recomputes the per-segment summary of the paper's capture-2 table and prints it

    python scripts/smoke_test.py
Exit code 0 = all checks passed.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = ["CC-speed", "CC", "CSF-speed", "CSF", "Router"]
SEGMENTS = [("standing still", 155, 188), ("upper body", 188, 232), ("walking", 232, 263), ("squat, jump", 263, 297),
            ("sideways, turning", 297, 354), ("crawl, lie, kneel", 354, 410), ("fast, frame edge", 410, 453),
            ("two people", 453, 499)]


def main():
    os.chdir(ROOT)
    out = os.path.join(tempfile.gettempdir(), "vsynclock_smoke_rows.json")
    print("[1/3] running take2_eval.py (a few minutes on a CPU) ...", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore", "scripts/realdata/take2_eval.py", "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit("take2_eval.py failed")

    print("[2/3] comparing with the shipped results ...", flush=True)
    new = json.load(open(out, encoding="utf-8"))
    ref = json.load(open("data/take2/rows_L300.json", encoding="utf-8"))
    assert len(new) == len(ref), f"{len(new)} windows vs {len(ref)} shipped"
    worst = 0.0
    for a, b in zip(new, ref):
        assert a["pair"] == b["pair"] and a["s0"] == b["s0"], "window order differs"
        for m in METHODS + ["truth"]:
            worst = max(worst, abs(a[m] - b[m]))
    print(f"      {len(new)} windows, largest difference {worst:.2e} frames")
    # bitwise equality is not expected across platforms and library builds (numpy 1.26 vs 2.x,
    # CPU/CUDA torch); the paper reports two decimals, so 1e-3 frames is a strict bound
    assert worst < 1e-3, "estimates differ from the shipped results"

    print("[3/3] capture-2 table (mean error in frames; windows with a correct coarse offset)")
    rows = [x for x in new if abs(x["truth"]) < 2]
    err = lambda x, m: abs(x[m] + x["truth"])
    print(f"      {'segment':<20}" + "".join(f"{m:>11}" for m in METHODS))
    for name, lo, hi in SEGMENTS:
        s = [x for x in rows if lo <= x["t_ref"] < hi]
        print(f"      {name:<20}" + "".join(f"{np.mean([err(x, m) for x in s]):11.2f}" for m in METHODS))
    print(f"      {'all':<20}" + "".join(f"{np.mean([err(x, m) for x in rows]):11.2f}" for m in METHODS))
    print("smoke test passed")


if __name__ == "__main__":
    main()
