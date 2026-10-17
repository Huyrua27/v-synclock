# Estimators on the final input (speed + body-axis vy) on identical samples: CC, CSF (run_vy_suite --tag final)
# and DTW, Caspi-style (same benchmarks, --features speed+vyb). Per-unit means, paired bootstrap vs CC.
import json, collections, numpy as np
SETS = {"TotalCapture 10 s test": ("data/final_rows/tc_rows_L300.json", "data/final_rows/tc_rows_L300_dtwcaspi.json",
                                   lambda r: (r["subj"], r["seq"]), lambda r: (r["subj"], r["seq"], r["c0"], r["pair"], r["j"], r["k"])),
        "AthletePose3D": ("data/final_rows/ap3d_rows.json", "data/final_rows/ap3d_rows_dtwcaspi.json",
                          lambda r: (r["subj"], r["trial"]), lambda r: (r["subj"], r["trial"], r["pair"], r["j"], r["k"])),
        "SportsPose test": ("data/final_rows/sp_rows_test.json", "data/final_rows/sp_rows_test_dtwcaspi.json", None, None)}
out = {}
for name, (fa, fb, unit, key) in SETS.items():
    rows = json.load(open(fa)) + json.load(open(fb))
    if isinstance(rows[0], dict) and unit is None:
        k0 = rows[0].keys()
        uk = "trial" if "trial" in k0 else "seq"
        unit = lambda r, uk=uk: (r.get("subj", r.get("subject")), r[uk])
        key = lambda r: tuple(r.get(k) for k in ("subj", "subject", "trial", "seq", "pair", "u", "v", "s0", "j", "k", "sh") if k in r)
    by = collections.defaultdict(dict)
    for r in rows:
        by[key(r)][r["method"]] = r
    keys = [k for k, v in by.items() if all(m in v for m in ("CC", "CSF", "DTW", "Caspi-style"))]
    res = {}
    for m in ("CC", "CSF", "DTW", "Caspi-style"):
        u = collections.defaultdict(list)
        for k in keys:
            u[unit(by[k][m])].append(by[k][m]["err"])
        res[m] = {x: np.mean(v) for x, v in u.items()}
    U = sorted(res["CC"])
    rng = np.random.default_rng(0)
    out[name] = {"samples": len(keys), "units": len(U)}
    line = f"{name:24s} n={len(keys):6d} units={len(U):3d} |"
    for m in ("CC", "CSF", "DTW", "Caspi-style"):
        e = np.array([res[m][x] for x in U]); gt1 = np.mean([by[k][m]["err"] > 1 for k in keys])
        d = e - np.array([res["CC"][x] for x in U]); b = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        out[name][m] = dict(mean=float(e.mean()), gt1=float(gt1), minus_cc=[float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
        line += f" {m} {e.mean():.3f} ({gt1:.0%})" + ("" if m == "CC" else f" [{d.mean():+.3f} {np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]")
    print(line)
json.dump(out, open("docs/paper/clips/final/estimators_final_input.json", "w"), indent=1)
