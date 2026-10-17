# -*- coding: utf-8 -*-
"""Re-run the experiments with a given motion input (--tag vy | final | base) in
small, resumable jobs. Each job writes its own file under docs/paper/clips/vy/ (row dumps under
data/vy_rows/, licensed-data derivatives, git-ignored) and is skipped once that file exists, so
the suite can be stopped and resumed at any time.

    python scripts/run_vy_suite.py --list            # status and rough duration of every job
    python scripts/run_vy_suite.py --next 2          # run the next two pending jobs
    python scripts/run_vy_suite.py --job tc_graph_test
"""
import argparse
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# --tag selects the configuration (set in main() before JOBS is built)
CONFIGS = {
    "vy":    dict(features="speed+vy",  ck="checkpoints/csf_b1_t20_vy.pt", ckn="checkpoints/csf_b1_t20_vy_noisy.pt"),
    "final": dict(features="speed+vyb", ck="checkpoints/csf_b1_t20_vy.pt", ckn="checkpoints/csf_b1_t20_vy_noisy.pt"),
    "base":  dict(features="speed",     ck="checkpoints/csf_b1_t20.pt",    ckn="checkpoints/csf_b1_t20_noisy.pt"),
}
_tag = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--tag"), "vy")
_cfg = CONFIGS[_tag]
OUT = f"docs/paper/clips/{_tag}"
ROWS = f"data/{_tag}_rows"
CK, CKN = _cfg["ck"], _cfg["ckn"]                   # noise-augmented CKN, as csf_b1_t20_noisy.pt
SEQ3 = ["--sequences", "160422_haggling1", "160226_haggling1", "160906_ian5"]
NOISE = ["--noise-px", "3", "--outlier-p", "0.01"]
CONS = ["--prior", "consensus"]
TAKE = ["--take-dir", "captures/take1", "--pose-dir", "data/realdata/s1_t1/pose"]
TC, TCD = "data/tc/tc_keypoints", "data/tc/tc_keypoints_dev"
F = ["--features", _cfg["features"]]
PY = [sys.executable, "-W", "ignore"]


def bench(kp, L, clips, rows):
    return PY + ["scripts/tc_benchmark.py", "--kp-dir", kp, "--clip-len", str(L), "--clips", str(clips),
                 "--methods", "CC", "CSF", "--checkpoint", CK, "--out", rows] + F


# name: (rough minutes, [commands], file whose existence marks the job done)
JOBS = {
    "tc_test_L300": (30, [bench(TC, 300, 4, f"{ROWS}/tc_rows_L300.json"),
                          PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_rows_L300.json", "--out", f"{OUT}/tc_summary_L300.json"]],
                     f"{OUT}/tc_summary_L300.json"),
    "tc_dev_L300": (20, [bench(TCD, 300, 4, f"{ROWS}/tc_dev_rows_L300.json"),
                         PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_dev_rows_L300.json", "--out", f"{OUT}/tc_dev_summary_L300.json"]],
                    f"{OUT}/tc_dev_summary_L300.json"),
    "tc_test_L60": (35, [bench(TC, 60, 8, f"{ROWS}/tc_rows_L60.json"),
                         PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_rows_L60.json", "--out", f"{OUT}/tc_summary_L60.json"]],
                    f"{OUT}/tc_summary_L60.json"),
    "tc_test_L150": (30, [bench(TC, 150, 4, f"{ROWS}/tc_rows_L150.json"),
                          PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_rows_L150.json", "--out", f"{OUT}/tc_summary_L150.json"]],
                     f"{OUT}/tc_summary_L150.json"),
    "tc_dev_L60": (20, [bench(TCD, 60, 8, f"{ROWS}/tc_dev_rows_L60.json"),
                        PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_dev_rows_L60.json", "--out", f"{OUT}/tc_dev_summary_L60.json"]],
                   f"{OUT}/tc_dev_summary_L60.json"),
    "tc_dev_L150": (20, [bench(TCD, 150, 4, f"{ROWS}/tc_dev_rows_L150.json"),
                         PY + ["scripts/tc_analysis.py", "--rows", f"{ROWS}/tc_dev_rows_L150.json", "--out", f"{OUT}/tc_dev_summary_L150.json"]],
                    f"{OUT}/tc_dev_summary_L150.json"),
    "tc_view": (5, [PY + ["scripts/tc_view_angle.py", "--rows", f"{ROWS}/tc_rows_L300.json", "--out", f"{OUT}/tc_view_L300.json"]],
                f"{OUT}/tc_view_L300.json"),
    "tc_null": (30, [PY + ["scripts/tc_null_rows.py", "--rows", f"{ROWS}/tc_rows_L300.json", "--checkpoint", CK,
                           "--out", f"{OUT}/tc_null_L300.json"] + F], f"{OUT}/tc_null_L300.json"),
    "tc_graph_test": (10, [PY + ["scripts/tc_camera_graph.py", "--clip-len", "300", "--clips", "4",
                                 "--out", f"{OUT}/tc_graph_L300.json"] + F], f"{OUT}/tc_graph_L300.json"),
    "tc_graph_dev": (8, [PY + ["scripts/tc_camera_graph.py", "--kp-dir", TCD, "--clip-len", "300", "--clips", "4",
                               "--out", f"{OUT}/tc_dev_graph_L300.json"] + F], f"{OUT}/tc_dev_graph_L300.json"),
    "tc_graph_test_L60": (10, [PY + ["scripts/tc_camera_graph.py", "--clip-len", "60", "--clips", "8",
                                     "--out", f"{OUT}/tc_graph_L60.json"] + F], f"{OUT}/tc_graph_L60.json"),
    "tc_drift3d": (45, [PY + ["scripts/tc_drift_3d.py", "--checkpoint", CK, "--prior", "2", "--no-test",
                              "--out", f"{OUT}/tc_drift3d.json"] + F], f"{OUT}/tc_drift3d.json"),
    "ap3d": (25, [PY + ["scripts/ap3d_benchmark.py", "--methods", "CC", "CSF", "--checkpoint", CK,
                        "--out", f"{ROWS}/ap3d_rows.json"] + F], f"{ROWS}/ap3d_rows.json"),
    "ap3d_graph": (5, [PY + ["scripts/ap3d_camera_graph.py", "--subjects", "S2", "S3", "S4",
                             "--out", f"{OUT}/ap3d_graph.json"] + F], f"{OUT}/ap3d_graph.json"),
    "sp": (25, [PY + ["scripts/sp_benchmark.py", "--subjects", "S06", "S12", "S19", "--methods", "CC", "CSF",
                      "--checkpoint", CK, "--out", f"{ROWS}/sp_rows_test.json"] + F], f"{ROWS}/sp_rows_test.json"),
    "sp_graph": (10, [PY + ["scripts/sp_benchmark.py", "--subjects", "S06", "S12", "S19", "--graph",
                            "--out", f"{OUT}/sp_graph_test.json"] + F], f"{OUT}/sp_graph_test.json"),
    # Panoptic controlled benchmark (same commands as the paper's speed-only results)
    "pan_fixed": (5, [PY + ["scripts/eval_b1.py", "--checkpoint", CK, "--split", "test"] + F], None),
    "pan_fixed_noisy": (5, [PY + ["scripts/eval_b1.py", "--checkpoint", CKN, "--split", "validation"] + NOISE + F], None),
    "pan_drift_clean": (25, [PY + ["scripts/drift_experiment.py", "--checkpoint", CK, "--window", "20", "--drifts", "0,1,2,3",
                                   "--dump-clips", f"{OUT}/drift_clean_cons.json"] + CONS + F], None),
    "pan_drift_noisy": (25, [PY + ["scripts/drift_experiment.py", "--checkpoint", CKN, "--window", "20", "--drifts", "1,2,3",
                                   "--dump-clips", f"{OUT}/drift_noisy_cons.json"] + NOISE + CONS + F], None),
    "pan_tri_clean": (25, [PY + ["scripts/drift_triangulation.py", "--checkpoint", CK, "--window", "20", "--drifts", "0,0.5,1,2,3",
                                 "--clip-len", "160", "--min-speed", "20", "--max-frames", "4000",
                                 "--dump-clips", f"{OUT}/tri_clean_cons.json"] + SEQ3 + CONS + F], None),
    "pan_tri_noisy": (25, [PY + ["scripts/drift_triangulation.py", "--checkpoint", CKN, "--window", "20", "--drifts", "0,1,2,3",
                                 "--clip-len", "160", "--min-speed", "20", "--max-frames", "4000",
                                 "--dump-clips", f"{OUT}/tri_noisy_cons.json"] + SEQ3 + NOISE + CONS + F], None),
    "pan_tri_testonly": (25, [PY + ["scripts/drift_triangulation.py", "--checkpoint", CK, "--window", "20", "--drifts", "0,0.5,1,2,3",
                                    "--clip-len", "160", "--sequences", "160422_haggling1", "160226_haggling1", "--min-speed", "20",
                                    "--max-frames", "12000", "--max-clips", "12",
                                    "--dump-clips", f"{OUT}/tri_testonly_clean_cons.json"] + CONS + F], None),
    # streaming V-SyncLock on the 8-camera TotalCapture rig
    "online_rig": (25, [PY + ["scripts/tc_online_rig.py", "--out", f"{OUT}/tc_online_rig.json"] + F], None),
    # take 1 (our capture)
    "take1_real5": (20, [PY + ["scripts/realdata/realdata_eval.py"] + TAKE + ["--calib", "data/realdata/s1_t1/calib.json",
                               "--ref", "iphone_cam1", "--start-sec", "130", "--checkpoint", CK,
                               "--out", f"{OUT}/take1_eval_cons.json"] + CONS + F], None),
    "take1_semireal": (15, [PY + ["scripts/realdata/semireal_webcam.py"] + TAKE + ["--start-sec", "130", "--checkpoints", CK,
                                  "--dump-clips", f"{OUT}/semireal_cons.json"] + CONS + F], None),
}


def done(name):
    marker = os.path.join(ROOT, OUT, f"{name}.done")
    return os.path.exists(marker) or (JOBS[name][2] is not None and os.path.exists(os.path.join(ROOT, JOBS[name][2])))


def run(name):
    minutes, cmds, _ = JOBS[name]
    os.makedirs(os.path.join(ROOT, OUT), exist_ok=True); os.makedirs(os.path.join(ROOT, ROWS), exist_ok=True)
    log = os.path.join(ROOT, OUT, f"{name}.log")
    print(f"== {name} (~{minutes} min), log: {OUT}/{name}.log", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            if r.returncode:
                print(f"   FAILED ({' '.join(cmd[2:4])}), see the log", flush=True)
                return False
    open(os.path.join(ROOT, OUT, f"{name}.done"), "w").close()
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--next", type=int, default=0)
    ap.add_argument("--job", nargs="*", default=[])
    ap.add_argument("--tag", choices=tuple(CONFIGS), default="vy",
                    help="vy: speed + image vy; final: speed + body vy (default V-SyncLock); base: speed only")
    a = ap.parse_args()
    pending = [n for n in JOBS if not done(n)]
    if a.list or not (a.next or a.job):
        for n, (m, _, _) in JOBS.items():
            print(f"  {'done   ' if done(n) else 'pending'}  {n:18s} ~{m} min")
        print(f"{len(pending)} pending, ~{sum(JOBS[n][0] for n in pending)} min")
        return
    for n in (a.job or pending[:a.next]):
        if done(n):
            print(f"== {n}: already done, skipped"); continue
        if not run(n):
            break


if __name__ == "__main__":
    main()
