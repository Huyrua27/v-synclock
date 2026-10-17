# V-SyncLock

Sub-frame synchronization and drift correction of consumer camera rigs from 2D keypoints.

V-SyncLock synchronizes the cameras of an ad-hoc rig (phones, webcams) after capture, from the
2D keypoints a markerless pipeline already produces. Its input is per-joint image speed together
with the signed velocity of every joint along the body axis; on this signal it estimates
sub-frame offsets and clock drift per camera pair (cross-correlation, or a learned estimator
called only where correlation is temporally inconsistent), gates pairs with a null calibrated
on the recording itself, and places the rig by cycle-consistent least squares, offline or as a
causal streaming stage.

This repository contains the code, the trained models, the result logs and the capture tools
needed to reproduce every table and figure of the paper.

## Layout

```
vsynclock/            package
  benchmark/          signals (cross_correlation.py: speed + body-axis velocity), DTW, Caspi-Irani-style,
                      coarse-to-fine drift fitting (drift.py), observability, metrics
  model/              ContinuSyncFormer (CSF) and its training code
  online.py           streaming synchronizer (gate, placement, clock-rate prior)
  panoptic/           CMU Panoptic loader, projection, triangulation
  realworld/          2D pose extraction (MediaPipe -> COCO-19)
scripts/              one script per experiment; run_vy_suite.py runs the benchmark suite
scripts/realdata/     our two five-device captures (calibration, evaluation, torch ground truth)
notebooks/            Colab notebooks extracting keypoints for AthletePose3D and TotalCapture
tools/                record_webcams.py: multi-webcam recorder with host timestamps
checkpoints/          trained CSF models (speed+vy input; speed input for the baselines)
results/              logs and summaries of the paper's runs (final = paper, base = speed input,
                      vy = image-vertical input); docs/FROZEN_RESULTS.md lists every number
captures/, data/      our captures: 2D keypoints, timestamps, calibration, torch ground truth
docs/                 DEV_PROTOCOL.md (pre-registered selection rules and all outcomes, incl. negative)
```

## Install

Python 3.12 on a CPU is enough for everything except training (pinned versions in
`requirements.txt`).

```bash
pip install -r requirements.txt
python scripts/smoke_test.py      # reproduces the capture-2 results from the shipped data (~5 min)
```

### Docker

```bash
docker build -t vsynclock .
docker run --rm vsynclock                                        # smoke test (default command)
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" \
    vsynclock python scripts/run_vy_suite.py --tag final           # full suite, after placing the datasets in data/
```

## Data

The licenses of AthletePose3D, TotalCapture and SportsPose forbid redistributing derived data,
so the repository regenerates every input from the original downloads:

| Dataset | Obtain | Keypoints | Expected location |
|---|---|---|---|
| CMU Panoptic | `python scripts/download_panoptic_minimal.py --sequence 160422_haggling1 --sequence 160226_haggling1 --sequence 160906_ian5` | 3D ground truth, projected | `data/panoptic_raw/` |
| TotalCapture | request access from the dataset authors | `notebooks/totalcapture_colab.ipynb` | `data/tc/tc_keypoints/` (test), `data/tc/tc_keypoints_dev/` (dev) |
| AthletePose3D | request access from the dataset authors | `notebooks/athletepose3d_colab.ipynb` | `data/ap3d/keypoints/` |
| SportsPose | request access from the dataset authors | `python scripts/sp_extract_pose.py --subjects S06 S12 S19` | `data/sportpose/` |

### Licenses and citation of the datasets

No video, keypoints, 3D poses, calibration or per-sample data of the datasets below are
distributed here; `results/` contains only aggregate quantities reported in the paper. Each
dataset must be obtained from its authors and used under its own terms, and must be cited:

- **TotalCapture** (CVSSP, University of Surrey): research use only, no commercial use, the data
  should not be redistributed. Publications must reference Trumble et al., *Total Capture: 3D
  Human Pose Estimation Fusing Video and Inertial Sensors*, BMVC 2017, and include the repository
  link https://cvssp.org/data/totalcapture/.
- **AthletePose3D**: non-commercial scientific research only; the data may not be reproduced,
  modified and/or made available in any form to any third party without the authors' written
  permission. Cite Yeung et al., *AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation
  and Kinematic Validation in Athletic Movements*, arXiv:2503.07499, 2025.
- **SportsPose**: academic use only, non-assignable and non-transferable. Citing the dataset URL
  instead of the publication is not compliant; cite Ingwersen et al., *SportsPose: A Dynamic 3D
  Sports Pose Dataset*, Proceedings of the IEEE/CVF International Workshop on Computer Vision in
  Sports, 2023.
- **CMU Panoptic**: research purposes only, no commercial use; the dataset or a modified version
  may not be redistributed without permission. Cite Joo et al., *Panoptic Studio: A Massively
  Multiview System for Social Motion Capture*, ICCV 2015.

The CSF checkpoints in `checkpoints/` are models trained on CMU Panoptic only (they contain no
Panoptic data) and are provided for non-commercial research use, in line with the Panoptic terms.

### Our captures

Our own captures are included as 2D keypoints, webcam host timestamps, iPhone frame
timestamps, ChArUco calibration and the torch (optical clock) ground truth
(`data/realdata/`, `data/take2/`, `captures/`). The raw videos show the performers and are not
released for privacy; `scripts/realdata/torch_*.py` document how the ground truth was extracted
from them.

## Reproducing the paper

All runs are deterministic (fixed seeds); confidence intervals use 10,000 paired bootstrap
resamples. Times are for one laptop CPU.

```bash
python scripts/run_vy_suite.py --tag final     # all real-image and Panoptic results (~2 h, resumable)
python scripts/run_vy_suite.py --tag base      # the same with the per-joint speed input (baselines)
```

| Paper | Command | Output |
|---|---|---|
| Table: signal formulations | `scripts/signal_baselines.py --dataset tc` (and `ap3d`, `sp`, `take2`) | `results/vy/signal_baselines_*.json` |
| Fig.: camera roll | `scripts/roll_robustness.py` | `results/vy/roll_*.json` |
| Table: estimators (CC, CSF) | `run_vy_suite.py --tag final` / `--tag base` | `results/final/tc_summary_L300.json`, `ap3d.log`, `sp.log` |
| Table: estimators (DTW, Caspi-style) | `tc_benchmark.py`, `ap3d_benchmark.py`, `sp_benchmark.py` with `--methods DTW Caspi-style --features speed+vyb`, then `scripts/dtwcaspi_summary.py` | `results/final/estimators_final_input.json` |
| Table: two-stage router, Fig. S: threshold sensitivity | `scripts/tc_hybrid2.py --round 2 --features speed+vyb` (dev, then `--select D 3.0` on test); `scripts/make_router_sensitivity.py` | `results/final/tc_*hybrid2r3_L300.json` |
| Tables: placement, gate | `run_vy_suite.py --tag final` (jobs `tc_graph_*`, `sp_graph`, `ap3d_graph`, `tc_null`) | `results/final/*graph*`, `tc_null.log` |
| Table: Panoptic | `run_vy_suite.py` jobs `pan_*` | `results/final/pan_*.log` |
| Tables: drift, TotalCapture 3D | jobs `pan_drift_*`, `tc_drift3d`, `take1_semireal` | `results/final/tc_drift3d.log`, `take1_semireal.log` |
| Table + Fig.: streaming | `scripts/tc_online_rig.py` | `results/final/online_rig.log`, `tc_online_rig.json` |
| Table: rig cost | `scripts/rig_profile.py` | `results/final/rig_profile.json` |
| Capture 1 (five devices) | `scripts/realdata/realdata_eval.py` (job `take1_real5`), `scripts/realdata/semireal_webcam.py` | `results/final/take1_*.log` |
| Capture 2 (torch ground truth) | `scripts/realdata/take2_eval.py`, `scripts/realdata/take2_rig.py`, `scripts/signal_baselines.py --dataset take2` | `data/take2/*.json` |
| Body axis vs image axis on dev | `scripts/body_axis_rule_dev.py` | `results/final/body_axis_rule_dev.txt` |
| Figures | `scripts/make_v2_figures.py`, `scripts/make_overview.py` | PDF figures |

Training the learned estimator (CSF, 0.8 M parameters) on Panoptic:

```bash
python scripts/train_model.py --data panoptic --motion --features speed+vy --clip-len 20 --select frmerr \
    --epochs 15 --lr 1e-4 --out checkpoints/csf_b1_t20_vy.pt
# noise-augmented variant: add --noise-px 3 --outlier-p 0.01 (csf_b1_t20_vy_noisy.pt)
```

## Using V-SyncLock on your own rig

```python
from vsynclock.online import StreamingSynchronizer, cc_profile
sync = StreamingSynchronizer(["cam0", "cam1", "cam2"], ref="cam0", fps=30.0, profile=cc_profile("speed+vyb"))
for c in cams:                      # (T, 19, 2) COCO-19 keypoints per camera, resampled on device timestamps
    sync.push(c, keypoints[c])      # and coarse-aligned to the reference by an integer offset
sync.update()                       # offline: one update on all frames; streaming: maybe_update() per chunk
delta = sync.offset("cam1", frames) # camera frame t shows reference time t + delta(t)
```

`scripts/realdata/take2_rig.py` shows the complete chain on a real capture (timestamps, coarse
alignment, rig placement, triangulation).
