# Frozen results (2026-10-15)

Single source of numbers for the rewrite. Every number below was produced by the current code:
`python scripts/run_vy_suite.py --tag final` (V-SyncLock default input: per-joint speed + signed vertical
velocity along the body axis, "speed+vyb"; CSF = csf_b1_t20_vy.pt with body-axis de-roll,
noise-augmented csf_b1_t20_vy_noisy.pt), `--tag base` (speed input, old checkpoints) and
`--tag vy` (speed + image-vertical vy). Logs and per-clip files: docs/paper/clips/{final,base,vy}/.
Selection history and rules: DEV_PROTOCOL.md. Errors in frames at 30 fps unless stated.

## Reproducibility of the previous paper numbers
- Panoptic drift (clean, noise), Panoptic 3D (clean, noise, test-only) and take-1 semi-real:
  `--tag base` reproduces the stored per-clip files of the paper **exactly**.
- Panoptic fixed offset (test): now CSF 0.106 / CC 0.148 / DTW 0.153 (paper: 0.13 / 0.21 / 0.16);
  fixed offset under noise (validation): CSF 0.411 / CC 0.620 / DTW 0.686 (supplement: 0.399 /
  0.485 / 0.479). Not reproducible with the current code (sampling/noise changed later): replace.
- TotalCapture / AthletePose3D / SportsPose speed-input numbers are reproduced exactly by
  scripts/signal_baselines.py (TC 10 s CC 1.320, AP3D CC 3.241, SP CC 1.001).

## Estimators on real images (final)
| | CC speed (old) | CSF speed (old) | DTW | **CC (final)** | **CSF (final)** |
|---|---|---|---|---|---|
| TotalCapture 10 s test (15 seq) | 1.320 | 0.974 | - | 0.483 (9% >1) | **0.432** (8%) |
| TotalCapture 10 s dev (9 seq) | 1.378 | 1.098 | - | 0.507 | **0.462** |
| TC test 2 / 5 / 10 s | | | | 0.80 / 0.63 / 0.48 | 0.84 / 0.61 / 0.43 |
| TC dev 2 / 5 / 10 s (rule 1 -> 10 s) | | | | 0.79 / 0.62 / 0.51 | 0.89 / 0.54 / 0.46 |
| AthletePose3D (233 trials) | 3.241 | 3.434 | 1.048 | **0.184** (0% >1) | 0.362 (2%) |
| SportsPose (76 trials) | 1.001 | 1.088 | 0.922 | 0.605 | **0.578** (paired -0.027 [-0.071, +0.014]) |
| SportsPose by activity (CC/CSF) | | | | jump .13/.11, soccer .55/.62, tennis 1.06/.89, baseball .63/.58, volley .65/.69 | |

## Two-stage router (final input; TotalCapture 10 s; rule of round 2, re-run as round 3)
CC on the clip and its quarters; CSF only when the quarter peaks disagree (D >= 3 frames, dev-selected,
grid edge). Test: 0.432 with 4% of clips sent to CSF = CSF alone 0.432; CC 0.483 (router - CC
-0.052 [-0.102, -0.012]); periodicity router 0.445 at 77% CSF. Cost 5.5x CC on 10-s clips (CSF 38x).
Dev: delta 0.75-1.25 beats both estimators alone (0.433 vs CC 0.507, CSF 0.462).

## Signal formulations (identical samples; image-vertical variants; scripts/signal_baselines.py)
TC / AP3D / SP, parabolic readout: Pose2Sim-style sum|vy| 3.15 / 2.89 / 1.92; OpenCap-style ankle-mean
signed vy 2.29 / 1.50 / 3.03; per-joint |vy| 2.57 / 3.34 / 1.59; speed 1.32 / 3.24 / 1.00; signed vy
0.49 / 0.13 / 1.03; speed + signed vy 0.44 / 0.18 / 0.65. Integer readout adds 0.06-0.12.

## Camera roll (scripts/roll_robustness.py)
Error at roll 0 / 20 / 45 / 90 deg: image vy TC 0.44/0.50/0.70/1.77, AP3D 0.18/0.20/0.29/3.13,
SP 0.65/0.63/0.81/1.76; body vy TC 0.48, AP3D 0.18, SP 0.61 at every angle (adopted as default).

## Viewpoint (final, CC/CSF)
- TotalCapture, subject-centred angle: failure >1 frame 2-14% in every bin, no U shape; AUC for
  success: angle 0.46-0.48, peak 0.73 (CC) / 0.67 (CSF).
- SportsPose: still peaks at 75-90 deg (CC 28% vs 8% at 0-30 and 150-180), weaker than with speed;
  AUC angle 0.51-0.53, peak 0.76.
- AthletePose3D: 0% failures on narrow and wide pairs (speed: 82% on wide pairs).

## Observability gate (TotalCapture 10 s test, null q = 0.95, CC)
No gate: precision 91%, mean error 0.49; gated: 60% kept, precision 96%, 73% of failures caught,
kept error 0.32. Walking: 85% of estimates correct but only 17% kept (rhythm raises the null).

## Camera graph (mean error of placed cameras; share > 1 frame)
| | star | tree | LS all | gated tree | **gated LS** | placed |
|---|---|---|---|---|---|---|
| TC 10 s test (old speed: 1.06 / 1.41 / 0.82 / 0.91 / 0.74) | 0.37 | 0.59 | 0.34 | 0.43 | **0.32** (4%) | 81% |
| TC 10 s dev | 0.73 | 1.00 | 0.64 | 0.72 | **0.55** | 72% |
| TC 2 s test | 0.73 | 0.77 | 0.51 | 0.59 | **0.50** | 76% |
| SportsPose (old: 1.32 / 1.18 / 0.92 / 0.87 / 0.81) | 0.76 | 0.85 | 0.63 | 0.54 | **0.52** | 61% |
| AthletePose3D S2-S4 (old gated: 39% -> 5% >1) | 0.16 | 0.17 | 0.16 | 0.17 | 0.16 (0%) | 100% |

## Drift + 3D on TotalCapture (48 clips, cameras 1/8/4/7, MPJPE mm to genlocked triangulation)
| d (frames / 10 s) | naive | CC-const | Caspi | CC-c2f | CSF-c2f | CC-c2f+prior | CSF-c2f+prior | oracle |
|---|---|---|---|---|---|---|---|---|
| 0 | 11.9 | 7.6 | 28.1 | 8.4 | 9.0 | 7.3 | 7.3 | 3.3 |
| 1 | 23.0 | 9.2 | 28.4 | 9.0 | 8.2 | 9.5 | 7.9 | 1.9 |
| 2 | 33.8 | 13.5 | 29.4 | 9.8 | 8.8 | 13.1 | 11.7 | 1.8 |
| 3 | 44.3 | 18.0 | 29.9 | 9.7 | 9.0 | 17.0 | 15.8 | 2.0 |
Drift error of c2f 0.33-0.46 (CC) / 0.35-0.40 (CSF) frames (old speed: 0.64-0.81). Old speed 3D at
d = 3: CC-c2f 18.3, CSF-c2f 17.6. With body vy the slope prior is not needed offline.

## Panoptic controlled benchmark (final vs base, same clips)
- Fixed offset, test: CSF 0.080 (base 0.106), CC 0.259 (base 0.148: body vy HURTS CC here; image
  vy 0.180), DTW 0.153. Under noise (validation): CSF 0.345 (0.411), CC 0.521 (0.620), DTW 0.686.
- Drift, clean (8 clips, d = 1/2/3): CC-c2f 0.055/0.050/0.063 (base 0.123/0.132/0.134), CSF-c2f
  0.048/0.039/0.050 (base 0.115/0.128/0.121), CC-const 0.26/0.53/0.80.
- Drift, noise (12 clips, d = 3): CC-c2f 0.40 (0.38), CSF-c2f 0.37 (0.26), CC-const 0.87.
- 3D, clean (11 clips, mm, d = 3): CC-c2f 0.89 (base 2.88), CSF-c2f 0.65 (1.51), oracle 0.36, naive 13.2.
- 3D, test sequences only (4 clips, d = 3): CC-c2f 0.21 (0.31), CSF-c2f 0.27 (0.23), oracle 0.24.
- 3D under noise (d = 3): CC-c2f 9.14 (10.37), CSF-c2f 9.13 (9.30), oracle 8.62, naive 14.13.

## Take 1 (our five-device capture)
- Semi-real drift on the host-synchronized webcam pair (10 clips, d = 0/1/2/3): CC-const
  0.19/0.28/0.52/0.79; CC-c2f 0.34/0.42/0.42/0.47 (base 0.41/0.55/0.63/0.72); CSF-c2f
  0.35/0.27/0.32/0.39 (base 0.60/0.57/0.55/0.46).
- Five cameras, reprojection on fast frames: 9.48-9.54 px for every sub-frame method (coarse only
  10.02): still not rankable. Estimated drift of phone B: +0.59 (CC-c2f) to +0.90 (CSF-c2f) frames/min.

## Streaming V-SyncLock (vsynclock/online.py; 8-camera TC rig, 15 sequences, per-camera drift +-3 frames/min)
Mean offset error of placed cameras at 10-30 / 30-60 / 60+ s: uncorrected 0.97/1.40/2.54; offline
0.23/0.30/0.27 (86% placed); causal 0.39/0.37/0.39 (69%); causal + slope prior 0.30/0.33/0.35 (69%).
Refit of 28 pairs: 2.2 s mean, 3.1 s p95, 3.9 s max per 5 s of video on one CPU thread (real time).
UPDATED 2026-10-16 after the streaming fix found on take 2 (gate-before-update, slope clip +-6 fr/min;
see DEV_PROTOCOL "Take 2"); before the fix: causal 0.68/0.51/0.35, + prior 0.28/0.33/0.30, refit 2.1 s
(docs/paper/clips/final/tc_online_rig_prefix.json).
MPJPE across modes is not comparable yet (different placed-camera sets): to fix before use.

## Cost (scripts/rig_profile.py, speed input; vyb adds the body-axis projection only)
C = 8 cameras, one minute of video, one thread, final input (clips/final/rig_profile.json): CC sync 32.4 s
(shared 18.8 + pass 2 13.6), CSF 129 s, MediaPipe 472 s. (Speed input, earlier: CC-c2f 20 s, CSF-c2f 131 s.)

## Take 2 (five devices, torch ground truth; DEV_PROTOCOL "Take 2"; data/take2/, git-ignored)
- Ground truth: per-edge MAD 3.5-8.7 ms; webcam pair vs host clock 1.45 ms; drift <= 0.06 fr/min;
  rolling-shutter-level per-camera constant 0.15-0.3 fr.
- 10-s windows, 8 pairs with correct coarse, 236 windows: CC speed 1.62 (36% >1), CC final 1.08 (26%),
  CSF speed 1.39 (33%), CSF final 0.88 (21%), router 0.90 (24%). Segments (CC final / CSF final):
  standing 1.02/0.51, upper 0.50/0.41, walking 0.22/0.20 (speed 1.38/1.04), squat-jump 0.21/0.20,
  sideways 0.38/0.67, crawl-lie 1.06/0.50, fast 0.36/0.33, two people 2.21/2.55.
- Signals: speed+vy 1.07, Pose2Sim-style 1.57 (-0.50 [-0.82,-0.16]), OpenCap-style 3.47, speed 1.54.
- Coarse: whole-take pairwise fails 2/10 pairs; star from iphone1 OK. Rig offline CC 0.18 (coarse only
  0.27), drift error <= 0.07 fr/min; CSF offline 0.56 (-0.2 fr/min spurious slope). Streaming after 60 s
  0.48-1.0. 3D LOO median 20.2 px for every timing (calibration-limited).

## Estimators with the final input, incl. DTW / Caspi-style (2026-10-17)
TC 10 s test: CC 0.483, CSF 0.432, DTW 0.663, Caspi 0.709. AP3D: CC 0.184, CSF 0.362, DTW 0.130 (best,
-0.054 [-0.064,-0.045] vs CC), Caspi 0.211. SP: CC 0.605, CSF 0.578, DTW 0.654, Caspi 0.687.
Body axis vs image axis on dev: TC dev CC -0.019, CSF -0.003; AP3D S3+S4 CC +0.003, CSF +0.026.
Streaming cache fix: two dev rounds, not adopted (DEV_PROTOCOL); streaming numbers above unchanged.
