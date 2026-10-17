# TotalCapture development-split protocol (written 2026-10-12, before any dev result was seen)

Purpose: three design choices in the paper were made after looking at the TotalCapture **test**
split (clip length, hybrid routing threshold tau, cycle-consistent least-squares placement).
They are re-selected here on the TotalCapture **development** split with rules fixed in advance,
then the test split is reported once at the dev-selected values.

## Data

- Dev split: subjects S1-S3 x {walking1, acting1, freestyle1} = 9 sequences, 8 cameras, 60 fps.
  These sequences belong to the official training partition and are disjoint from the test
  sequences (walking2, acting3, freestyle3); S1-S3 also appear in test (seen subjects, unseen
  sequences), as in the official protocol.
- Keypoints: same MediaPipe Tasks extraction as the test split (notebooks/totalcapture_colab.ipynb,
  SUBSET_NAME = 'dev'), stored privately in data/tc/tc_keypoints_dev.
- Nothing else changes: same scripts, seeds, offsets (j + k/2), estimators, CSF checkpoint
  (csf_b1_t20.pt, trained on Panoptic only), null q = 0.95, Huber scale 0.5 frame.

## Rules (fixed now)

1. **Clip length.** Candidates: 2 s, 5 s, 10 s (60 / 150 / 300 frames at 30 fps).
   For each estimator (CC, CSF), choose the length with the lowest dev mean error
   (sequence-level mean over all camera pairs and offsets). Ties (within 0.02 frames) go to
   the shorter length (less temporal blur under drift).
2. **Hybrid routing threshold tau.** Sweep tau in {0.20, 0.25, ..., 0.80} on dev 10-s clips
   (or the length selected in rule 1 if it differs). Choose the largest tau (cheapest) whose dev
   mean error is within 0.05 frames of CSF-only on dev. If no tau qualifies, report the hybrid
   as not useful.
3. **Camera placement.** On dev, gated (null q = 0.95) tree vs gated cycle-consistent least
   squares, at the clip length selected in rule 1 (and at 2 s for reference). Adopt least
   squares as V-SyncLock's placement if its dev mean error over placed cameras is lower than the
   tree's; otherwise keep the tree.

## Reporting

- The paper reports dev results for each rule and then the test result at the dev-selected
  value, and removes the "chosen after seeing test" caveat only for choices whose dev selection
  matches what the test-based analysis used. Any disagreement is reported as such.
- Commands: `scripts/tc_benchmark.py --kp-dir data/tc/tc_keypoints_dev` (clip lengths
  60/150/300), `scripts/tc_hybrid.py`, `scripts/tc_camera_graph.py` with the same flags as for test.

## Results (2026-10-12, after running the rules above once)

| Rule | Dev result | Selected | Same as the test-based analysis? |
|---|---|---|---|
| 1. Clip length | CC 1.83 / 1.43 / 1.38, CSF 1.97 / 1.28 / 1.10 frames (2 / 5 / 10 s) | 10 s (both) | yes |
| 2. Hybrid tau | CSF-only 1.112; tau=0.40 -> 1.129 (within 0.05), tau=0.45 -> 1.171 (not) | tau = 0.40 | **no** (test analysis used 0.5 a priori) |
| 3. Placement | 10 s: gated LS 0.80 vs gated tree 1.11 (2 s: 0.72 vs 0.77) | least squares | yes |

Test at the selected tau = 0.40: hybrid - CC = -0.322 [-0.518, -0.150], hybrid - CSF = +0.023
[-0.008, +0.058] frames per sequence, cost 8.0x CC (CSF-only 10.1x).
Files: docs/paper/clips/tc_dev_{rows,summary}_L{60,150,300}.json, tc_dev_hybrid_L300.json,
tc_dev_graph_L{60,300}.json, tc_hybrid_L300_tau040.json.

---

# SportsPose replication plan (written 2026-10-13, before any SportsPose synchronization result)

- Data: indoor subjects S06, S12, S19 (test split of Ingwersen et al. 2025, "Two Views"), 25-26
  trials each, 7 hardware-synchronized cameras, 270 frames at 90 fps (3.0 s). S00 is used only to
  verify the pipeline (rotation, calibration/camera mapping, video_index alignment), not to tune.
- Synchronized ground truth: camera v at pose frame i is video frame video_index[v, i] (21/102
  trials have one camera starting one frame early/late; timestamps confirm). The first and last
  two pose frames are dropped (boundary artefacts).
- Offsets: decimation 90 -> 30 fps, j + k/3 frames, j in {-1, 0, 1}, k in {0, 1, 2}; whole trial
  (~86 frames at 30 fps); all 21 camera pairs; CC, DTW, Caspi-style, CSF (csf_b1_t20.pt, no
  retraining). Unit: trial; CIs by bootstrap over trials, also per subject.
- Viewpoint: subject-centred angle from the SportsPose 3D hips (COCO 11/12) and camera centres
  -R^T T; same bins as TotalCapture.
- Camera graph: per-camera random phases 0..5 source frames; star / tree / least squares (all
  edges) / gated tree / gated least squares. Gate: ground-truth-free null from window pairs of the
  same camera pair taken from different trials of the same subject (a within-trial null is
  impossible at 3 s), q = 0.95, fixed. Huber 0.5, as for TotalCapture.
- Predictions stated in advance (from the paper): (1) on short clips of non-periodic sports
  motion, classical whole-clip methods are at least as accurate as CSF; (2) failure is not
  monotonic in the viewpoint angle; (3) least squares is at least as good as the tree when the
  gated graph has cycles. Any outcome is reported.

---

# Two-stage (post-CC) routing and rig-level edge pruning (written 2026-10-14, before any result)

Motivation (external review): the periodicity router sends 77% of test windows to CSF (8.0x CC
cost). A router that runs CC first and calls CSF only when the CC lag profile is ambiguous may be
cheaper at equal accuracy.

## Two-stage routing (TotalCapture, 10-s clips, same rows/offsets/checkpoint as rule 2)

- Stage 1: CC on every sample. Ambiguity margin m = p1 - p2, where p1 is the maximum of the CC
  lag profile (integer lags -8..8) and p2 the largest local maximum at least 2 lags away from the
  argmax (p2 = -1 if none).
- Stage 2: CSF replaces CC iff margin < mu (variant "M"), or iff margin < mu AND periodicity >
  0.40 (variant "M+P", periodicity and tau frozen from rule 2).
- Grid mu in {0.00, 0.02, ..., 0.40}. Cost per sample = (0.7 + share_CSF * 7.1) / 0.7 x CC
  (CC always runs).
- Selection on dev: among (variant, mu) whose dev mean error is within 0.05 frames of CSF-only on
  dev, choose the lowest cost; ties (cost within 0.1x) go to "M" (simpler), then to the smaller mu.
  If none qualifies, report two-stage routing as not useful.
- Test: report the selected rule once (mean, >1 frame, CSF share, cost), with paired per-sequence
  bootstrap CIs vs CC, CSF and the periodicity hybrid (tau = 0.40).

## Edge pruning for rig-level cost (TotalCapture 10-s clips, gated null q=0.95 + least squares)

- After pass 1, keep for every camera its k strongest edges (union); gate and place on those
  only. k in {2, 3}, both reported on dev and test (no selection); adopted in the paper's cost
  analysis only if, on dev, the mean placement error is within 0.05 frames of using all edges
  and the share of placed cameras drops by at most 5 points.

## Results: two-stage routing (dev, 2026-10-14, run once)

CC 1.378, CSF-only 1.098 (10.1x), periodicity hybrid tau=0.40 1.112 (8.2x). Two-stage: variant M
saturates at 1.194 (mu=0.16-0.20, 23% CSF, 3.3-3.4x); M+P at 1.206 (2.9x). No rule is within
0.05 frames of CSF-only, so by the rule above two-stage routing is **not adopted** and the test
split is not evaluated. Descriptively: an ambiguous CC profile flags about a quarter of samples
and recovers about two thirds of CSF's dev gain over CC at a third of its cost; the remaining
gain comes from samples where CC is confidently wrong (aliased peak with a clear margin).
File: docs/paper/clips/tc_dev_hybrid2_L300.json.

## Slope prior (shrinking the Theil-Sen slope towards zero): not run

Checked against existing results before running anything: at d = 0 (constant 0.5-frame offset)
even the constant-offset correction (CC-const, slope fixed at 0) gives 18.3 mm vs 11.9 mm for no
correction, and its offset error (0.78 frames) exceeds the 0.5-frame offset itself
(docs/paper/clips/tc_drift3d.json). The d = 0 loss is the estimator's level noise, not a spurious
slope, so no shrinkage of the slope can remove it; the experiment is not run.

## Causal (online) mode (TotalCapture test, written 2026-10-14 before any result; no selection)

- Test sequences with cameras 1 (reference), 8, 4, 7; whole sequence decimated to 30 fps; drift
  imposed on 8, 4, 7: delta(t) = 0.5 + r * t, r in {1, 3} frames per minute (Phone B drifts about
  1 frame/min), exact by cubic resampling of the 60-fps keypoints, as in tc_drift_3d.py.
- Offline: CC-c2f (consensus prior, window 20, stride 6, radius 3) on the whole sequence.
- Causal: refit every 5 s on the windows fully observed so far; buffers: growing (all past),
  sliding 30 s, sliding 60 s. The line fitted at time T is used for frames in (T, T + 5 s].
  No output correction before the first fit at T = 10 s (frames 0-10 s are left uncorrected).
- Report: mean |offset error| per time bin (0-10, 10-30, 30-60, 60+ s), and 3D MPJPE against the
  triangulation of the genlocked detections for frames after 10 s. No parameter is selected.
- Amendment (same day, before running): each sequence is capped at its first 120 s, because the
  lag profiles span +-8 frames and 3 frames/min over a 3-min sequence would leave that range (a
  deployed system would re-centre windows on the running estimate; not modelled here).

## Results: edge pruning (2026-10-14, run once; 10-s clips, --clips 4 as for the paper's tables)

| split | all edges (gated LS) | k = 2 | k = 3 |
|---|---|---|---|
| dev  | 0.80, 65% placed | 0.90, 63% | 0.89, 65% |
| test | 0.74, 81% placed | 0.77, 80% | 0.74, 81% |

Dev error rises by 0.09-0.10 frames (> 0.05), so pruning is **not adopted**; the paper reports it
only as a cost/accuracy trade-off. Runtime (scripts/rig_profile.py, 60 s of S2_acting3, one
thread, s per minute of video; docs/paper/clips/rig_profile.json): C = 8 cameras, 28 pairs:
coarse 0.07, pass 1 2.2, null gate 8.4, pass 2 CC-c2f 9.2 (1.9 on 8 processes) or CSF-c2f 120
(35 on 8 processes), placement < 0.01, resampling 0.01, triangulation 0.35; with k = 2 (10 edges)
pass 2 is 3.2 (CC) / 46 (CSF). MediaPipe: 59 s per camera.
Files: docs/paper/clips/tc_{dev_,}graph_L300_prune.json.
- Exploratory addition (after the first run, so not pre-specified): "growing_const", the causal
  growing buffer with a constant offset (median of the pass-2 offsets, no slope), because the
  first run showed slopes of -4 to +2 frames/min from 10-20 s buffers when the truth is +1.

## Results: causal mode (2026-10-14; 15 test sequences, 50-80 s each, cameras 1/8/4/7)

Mean |offset error| (frames) per time bin 0-10 / 10-30 / 30-60 / 60+ s, and MPJPE (mm) vs the
genlocked triangulation after 10 s:

| r (frames/min) | naive | offline | growing | sliding 30 s | sliding 60 s | growing_const (exploratory) |
|---|---|---|---|---|---|---|
| 1 | .58/.83/1.20/1.58, 25.3 | .26/.35/.54/.44, 10.9 | .58/.80/.62/.44, 18.7 | .58/.80/.72/.60, 20.1 | = growing | .58/.51/.71/1.00, 15.3 |
| 3 | .75/1.50/2.60/3.74, 48.8 | .37/.41/.76/.66, 14.6 | .75/.90/.85/.60, 22.0 | .75/.90/.88/.93, 22.8 | = growing | .75/.93/1.57/2.21, 30.1 |

Reading: a causal line needs about 30-60 s of history before its slope is usable (10-30 s: no
better than no correction at 1 frame/min); afterwards it approaches the offline error. A constant
offset is better in the first 30 s, worse later. Offline remains best (10.9 vs 18.7 mm at 1
frame/min). Sliding 60 s = growing because sequences are <= 80 s. File: docs/paper/clips/tc_online.json.

## Two-stage routing, round 2 (written 2026-10-14 after a dev-only exploration; test untouched)

Dev exploration (scratch analysis, dev rows only): the CC margin misses samples where CC is
confidently wrong. Temporal consistency does better: run CC also on the four quarters of the clip
and take D = median over quarters of |quarter peak lag - whole-clip peak lag| (frames, parabolic
refinement). Dev AUC for "CSF better by > 0.5 frame": D 0.78, half-split 0.74, margin 0.69,
periodicity 0.58.

- Rule: stage 1 = CC on the clip and its four quarters; stage 2: CSF replaces CC iff D >= delta.
- Grid delta in {0.25, 0.50, ..., 3.00}. Selection on dev with the round-1 criterion: lowest cost
  (= smallest CSF share, i.e. largest delta) whose dev mean error is within 0.05 frames of
  CSF-only. If none qualifies, report as not useful.
- Cost, measured on 10-s clips, one thread: CC 1.19 ms, four quarters 3.60 ms, CSF 45.4 ms
  (CSF-only = 38x CC on whole clips). Reported both in this clip model and in the paper's
  per-window model (CC 0.7, quarters 3x CC, CSF 7.1 ms).
- Test: the selected delta once, with per-sequence bootstrap CIs vs CC, CSF and the periodicity
  hybrid (tau = 0.40), and per activity.

## Results: two-stage routing, round 2 (2026-10-14)

Dev: every delta in the grid qualifies (1.060-1.104 vs CSF-only 1.098); delta = 1.25-1.75 is even
below CSF-only (1.060). Selected by the rule: delta = 3.00 (19% CSF; note: the edge of the grid).
Test, run once at delta = 3.00 (15 sequences, 8136 samples):

| | mean (frames) | >1 frame | CSF share | cost, window model | cost, clip model |
|---|---|---|---|---|---|
| CC | 1.320 | 32% | 0% | 1.0x | 1.0x |
| CSF | 0.974 | 28% | 100% | 10.1x | 38x |
| periodicity hybrid (tau 0.40) | 0.998 | 29% | 77% | 8.0x | ~29x |
| two-stage D >= 3.0 | 1.034 | 30% | 19% | 5.9x | 11.1x |

Paired per-sequence differences: two-stage - CC = -0.285 [-0.427, -0.163]; - CSF = +0.060
[-0.003, +0.125]; - periodicity hybrid = +0.037 [-0.029, +0.104]. By activity (two-stage / CC /
CSF, CSF share): walking 1.86 / 2.45 / 1.66 (36%), acting 0.62 / 0.80 / 0.69 (11%), freestyle
0.64 / 0.72 / 0.60 (8%). Reading: consistent with the periodicity hybrid at a fraction of the CSF
calls; on test it is 0.06 frames behind CSF-only (CI touches zero), slightly outside the dev
tolerance. Files: docs/paper/clips/tc_{dev_,}hybrid2r2_L300.json.

# Drift-slope prior (written 2026-10-14, before any result; supersedes "not run" above)

- Estimator: after the coarse-to-fine pass-2 offsets (centres c_i, offsets o_i) and the Theil-Sen
  line (slope s), shrink the slope towards zero with a Gaussian prior N(0, sigma_p^2):
  s' = s * sigma_p^2 / (sigma_p^2 + se^2), se = 1.4826 MAD(o_i - s c_i - b) / sqrt(sum (c_i - mean c)^2)
  * sqrt(W / stride) (overlapping windows); intercept refit as median(o_i - s' c_i).
- sigma_p = 2 frames per minute of 30-fps video, fixed a priori from the physical scale of
  consumer clock drift (Phone B: 0.5-1.1 frames/min after timestamp resampling); not tuned.
- Evaluations (no selection; all reported):
  (a) TotalCapture drift + 3D (tc_drift_3d.py, test, 10-s clips, d = 0..3 frames per 10 s, i.e.
      0-18 frames/min, far above the prior's scale): CC-c2f+prior, CSF-c2f+prior vs. the unshrunk
      versions. Expected: helps at d = 0, hurts at d >= 1 (the imposed drift is unphysical).
  (b) Causal mode (tc_online.py, r = 1 and 3 frames/min): growing buffer + prior.

## Results: drift-slope prior (2026-10-14, run once; sigma_p = 2 frames/min)

The earlier "not run" argument was wrong: CC-const is a different (whole-clip, noisier) estimator,
and most of the coarse-to-fine loss at d = 0 is slope noise.
(a) TotalCapture drift + 3D, 48 test clips, MPJPE (mm) to the genlocked triangulation:

| d (frames/10 s) | naive | CC-c2f | CC-c2f+prior | CSF-c2f | CSF-c2f+prior |
|---|---|---|---|---|---|
| 0 | 11.9 | 16.5 | 9.5 | 17.3 | 11.8 |
| 1 | 23.0 | 17.4 | 11.8 | 17.6 | 13.0 |
| 2 | 33.8 | 19.2 | 17.3 | 17.9 | 16.4 |
| 3 | 44.3 | 18.3 | 20.8 | 17.6 | 19.8 |

Paired (clips): CC prior - c2f = -7.0 [-10.0,-4.3], -5.6 [-8.5,-3.2], -1.9 [-4.2,+0.3],
+2.5 [+0.6,+4.4]; CC prior - naive at d=0: -2.4 [-4.0,-0.5]. As expected the prior hurts only at
18 frames/min, an order of magnitude above consumer clocks.
(b) Causal: growing + prior gives the best causal result (r=1: 0.47/0.58/0.42 frames at
10-30/30-60/60+ s, 13.4 mm vs 18.7 growing, 15.3 constant; r=3: 20.7 vs 22.0 mm).
Files: docs/paper/clips/tc_drift3d_prior.json, tc_online_prior.json.

# Signed motion features (2026-10-14)

Question (external review): do signed left-right velocity features break the gait aliasing?
Probe (scripts/tc_feature_probe.py; feature sets defined in the script before running): CC on the
TotalCapture **dev** rows (10-s clips, all pairs, offsets j + k/2), each feature channel z-scored
over the clip. Mean error (frames, per-sequence) / share > 1 frame:

| features | all | walking | acting | freestyle |
|---|---|---|---|---|
| speed (current) | 1.437 (36%) | 2.212 (60%) | 1.549 (35%) | 0.549 (10%) |
| vy (signed vertical image velocity) | 0.637 (15%) | 0.692 (19%) | 0.794 (14%) | 0.426 (11%) |
| speed + vy | **0.586 (13%)** | 0.668 (19%) | 0.711 (11%) | 0.378 (8%) |
| speed + lr_speed | 1.114 (34%) | 1.567 (53%) | 1.184 (33%) | 0.591 (13%) |
| speed + lr_vy | 1.280 (34%) | 1.909 (54%) | 1.293 (32%) | 0.639 (16%) |
| speed + vy + lr_vy | 0.640 (15%) | 0.701 (20%) | 0.750 (14%) | 0.469 (12%) |

Left-right differences help little (left and right limbs reach peak speed together); the signed
vertical velocity does: it is negative on the way up and positive on the way down, so the two
strokes of a limb no longer look alike. Assumption: cameras roughly upright (image y ~ gravity),
true for every dataset here; phones are rotated by their orientation metadata.

Selection (rule: lowest dev mean error among the probed sets): **speed + vy**, fixed from here on.
Next, each evaluated once with no further selection: CC(speed+vy) on the TotalCapture test split;
on AthletePose3D, SportsPose and the Panoptic drift/3D benchmarks; and a CSF retrained on
Panoptic with speed + vy input (same recipe as csf_b1_t20: --motion --clip-len 20 --select frmerr
--epochs 15 --lr 1e-4, seeds 1-4), compared with CSF(speed).

Test (TotalCapture, 15 sequences, 8136 samples, run once with speed + vy fixed): CC(speed) 1.413
(33% > 1), CC(speed + vy) 0.539 (9%), vy alone 0.585; walking 2.444 -> 0.685, acting 1.099 -> 0.559,
freestyle 0.696 -> 0.374. Paired per sequence: CC(speed+vy) - CC = -0.780 [-1.167, -0.417];
CC(speed+vy) - CSF(speed) = -0.435 [-0.695, -0.193]; better on 13/15 sequences in both.
Files: docs/paper/clips/tc_{dev,test}_feature_probe.json.
- SportsPose amendment (after its first CC-vy run, 3.52 frames): SportsPose stores keypoints in
  sensor axes and several cameras are mounted rotated (calib.pkl numtimesrot90clockwise), so that
  run used a horizontal axis as "vertical" on those cameras. The protocol's definition is the
  display-upright vertical (as phones give by orientation metadata); the rerun applies it
  (sp_benchmark.upright). Reported as such; the first run is kept in the record.

## Results: speed + vy on all real-image benchmarks (2026-10-14, each run once)

| benchmark (unit) | CC | DTW | CSF | **CC-vy** | **CSF-vy** |
|---|---|---|---|---|---|
| TotalCapture 10 s, test (15 seq) | 1.320 (32%) | - | 0.974 (28%) | 0.539 (9%) | **0.385 (7%)** |
| TotalCapture 10 s, dev (9 seq) | 1.378 | - | 1.098 | 0.586 | **0.465** |
| AthletePose3D, pairs <= 69 deg (trials) | 0.35 | 0.28 | 0.42 | **0.10** (0% > 1) | - |
| AthletePose3D, pairs >= 98 deg | 4.69 (82%) | 1.43 (36%) | 4.94 (86%) | **0.22 (0%)** | - |
| SportsPose (76 trials) | 1.001 | 0.922 | 1.088 | **0.648** | - |

Paired: TC test CSF-vy - CC-vy = -0.155 [-0.303, -0.043], CSF-vy - CSF = -0.590 [-0.801, -0.407],
CC-vy - CSF = -0.435 [-0.693, -0.185]; SportsPose CC-vy - DTW = -0.274 [-0.377, -0.166].
TotalCapture by activity (test): CSF-vy walking 0.59 / acting 0.26 / freestyle 0.30; CC-vy 0.68 /
0.56 / 0.39. SportsPose CC-vy by activity: jump 0.13, soccer 0.62, tennis 1.14, baseball 0.67,
volley 0.69. CSF-vy = csf_b1_t20_vy.pt (seed 1; seeds 2-4 trained, not yet evaluated).
Reading: vertical motion is seen by every horizontal view, whereas image speed depends on the
viewing direction; this removes most of the wide-baseline (near-orthogonal) failures.
- Note: the probe (tc_feature_probe.py) refines the CC peak within +-0.5 lag; the benchmark CC
  (CrossCorrelation, used for every CC number in the paper) within +-1. Official numbers come from
  the benchmark (run_vy_suite.py): TC 10-s test CC-vy 0.441 (8% > 1), CSF-vy 0.385 (7%); dev 0.526 / 0.465.

## Results: real-image suite with speed + vy (2026-10-15; scripts/run_vy_suite.py, docs/paper/clips/vy/)

Same protocols, seeds and clips as the speed-only results; CC = CrossCorrelation(speed+vy),
CSF = csf_b1_t20_vy.pt. Old (speed) -> new (speed + vy):

- TC clip length, dev CC 1.83/1.43/1.38 -> 0.80/0.62/0.53, CSF 1.97/1.28/1.10 -> 0.89/0.53/0.47
  (2/5/10 s): rule 1 still selects 10 s. Test 10 s: CC 1.32 -> 0.44, CSF 0.97 -> 0.385.
- TC subject-centred view angle: failure 4-11% in every bin (no U shape); AUC angle 0.44-0.52,
  peak 0.73 (CC) / 0.68 (CSF).
- TC null gate q0.95 (10 s): precision 92 -> 96%, kept-error 0.45 -> 0.32 (CC); walking: 85% of
  estimates correct but only 16% kept (shared rhythm raises the null).
- TC camera graph 10 s test: star 0.37, tree 0.73, LS all 0.35, gated tree 0.48, gated LS 0.31
  (81% placed); dev: 0.52 / 1.39 / 0.51 / 0.82 / 0.46 (71%); 2 s test: LS all 0.50, gated LS 0.50.
- TC drift + 3D (MPJPE mm, d = 0/1/2/3 frames per 10 s): naive 11.9/23.0/33.8/44.3; CC-const
  7.2/8.5/12.8/17.4; CC-c2f 8.0/8.8/9.5/9.4; CSF-c2f 8.8/7.9/8.3/8.5; with the slope prior
  CC 7.6/9.7/13.0/16.6, CSF 7.3/7.6/11.2/15.4; drift error of c2f 0.31-0.44 frames (was 0.64-0.81).
  With vy features the slope prior is no longer needed (it helped only noisy estimates).
- AthletePose3D (233 trials): CC 0.182 (0% > 1), CSF 0.344 (2%).
- SportsPose (76 trials): CC 0.648, CSF 0.612 (paired -0.035 [-0.077, +0.005]); view angle still
  peaks at 75-90 deg (CC 30% vs 8% at the extremes), weaker than before; AUC peak 0.76.
  Graph: star 0.76, tree 0.79, LS all 0.59, gated tree 0.54, gated LS 0.50 (62% placed)
  (was 1.32 / 1.18 / 0.92 / 0.87 / 0.81).

## Results: Panoptic controlled benchmark and take 1 with speed + vy (2026-10-15)

Same commands, sequences and clips as the paper's speed-only results (run_vy_suite.py pan_*,
take1_*); CSF = csf_b1_t20_vy.pt, noise-augmented CSF = csf_b1_t20_vy_noisy.pt (t20 recipe +
--noise-px 3 --outlier-p 0.01, seed 1). Old (speed) -> new (speed + vy):

- Fixed offset, Panoptic test: CSF 0.126 -> 0.076, CC 0.21 -> 0.180, DTW 0.153.
- Fixed offset under noise (validation): with the CURRENT code the speed models give CSF 0.411,
  CC 0.620, DTW 0.686 -- the paper's supplementary table (0.399 / 0.485 / 0.479) is NOT
  reproducible with the current scripts (likely run before the noise model changed); vy: CSF
  0.332, CC 0.423 (DTW unchanged 0.686). The supplementary table must be replaced.
- Drift, clean (8 clips, frames, d = 1/2/3): CC-c2f 0.053/0.052/0.069, CSF-c2f 0.045/0.035/0.049
  (was 0.10-0.13 / 0.12-0.13); CC-const 0.26/0.53/0.80.
- Drift, noise (12 clips, d = 1/2/3): CC-c2f 0.22/0.25/0.56, CSF-c2f 0.31/0.46/0.25.
- 3D, clean (11 clips, mm, d = 0..3): CC-c2f 1.07/0.83/1.03/0.97/0.90, CSF-c2f 0.86/0.76/0.84/0.86/0.89,
  oracle 0.20-0.37, naive 3.65-13.20 (was CC-c2f 2.88, CSF-c2f 1.51 at d = 3).
- 3D, test sequences only (4 clips, d = 3): CC-c2f 0.25, CSF-c2f 0.32, oracle 0.24.
- 3D under noise (d = 3): CC-c2f 9.03, CSF-c2f 9.07, oracle 8.62, naive 14.13.
- Take 1 semi-real (10 clips, d = 0/1/2/3): CC-const 0.19/0.28/0.52/0.79, CC-c2f 0.36/0.43/0.44/0.43,
  CSF-c2f 0.32/0.27/0.33/0.33 (was CC-c2f 0.72, CSF-c2f 0.46 at d = 3).
- Take 1, five cameras, reprojection fast frames: 9.46-9.52 px for every sub-frame method (still
  not rankable); Phone B drift +0.66 to +0.87 frames/min.

## Results: streaming V-SyncLock (vsynclock/online.py, scripts/tc_online_rig.py, 2026-10-15)

TotalCapture test, all 8 cameras, 15 sequences (<= 120 s), per-camera drift 0.5-scale offsets in
[-1.5, 1.5] and rates in [-3, 3] frames/min, 1-s chunks, refit every 5 s. Mean |offset error| of
placed cameras at 10-30 / 30-60 / 60+ s: uncorrected 0.97/1.40/2.54; offline 0.20/0.25/0.31 (81%
placed); causal 0.62/0.45/0.30 (70%); causal + slope prior (2 frames/min) 0.30/0.29/0.29 (71%).
Refit of all 28 pairs: 1.4 s mean, 1.8 s p95 per 5 s of video on one CPU thread (real time).
MPJPE is not comparable across modes yet (different placed-camera sets); to be fixed.

# Signal-formulation baselines (written 2026-10-15, before any result; no selection)

Prior tools already use vertical keypoint motion (OpenCap utilsSync.py: mean signed vertical
velocity of the two ankles, integer lag; Pose2Sim synchronization.py: summed |vertical speed| of the
chosen keypoints after a Butterworth low-pass, integer lag; Disney US12574586B2: per-keypoint vertical
velocity). Six formulations are compared on identical samples, with the same +-8-lag
cross-correlation, each with integer-lag (argmax, as the tools) and parabolic sub-frame readout:
 1 pose2sim  : sum_j |vy_j| over all joints, 4th-order zero-phase Butterworth 6 Hz, one channel
 2 opencap   : mean of signed vy over the two ankles, one channel
 3 abs_vy    : per-joint |vy_j|
 4 speed     : per-joint |dx_j/dt| (previous V-SyncLock input)
 5 vy        : per-joint signed vy_j
 6 speed+vy  : per-joint speed and signed vy (current V-SyncLock input)
Every channel is z-scored over the clip. Data: TotalCapture 10-s test rows (15 sequences),
AthletePose3D (233 trials, whole clip), SportsPose (76 trials, display-upright axes). Unit:
sequence / trial; paired bootstrap CIs of 6 (parabolic) minus each other variant. Hypotheses stated
in advance: unsigned variants (1, 3, 4) alias on rhythmic motion; averaging left and right ankles (2)
partly cancels in gait. All outcomes are reported.

## Results: signal-formulation baselines (2026-10-15; scripts/signal_baselines.py)

First run had a bug (joints missing for a whole clip stayed NaN in the probe features, unlike
motion_features); fixed and rerun; after the fix the speed and speed+vy numbers equal the
benchmark's exactly. Mean error (frames, per sequence/trial), parabolic readout (integer in brackets):

| variant | TotalCapture 10 s | AthletePose3D | SportsPose |
|---|---|---|---|
| Pose2Sim-style sum|vy| filtered | 3.15 (3.17) | 2.89 (2.95) | 1.92 (1.96) |
| OpenCap-style mean signed vy of ankles | 2.29 (2.31) | 1.50 (1.52) | 3.03 (3.05) |
| per-joint |vy| | 2.57 (2.60) | 3.34 (3.41) | 1.59 (1.64) |
| per-joint speed | 1.32 (1.35) | 3.24 (3.27) | 1.00 (1.06) |
| per-joint signed vy | 0.49 (0.55) | **0.13** (0.27) | 1.03 (1.08) |
| per-joint speed + signed vy | **0.44** (0.50) | 0.18 (0.30) | **0.65** (0.71) |

Paired, speed+vy (parabolic) minus: Pose2Sim-style -2.71 / -2.71 / -1.27; OpenCap-style -1.85 /
-1.32 / -2.38; |vy| -2.12 / -3.16 / -0.94; speed -0.88 / -3.06 / -0.35; vy -0.05 [-0.10, 0.00] /
+0.05 [+0.04, +0.06] / -0.39; integer readout of speed+vy -0.06 / -0.12 / -0.07 (all CIs exclude 0
unless shown). Both hypotheses hold: unsigned formulations (sum|vy|, |vy|, speed) alias or fail
across views; averaging the ankles fails in gait (TC walking 2.10) and on SportsPose. Signed vy
alone is best on AthletePose3D by 0.05; speed + vy is the only formulation good on all three.
Caveat: the tools are re-implemented as signals under our correlator, not run end-to-end (no
Pose2Sim max-speed window, no OpenCap reprojection refinement or activity-specific modes).
Files: docs/paper/clips/vy/signal_baselines_{tc,ap3d,sp}.json.

# Camera-roll robustness (written 2026-10-15, before any result)

The vy input assumes roughly upright cameras. Test: rotate the 2D keypoints of camera B of every
sample by a roll angle theta in {0, 5, 10, 15, 20, 30, 45, 60, 90} degrees (camera A unchanged) on
the TotalCapture 10-s test rows, AthletePose3D and SportsPose samples of signal_baselines.py.
Variants (CC, parabolic readout): speed (roll-invariant), speed + image vy (current), and
speed + body vy, where "vertical" is the per-camera, per-clip body axis: the median direction from
mid-hip (joint 2) to neck (joint 0), velocities projected on it (sign: downwards positive, as image y).
Adoption rule: body vy replaces image vy as the default if, at theta = 0, its mean error is within
0.05 frames of image vy on all three datasets; otherwise image vy stays and body vy is reported as
the option for rolled cameras.

## Results: camera roll (2026-10-15; scripts/roll_robustness.py, docs/paper/clips/vy/roll_*.json)

Mean error (frames) at roll 0 / 20 / 45 / 90 deg of camera B:
- TotalCapture: speed 1.32 (invariant); image vy 0.441 / 0.498 / 0.702 / 1.770; body vy 0.483 / 0.487 / 0.485 / 0.483.
- AthletePose3D: speed 3.24; image vy 0.182 / 0.197 / 0.288 / 3.129; body vy 0.184 at every angle.
- SportsPose: speed 1.00; image vy 0.648 / 0.626 / 0.809 / 1.764; body vy 0.605 at every angle.
Image vy tolerates about 20 deg of roll; body vy is roll-invariant. Rule: at theta = 0 body vy is
within 0.05 of image vy on all three (+0.042, +0.002, -0.043) -> **body vy is adopted as the
default**. Implementation: de-roll each camera by its median hip->neck axis before computing the
features (identical to body vy for CC; lets the CSF input stay image-vertical). Assumes a mostly
upright subject (median over the clip/recording).

# Two-stage routing, round 3: on the final input (written 2026-10-15, before any result)

Same rule as round 2 (temporal consistency D of the CC peak over the clip's four quarters; CSF iff
D >= delta; grid delta in {0.25, ..., 3.00}; select the cheapest delta within 0.05 frames of CSF
alone on dev; test once), now with CC = speed + body vy and CSF = csf_b1_t20_vy.pt (de-rolled), on
the rows of run_vy_suite.py --tag final (data/final_rows/tc_{dev_,}rows_L300.json). The periodicity
router (tau = 0.40) and CC / CSF alone are reported as references on the same rows. If no delta
qualifies, the router is dropped from the paper body.

## Results: two-stage routing, round 3 (final input, 2026-10-15)

Dev (sequence means): CC 0.507, CSF 0.462, periodicity router 0.510 (79% CSF); every delta
qualifies, delta = 0.75-1.25 is below both estimators alone (0.433-0.436 at 14-29% CSF calls);
selected by the rule: delta = 3.00 (4% CSF calls), the edge of the grid.
Test, once: two-stage 0.432 (8% > 1) with 4% of the clips sent to CSF = CSF alone 0.432 (100%);
CC 0.483; periodicity router 0.445 (77%). Paired: two-stage - CC = -0.052 [-0.102, -0.012];
- CSF = -0.000 [-0.036, +0.034]; - periodicity router = -0.014 [-0.044, +0.013]. Cost on 10-s
clips: 5.5x CC (CSF alone 38x). By activity (two-stage / CC / CSF, CSF share): walking 0.57 /
0.67 / 0.59 (5%), acting 0.26 / 0.26 / 0.28 (1%), freestyle 0.46 / 0.55 / 0.42 (5%).
Reading: with the final input, CC's temporal self-inconsistency singles out the few clips where the
learned estimator matters; the router keeps CSF's accuracy at 1/7 of its cost. It stays in the body.
Files: docs/paper/clips/final/tc_{dev_,}hybrid2r3_L300.json.

## Take 2: five devices with flash ground truth (2026-10-16)

Capture captures/take2: iPhone 1-3 (~580 s, iPhone 2 variable frame rate) and webcams
0-1 (300 s, host clock, 5-18 dropped-frame gaps). Phone torch on a ~2 m tripod blinking at 0.7 Hz for
the whole take; a calibration board first (0-155 s), then nine clap-separated segments: standing
155-188, upper body 188-232, walking 232-263, vertical (squat/jump) 263-297, horizontal/turn 297-354,
non-upright (crawl/lie/kneel) 354-410, fast/edge of frame 410-453, two people 453-499, free 499-531 s
(iphone1 time). Nothing was tuned on this take; every estimator and threshold is the frozen one.

Ground truth (scratchpad t4_gt.py -> data/take2/flash_gt.json): sub-frame mid-level crossings of the
torch on each camera's own clock (iPhone pts, webcam host timestamps), 800 edges per iPhone / 400 per
webcam, matched after a motion-energy coarse alignment (peak 2-4x the runner-up), Huber line per
camera vs iphone1. Offsets +3.385 / +2.836 / +4.851 / +4.834 s; drift -0.02 / 0.00 / -0.056 / -0.051
frames/min (i.e. essentially none); per-edge MAD 3.5-8.7 ms. Independent check: the two webcams share
the host clock; the flash says webcam1 - webcam0 differs from the host start difference by 1.45 ms.
Whole-take per-pair medians of the window estimates differ from the flash by a per-camera constant
of 0.15-0.3 frames (5-10 ms): the torch sits near the top of the image and the body mid-frame, so
rolling-shutter readout bounds the ground-truth accuracy at about this level.

Pairwise, TotalCapture protocol (scripts/realdata/take2_eval.py -> data/take2/rows_L300.json): 10-s
windows, all pairs, one whole-take integer coarse offset per pair. The coarse stage fails on the two
iphone3-webcam pairs (13 and 44 frames off); the other 8 pairs, 236 windows: CC speed 1.62 (36% > 1),
CC final 1.08 (26%), CSF speed 1.39 (33%), CSF final 0.88 (21%), two-stage router 0.90 (24%).
By segment (CC final / CSF final): standing 1.02 / 0.51, upper body 0.50 / 0.41, walking 0.22 / 0.20
(speed input 1.38 / 1.04), vertical 0.21 / 0.20, horizontal 0.38 / 0.67, non-upright 1.06 / 0.50,
fast 0.36 / 0.33, two people 2.21 / 2.55 (MediaPipe tracks one person per view), free 0.50 / 0.42.
Signal formulations (scripts/signal_baselines.py --dataset take2): speed+vy 1.07 vs Pose2Sim-style
1.57 (-0.50 [-0.82, -0.16]), OpenCap-style 3.47, speed 1.54; |vy| and vy alone tie (n.s.).
By pair angle (action segments, without two people): speed input 1.1-1.4 at 73-162 deg, final CC
0.23-0.92, CSF 0.21-0.84 (worst pair: 162 deg).

Rig (scripts/realdata/take2_rig.py -> data/take2/rig.json; star coarse from iphone1 succeeds for all
cameras): mean |offset error| of the four cameras, coarse only 0.274 (lucky near-integer offsets);
CC offline 0.184; CSF offline 0.563 (its window lines invent a -0.2 frames/min drift); prior has no
effect offline. Estimated drift with CC: |error| <= 0.07 frames/min.
3D (calib via static floor board; iphone1 extrinsics from the flash-synchronized waving board,
scripts/realdata/take2_calib_ref.py, centre spread 11 cm): LOO reprojection median 20.2 px for every
timing including the flash ground truth -> calibration-limited, timing not rankable (as in take 1).

Streaming bug found on this take (post-hoc fix, disclosed): with no motion in the first 30 s the first
causal fits produced lines of -600 frames/min, which then centred every later window (lock-in). Fix in
vsynclock/online.py: (i) a pair's line is updated only when the pair passes the null gate, (ii) the slope
is clipped to +-6 frames/min (physical bound; real clocks drift < 0.2). TotalCapture online rig re-run
(data/take2/tc_online_rig_fix.log): offline unchanged (0.23/0.30/0.27); causal 0.68/0.51/0.35 ->
0.39/0.37/0.39; causal + prior 0.28/0.33/0.30 -> 0.30/0.33/0.35. Take 2 streaming after the fix, mean
error after 60 s: CC 0.55-1.00, CSF 0.48-0.72 frames, i.e. far worse than offline (0.18): on real
footage with long low-information stretches the causal mode does not reach offline accuracy.

## Pre-registered: streaming cache fix (2026-10-17, written before any dev result)

Diagnosis (take 2, CC streaming): errors grow to -0.9..-1.35 frames late in the take with a
spurious -0.2..-0.3 frames/min slope, while offline on the same data gives ~0 slope and 0.18 frames.
Cause: each window is correlated once, centred on the line in force at the time, and cached forever;
windows from the first two minutes were centred on lines off by several frames (true lag at/over the
+-8 edge) and keep biasing Theil-Sen. Secondary: pairs of a camera that stopped recording (webcams at
300 s) keep entering placement with stale lines.
Variants (vsynclock/online.py, same code, flags): B0 = current (gate-before-update, slope clip +-6, the
baseline snapshot is data/take2/online_B0.py on 15e4d17 + working tree); fix = B0 + (i) stale-window
re-centring: after a pair's line is accepted, every cached window whose lag centre differs from the
new line by > 3 frames (= rho), or whose peak is at the profile edge with a moved centre, is
recomputed around the new line and REPLACES the old entry; one refit, no further iteration; (ii)
inactive cameras: a camera with no detection for 5 s keeps its last timing for its recorded frames but
its pairs leave fit and placement. Nothing else changes (input, coarse, prior, gate, placement).
Dev (TotalCapture dev split, 9 sequences, first 120 s, seed 0, priors 0 and 2), B0 vs fix:
  D1 unmodified replay, drift +-3; D2 + 30 s near-static lead-in (first pose + 1 px jitter; a
  controlled stress test, cleaner than real detections); D3 two cameras stop after 40 s; D4 drift
  +-0 and D5 drift +-1 frames/min.
Rule: adopt fix iff (a) on D1, D4, D5 its causal error (mean of the 10-30/30-60/60+ bins, each prior)
is not worse than B0 by > 0.02 frames and its placed share not lower by > 2 points, and (b) on D2 or
D3 it lowers the causal error by >= 0.05 frames for at least one prior. Then TotalCapture test (the
frozen streaming protocol) once, and take 2 once. Take 2 was used to find the bug, so it is reported
as the regression it fixes, not as an independent test.

## Result: streaming cache fix, round 1 (dev only; test untouched)
Causal error, mean of 10-30/30-60/60+ bins (prior 0 / prior 2), B0 -> fix: D1 0.383->0.373 /
0.343->0.347; D4 0.387->0.377 / 0.290->0.287; D5 0.383->0.377 / 0.293->0.280 (placed share unchanged)
=> (a) holds. D2 (static lead-in) 0.575->0.575 / 0.585->0.585; D3 (two cameras stop) 0.397->0.447 /
0.343->0.367, offline placed 84%->66% => (b) fails. Fix NOT adopted.
Reading: (1) the static, jittered lead-in never passes the gate, so B0 does not lock in: D2 does not
reproduce the take-2 mechanism (structured but misleading motion during the board segment that passes
the gate with a wrong line). (2) The inactive-camera rule has an implementation flaw: offline, all data
arrive at once, a camera that stopped is inactive at the only update and is never placed; the
intended behaviour is to place it from its own edges after the active cameras, without letting it pull
them. Files: data/streamfix/D{1..5}_{B0,fix}.{json,log}.

## Pre-registered: streaming cache fix, round 2 (2026-10-17, before any round-2 result)
Changes after round 1 (dev only): (i) implementation fix of the inactive-camera rule: cameras still
recording are placed from the edges among themselves; a camera that stopped keeps its last timing,
or, if it has none (offline), is placed from its own edges to the placed cameras without moving them;
(ii) D2 replaced by D2b: a 30-s lead-in in which every camera shows real motion from a different random
time of its own sequence (structured, misleading motion without common timing: the take-2 mechanism).
Same variants B0 / fix, same rule (a)/(b), D3 re-run with the corrected rule, D1/D4/D5 re-run for the
fix only if its code path changed for them (it did not: with all cameras recording the placement is
unchanged), so round-1 D1/D4/D5 stand. If (b) fails again, B0 is kept and the take-2 streaming result
is reported as a limitation with its diagnosis.

## Result: streaming cache fix, round 2 (dev only; test untouched) -> B0 kept
D2b (misleading desynchronized lead-in) reproduces the take-2 failure in B0 (causal 1.55 / 1.38 frames
at 30-60 s), but the fix does not repair it: mean of bins prior 0 / prior 2, B0 1.075 / 1.000 -> fix
1.085 / 1.135, with ~2,200-2,400 re-centred windows per sequence (the line oscillates and drags the
recomputation along). D3 with the corrected inactive rule: 0.397 / 0.343 -> 0.447 / 0.367 (offline
placed 84% -> 74%, from 66% in round 1). Rule (b) fails again: B0 is kept; vsynclock/online.py defaults are
recenter=False, inactive_after=None (both remain as options). Streaming on take 2 is reported as a
limitation with this diagnosis: misleading early motion that passes the gate yields wrong lines whose
windows bias later fits; neither re-centring the cache nor dropping stopped cameras repairs it.
Files: data/streamfix/D2b_*, D3r2_*.

## DTW and Caspi-style with the final input (2026-10-17; reviewer request, no selection involved)
vsynclock.benchmark.baselines DTW / CaspiIrani got a `features` argument (default "speed": earlier numbers
unchanged); the three benchmarks pass --features. Same samples as --tag final (8,136 / 16,776 / 14,364).
Per-unit means (share > 1), minus CC [95% paired CI] (scripts/dtwcaspi_summary.py ->
docs/paper/clips/final/estimators_final_input.json):
  TC 10 s test: CC 0.483 (9%), CSF 0.432 (8%), DTW 0.663 (13%) [+0.180 +0.105,+0.249], Caspi 0.709 (16%) [+0.225]
  AP3D:         CC 0.184 (0%), CSF 0.362 (2%), DTW 0.130 (0%) [-0.054 -0.064,-0.045], Caspi 0.211 (0%) [+0.026]
  SP test:      CC 0.605 (17%), CSF 0.578 (16%), DTW 0.654 (19%) [+0.049 -0.015,+0.111], Caspi 0.687 (19%) [+0.082]
Every estimator gains from the input (DTW speed: AP3D 1.05, SP 0.92). DTW is best on the short AP3D
sprints; CC/CSF elsewhere.

## Body axis vs image axis on development data (2026-10-17; reviewer request)
The decision row "vertical axis" had been evaluated on test samples. On development data
(scripts/body_axis_rule_dev.py -> docs/paper/clips/final/body_axis_rule_dev.txt), body - image:
TC dev CC -0.019 [-0.084,+0.052], CSF -0.003; AP3D dev (S3+S4) CC +0.003, CSF +0.026: the rule
(<= +0.05) holds on dev for both estimators; SportsPose has no development split.

## Pre-registered: streaming Hough-vote acquisition/tracking (2026-10-17, before any result) - LAST streaming attempt
Idea (Hough track-before-detect, Carlson et al. 1994; acquisition/tracking of delay-lock loops): never
pick a per-window peak for the coarse line. Every long (80-frame) window adds its median-subtracted lag
profile, on ABSOLUTE lags, to an accumulator over (offset beta in [-12, 12] frames, step 0.25; slope in
[-6, 6] frames/min, step 0.25): H(beta, s) += phi_i(beta + s t_i) (0 outside the window's range). The
coarse line is argmax H (order-independent; a mis-centred window simply casts no vote near the true
line). Pass 2 is unchanged (search +-rho around that line, Theil-Sen, prior, slope clip).
Acquisition/tracking: a pair is "locked" after its voted line passed the gate and moved by < 1 frame
(at the current time) in 2 consecutive updates; only locked pairs centre NEW windows on their line
(to follow long drift); otherwise windows are centred at 0 (the coarse alignment). A move >= 1 frame
unlocks. Flag vote=True in vsynclock/online.py; nothing else changes. Re-centring of cached windows and the
inactive-camera rule stay off.
Dev: D1, D2b, D3, D4, D5 as before, B0 vs vote. Rule as before: (a) on D1/D4/D5 not worse than B0 by
> 0.02 frames (each prior) and placed share not lower by > 2 points; (b) on D2b or D3 better by >= 0.05
for at least one prior. Pass -> TotalCapture test (frozen streaming protocol) once, take 2 once (take 2
reported as the failure it addresses). Fail -> B0 kept, streaming on real footage stays a limitation.

## Result: streaming Hough-vote (dev only; test untouched) -> B0 kept; streaming attempts closed
Causal error, mean of bins (prior 0 / prior 2), B0 -> vote: D1 0.383/0.343 -> 0.437/0.390; D4
0.387/0.290 -> 0.413/0.293; D5 0.383/0.293 -> 0.430/0.337; D2b 1.075/1.000 -> 1.375/1.255; D3
0.397/0.343 -> 0.373/0.350; placed share unchanged. (a) and (b) fail. Offline, voting is slightly better
than B0 (0.23-0.25 vs 0.24-0.27 frames per bin), but causally the accumulator is dominated by the
early windows and the lock/unlock logic slows drift tracking (60+ s bin 0.26 -> 0.41 on D1). vote=False
stays the default. Per the agreement this was the last streaming attempt: streaming on real footage
remains a limitation, with the diagnosis (misleading early motion that passes the gate; path-dependent
window centring) and three dev-tested remedies that did not repair it (cache re-centring, inactive
cameras, Hough voting). Files: data/streamfix/D*_vote.*
