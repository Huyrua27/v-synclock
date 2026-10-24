set -x
K=data/tc/MIST_TC_keypoints_confirm; O=data/refine/confirm; CK=checkpoints/csf_b1_t20_vy.pt; F="--features speed+vyb"
P="python -W ignore"
$P scripts/tc_benchmark.py --kp-dir $K --clip-len 300 --clips 4 --methods CC CSF --checkpoint $CK --out $O/rows_final.json $F > $O/c1_final.log 2>&1
$P scripts/tc_analysis.py --rows $O/rows_final.json --out $O/summary_final.json > $O/c1_summary.log 2>&1
$P scripts/tc_benchmark.py --kp-dir $K --clip-len 300 --clips 4 --methods CC CSF --checkpoint $CK --out $O/rows_imageaxis.json --features speed+vy > $O/c1_imageaxis.log 2>&1
$P scripts/tc_benchmark.py --kp-dir $K --clip-len 300 --clips 4 --methods CC CSF --checkpoint checkpoints/csf_b1_t20.pt --out $O/rows_speed.json --features speed > $O/c1_speed.log 2>&1
$P scripts/tc_benchmark.py --kp-dir $K --clip-len 300 --clips 4 --methods DTW Caspi-style --checkpoint $CK --out $O/rows_dtwcaspi.json $F > $O/c1_dtwcaspi.log 2>&1
$P scripts/tc_hybrid2.py --rows $O/rows_final.json --kp-dir $K --features speed+vyb --round 2 --select D 3 --out $O/router_d3.json > $O/c2_d3.log 2>&1
$P scripts/tc_hybrid2.py --rows $O/rows_final.json --kp-dir $K --features speed+vyb --round 2 --select D 0.75 --out $O/router_d075.json > $O/c2_d075.log 2>&1
$P scripts/tc_null_rows.py --rows $O/rows_final.json --kp-dir $K --checkpoint $CK --out $O/gate.json $F > $O/c3_gate.log 2>&1
$P scripts/tc_camera_graph.py --kp-dir $K --clip-len 300 --clips 4 --out $O/graph.json $F > $O/c3_graph.log 2>&1
$P scripts/tc_drift_3d.py --kp-dir $K --checkpoint $CK --prior 2 --no-test --out $O/drift3d.json $F > $O/c4_drift3d.log 2>&1
echo ALLDONE
