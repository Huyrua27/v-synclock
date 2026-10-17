# Body-axis vs image-vertical input on development data (rule: body may replace image if <= +0.05 frames).
# Rows from run_vy_suite.py --tag vy (image axis) and --tag final (body axis).
import json, numpy as np, collections
def unit_means(rows, method, key, keep=lambda r: True):
    d = collections.defaultdict(list)
    for r in rows:
        if r['method'] == method and keep(r):
            d[key(r)].append(r['err'])
    return {k: np.mean(v) for k, v in d.items()}
def compare(name, A, B, method, key, keep=lambda r: True):
    ua, ub = unit_means(A, method, key, keep), unit_means(B, method, key, keep)
    U = sorted(set(ua) & set(ub)); a = np.array([ua[u] for u in U]); b = np.array([ub[u] for u in U])
    d = b - a; rng = np.random.default_rng(0); bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    print(f'{name:28s} {method:4s} units {len(U):3d}  image {a.mean():.3f}  body {b.mean():.3f}  body-image {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  rule(<=+0.05): {"pass" if d.mean() <= 0.05 else "FAIL"}')
L = lambda p: json.load(open(p))
tcv, tcf = L('data/vy_rows/tc_dev_rows_L300.json'), L('data/final_rows/tc_dev_rows_L300.json')
for m in ('CC', 'CSF'):
    compare('TotalCapture dev (9 seq)', tcv, tcf, m, lambda r: (r['subj'], r['seq']))
apv, apf = L('data/vy_rows/ap3d_rows.json'), L('data/final_rows/ap3d_rows.json')
for m in ('CC', 'CSF'):
    compare('AthletePose3D dev S3+S4', apv, apf, m, lambda r: (r['subj'], r['trial']), lambda r: r['subj'] in ('S3', 'S4'))
    compare('AthletePose3D test S2', apv, apf, m, lambda r: (r['subj'], r['trial']), lambda r: r['subj'] == 'S2')
tv, tf = L('data/vy_rows/tc_rows_L300.json'), L('data/final_rows/tc_rows_L300.json')
for m in ('CC', 'CSF'):
    compare('TotalCapture test (15 seq)', tv, tf, m, lambda r: (r['subj'], r['seq']))
