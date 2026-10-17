"""Flash ground truth for take 2: edges per camera, coarse alignment by motion energy, matched-edge drift fit."""
import csv, json, sys
import numpy as np

S, D = sys.argv[1], sys.argv[2]
CAMS = ['iphone1', 'iphone2', 'iphone3', 'webcam0', 'webcam1']
REF = 'iphone1'


def load(f):
    a = np.load(f'{S}/t4_{f}_full.npy')
    t = a[:, 0]
    if f.startswith('webcam'):  # host-clock timestamps
        t = np.array([float(r['t_rel_s']) for r in csv.DictReader(open(f'{D}/{f}_timestamps.csv', encoding='utf-8-sig'))])
        assert len(t) == len(a), (f, len(t), len(a))
    return t, a[:, 1], a[:, 4]


def edges(a, t):
    lo_l, hi_l = np.percentile(a, 10), np.percentile(a, 90)
    amp = hi_l - lo_l
    d = np.diff(a)
    big = np.where(np.abs(d) > 0.3 * amp)[0]
    out = []
    for g in np.split(big, np.where(np.diff(big) > 3)[0] + 1):
        if len(g) == 0:
            continue
        lo, hi = g[0], g[-1] + 1
        if lo < 5 or hi > len(a) - 6:
            continue
        pre, post = np.median(a[lo - 4:lo + 1]), np.median(a[hi:hi + 5])
        s = int(np.sign(post - pre))
        if np.sign(d[g]).min() != np.sign(d[g]).max():
            continue                                   # mixed directions: occlusion / flicker
        mid = 0.5 * (pre + post)
        k = lo + int(np.argmax(s * (a[lo:hi + 1] - mid) >= 0))
        fr = k - 1 + (mid - a[k - 1]) / (a[k] - a[k - 1])
        out.append(dict(frame=fr, t=float(np.interp(fr, np.arange(len(t)), t)), sign=s, pre=pre, post=post))
    # plateau sanity: on/off levels near the robust levels of this camera
    on = np.median([e['post'] if e['sign'] > 0 else e['pre'] for e in out])
    off = np.median([e['pre'] if e['sign'] > 0 else e['post'] for e in out])
    A = on - off
    good = []
    for e in out:
        hi_v, lo_v = (e['post'], e['pre']) if e['sign'] > 0 else (e['pre'], e['post'])
        if abs(hi_v - on) < 0.25 * A and abs(lo_v - off) < 0.25 * A:
            good.append(e)
    # cadence sanity: neighbouring edges of opposite sign ~ half a period apart
    ts = np.array([e['t'] for e in good])
    keep = []
    for i, e in enumerate(good):
        gaps = [abs(ts[j] - ts[i]) for j in (i - 1, i + 1) if 0 <= j < len(ts) and good[j]['sign'] != e['sign']]
        if gaps and min(abs(g - 0.714) for g in gaps) < 0.12:
            keep.append(e)
    return out, keep, (on, off)


def coarse(tr, mr, tc, mc, hz=20.0, max_lag=120.0):
    def grid(t, m):
        g = np.arange(t[0], t[-1], 1 / hz)
        x = np.interp(g, t, np.log1p(m))
        x = x - np.convolve(x, np.ones(int(2 * hz)) / (2 * hz), 'same')   # high-pass ~2 s
        return g, (x - x.mean()) / (x.std() + 1e-9)
    gr, xr = grid(tr, mr)
    gc, xc = grid(tc, mc)
    best = (-1, 0)
    lags = np.arange(-max_lag, max_lag, 1 / hz)
    scores = []
    for L in lags:   # cam time = ref time + L
        i0 = max(0, int(round((gc[0] - L - gr[0]) * hz)))
        j0 = max(0, int(round((gr[0] + L - gc[0]) * hz)))
        n = min(len(xr) - i0, len(xc) - j0)
        sc = float(np.mean(xr[i0:i0 + n] * xc[j0:j0 + n])) if n > 30 * hz else 0.0
        scores.append(sc)
    scores = np.array(scores)
    k = scores.argmax()
    second = np.max(scores[np.abs(lags - lags[k]) > 1.0])
    return float(lags[k]), float(scores[k]), float(second)


data = {f: load(f) for f in CAMS}
E = {}
for f in CAMS:
    t, a, m = data[f]
    raw, keep, lv = edges(a, t)
    E[f] = keep
    print(f'{f}: frames {len(t)}, dur {t[-1]:.1f}s, edges raw {len(raw)} kept {len(keep)}, on/off {lv[0]:.0f}/{lv[1]:.0f}')

res = {}
tr, ar, mr = data[REF]
er = E[REF]
for f in CAMS:
    if f == REF:
        continue
    tc, ac, mc = data[f]
    L, sc, sc2 = coarse(tr, mr, tc, mc)
    ec = E[f]
    tcs = np.array([e['t'] for e in ec])
    pairs = []
    for e in er:
        j = np.argmin(np.abs(tcs - (e['t'] + L)))
        if abs(tcs[j] - (e['t'] + L)) < 0.3 and ec[j]['sign'] == e['sign']:
            pairs.append((e['t'], ec[j]['t'], e['sign']))
    p = np.array(pairs)
    x, o = p[:, 0], p[:, 1] - p[:, 0]
    # robust line (IRLS Huber)
    w = np.ones_like(x)
    for _ in range(20):
        A = np.c_[x - x.mean(), np.ones_like(x)] * np.sqrt(w)[:, None]
        b, c0 = np.linalg.lstsq(A, o * np.sqrt(w), rcond=None)[0]
        r = o - (b * (x - x.mean()) + c0)
        sc_ = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = np.minimum(1, 1.345 * sc_ / (np.abs(r) + 1e-12))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.polyfit(x[i], o[i], 1)[0])
    lo_b, hi_b = np.percentile(bs, [2.5, 97.5])
    rise, fall = p[:, 2] > 0, p[:, 2] < 0
    res[f] = dict(coarse_lag=L, coarse_score=sc, coarse_second=sc2, n=len(p), span=float(np.ptp(x)),
                  offset_mid_s=float(c0), t_mid=float(x.mean()), slope_ms_per_min=float(b * 6e4),
                  slope_ci=[float(lo_b * 6e4), float(hi_b * 6e4)], slope_frames_per_min=float(b * 6e4 / 33.333),
                  resid_mad_ms=float(np.median(np.abs(r)) * 1e3), resid_std_ms=float(r.std() * 1e3),
                  rise_minus_fall_ms=float((np.median(r[rise]) - np.median(r[fall])) * 1e3))
    np.save(f'{S}/t4_pairs_{f}.npy', p)
    print(f, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[f].items()}))
json.dump(res, open(f'{S}/t4_gt.json', 'w'), indent=1)
