# -*- coding: utf-8 -*-
"""Streaming (causal) V-SyncLock: synchronize a rig while the keypoints arrive.

Frames are pushed per camera (already on each device's own uniform time grid, i.e. after
timestamp resampling, and roughly aligned to the reference by an integer coarse offset).
For every camera pair the synchronizer keeps

* an incremental cache of lag profiles: each long (pass-1) and short (pass-2) window is
  correlated once, as soon as both streams cover it; windows are re-centred on the running
  drift estimate, so arbitrarily long drift stays inside the +-max_lag profile;
* an accumulating ground-truth-free null: lag-profile peaks of long-window pairs taken at
  least `min_gap` seconds apart; the gate threshold is its q-quantile;

and every `refit_every` seconds it refits each pair's drift line with the coarse-to-fine
search (consensus prior, narrow pass 2, Theil-Sen, optional clock-rate slope prior), gates the
pairs by their pass-1 peak against the null, and places the cameras by weighted least squares
with a Huber loss. The result is a line delta_c(t) = slope_c t + intercept_c per placed camera
(frames, reference time), meaning camera c's frame t shows reference time t + delta_c(t).

    sync = StreamingSynchronizer(["cam1", "cam2", "cam3"], ref="cam1")
    for t in range(T):
        for c in cams:
            sync.push(c, xy[c][t])           # (J, 2) keypoints, NaN for missing joints
        est = sync.maybe_update()            # None, or {cam: (slope, intercept)} after a refit
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .benchmark.baselines.cross_correlation import lag_scores, motion_features
from .benchmark.drift import peak_in_range, shrink_slope, theilsen_line
from .core.types import KeypointSequence


def cc_profile(features="speed+vyb", max_lag=8):
    return lambda a, b: lag_scores(motion_features(a, features), motion_features(b, features), max_lag)


def _ts(c, v, cap=150):
    """Theil-Sen line on at most `cap` evenly spaced points (it is quadratic in their number)."""
    c, v = np.asarray(c, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if len(c) > cap:
        k = np.linspace(0, len(c) - 1, cap).round().astype(int)
        c, v = c[k], v[k]
    return theilsen_line(c, v)


class _Pair:
    def __init__(self):
        self.long, self.short = [], []           # (centre, lags, scores) in reference time
        self.long_meta, self.short_meta = [], [] # (start frame, length, lag centre used) per window
        self.next_long = self.next_short = 0     # start frame of the next window to compute
        self.null = []
        self.line = (0.0, 0.0)                   # current drift estimate of this pair
        self.peak = 0.0
        self.H = None                            # Hough accumulator over (offset, slope) (vote mode)
        self.locked, self.stable = False, 0      # acquisition / tracking state (vote mode)


class StreamingSynchronizer:
    def __init__(self, cameras, ref, fps=30.0, profile=None, window=20, stride=6, long_window=80,
                 long_stride=20, radius=3.0, refit_every=5.0, first_fit=10.0, prior=None,
                 null_q=0.95, null_n=200, null_per_update=20, min_gap=3.0, huber=0.5, max_slope=6.0, recenter=False, recenter_tol=3.0, inactive_after=None, vote=False,
                 vote_offset=12.0, vote_offset_step=0.25, vote_slope_step=0.25, lock_after=2, seed=0):
        self.cams, self.ref = list(cameras), ref
        self.fps, self.prof = fps, profile or cc_profile()
        self.w, self.s, self.lw, self.ls = window, stride, long_window, long_stride
        self.radius, self.refit, self.first = radius, refit_every, first_fit
        self.prior = None if prior is None else prior / (60.0 * fps)    # frames/min -> frames/frame
        self.null_q, self.null_n, self.null_k, self.gap = null_q, null_n, null_per_update, min_gap
        self.huber = huber
        self.max_slope = None if max_slope is None else max_slope / (60.0 * fps)   # frames/min -> frames/frame
        self.recenter, self.recenter_tol = recenter, recenter_tol
        self.inactive_after = inactive_after      # seconds without new detections -> camera inactive
        self.last_seen = {c: -1 for c in self.cams}
        self.vote, self.lock_after = vote, lock_after
        if vote:                                  # Hough grid: offset (frames) x slope (frames per frame)
            self.gb = np.arange(-vote_offset, vote_offset + 1e-9, vote_offset_step)
            ms = max_slope if max_slope is not None else 6.0
            self.gs = np.arange(-ms, ms + 1e-9, vote_slope_step) / (60.0 * fps)
        self.n_recomputed = 0
        self.rng = np.random.default_rng(seed)
        self.buf = {c: [] for c in self.cams}
        self.pairs = {p: _Pair() for p in itertools.combinations(self.cams, 2)}
        self.next_fit = first_fit * fps
        self.estimate = {ref: (0.0, 0.0)}
        self.placed = {ref}
        self.timing = []                          # seconds per update (profiling)

    # ---- input ---------------------------------------------------------------------------
    def push(self, cam, xy):
        """Append one frame (J, 2) or a chunk (n, J, 2) of camera `cam`."""
        xy = np.asarray(xy, dtype=np.float64)
        n0 = len(self.buf[cam])
        chunk = xy if xy.ndim == 3 else xy[None]
        self.buf[cam].extend(chunk)
        seen = np.flatnonzero(np.isfinite(chunk).any(axis=(1, 2)))
        if len(seen):
            self.last_seen[cam] = n0 + int(seen[-1])

    def _stream(self, c, lo, hi):
        if lo < 0 or hi > len(self.buf[c]):
            return None
        x = np.array(self.buf[c][lo:hi])
        if not np.isfinite(x).any():
            return None
        if not np.isfinite(x).all():                                   # gap-fill per joint
            t = np.arange(len(x))
            for j in range(x.shape[1]):
                for k in range(2):
                    v = x[:, j, k]; ok = np.isfinite(v)
                    x[:, j, k] = np.interp(t, t[ok], v[ok]) if ok.sum() >= 2 else 0.0
        return KeypointSequence(x, self.fps)

    def _window(self, u, v, t0, w, line):
        """Profile of u[t0:t0+w] against v, re-centred on the pair's current line."""
        k0 = int(round(line[0] * (t0 + w / 2) + line[1]))
        a, b = self._stream(u, t0 + k0, t0 + k0 + w), self._stream(v, t0, t0 + w)
        if a is None or b is None:
            return None
        try:
            lags, sc = self.prof(a, b)
        except ValueError:                                             # no motion
            return None
        return t0 + w / 2, np.asarray(lags, dtype=np.float64) + k0, np.asarray(sc, dtype=np.float64)

    @staticmethod
    def _centre(line, t0, w):
        return int(round(line[0] * (t0 + w / 2) + line[1]))

    def _advance(self, p, P):
        u, v = p
        n = min(len(self.buf[u]), len(self.buf[v])) - 8               # margin for the re-centring
        cline = P.line if (not self.vote or P.locked) else (0.0, 0.0)
        while P.next_long + self.lw <= n:
            r = self._window(u, v, P.next_long, self.lw, cline)
            if r:
                P.long.append(r); P.long_meta.append((P.next_long, self.lw, self._centre(cline, P.next_long, self.lw)))
                if self.vote:
                    self._cast(P, r)
            P.next_long += self.ls
        while P.next_short + self.w <= n:
            r = self._window(u, v, P.next_short, self.w, cline)
            if r:
                P.short.append(r); P.short_meta.append((P.next_short, self.w, self._centre(cline, P.next_short, self.w)))
            P.next_short += self.s
        gap, tries = int(self.gap * self.fps), 0
        while len(P.null) < self.null_n and tries < 3 * self.null_k and n > self.lw + gap:
            tries += 1
            t1, t2 = self.rng.integers(0, n - self.lw, 2)
            if abs(int(t1) - int(t2)) < gap:
                continue
            a, b = self._stream(u, int(t1), int(t1) + self.lw), self._stream(v, int(t2), int(t2) + self.lw)
            if a is None or b is None:
                continue
            try:
                P.null.append(float(np.max(self.prof(a, b)[1])))
            except ValueError:
                pass
            if tries >= self.null_k and len(P.null) >= 20:
                break

    def _cast(self, P, r):
        """Add one window's lag profile to the pair's Hough accumulator: every line (beta, s) receives
        the profile value at the lag it predicts for this window, beta + s * t. The profile is
        median-subtracted and counts 0 outside the window's lag range, so a window centred far from
        a line neither supports nor penalizes it."""
        c, lags, sc = r
        z = sc - np.median(sc)
        lam = self.gb[:, None] + self.gs[None, :] * c
        v = np.interp(lam, lags, z, left=0.0, right=0.0)
        P.H = v if P.H is None else P.H + v

    def _voted(self, P):
        i, j = np.unravel_index(int(np.argmax(P.H)), P.H.shape)
        return float(self.gs[j]), float(self.gb[i])

    def _recentre(self, p, P):
        """Recompute cached windows whose lag centre is stale under the pair's current line.

        A window is correlated over +-max_lag around the line in force when it was first
        computed. If the line has since moved by more than `recenter_tol` frames at that window
        (or the window's peak sits at the edge of its profile and the centre has moved), the
        true lag may lie at or beyond the edge, and the cached profile would bias every later fit.
        The window is recomputed around the current line and replaces the old entry (never added
        as a second observation); a window that no longer yields a profile is dropped."""
        u, v = p
        n_re = 0
        for win, meta in ((P.long, P.long_meta), (P.short, P.short_meta)):
            keep_w, keep_m = [], []
            for r, (t0, w, k0) in zip(win, meta):
                k1 = self._centre(P.line, t0, w)
                edge = int(np.argmax(r[2])) in (0, len(r[2]) - 1)
                if abs(k1 - k0) > self.recenter_tol or (edge and k1 != k0):
                    r2 = self._window(u, v, t0, w, P.line)
                    n_re += 1
                    if r2 is None:
                        continue
                    r, k0 = r2, k1
                keep_w.append(r); keep_m.append((t0, w, k0))
            win[:], meta[:] = keep_w, keep_m
        self.n_recomputed += n_re
        return n_re

    # ---- estimation ----------------------------------------------------------------------
    def _fit(self, P):
        if len(P.short) < 2:
            return None
        if self.vote:
            if P.H is None:
                return None
            slope, inter = self._voted(P)
            c = np.array([x for x, _, _ in P.short])
            off = np.array([peak_in_range(l, s, slope * x + inter, self.radius) for x, l, s in P.short])
            if self.prior is not None:
                if len(c) > 150:
                    k = np.linspace(0, len(c) - 1, 150).round().astype(int)
                    return self._clamp(c, shrink_slope(c[k], off[k], self.prior, self.w, self.s * len(c) / 150))
                return self._clamp(c, shrink_slope(c, off, self.prior, self.w, self.s))
            return self._clamp(c, _ts(c, off))
        if len(P.long) >= 2:
            slope, inter = _ts([c for c, _, _ in P.long], [peak_in_range(l, s) for _, l, s in P.long])
        elif P.long:
            slope, inter = 0.0, peak_in_range(P.long[0][1], P.long[0][2])
        else:
            slope, inter = P.line
        c = np.array([x for x, _, _ in P.short])
        free = np.array([peak_in_range(l, s) for _, l, s in P.short])
        cands = [(slope, inter), _ts(c, free)]
        sup = [np.mean(np.abs(free - (m * c + q)) <= self.radius) for m, q in cands]
        slope, inter = cands[int(np.argmax(sup))]
        off = np.array([peak_in_range(l, s, slope * x + inter, self.radius) for x, l, s in P.short])
        if self.prior is not None:
            if len(c) > 150:
                k = np.linspace(0, len(c) - 1, 150).round().astype(int)
                return self._clamp(c, shrink_slope(c[k], off[k], self.prior, self.w, self.s * len(c) / 150))
            return self._clamp(c, shrink_slope(c, off, self.prior, self.w, self.s))
        return self._clamp(c, _ts(c, off))

    def _clamp(self, c, line):
        """Physical bound on the drift slope: consumer clocks drift far less than a few frames per
        minute, so a steeper line comes from windows without timing evidence. Clip the slope and keep
        the line through the same point at the centre of the evidence."""
        slope, inter = line
        if self.max_slope is None or abs(slope) <= self.max_slope:
            return line
        cm = float(np.mean(c))
        s2 = float(np.clip(slope, -self.max_slope, self.max_slope))
        return s2, slope * cm + inter - s2 * cm

    def _place(self, edges):
        nodes = [c for c in self.cams if c != self.ref]
        idx = {c: i for i, c in enumerate(nodes)}
        comp, frontier = {self.ref}, [self.ref]
        while frontier:
            x = frontier.pop()
            for (u, v) in edges:
                for a, b in ((u, v), (v, u)):
                    if a == x and b not in comp:
                        comp.add(b); frontier.append(b)
        E = [(u, v, P) for (u, v), P in edges.items() if u in comp and v in comp]
        if len(comp) == 1 or not E:
            return {self.ref: (0.0, 0.0)}, comp
        A = np.zeros((len(E), len(nodes)))
        for k, (u, v, _) in enumerate(E):
            # pair line: v's frame t shows the instant of u's frame t + d_uv, so with o_c = frame index
            # of camera c minus that of the reference at the same instant, o_u - o_v = d_uv
            if u in idx: A[k, idx[u]] += 1
            if v in idx: A[k, idx[v]] -= 1
        out = {self.ref: (0.0, 0.0)}
        base = np.array([max(P.peak, 1e-3) for _, _, P in E])
        sol = []
        for j in (0, 1):                                              # slope, intercept
            b = np.array([P.line[j] for _, _, P in E]); rw = np.ones(len(E))
            for _ in range(10):
                W = np.sqrt(base * rw)
                x = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
                r = np.abs(A @ x - b)
                if j == 1:
                    rw = np.where(r <= self.huber, 1.0, self.huber / np.maximum(r, 1e-9))
            sol.append(x)
        for c in comp - {self.ref}:                                   # delta_c = -o_c
            out[c] = (-float(sol[0][idx[c]]), -float(sol[1][idx[c]]))
        return out, comp

    def _active(self, c):
        """A camera that has delivered no detection for `inactive_after` seconds (e.g. it stopped
        recording) keeps its last timing for the frames it did record, but its pairs no longer
        enter the fit and placement of the cameras that are still recording."""
        if self.inactive_after is None or c == self.ref:
            return True
        return len(self.buf[self.ref]) - 1 - self.last_seen[c] <= self.inactive_after * self.fps

    def update(self):
        t0 = time.perf_counter()
        edges = {}
        active = {c for c in self.cams if self._active(c)}
        for p, P in self.pairs.items():
            self._advance(p, P)
            fit = self._fit(P)
            if fit is None:
                continue
            recent = P.long[-12:]
            P.peak = float(np.mean([np.max(s) for _, _, s in recent])) if recent else 0.0
            thr = np.quantile(P.null, self.null_q) if len(P.null) >= 20 else np.inf
            if P.peak >= thr:                 # accept a line only from pairs with timing evidence;
                if self.vote:                 # acquisition -> tracking after `lock_after` stable updates
                    tn = float(len(self.buf[self.ref]))
                    moved = abs((fit[0] - P.line[0]) * tn + fit[1] - P.line[1])
                    P.stable = P.stable + 1 if moved < 1.0 else 0
                    P.locked = P.stable >= self.lock_after
                P.line = fit                  # otherwise keep the previous one (windows stay centred on it)
                if self.recenter and self._recentre(p, P):
                    refit = self._fit(P)      # one refit on the recomputed windows (no further iteration)
                    if refit is not None:
                        P.line = refit
                edges[p] = P
        # 1) cameras that are still recording are placed from the edges among themselves
        est, comp = self._place({p: P for p, P in edges.items() if p[0] in active and p[1] in active})
        # 2) a camera that stopped keeps its last timing; if it has none yet (e.g. offline, where all
        #    data arrive at once), it is placed from its own edges to the placed cameras, which it
        #    does not move
        for c in self.cams:
            if c in active or c in est:
                continue
            if c in self.estimate:
                est[c] = self.estimate[c]
                continue
            sl, ic, w = [], [], []
            for (u, v), P in edges.items():
                if c == u and v in est:        # o_u - o_v = d_uv and delta = -o  ->  delta_u = delta_v - d_uv
                    sl.append(est[v][0] - P.line[0]); ic.append(est[v][1] - P.line[1]); w.append(max(P.peak, 1e-3))
                elif c == v and u in est:      # delta_v = delta_u + d_uv
                    sl.append(est[u][0] + P.line[0]); ic.append(est[u][1] + P.line[1]); w.append(max(P.peak, 1e-3))
            if w:
                w = np.asarray(w)
                est[c] = (float(np.average(sl, weights=w)), float(np.average(ic, weights=w)))
        self.estimate, self.placed = est, comp | set(est)
        self.timing.append(time.perf_counter() - t0)
        return est

    def maybe_update(self):
        """Refit when `refit_every` seconds of new reference frames have arrived."""
        if len(self.buf[self.ref]) >= self.next_fit:
            self.next_fit += self.refit * self.fps
            return self.update()
        return None

    def offset(self, cam, t):
        """Current estimate of delta_cam at reference frame(s) t; NaN if not placed."""
        if cam not in self.estimate:
            return np.full(np.shape(t), np.nan)
        s, b = self.estimate[cam]
        return s * np.asarray(t, dtype=np.float64) + b
