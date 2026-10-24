# -*- coding: utf-8 -*-
"""Per-frame presentation times of the video track of a .mov/.mp4, read from the container (no decoding).

Parses moov/trak/mdia (mdhd timescale, hdlr 'vide') and stbl/stts (+ ctts if present) and the edit list
offset is ignored (times start at the first sample). Also reports creation time and the nominal rate.

    python scripts/realdata/mov_timing.py "data/test4/Test 4/1"
"""
import datetime
import glob
import os
import struct
import sys

import numpy as np


def boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack(">I4s", f.read(8))
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]; hdr = 16
        elif size == 0:
            size = end - pos
        yield typ.decode("latin1"), pos + hdr, pos + size
        pos += size


def find(f, start, end, path):
    for typ, s, e in boxes(f, start, end):
        if typ == path[0]:
            return (s, e) if len(path) == 1 else find(f, s, e, path[1:])
    return None


def video_times(path):
    with open(path, "rb") as f:
        f.seek(0, 2); n = f.tell()
        moov = find(f, 0, n, ["moov"])
        mvhd = find(f, *moov, ["mvhd"])
        f.seek(mvhd[0]); ver = f.read(1)[0]; f.read(3)
        created = struct.unpack(">Q" if ver else ">I", f.read(8 if ver else 4))[0]
        for typ, s, e in boxes(f, *moov):
            if typ != "trak":
                continue
            hdlr = find(f, s, e, ["mdia", "hdlr"])
            f.seek(hdlr[0] + 8)
            if f.read(4) != b"vide":
                continue
            mdhd = find(f, s, e, ["mdia", "mdhd"])
            f.seek(mdhd[0]); v = f.read(1)[0]; f.read(3)
            f.read(16 if v else 8)
            scale = struct.unpack(">I", f.read(4))[0]
            stbl = find(f, s, e, ["mdia", "minf", "stbl"])
            stts = find(f, *stbl, ["stts"])
            f.seek(stts[0] + 4); cnt = struct.unpack(">I", f.read(4))[0]
            d = np.frombuffer(f.read(8 * cnt), dtype=">u4").reshape(-1, 2)
            deltas = np.repeat(d[:, 1].astype(np.int64), d[:, 0].astype(np.int64))
            dts = np.concatenate([[0], np.cumsum(deltas)[:-1]])
            ctts = find(f, *stbl, ["ctts"])
            pts = dts.astype(np.int64)
            if ctts:
                f.seek(ctts[0]); cv = f.read(1)[0]; f.read(3)
                c = struct.unpack(">I", f.read(4))[0]
                o = np.frombuffer(f.read(8 * c), dtype=">i4").reshape(-1, 2)   # offsets are signed in practice
                pts = pts + np.repeat(o[:, 1].astype(np.int64), o[:, 0].astype(np.int64))
            pts = np.sort(pts) / scale
            when = datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=created)
            return pts - pts[0], scale, when
    raise ValueError("no video track")


def main():
    for d in sys.argv[1:]:
        for p in sorted(set(glob.glob(os.path.join(d, "*.mov")) + glob.glob(os.path.join(d, "*.mp4")))):
            t, scale, when = video_times(p)
            dt = np.diff(t)
            med = np.median(dt)
            print(f"{os.path.basename(p):14s} created {when:%Y-%m-%d %H:%M:%S}  frames {len(t):6d}  dur {t[-1]:7.2f}s  "
                  f"mean fps {(len(t) - 1) / t[-1]:7.3f}  median fps {1 / med:6.2f}  "
                  f"gaps>1.5dt {int((dt > 1.5 * med).sum()):3d}  min/max dt {dt.min() * 1e3:5.1f}/{dt.max() * 1e3:6.1f} ms  "
                  f"index-vs-clock at end {(np.arange(len(t)) / (1 / med) - t)[-1] / med:+.1f} fr")
            np.save(p[:-4] + "_pts.npy", t)


if __name__ == "__main__":
    main()
