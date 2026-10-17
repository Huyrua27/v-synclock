#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-SyncLock: record N webcams simultaneously with a host timestamp per frame (soft-sync layer)
=======================================================================
- OPEN + CONFIGURE sequentially on the main thread (opening several identical webcams at once is racy
  with Windows DSHOW: failed opens / stuck formats). Open one by one, verify MJPG + size,
  then hand each camera to a thread that only READS/WRITES.
- threading.Barrier makes the threads enter their read loops at nearly the same moment.
- Every frame is stamped with time.perf_counter_ns(), the process-wide clock
  (webcams are directly comparable). The host-timestamp layer complements optical cues when the webcams
  have no timecode.
- MJPG fourcc: the webcam compresses at the source, which greatly reduces USB bandwidth (raw YUY2 ~55 MB/s
  vs MJPG ~5-8 MB/s at 720p30). Required for several webcams at 30 fps.

Requires: opencv-python, numpy.

EXAMPLES
-----
python record_webcams.py --cams 0 1 2 --fps 30 --width 1280 --height 720 --duration 60
python record_webcams.py --preview            # preview only (no recording): framing / identification
python record_webcams.py --show --duration 60 # record and show live
"""
import argparse, csv, os, sys, threading, time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):          # console Windows cp125x → ép UTF-8
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Needs opencv:  pip install opencv-python numpy")

IS_WIN = sys.platform.startswith("win")


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def backend_id(name):
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}.get(
        name, cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY)


def list_cams(a, max_index=10):
    """Scan indices 0..max for real cameras and which of them accept MJPG at 720p."""
    print(f"Quét camera index 0..{max_index-1} (backend={a.backend}):")
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend_id(a.backend))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f0 = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            mtest = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ok = "MJPG@720 OK" if (mtest == "MJPG" and mw == 1280) else f"NO ({mw}x.. {mtest})"
            print(f"  index {i}: opens - default {w}x{h} {f0} - MJPG720 test -> {ok}")
            found.append(i)
        cap.release()
    print(f"\nUsable indices: {found}")
    print("-> pick the indices with 'MJPG@720 OK' and run:  python record_webcams.py --cams a b c")


def open_cam(index, a):
    """Open and configure one webcam (sequentially, on the main thread). Returns (cap, info) or (None, reason)."""
    cap = cv2.VideoCapture(index, backend_id(a.backend))
    if not cap.isOpened():
        return None, "cannot open (camera busy? used by another app?)"
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    # set SIZE first, then MJPG, then fps. Do NOT grab a frame here (grabbing a busy camera hangs).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FPS,          a.fps)
    time.sleep(0.1)
    if fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) != "MJPG":   # retry once, without reading
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        time.sleep(0.1)
    ok = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == a.width
          and fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) == "MJPG")
    if a.exposure is not None:                # DSHOW 0.25=manual, 0.75=auto (quirk UVC)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(a.exposure))
    elif a.auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS); fcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    info = f"{aw}x{ah} @{afps:.0f} {fcc}" + ("" if ok else "  WARNING: MJPG/size not applied -> raw stream uses a lot of bandwidth!")
    return cap, info


class CamRecorder(threading.Thread):
    def __init__(self, cam_index, cap, args, barrier, stop_event, latest):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.cap = cap
        self.a = args
        self.barrier = barrier
        self.stop_event = stop_event
        self.latest = latest
        self.stamps = []

    def run(self):
        cap = self.cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or self.a.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.a.height
        path = os.path.join(self.a.outdir, f"webcam{self.cam_index}.mp4")
        writer = None if self.a.no_record else cv2.VideoWriter(
                 path, cv2.VideoWriter_fourcc(*"mp4v"), self.a.fps, (w, h))
        for _ in range(3):                    # warm-up auto-gain
            cap.read()
        try:
            self.barrier.wait(timeout=8)      # start together
        except threading.BrokenBarrierError:
            pass

        idx = 0
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            ts = time.perf_counter_ns()
            if not ok:
                continue
            if writer is not None:
                writer.write(frame)
            self.stamps.append((idx, ts))
            if self.a.show:
                self.latest[self.cam_index] = frame
            idx += 1

        if writer is not None:
            writer.release()
        # no cap.release() here: the main thread releases all handles (avoids leaks when killed)
        if not self.a.no_record:
            self._dump_csv(path)

    def _dump_csv(self, video_path):
        if not self.stamps:
            return
        t0 = self.stamps[0][1]
        with open(os.path.splitext(video_path)[0] + "_timestamps.csv", "w",
                  newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "t_host_ns", "t_rel_s"])
            for i, ts in self.stamps:
                wr.writerow([i, ts, f"{(ts - t0) / 1e9:.6f}"])

    def stats(self):
        if len(self.stamps) < 2:
            return dict(cam=self.cam_index, frames=len(self.stamps), dur="?", fps_eff=0.0, dropped="?")
        dur = (self.stamps[-1][1] - self.stamps[0][1]) / 1e9
        fps_eff = (len(self.stamps) - 1) / dur if dur > 0 else 0
        return dict(cam=self.cam_index, frames=len(self.stamps), dur=round(dur, 2),
                    fps_eff=round(fps_eff, 3), dropped=max(0, round(self.a.fps * dur - len(self.stamps))))


def main():
    ap = argparse.ArgumentParser(description="Ghi N webcam + host-timestamp (V-SyncLock)")
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=None, help="seconds; empty = stop with q / Ctrl-C")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure (UVC ~ -log2 s, e.g. -6 = 1/64 s)")
    ap.add_argument("--auto-exposure", action="store_true")
    ap.add_argument("--show", action="store_true", help="show the cameras live while recording (q = stop)")
    ap.add_argument("--preview", action="store_true", help="preview only, no recording (framing / identification)")
    ap.add_argument("--backend", default="dshow" if IS_WIN else "any", choices=["dshow", "msmf", "any"],
                    help="camera backend; try 'msmf' if a camera refuses MJPG")
    ap.add_argument("--list", action="store_true", help="scan and list camera indices, then exit")
    a = ap.parse_args()
    if a.list:
        list_cams(a)
        return
    a.no_record = a.preview
    if a.preview:
        a.show = True
    a.outdir = a.outdir or ("capture_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not a.no_record:
        os.makedirs(a.outdir, exist_ok=True)

    # ---- open sequentially (avoids races between identical cameras) ----
    print(f"[V-SyncLock] Opening {len(a.cams)} webcams {a.cams} @{a.fps}fps {a.width}x{a.height} "
          f"{'(PREVIEW, not recording)' if a.no_record else '-> '+a.outdir}")
    opened = []
    try:
        for c in a.cams:
            cap, info = open_cam(c, a)
            if cap is None:
                print(f"  cam{c}: OPEN FAILED ({info})")
            else:
                print(f"  cam{c}: {info}")
                opened.append((c, cap))
            time.sleep(0.3)                   # pause between opens
    except KeyboardInterrupt:
        for _, cap in opened:
            try: cap.release()
            except Exception: pass
        sys.exit("\nCancelled while opening cameras (handles released).")
    if not opened:
        for _, cap in opened:
            try: cap.release()
            except Exception: pass
        sys.exit("No webcam could be opened.\n"
                 "  - Under WSL: run with the WINDOWS Python (not python3 inside WSL).\n"
                 "  - If Windows reports the camera is in use: end stale python.exe processes (Task Manager) "
                 "or unplug and replug the webcams, then run again.")

    barrier = threading.Barrier(len(opened))
    stop = threading.Event()
    latest = {}
    recs = [CamRecorder(c, cap, a, barrier, stop, latest) for c, cap in opened]
    recs_by_cam = {r.cam_index: r for r in recs}
    print("       starting together...")
    for r in recs:
        r.start()

    def mosaic(elapsed):
        TH = 360; tiles = []
        for c, _ in opened:
            fr = latest.get(c); rec = recs_by_cam[c]
            live = len(rec.stamps) / elapsed if elapsed > 0 else 0
            if fr is None:
                tile = np.zeros((TH, 640, 3), np.uint8)
                cv2.putText(tile, "cho frame...", (12, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
            else:
                hh, ww = fr.shape[:2]
                tile = cv2.resize(fr, (int(TH * ww / hh), TH))
            col = (0, 220, 0) if live >= a.fps * 0.8 else (0, 0, 255)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(tile, f"cam{c}  {live:.0f}/{a.fps}fps  {len(rec.stamps)}f",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            tiles.append(tile)
        return np.hstack(tiles) if tiles else None

    t_start = time.perf_counter()
    try:
        if a.show:
            title = "V-SyncLock preview (KHONG ghi) - q=dung" if a.no_record else "V-SyncLock 3-cam REC - q=dung"
            while not stop.is_set():
                el = time.perf_counter() - t_start
                mos = mosaic(el)
                if mos is not None:
                    cv2.imshow(title, mos)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop.set()
                if a.duration and el >= a.duration:
                    stop.set()
            cv2.destroyAllWindows()
        else:
            if a.duration:
                while (time.perf_counter() - t_start) < a.duration and any(r.is_alive() for r in recs):
                    time.sleep(0.05)
            else:
                print("       recording... press Ctrl-C to stop.")
                while any(r.is_alive() for r in recs):
                    time.sleep(0.2)
            stop.set()
    except KeyboardInterrupt:
        print("\n[V-SyncLock] Stopping...")
        stop.set()
    finally:
        stop.set()
        for r in recs:
            r.join(timeout=5)
        for _, cap in opened:                 # always release handles so the next run does not find the camera busy
            try: cap.release()
            except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

    print("\n=== RESULT ===")
    for r in recs:
        s = r.stats()
        flag = "  WARNING: dropped frames (USB bandwidth?)" if isinstance(s["dropped"], int) and s["dropped"] > 5 else ""
        print(f"  cam{s['cam']}: {s['frames']} frames - {s['dur']} s - effective fps {s['fps_eff']} - dropped ~{s['dropped']}{flag}")
    if not a.no_record:
        print(f"\nVideo + *_timestamps.csv → {a.outdir}")
    print("→ Dùng t_host_ns canh chéo webcam (coarse), flash-fade cho sub-frame.")


if __name__ == "__main__":
    main()
