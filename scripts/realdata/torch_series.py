import cv2, numpy as np, sys, time
D, S, f, x, y = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
r = 9 if f.startswith('iphone') else 6
c = cv2.VideoCapture(f'{D}/{f}.mp4'); v = []; pts = []; t0 = time.time(); prev = None
while c.grab():
    pts.append(c.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    ok, b = c.retrieve()
    p = b[y - 3 * r:y + 3 * r, x - 3 * r:x + 3 * r].astype(np.float32).mean(-1)
    sm = cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32).mean(-1)
    md = 0.0 if prev is None else float(np.abs(sm - prev).mean()); prev = sm
    v.append([p[2 * r:4 * r, 2 * r:4 * r].mean(), p.mean(), p.max(), md])
np.save(f'{S}/t4_{f}_full.npy', np.c_[np.array(pts), np.array(v)])
print(f, len(v), round(time.time() - t0), 's')
