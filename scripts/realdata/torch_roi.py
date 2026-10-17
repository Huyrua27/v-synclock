import cv2, numpy as np, sys
D, S = sys.argv[1], sys.argv[2]
for f in ['iphone1','iphone2','iphone3','webcam0','webcam1']:
    c = cv2.VideoCapture(f'{D}/{f}.mp4'); fps = c.get(5); g = []
    for i in range(int(40 * fps)):
        ok, b = c.read()
        if not ok: break
        g.append(cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (320, 180), interpolation=cv2.INTER_AREA))
    g = np.array(g, np.float32); n = len(g)
    F = np.fft.rfft(g - g.mean(0), axis=0); fr = np.fft.rfftfreq(n, 1 / fps)
    band = (fr > 0.6) & (fr < 0.8)
    pw = (np.abs(F[band]) ** 2).max(0) / ((np.abs(F[1:]) ** 2).mean(0) + 1e-6)
    y, x = np.unravel_index(pw.argmax(), pw.shape)
    k = band.nonzero()[0][np.abs(F[band, y, x]).argmax()]
    print(f, 'peak at', x * 4 * (1 if f.startswith('web') else 1.5), y * 4 * (1 if f.startswith('web') else 1.5), 'freq', round(fr[k], 4), 'score', round(float(pw.max()), 1))
    im = cv2.cvtColor(g[n // 2].astype(np.uint8), cv2.COLOR_GRAY2BGR); cv2.circle(im, (int(x), int(y)), 6, (0, 0, 255), 1)
    cv2.imwrite(f'{S}/t4_{f}_roi.jpg', im)
    np.save(f'{S}/t4_{f}_led40.npy', g[:, max(0, y - 1):y + 2, max(0, x - 1):x + 2].mean((1, 2)))
