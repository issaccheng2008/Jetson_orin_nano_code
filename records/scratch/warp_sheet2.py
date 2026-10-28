import cv2, json, glob, os, re, numpy as np

os.chdir(os.path.expanduser("~/shape_dump_3"))
S = 300
tiles = []
for f in sorted(glob.glob("*.json")):
    d = json.load(open(f))
    q = d.get("quad")
    jpg = f[:-5] + ".jpg"
    if not isinstance(q, str) or not os.path.exists(jpg):
        continue
    n = [float(v) for v in re.findall(r"-?[0-9.]+", q)]
    pts = np.array(n, dtype=np.float32).reshape(-1, 2)
    im = cv2.imread(jpg)
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], dtype=np.float32)
    w = cv2.warpPerspective(im, cv2.getPerspectiveTransform(pts[:4], dst), (S, S))
    w = cv2.copyMakeBorder(w, 40, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    hud = d.get("hu_dist")
    lab = "%s %s hu:%s %.2f" % (f[:4], d.get("shape_rules"), d.get("hu_best"), hud or 0)
    cv2.putText(w, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    tiles.append(w)

per = 2
for gi in range(0, len(tiles), per * 3):
    rows = []
    for i in range(gi, min(gi + per * 3, len(tiles)), per):
        r = list(tiles[i:i + per])
        while len(r) < per:
            r.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(r))
    sheet = np.vstack(rows)
    out = "/home/isaac/warp_%d.png" % (gi // (per * 3))
    cv2.imwrite(out, sheet)
    print(out, sheet.shape)
