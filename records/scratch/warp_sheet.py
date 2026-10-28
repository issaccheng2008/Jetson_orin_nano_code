import cv2, json, glob, os, re, numpy as np

os.chdir(os.path.expanduser("~/shape_dump_3"))
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
    s = 200.0
    dst = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)
    w = cv2.warpPerspective(im, cv2.getPerspectiveTransform(pts[:4], dst), (200, 200))
    w = cv2.copyMakeBorder(w, 30, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    hud = d.get("hu_dist")
    lab = "%s %s hu:%s %.2f" % (f[:4], d.get("shape_rules"), d.get("hu_best"), hud or 0)
    cv2.putText(w, lab, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    tiles.append(w)

rows = []
for i in range(0, len(tiles), 4):
    r = list(tiles[i:i + 4])
    while len(r) < 4:
        r.append(np.full_like(tiles[0], 255))
    rows.append(np.hstack(r))
sheet = np.vstack(rows)
cv2.imwrite("/home/isaac/warp_sheet.png", sheet)
print(len(tiles), sheet.shape)
