import cv2, json, glob, os, re, numpy as np

os.chdir(os.path.expanduser("~/shape_dump_3"))
S = 460
want = {"0026", "0031", "0033", "0036", "0037", "0039", "0040", "0042"}
for f in sorted(glob.glob("*.json")):
    if f[:4] not in want:
        continue
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
    w = cv2.copyMakeBorder(w, 44, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    hud = d.get("hu_dist")
    cv2.putText(w, "%s %s hu:%s %.2f clo=%s" % (f[:4], d.get("shape_rules"),
                d.get("hu_best"), hud or 0, d.get("closure")),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imwrite("/home/isaac/t_%s.png" % f[:4], w)
    print("/home/isaac/t_%s.png" % f[:4], w.shape)
