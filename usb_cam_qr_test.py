"""USB摄像头 QR 实测 — raw + 2x upscale 双策略"""
import cv2
import time
import os
import numpy as np

# ── 可调参数 ──
CAM_IDX = int(os.environ.get("CAM_IDX", "0"))
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "auto").strip().lower()
COOLDOWN_MS = 2000


def _display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _window_enabled():
    if SHOW_WINDOW in ("1", "true", "yes", "on"):
        return _display_available()
    if SHOW_WINDOW in ("0", "false", "no", "off"):
        return False
    return _display_available()


WINDOW_ENABLED = _window_enabled()

detector = cv2.QRCodeDetector()
last_qr = None
last_qr_n = 0
last_send_ms = 0
fps_t0 = time.time()
fps_n = 0
fps_val = 0.0


def _try_decode(img_gray):
    try:
        data, pts, _ = detector.detectAndDecode(img_gray)
        if pts is not None and data:
            data = data.strip()
            if data in ("1","2","3","4","5","6"):
                return data, pts
    except cv2.error:
        pass
    return None, None


def decode_multi(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # S1: raw
    data, pts = _try_decode(gray)
    if data: return data, pts, "raw"

    # S2: 2x upscale
    h, w = gray.shape[:2]
    up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    data, pts = _try_decode(up)
    if data and pts is not None:
        pts = pts * 0.5
        return data, pts, "upscale"

    return None, None, None


cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAM_IDX}")

if WINDOW_ENABLED:
    print("Hold QR code in front of camera. Press ESC to quit.\n")
else:
    if SHOW_WINDOW in ("1", "true", "yes", "on"):
        print("SHOW_WINDOW=1 was requested, but no DISPLAY/WAYLAND_DISPLAY is available.")
    print("No GUI display detected; running headless. Press Ctrl+C to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, pts, strategy = decode_multi(frame)
        now = int(time.time() * 1000)

        fps_n += 1
        if fps_n % 30 == 0:
            fps_val = 30 / max(time.time() - fps_t0, 1e-3)
            fps_t0 = time.time()

        if data:
            box = pts.astype(int).reshape(-1, 2)
            xs, ys = box[:, 0], box[:, 1]
            qr_w, qr_h = int(xs.max() - xs.min()), int(ys.max() - ys.min())

            if WINDOW_ENABLED:
                cv2.polylines(frame, [box], True, (0, 255, 0), 2)
                cv2.putText(frame, f"QR={data}  {qr_w}x{qr_h}px  [{strategy}]  FPS={fps_val:.0f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if data == last_qr:
                last_qr_n += 1
            else:
                last_qr = data
                last_qr_n = 1

            if last_qr_n >= 1 and (now - last_send_ms) >= COOLDOWN_MS:
                print(f"  >>> QR={data}  size={qr_w}x{qr_h}px  strategy={strategy}")
                last_send_ms = now
                last_qr_n = 0
        else:
            if WINDOW_ENABLED:
                cv2.putText(frame, f"No QR  FPS={fps_val:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            last_qr = None
            last_qr_n = 0

        if WINDOW_ENABLED:
            cv2.imshow("USB Camera QR Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
except KeyboardInterrupt:
    pass

cap.release()
if WINDOW_ENABLED:
    cv2.destroyAllWindows()
