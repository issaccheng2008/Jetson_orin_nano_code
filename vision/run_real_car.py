"""Standalone real-car controller — V1 detector + PID + 4-wheel differential drive.

USB camera → LineDetector → dual-mode PID + lost recovery → 4 wheel speeds (rad/s).
Optional serial output to MCU at 115200 baud, ~10 Hz.
"""

import math, os, sys, time, json, atexit
import numpy as np
import cv2

# ── Path to V1 detector ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_V1_DIR = os.path.join(_SCRIPT_DIR, "v1_production")
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from line_detector_v1_warp import LineDetector


def clamp(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v


# ═══════════════════════════════════════════════════════════════════════
# Config — all overridable via env vars
# ═══════════════════════════════════════════════════════════════════════

# Camera
CAM_IDX = int(os.environ.get("CAM_IDX", "0"))
CAM_W = int(os.environ.get("CAM_W", "1280"))
CAM_H = int(os.environ.get("CAM_H", "720"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))

# Detector
CAM_HEIGHT_CM  = float(os.environ.get("CAM_HEIGHT_CM",  "40.0"))
CAM_PITCH_DEG  = float(os.environ.get("CAM_PITCH_DEG",  "45.0"))
CAM_VFOV_DEG   = float(os.environ.get("CAM_VFOV_DEG",   "49.0"))  # 100° diag / 16:9

# PID — dual-mode (straight / curve)
KP_S = float(os.environ.get("JETSON_PID_STRAIGHT_KP", "0.83"))
KI_S = float(os.environ.get("JETSON_PID_STRAIGHT_KI", "0.004"))
KD_S = float(os.environ.get("JETSON_PID_STRAIGHT_KD", "0.095"))
KP_C = float(os.environ.get("JETSON_PID_CURVE_KP",    "0.78"))
KI_C = float(os.environ.get("JETSON_PID_CURVE_KI",    "0.002"))
KD_C = float(os.environ.get("JETSON_PID_CURVE_KD",    "0.16"))
I_MAX  = float(os.environ.get("JETSON_PID_I_CLAMP",   "60.0"))
PID_DT  = float(os.environ.get("JETSON_PID_DT",       "0.033"))  # nominal 30 FPS

# Steer
STEER_SAT = float(os.environ.get("JETSON_STEER_SAT",   "45.0"))
STEER_SCL = float(os.environ.get("JETSON_STEER_SCALE", "0.6"))
RATE_LIM  = float(os.environ.get("JETSON_STEER_RATE_LIMIT", "5.0"))

# Speed-scaling exponent for steer (0=no scaling, 1.0=curvature-invariant)
# Physics: kappa = delta/speed, so steer ∝ speed^1.0 for same trajectory
ST2SPD_EXP = float(os.environ.get("JETSON_STEER_SPEED_EXP", "1.0"))

# Speed (real car — cm/s)
BASE_SPD  = float(os.environ.get("REAL_CAR_SPEED",       "15.0"))
MIN_SPD   = float(os.environ.get("REAL_CAR_MIN_SPEED",   "5.0"))
LOST_SPD  = float(os.environ.get("REAL_CAR_LOST_SCALE",  "0.92"))

# Lost recovery
HOLD_FRAMES = int(os.environ.get("LOST_HOLD_FRAMES", "6"))
SRCH_TURN   = float(os.environ.get("LOST_SEARCH_TURN", "11.0"))

# Differential drive — real car
ST2WHL       = float(os.environ.get("REAL_CAR_ST2WHL", "0.1"))       # steer→轮速差(cm/s)
WHEEL_RADIUS = float(os.environ.get("REAL_CAR_WHEEL_RADIUS", "3.0")) # cm

# Serial
SERIAL_PORT  = os.environ.get("SERIAL_PORT", "COM10")
SERIAL_BAUD  = int(os.environ.get("SERIAL_BAUD", "115200"))
#SERIAL_ENABLED = True   # 启动即开串口, 's' 键切换
SERIAL_ENABLED = os.environ.get("SERIAL_ENABLED", "0") == "1"

# Misc
MAX_SEC = float(os.environ.get("REAL_CAR_MAX_SEC", "0"))  # 0 = no limit
PRINT_INTERVAL = 0.5  # seconds between console prints


# ═══════════════════════════════════════════════════════════════════════
# Serial setup (optional)
# ═══════════════════════════════════════════════════════════════════════

_ser = None
_serial_first_sent = False

def _serial_open():
    global _ser
    try:
        import serial
        _ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
        print(f"[serial] opened {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        print(f"[serial] WARNING: cannot open {SERIAL_PORT}: {e}")
        _ser = None

def _serial_close():
    global _ser
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
        _ser = None

def _serial_send(frame: bytes):
    """Send binary frame to MCU. Returns True on success."""
    global _ser, SERIAL_ENABLED, _serial_first_sent
    if not SERIAL_ENABLED or _ser is None:
        return False
    try:
        n = _ser.write(frame)
        if not _serial_first_sent:
            _serial_first_sent = True
            print(f"[serial] first frame sent ({n} bytes): {frame.hex().upper()}")
        return True
    except Exception as e:
        print(f"[serial] write error: {e}")
        _serial_close()
        return False

atexit.register(_serial_close)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global SERIAL_ENABLED

    # ── Camera ──
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAM_IDX}")
        sys.exit(1)

    # DSHOW sometimes reports 0x0 before first grab — read one frame to wake it
    for _ in range(3):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.05)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w <= 0 or actual_h <= 0:
        actual_w, actual_h = CAM_W, CAM_H
    print(f"Camera {CAM_IDX}: requested {CAM_W}x{CAM_H}, got {actual_w}x{actual_h}")

    # ── Detector ──
    ld = LineDetector(actual_w, actual_h,
        cam_height_cm=CAM_HEIGHT_CM,
        cam_pitch_deg=CAM_PITCH_DEG,
        cam_vfov_deg=CAM_VFOV_DEG)

    # ── State ──
    integral      = 0.0
    last_err      = 0.0
    last_steer    = 0.0
    last_lock_ok  = False
    last_curve    = False   # detect curve→straight transitions
    last_print_t  = -99.0
    last_frame_t  = 0.0     # actual dt measurement
    t0            = None

    print(f"run_real_car: {actual_w}x{actual_h}  "
          f"KP_s={KP_S:.3f} KP_c={KP_C:.3f}  "
          f"speed={BASE_SPD:.1f} cm/s  st2whl={ST2WHL:.2f}  wheel_r={WHEEL_RADIUS:.1f}cm  "
          f"max_sec={MAX_SEC:.0f}")
    print("Keys: 'q'=quit  's'=toggle serial")

    # 启动时自动打开串口
    if SERIAL_ENABLED:
        _serial_open()

    while True:
        t = time.time()
        if t0 is None:
            t0 = t
        dt = t - last_frame_t if last_frame_t > 0 else PID_DT
        dt = clamp(dt, 0.01, 0.2)
        last_frame_t = t

        # ── Grab frame ──
        ok, bgr = cap.read()
        if not ok:
            print("[WARN] frame grab failed, retrying...")
            time.sleep(0.01)
            continue

        # ── Detector ──
        _dev, _hdg, conf, _vis, dbg = ld.process(bgr)
        err     = float(dbg.get("fused_err", 0.0))
        curve   = bool(dbg.get("curve_mode", False))
        lost    = int(dbg.get("lost_frames", 0))
        lock_ok = bool(dbg.get("bottom_lock_valid", False))

        # ── Reacquisition reset ──
        if lock_ok and not last_lock_ok:
            integral = 0.0
            last_err = 0.0
        last_lock_ok = lock_ok

        # ── PID (dual-mode) or lost recovery ──
        # Speed factor (10 cm/s = 1.0)
        spd_f = max(BASE_SPD / 10.0, 0.5)

        if lost == 0:
            kp, ki, kd = (KP_C, KI_C, KD_C) if curve else (KP_S, KI_S, KD_S)

            # Speed-adaptive gains
            # KD kept as-is (derr already grows with speed, don't double-scale)
            integral += err * dt / spd_f  # KI: slow accumulation at high speed
            integral = clamp(integral, -I_MAX / spd_f, I_MAX / spd_f)  # tighten clamp

            derr = (err - last_err) / max(dt, 1e-3)
            last_err = err

            # Curve exit boost: clear integral + extra KD to snap back to straight
            curve_exit = last_curve and not curve
            if curve_exit:
                integral *= 0.2
                kd *= 1.6
            last_curve = curve

            pid_out = kp * err + ki * integral + kd * derr
            steer = STEER_SAT * math.tanh(pid_out / STEER_SCL)
            steer *= spd_f ** ST2SPD_EXP  # speed-scaled steering (kappa = delta/v)
            if conf < 0.25:
                integral *= 0.85
        else:
            hold_frames = max(3, int(HOLD_FRAMES / spd_f ** 0.5))
            if lost <= hold_frames:
                steer = last_steer * 0.85
            else:
                phase = (lost // 8) % 2
                steer = abs(SRCH_TURN) * spd_f ** 0.5 if phase == 0 else -abs(SRCH_TURN) * spd_f ** 0.5
            integral *= 0.5

        # ── Steer rate limit (speed-scaled) ──
        ds = steer - last_steer
        rate_lim_spd = RATE_LIM * spd_f ** 0.5
        if abs(ds) > rate_lim_spd:
            steer = last_steer + math.copysign(rate_lim_spd, ds)
        last_steer = steer

        # ── Speed control ──
        spd = BASE_SPD
        if lost > 0:
            spd *= LOST_SPD
        if conf < 0.5:
            spd *= (0.5 + 0.5 * conf)
        spd = max(spd, MIN_SPD)

        # ── Convert to 4 wheel speeds (rad/s) ──
        # steer 直接当轮速差, 和仿真一样: delta = steer × STEER_TO_WHEEL
        delta    = steer * ST2WHL
        v_right  = spd + delta                     # cm/s at right wheels
        v_left   = spd - delta                     # cm/s at left wheels

        fl = v_left  / WHEEL_RADIUS  # rad/s
        fr = v_right / WHEEL_RADIUS
        rl = fl
        rr = fr

        # ── Serial output (every frame, camera-rate) ──
        def _rad2byte(v):
            return max(-127, min(127, int(v * 10.0))) & 0xFF
        frame = bytes([0xFF,
                       _rad2byte(fr), _rad2byte(fl),
                       _rad2byte(rr), _rad2byte(rl),
                       0xEE])
        _serial_send(frame)

        # ── Console print (every PRINT_INTERVAL seconds) ──
        if t - last_print_t > PRINT_INTERVAL:
            last_print_t = t
            print(f"{frame[0]:02X}{frame[1]:02X}{frame[2]:02X}{frame[3]:02X}{frame[4]:02X}{frame[5]:02X}")

        # ── Display (4 windows, same as vision_main.py) ──
        def _put(img, s, y, color=(0, 255, 0)):
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 1.Original — raw camera frame + overlay text
        frame_disp = bgr.copy()
        _put(frame_disp, f"steer:{steer:+.1f}  curve:{1 if curve else 0}  lost:{lost}", 25)
        _put(frame_disp, f"spd:{spd:.1f} cm/s  conf:{conf:.2f}", 50)
        _put(frame_disp, f"FL:{fl:+.1f}  FR:{fr:+.1f}  RL:{rl:+.1f}  RR:{rr:+.1f} rad/s", 75)
        _put(frame_disp, f"SERIAL:{'ON' if SERIAL_ENABLED else 'OFF'}", 100, (255, 255, 0))
        _put(frame_disp, "Q=quit S=toggle_serial", 125, (200, 200, 200))
        cv2.imshow("1.Original", frame_disp)

        # 2.Warp (birdseye)
        if "bird" in dbg and dbg["bird"] is not None:
            bird_bgr = cv2.cvtColor(dbg["bird"], cv2.COLOR_GRAY2BGR)
            cv2.imshow("2.Warp (birdseye)", cv2.resize(bird_bgr, (320, 400), interpolation=cv2.INTER_NEAREST))

        # 3.Adaptive (binary)
        if "binary_raw" in dbg and dbg["binary_raw"] is not None:
            b_raw = cv2.cvtColor(dbg["binary_raw"], cv2.COLOR_GRAY2BGR)
            cv2.imshow("3.Adaptive (binary)", cv2.resize(b_raw, (320, 400), interpolation=cv2.INTER_NEAREST))

        # 4.Close+Fit — V1 annotated birdseye
        if _vis is not None:
            cv2.imshow("4.Close+Fit", cv2.resize(_vis, (320, 400), interpolation=cv2.INTER_NEAREST))

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            SERIAL_ENABLED = not SERIAL_ENABLED
            if SERIAL_ENABLED and _ser is None:
                _serial_open()
            print(f"[key] serial {'ON' if SERIAL_ENABLED else 'OFF'}")

        # ── Time limit ──
        if MAX_SEC > 0 and (t - t0) > MAX_SEC:
            print(f"Done. {t - t0:.1f}s elapsed (MAX_SEC={MAX_SEC:.0f}).")
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    _serial_close()
    print("Exited.")


if __name__ == "__main__":
    main()
