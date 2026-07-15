"""Jetson Nano 视觉 Demo — 巡线 + QR + 红条 实时可视化
双击 run_vision_demo.bat 运行。ESC 退出。
"""
import cv2
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1_production"))
from qr_detector import QRDetector
from line_detector_v1_warp import LineDetector


# ── 可调参数 ──
CAM_IDX = int(os.environ.get("CAM_IDX", "0"))
CAM_W = int(os.environ.get("CAM_W", "1280"))
CAM_H = int(os.environ.get("CAM_H", "720"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))
COOLDOWN_MS = 2000


def main():
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Failed to open camera {CAM_IDX}.")
        return

    # Read one frame so the camera applies the requested configuration.
    ret, test_frame = cap.read()
    if not ret:
        print(f"Camera {CAM_IDX} opened, but could not capture a frame.")
        cap.release()
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    print(
        f"Actual camera configuration: "
        f"{actual_w}x{actual_h}, {actual_fps:.1f} FPS, {fourcc_text}"
    )

    # 检测器
    qr = QRDetector(stable_frames=1, cooldown_ms=COOLDOWN_MS,
                    min_edge_px=20, max_edge_px=400, debug=False)
    ld = LineDetector(cam_w=actual_w, cam_h=actual_h,
                      cam_height_cm=40.0, cam_pitch_deg=45.0, cam_vfov_deg=44.4)

    print(f"Jetson Vision Demo  ({actual_w}x{actual_h})")
    print("  巡线: 几何原语拟合 (平行线 / 同心圆)")
    print("  QR:   raw + 2x upscale  红条: HSV mask")
    print("  Press ESC to quit\n")

    fps_t0 = time.time()
    fps_n = 0
    fps_val = 0.0
    last_qr_action = None
    last_qr_t = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_now = time.time()

        # FPS
        fps_n += 1
        if fps_n % 30 == 0:
            fps_val = 30 / max(t_now - fps_t0, 1e-3)
            fps_t0 = t_now

        # ── 巡线 ──
        dev_px, heading_deg, conf, vis_bird, dbg = ld.process(frame)
        status_line = f"FPS={fps_val:.0f}"

        if dev_px is not None and conf > 0.15:
            # 转弯方向判定
            mtype = dbg.get("model_type", "?") or "?"
            if abs(heading_deg) < 4:
                turn_text, turn_color = "STRAIGHT", (0, 255, 0)
            elif heading_deg > 0:
                turn_text, turn_color = "RIGHT >>>", (0, 200, 255)
            else:
                turn_text, turn_color = "<<< LEFT", (0, 200, 255)

            status_line += f" | [{mtype}] dev={dev_px:+.0f}px head={heading_deg:+.0f}deg [{turn_text}] c={conf:.2f}"

            # 转向指示（中央大箭头）
            cx, cy = actual_w // 2, actual_h // 2
            arrow_len = int(30 + abs(heading_deg) * 2.5)
            arrow_angle = np.radians(-heading_deg - 90)  # 上=0°，右转=右箭头
            dx = int(arrow_len * np.cos(arrow_angle))
            dy = int(arrow_len * np.sin(arrow_angle))
            cv2.arrowedLine(frame, (cx - dx, cy - dy), (cx + dx, cy + dy),
                            turn_color, 3, tipLength=0.4)

            # 偏离指示条（底部）
            bar_cx = actual_w // 2
            bar_y = actual_h - 25
            cv2.line(frame, (bar_cx - 80, bar_y), (bar_cx + 80, bar_y), (80, 80, 80), 2)
            cv2.circle(frame, (bar_cx, bar_y), 4, (255, 255, 255), -1)
            dev_indicator = int(bar_cx + dev_px * 0.5)
            dev_indicator = max(bar_cx - 80, min(bar_cx + 80, dev_indicator))
            cv2.circle(frame, (dev_indicator, bar_y), 7, turn_color, -1)
        else:
            status_line += " | NO LINE"

        # ── QR ──
        action, qr_dbg = qr.update(frame)
        if action is not None:
            last_qr_action = action
            last_qr_t = t_now
            cv2.putText(frame, f"QR={action}!", (actual_w - 150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        elif last_qr_action is not None and t_now - last_qr_t < 2.0:
            cv2.putText(frame, f"QR={last_qr_action}", (actual_w - 150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 180), 2)

        # ── 红条 ──
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | \
                   cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red_r = cv2.countNonZero(mask_red) / (actual_w * actual_h)
        if red_r > 0.05:
            status_line += f" | RED={red_r:.2f}"
            # 红条区域轮廓
            contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # ── 状态栏 ──
        cv2.rectangle(frame, (0, 0), (actual_w, 28), (30, 30, 30), -1)
        cv2.putText(frame, status_line, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("1.Original", frame)

        # 中间结果窗口
        if "bird" in dbg and dbg["bird"] is not None:
            bird_bgr = cv2.cvtColor(dbg["bird"], cv2.COLOR_GRAY2BGR)
            cv2.imshow("2.Warp (birdseye)", cv2.resize(bird_bgr, (320, 400), interpolation=cv2.INTER_NEAREST))
        if "binary_raw" in dbg and dbg["binary_raw"] is not None:
            b_raw = cv2.cvtColor(dbg["binary_raw"], cv2.COLOR_GRAY2BGR)
            cv2.imshow("3.Adaptive (binary)", cv2.resize(b_raw, (320, 400), interpolation=cv2.INTER_NEAREST))
        if vis_bird is not None:
            cv2.imshow("4.Close+Fit", cv2.resize(vis_bird, (320, 400), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
