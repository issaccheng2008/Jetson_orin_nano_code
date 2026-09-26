"""V1 = V0's band-scanning detection heuristics + IPM birdseye warp.

Key differences from V0:
  - Warps BGR to 320x400 birdseye at start of process(), then all processing on birdseye
  - No camera LUT — uniform cm_per_px on birdseye
  - Band definitions adapted for 400px birdseye (down 266-398, mid 132-264, up 0-130)
  - No PID/steer/lost controller code — pure vision pipeline
  - No JSON config loading — all params are hardcoded defaults
  - Constructor: V1(cam_w=1280, cam_h=720, cam_height_cm=32.5, cam_vfov_deg=55.876)
"""

import cv2
import numpy as np
import math

from utils import clamp


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════


def median(vals):
    n = len(vals)
    if n == 0:
        return None
    s = sorted(vals)
    m = n // 2
    if n & 1:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])


def stdev(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    var = 0.0
    for v in vals:
        d = v - mu
        var += d * d
    return math.sqrt(var / (n - 1))


def line_fit(ys, xs):
    """Least-squares line fit: x = a * y + b. Returns (a, b)."""
    n = len(xs)
    if n < 2:
        return 0.0, xs[0]
    mean_y = sum(ys) / n
    mean_x = sum(xs) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dy = ys[i] - mean_y
        num += dy * (xs[i] - mean_x)
        den += dy * dy
    if den == 0:
        return 0.0, mean_x
    a = num / den
    b = mean_x - a * mean_y
    return a, b


def confidence_weighted_ema(previous, fused, alpha, confidence):
    """Blend a fused reading into the error EMA, in proportion to confidence.

    The fusion never reads conf, so without this a frame the detector rates at
    0.07 moves the error exactly as hard as one it rates at 0.99. A 2026-09-26
    run had such a frame alone drive fused_err to a saturated birdseye-edge
    reading and err_cm to -30 two frames after a clean +1.0 cm one; the
    controller obeyed at full right lock and the line was never re-acquired.
    """
    update = (1.0 - alpha) * clamp(confidence, 0.0, 1.0)
    return (1.0 - update) * previous + update * fused


# ═══════════════════════════════════════════════════════════════════════
# LineDetector V1
# ═══════════════════════════════════════════════════════════════════════

class LineDetector:
    def __init__(self, cam_w=1280, cam_h=720, cam_height_cm=32.5, cam_pitch_deg=45.0, cam_vfov_deg=55.876,
                 z_calib=None):
        # ── Camera params ──
        self.cam_w = int(cam_w)
        self.cam_h = int(cam_h)
        self.cam_height = float(cam_height_cm)
        self.cam_pitch = np.radians(cam_pitch_deg)
        self.cam_vfov_deg = float(cam_vfov_deg)

        # ── 距离线性校正（系数在 cameras.json，见 scripts/fit_camera_distance.py）──
        if z_calib is None:
            try:
                from camera_config import distance_calib
                z_calib = distance_calib()
            except Exception:
                z_calib = (1.0, 0.0)
        self.z_a, self.z_b = float(z_calib[0]), float(z_calib[1])

        # ── Birdseye ──
        self.bird_h = 400
        self.bird_w = 320
        self.center_x = self.bird_w // 2  # 160

        # Build IPM matrix (same as V2/V3)
        self.M = self._build_birdseye_matrix(lookahead=(10.0, 80.0))
        self.cm_per_px = self._compute_cm_per_px()
        self.z_per_px = (80.0 - 20.0) / float(self.bird_h - 1)  # vertical cm per px
        self._asp = self.cm_per_px / self.z_per_px  # pixel aspect ratio (~1.84)

        # fused_err is the dimensionless near_err_px / (0.5*bird_w). Every consumer
        # (PID gains, STEP_LEN_CM, steer_full_scale_cm) is calibrated in cm, so the
        # conversion is published alongside it as fused_err_cm.
        self.err_scale_cm = 0.5 * self.bird_w * self.cm_per_px

        # Pinhole intrinsics (16:9, square pixels) — exact red-bar distance
        vfov_rad = np.radians(self.cam_vfov_deg)
        hfov_rad = 2.0 * np.arctan(np.tan(vfov_rad / 2.0) * self.cam_w / self.cam_h)
        self.fx_px = self.cam_w / (2.0 * np.tan(hfov_rad / 2.0))
        self.fy_px = self.cam_h / (2.0 * np.tan(vfov_rad / 2.0))
        self.cx_px = self.cam_w / 2.0
        self.cy_px = self.cam_h / 2.0

        # ── Threshold params ──
        self.th_offset = -12  # 反光把线打成亮斑时放宽，让不够黑的也进得来
        self.th_min = 25
        self.th_max = 80
        self.dark_margin = 24

        # ── ROI general params ──
        self.min_track_width = 24
        self.max_track_width = 300    # narrow bands, max plausible width
        self.min_pair_ratio = 0.35    # easier pair matching in tight band
        self.width_std_max = 20       # tighter band → lower width variance
        self.conf_min = 0.12
        self.min_line_width = 4
        self.max_line_width = 120
        self.lane_width_init_px = 140.0
        self.lane_width_tol_px = 50.0  # tighter band → tighter width tolerance
        self.max_center_jump_px = 45.0  # 50px band, can't jump beyond band height
        self.min_pair_lines = 2

        # ── Narrow gate detection ──
        self.narrow_gate_exit_ratio = 1.15
        self.narrow_red_enter_z_cm = 70.0    # 进入窄门时红条约在此距离内
        self.narrow_red_close_z_cm = 20.0    # red bar z < this → exiting
        self.narrow_line_close_z_cm = 50.0   # start line z < this → exiting

        # ── Simple Bottom Mode ──
        self.simple_bottom_mode = True
        self.bottom_start_ratio = 0.875   # y=350, bottom 1/8
        self.bottom_rows = 5
        self.bottom_step = 2
        self.single_line_conf = 0.30

        # ── 质心定位（阈值法配不成对时兜底）──
        # 走路抖动把线糊浅后，硬阈值可能整行取不到 run。匀速模糊的核近似对称，
        # 对称核不改变一阶矩，所以亮度凹陷的质心仍是无偏的线中心。
        self.centroid_conf = 0.85
        self.centroid_min_contrast = 5.0   # 灰度凹陷峰值下限，低于此认为没线

        # ── Two-band direction detection (lower 2 of 8 layers: 300-349, 350-399) ──
        self.two_band_mode = True
        # Band row ranges (400px birdseye, 50px per layer)
        self.band_low_y0 = 350   # layer 7 — near positioning (bottom 1/8)
        self.band_low_y1 = 399
        self.band_mid_y0 = 300   # layer 6 — lookahead / curve detection
        self.band_mid_y1 = 349
        # Band scan params
        self.band_rows_low = 8
        self.band_rows_mid = 8
        self.band_step_low = 2
        self.band_step_mid = 2
        # Band weights
        self.band_weight_low = 0.65
        self.band_weight_mid = 0.35

        # ── Obstacle detection ──
        self.cross_black_run_ratio = 0.25  # 鸟瞰图横线窄, 降低门槛
        self.cross_black_cover_ratio = 0.20
        self.red_detect_enable = True
        # HSV 判红：黑白地板上唯一带色调的物体，只看 Hue+饱和度，
        # 不管明暗（暗红/亮红都能检出；灰色地板 S≈0 被排除）
        self.red_h_max = 15         # 红色 Hue 上限（H≤15 含橙红）
        self.red_h_min = 165        # 红色 Hue 下限（跨 180 边界）
        self.red_s_min = 70         # 饱和度下限（灰色地板 S≈0）
        self.red_v_min = 40         # 亮度下限（纯黑排除）
        self.red_min_pixels = 50    # 红像素面积下限（滤噪点）
        self.red_bar_aspect_min = 1.5  # 红条 bbox 宽/高下限（横条；方块噪点排除）
        self.red_row_ratio = 0.35

        # ── Red bar detection ──
        self.red_bar_confirm_frames = 4   # 连续确认帧数（去抖）
        self.red_bar_release_frames = 4   # 连续丢失帧数才释放（迟滞，防抖动）

        # ── Bottom lock ──
        self.bottom_lock_enable = True
        self.bottom_lock_start_ratio = 0.875  # y=350, bottom 1/8
        self.bottom_lock_rows = 10
        self.bottom_lock_step = 2
        self.bottom_lock_min_pair_ratio = 0.55
        self.bottom_lock_sym_tol_px = 24.0
        self.bottom_lock_blend = 0.72  # stronger lock anchor, less band-scan bias in curves
        self.bottom_lock_conf_penalty = 0.45
        self.lock_reacquire_reset = True

        # ── Startup ──
        self.startup_settle_frames = 25
        self.startup_conf_min_scale = 0.70
        self.startup_min_weight_scale = 0.70
        self.startup_force_simple_bottom = True

        # ── Fusion params ──
        self.smooth_alpha = 0.72         # narrower bands → more noise → more smoothing
        self.min_weight = 0.10

        # ── Pixel domain gains（窄带适配：50px band separation, less lookahead）──
        self.pix_lookahead_gain = 0.25   # far band closer, less curvature info
        self.pix_curve_gain = 0.18       # narrower band → weaker curve signal
        self.pix_angle_gain = 0.15 / self._asp  # compensated: angle×gain unchanged
        self.curve_switch_px = 18.0      # ~1/3 of 50px band separation
        self.curve_angle_deg = 8.0       # fitted heading past which one band is a curve
        self.left_curve_outward_gain = 0.35
        self.left_curve_outward_px = 6.0

        # ── Shake robust ──
        self.robust_enable = True
        self.robust_diff_window = 5
        self.robust_diff_rms_trigger_px = 8.0  # narrower band → lower diff tolerance
        self.robust_alpha_high = 0.88
        self.robust_bottom_lock_blend_scale = 1.5
        self.robust_decay_frames = 8

        # ── Internal state ──
        self._state = self._initial_state()

    def _initial_state(self):
        """Every cross-frame memory, at its as-just-started value."""
        return {
            "smoothed_err": 0.0,
            "lost_frames": 0,
            "last_base_err": 0.0,
            "last_angle_err": 0.0,
            "last_far_dist": 0.0,
            "last_lane_center_x": float(self.center_x),
            "last_lane_width_px": float(self.lane_width_init_px),
            "last_band_mask": 0,
            "startup_frames": 0,
            "last_bottom_lock_valid": False,
            "near_err_history": [],
            "shake_active_frames": 0,
            "diff_rms_px": 0.0,
            "red_bar_count": 0,
            "red_bar_miss": 0,
            "red_bar_cx": 0.0,
            "red_bar_cy": 0.0,
            "red_bar_z_cm": 0.0,
        }

    def reset_state(self):
        """Drop every cross-frame memory, so the next frame starts from scratch.

        Two of these are self-reinforcing and are why a detector that locks onto the
        wrong thing stays locked: `last_lane_center_x` is the next frame's scan hint,
        and `smoothed_err` is an EMA with a long memory. The fallback path also
        republishes `last_base_err` / `last_angle_err` verbatim. A fresh process on
        the same curve tracks fine, so a fresh state should behave the same.
        """
        self._state = self._initial_state()

    # ═══════════════════════════════════════════════════════════
    # Birdseye matrix (IPM: pinhole back-projection of ground plane)
    # ═══════════════════════════════════════════════════════════

    def _build_birdseye_matrix(self, lookahead):
        """IPM (Inverse Perspective Mapping):
        1. Define ground-plane rectangle in physical coords
        2. Project to image via pinhole model -> trapezoid src
        3. dst is regular rectangle -> getPerspectiveTransform
        """
        near, far = lookahead
        near = max(near, 20.0)

        # Pinhole camera params (square pixels -> fx=fy)
        vfov_rad = np.radians(self.cam_vfov_deg)
        hfov_rad = 2.0 * np.arctan(np.tan(vfov_rad / 2.0) * self.cam_w / self.cam_h)
        fx = self.cam_w / (2.0 * np.tan(hfov_rad / 2.0))
        fy_calc = self.cam_h / (2.0 * np.tan(vfov_rad / 2.0))
        cx = self.cam_w / 2.0
        cy = self.cam_h / 2.0

        # Ground rectangle corners
        ground_w_far = 2.0 * far * np.tan(hfov_rad / 2.0)
        W = ground_w_far * 0.7

        world_pts = np.float32([
            [W / 2, near], [-W / 2, near],   # near right, near left
            [-W / 2, far], [W / 2, far],      # far left, far right
        ])

        # Project world -> image (pinhole model)
        cp = np.cos(self.cam_pitch)
        sp = np.sin(self.cam_pitch)
        src_pts = []
        for wx, wz in world_pts:
            Xc = wx
            Yc = self.cam_height * cp - wz * sp
            Zc = self.cam_height * sp + wz * cp
            if Zc < 0.01:
                Zc = 0.01
            u = fx * Xc / Zc + cx
            v = fy_calc * Yc / Zc + cy
            src_pts.append([u, v])
        src = np.float32([[clamp(p[0], 0, self.cam_w-1),
                           clamp(p[1], 0, self.cam_h-1)] for p in src_pts])

        # dst rectangle (near=bottom, far=top)
        dst = np.float32([
            [self.bird_w - 1, self.bird_h - 1], [0, self.bird_h - 1],  # near -> bottom
            [0, 0], [self.bird_w - 1, 0],                                # far -> top
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _compute_cm_per_px(self):
        """Horizontal cm per pixel (同 _build_birdseye_matrix 的 W 公式)"""
        hfov_rad = 2 * np.arctan(
            np.tan(np.radians(self.cam_vfov_deg / 2)) * self.cam_w / self.cam_h)
        far = 80.0
        ground_w_far = 2.0 * far * np.tan(hfov_rad / 2.0)
        W = ground_w_far * 0.7
        return W / self.bird_w

    def _to_true_z(self, z_model):
        """相机模型读数 → 地面真值 cm（系数在 cameras.json 的 distance_calib）。"""
        return self.z_a * z_model + self.z_b

    def _px_to_ground_cm(self, x, y):
        """Convert birdseye pixel (x, y) to ground cm.
        x_cm: horizontal offset from center (positive = right)
        z_cm: forward distance from robot
        """
        x_cm = (x - self.center_x) * self.cm_per_px
        z_cm = 20.0 + (self.bird_h - 1 - y) * self.z_per_px
        return x_cm, self._to_true_z(z_cm)

    # ═══════════════════════════════════════════════════════════
    # Otsu adaptive threshold
    # ═══════════════════════════════════════════════════════════

    def _otsu_threshold(self, gray):
        """Manual Otsu — same algorithm as V0 (not cv2.THRESH_OTSU)."""
        hist = [0] * 256
        h, w = gray.shape
        step_y = max(1, h // 30)
        step_x = max(1, w // 40)
        total = 0

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                g = int(gray[y, x])
                hist[g] += 1
                total += 1

        if total == 0:
            return 64

        sum_all = sum(i * hist[i] for i in range(256))

        sum_b = 0
        w_b = 0
        max_var = -1.0
        best_t = 64

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f
            d = m_b - m_f
            var_between = w_b * w_f * d * d
            if var_between > max_var:
                max_var = var_between
                best_t = t

        return best_t

    # ═══════════════════════════════════════════════════════════
    # Obstacle detection
    # ═══════════════════════════════════════════════════════════

    def _detect_row_blocker(self, gray, bgr, y, x0, x1, black_th, track_is_dark):
        n = max(1, x1 - x0 + 1)
        red_block = False
        black_block = False

        if self.red_detect_enable:
            row = bgr[y:y + 1, x0:x1 + 1]
            hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
            hh = hsv[:, :, 0].astype(np.int32)
            ss = hsv[:, :, 1].astype(np.int32)
            vv = hsv[:, :, 2].astype(np.int32)
            is_red = (((hh <= self.red_h_max) | (hh >= self.red_h_min))
                      & (ss >= self.red_s_min) & (vv >= self.red_v_min))
            if np.count_nonzero(is_red) / n >= self.red_row_ratio:
                red_block = True

        if not red_block:
            row_g = gray[y, x0:x1 + 1]
            if track_is_dark:
                is_track = row_g <= black_th
            else:
                is_track = row_g >= black_th
            padded = np.concatenate(([False], is_track, [False]))
            rises = np.where(np.diff(padded.astype(np.int8)) == 1)[0]
            falls = np.where(np.diff(padded.astype(np.int8)) == -1)[0]
            if len(rises) > 0:
                longest = int(np.max(falls - rises + 1))
                cover = int(np.sum(is_track))
                if (longest >= self.cross_black_run_ratio * n and
                        cover >= self.cross_black_cover_ratio * n):
                    black_block = True

        return red_block, black_block

    def _detect_start_line(self, gray, black_th, track_is_dark):
        """Find horizontal black start-line position in birdseye. Returns y or None."""
        y0 = int(self.bird_h * 0.5)
        y1 = self.bird_h - 1
        best_y, best_score = None, 0.0
        step = max(1, (y1 - y0) // 20)
        for y in range(y0, y1, step):
            row = gray[y, :]
            if track_is_dark:
                is_black = row <= black_th
            else:
                is_black = row >= black_th
            padded = np.concatenate(([False], is_black, [False]))
            rises = np.where(np.diff(padded.astype(np.int8)) == 1)[0]
            falls = np.where(np.diff(padded.astype(np.int8)) == -1)[0]
            if len(rises) > 0:
                longest = np.max(falls - rises + 1)
                cover = np.sum(is_black)
                row_score = longest / max(1, self.bird_w)
                if row_score > 0.25 and cover > 40 and row_score > best_score:
                    best_score = row_score
                    best_y = y
        return best_y

    def _detect_red_bar(self, bgr):
        """Find red bar in the ORIGINAL image.

        Returns (cx, v_foot, z_cm) or None:
          cx     - horizontal centroid (px, original image)
          v_foot - lowest red row of the largest bar-shaped blob
          z_cm   - exact ground distance of v_foot via pinhole back-projection

        Picks the largest bar-shaped connected component (not every red pixel):
        a stray red speck near the image bottom would otherwise pin v_foot to
        the last row and report the bar as always ~12cm away.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m_lo = cv2.inRange(hsv, (0, self.red_s_min, self.red_v_min),
                           (self.red_h_max, 255, 255))
        m_hi = cv2.inRange(hsv, (self.red_h_min, self.red_s_min, self.red_v_min),
                           (180, 255, 255))
        mask = cv2.bitwise_or(m_lo, m_hi)

        if cv2.countNonZero(mask) < self.red_min_pixels:
            return None

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_area, best_bottom = -1, 0, -1
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.red_min_pixels:
                continue
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if w < self.red_bar_aspect_min * h:
                continue
            bottom = int(stats[i, cv2.CC_STAT_TOP]) + h
            if area > best_area or (area == best_area and bottom > best_bottom):
                best, best_area, best_bottom = i, area, bottom
        if best < 0:
            return None

        x0 = int(stats[best, cv2.CC_STAT_LEFT])
        y0 = int(stats[best, cv2.CC_STAT_TOP])
        w0 = int(stats[best, cv2.CC_STAT_WIDTH])
        h0 = int(stats[best, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(labels[y0:y0 + h0, x0:x0 + w0] == best)
        cx = float(x0 + np.mean(xs))
        v_foot = float(y0 + np.max(ys))  # lowest row = bar near edge
        a = (v_foot - self.cy_px) / self.fy_px
        z_cm = self.cam_height * (np.cos(self.cam_pitch) - a * np.sin(self.cam_pitch)) \
            / (a * np.cos(self.cam_pitch) + np.sin(self.cam_pitch))
        return cx, v_foot, self._to_true_z(z_cm)

    # ═══════════════════════════════════════════════════════════
    # Run collection
    # ═══════════════════════════════════════════════════════════

    def _collect_track_runs_on_row(self, gray, y, x0, x1, black_th, track_is_dark):
        row = gray[y, x0:x1 + 1]
        if track_is_dark:
            mask = row <= black_th
        else:
            mask = row >= black_th

        padded = np.concatenate(([False], mask, [False]))
        rises = np.where(np.diff(padded.astype(np.int8)) == 1)[0] + x0
        falls = np.where(np.diff(padded.astype(np.int8)) == -1)[0] + x0 - 1
        widths = falls - rises + 1
        valid = (widths >= self.min_line_width) & (widths <= self.max_line_width)
        return list(zip(rises[valid].tolist(), falls[valid].tolist()))

    def _centroid_pair_center(self, gray_raw, y, hint_center, lane_width_hint,
                              x0, x1):
        """阈值法配不成对时，直接在鸟瞰灰度上找两条暗凹陷，用质心定中心。

        为什么需要它：扫描输入是黑帽响应再被连通域掩膜削过的（面积<300 或高<80
        就清零），走路模糊会把线打成小碎片，右线常在掩膜那步就没了，于是配不成对。
        而**未经处理的鸟瞰灰度里两条线仍然清楚**。

        为什么质心无偏：匀速运动模糊的核近似对称，对称核不改变一阶矩——线被抬灰、
        边缘展宽，但凹陷的质心仍是真实中心，只是对比度下降。阈值法一过阈就归零。

        不依赖 hint 定位窗口：hint 在线长期丢失时会自己退化成最小值，用它定窗口反而
        会找错地方。这里全行搜凹陷，只把期望线宽当「挑哪一对」的偏好。
        """
        row = gray_raw[y, x0:x1 + 1].astype(np.float32)
        bg = float(np.percentile(row, 60))
        d = np.clip(bg - row, 0.0, None)
        peak = float(d.max())
        if peak < self.centroid_min_contrast:
            return None

        thr = max(self.centroid_min_contrast, 0.25 * peak)
        mask = d >= thr
        segs = []
        i, n = 0, mask.size
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            # 贴着行边的暗段多半是鸟瞰图未定义的黑边，不是线
            if i > 0 and j < n:
                seg = d[i:j]
                wsum = float(seg.sum())
                if wsum > 1e-6:
                    xs = np.arange(i, j, dtype=np.float32)
                    segs.append(float((seg * xs).sum() / wsum))
            i = j
        if len(segs) < 2:
            return None

        want = (lane_width_hint if lane_width_hint >= self.min_track_width * 2
                else self.lane_width_init_px)
        best = None
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                lane_w = segs[b] - segs[a]
                if not (self.min_track_width <= lane_w <= self.max_track_width):
                    continue
                center = 0.5 * (segs[a] + segs[b])
                score = abs(lane_w - want) + 0.3 * abs(center - hint_center)
                if best is None or score < best[0]:
                    best = (score, lane_w, center)

        if best is None:
            return None
        return {
            "center_px": best[2] + x0,
            "lane_width_px": best[1],
            "conf": self.centroid_conf,
            "line_mode": 3,
        }

    # ═══════════════════════════════════════════════════════════
    # Pair selection
    # ═══════════════════════════════════════════════════════════

    def _choose_pair_center_from_runs(self, runs, hint_center, lane_width_hint, x0, x1):
        if len(runs) < 2:
            return None

        best = None
        best_score = 1e9
        for i in range(len(runs)):
            wi = runs[i][1] - runs[i][0] + 1
            if wi < 5:  # too thin to be a real track edge, noise
                continue
            li = 0.5 * (runs[i][0] + runs[i][1])
            for j in range(i + 1, len(runs)):
                wj = runs[j][1] - runs[j][0] + 1
                if wj < 5:
                    continue
                rj = 0.5 * (runs[j][0] + runs[j][1])
                lane_w = rj - li
                if self.min_track_width <= lane_w <= self.max_track_width:
                    if lane_width_hint > 0:
                        max_width_err = max(48.0, self.lane_width_tol_px * 1.6)
                        if abs(lane_w - lane_width_hint) > max_width_err:
                            continue
                    center = 0.5 * (li + rj)
                    if x0 <= center <= x1:
                        width_err = (
                            abs(lane_w - lane_width_hint) if lane_width_hint > 0 else 0.0
                        )
                        center_err = abs(center - hint_center)
                        score = 1.0 * center_err + 0.8 * width_err
                        if score < best_score:
                            best_score = score
                            best = {
                                "center_px": center,
                                "lane_width_px": lane_w,
                                "conf": 1.0,
                                "line_mode": 2,
                            }

        return best

    def _choose_single_run_near_hint(self, runs, hint_center):
        if not runs:
            return None
        best = None
        best_err = 1e9
        for run in runs:
            c = 0.5 * (run[0] + run[1])
            err = abs(c - hint_center)
            if err < best_err:
                best_err = err
                best = run
        return best

    def _infer_center_from_single_run(self, run, lane_width_meas, x0, x1):
        """Place the boundary that is missing from the one run the row still has.

        The side comes from the run's position against the middle of the birdseye, not
        against the centre being tracked. The middle is the column under the camera,
        and the lane centre stays within a few px of it even on the tightest real
        curve - ~12 px at 25 cm ahead on a 77.5 cm radius - while a boundary is half a
        lane (70 px) away, so the discrimination holds until the robot is already off
        the track. Reading the side off the tracked centre instead makes it a function
        of the previous estimate: a centre that has drifted to the wrong side of the
        run confirms its own error frame after frame. Against a constant there is no
        loop to close.
        """
        c = 0.5 * (run[0] + run[1])
        # Only a row that measured both boundaries may set the width: a width taken
        # off an inferred row is the number this function just made up, and feeding it
        # back makes the estimate chase itself.
        w = float(lane_width_meas) if lane_width_meas > 0 else float(self.lane_width_init_px)
        side = "left" if c < self.center_x else "right"
        # Bounded by construction: the side is a constant's comparison, so the centre
        # lands w/2 to that side of the frame middle and w <= max_track_width. The old
        # rule sided off the tracked centre, which can sit anywhere, and clamped the
        # result - reporting the frame edge as a measurement, which is how one row
        # handed the fusion a near_err_px of +/-160 on a 35 cm lane.
        center = c + 0.5 * w if side == "left" else c - 0.5 * w
        return {
            "center_px": center,
            "lane_width_px": w,
            "conf": self.single_line_conf,
            "line_mode": 1,
            "side": side,
        }

    # ═══════════════════════════════════════════════════════════
    # Band scanning (on birdseye)
    # ═══════════════════════════════════════════════════════════

    def _scan_band_midline(self, gray, bgr, black_th, track_is_dark,
                           hint_x, lane_width_hint,
                           y_start_ratio, y_end_ratio, max_rows, row_step,
                           gray_raw=None):
        """Scan a band of rows on the birdseye, find midline per row.

        gray_raw: 未做黑帽/掩膜处理的鸟瞰灰度，只给质心兜底用。gray 上的线是
        「亮峰压在 0 背景上」且被连通域掩膜削过，质心需要的是「灰底上的暗凹陷」。
        """
        row_step = max(1, row_step)
        img_w = self.bird_w
        img_h = self.bird_h
        img_cx = self.center_x
        x0 = 0
        x1 = img_w - 1
        y_start = int(clamp(y_start_ratio * img_h, 0, img_h - 1))
        y_end = int(clamp(y_end_ratio * img_h, 0, img_h - 1))
        if y_end < y_start:
            y_end = y_start

        centers_px = []
        centers_cm = []
        lane_widths = []
        ys = []
        zs_cm = []
        conf_sum = 0.0
        pair_rows = 0
        single_rows = 0
        red_block_rows = 0
        black_block_rows = 0
        left_seen = False
        right_seen = False

        last_center = hint_x
        last_width = lane_width_hint

        rows_done = 0
        y = y_start
        while y <= y_end and rows_done < max_rows:
            red_block, black_block = self._detect_row_blocker(
                gray, bgr, y, x0, x1, black_th, track_is_dark
            )
            if red_block:
                red_block_rows += 1
                rows_done += 1
                y += row_step
                continue
            if black_block:
                black_block_rows += 1
                rows_done += 1
                y += row_step
                continue

            runs = self._collect_track_runs_on_row(
                gray, y, x0, x1, black_th, track_is_dark
            )
            chosen = self._choose_pair_center_from_runs(
                runs, last_center, last_width, x0, x1
            )
            # 阈值配不成对时先试质心：它实打实量出两条线的位置，
            # 信息量高于下面的单线盲推（后者只测到一条线，另一条按线宽硬挪）。
            if chosen is None and gray_raw is not None:
                chosen = self._centroid_pair_center(
                    gray_raw, y, last_center, last_width, x0, x1
                )
            if chosen is None and len(runs) >= 1:
                best_run = self._choose_single_run_near_hint(runs, last_center)
                if best_run is not None:
                    chosen = self._infer_center_from_single_run(
                        best_run, last_width, x0, x1
                    )

            if chosen is not None:
                center_px = chosen["center_px"]
                lane_w = chosen["lane_width_px"]
                # 空间连续性：和上一行的 center 比，跳跃太大说明是噪声
                if len(centers_px) > 0:
                    jump = abs(center_px - last_center)
                    if jump > max(30.0, lane_w * 1.2):
                        rows_done += 1
                        y += row_step
                        continue
                if abs(center_px - last_center) > (self.max_center_jump_px * 2.2):
                    rows_done += 1
                    y += row_step
                    continue
                x_cm, z_cm = self._px_to_ground_cm(center_px, y)
                centers_px.append(center_px)
                centers_cm.append(x_cm)
                lane_widths.append(lane_w)
                ys.append(y)
                zs_cm.append(z_cm)
                conf_sum += chosen["conf"]
                mode = int(chosen.get("line_mode", 1))
                if mode >= 2:
                    # Paired off the runs, or measured as two dips in the raw gray, so
                    # both boundaries were really seen.
                    pair_rows += 1
                    left_seen = right_seen = True
                else:
                    single_rows += 1
                    left_seen = left_seen or chosen.get("side") == "left"
                    right_seen = right_seen or chosen.get("side") == "right"
                last_center = center_px
                if mode >= 2:
                    last_width = lane_w

            rows_done += 1
            y += row_step

        if len(centers_px) < 3:
            return None

        center_px = median(centers_px)
        center_cm = median(centers_cm)
        lane_width_px = median(lane_widths)
        dist_cm = median(zs_cm)
        width_std = stdev(lane_widths)
        a, _ = line_fit(ys, centers_px)
        angle = math.degrees(math.atan(a * self._asp))  # pixel→physical

        hit_ratio = len(centers_px) / float(max(1, max_rows))
        conf_raw = (conf_sum / float(max(1, len(centers_px)))) * hit_ratio
        conf = conf_raw * (1.0 - clamp(width_std / max(self.width_std_max, 1e-6), 0.0, 1.0))
        blocker_ratio = (red_block_rows + black_block_rows) / float(max(1, max_rows))
        if blocker_ratio > 0.25:
            conf *= (1.0 - 0.55 * clamp((blocker_ratio - 0.25) / 0.75, 0.0, 1.0))

        valid_rows = max(1, pair_rows + single_rows)
        pair_ratio = pair_rows / float(valid_rows)
        single_ratio = single_rows / float(valid_rows)

        return {
            "center_cm": center_cm,
            "center_px": center_px,
            "dist_cm": dist_cm,
            "lane_width_px": lane_width_px,
            "weight": 1.0,
            "angle": angle,
            "conf": conf,
            "pair_ratio": pair_ratio,
            "single_ratio": single_ratio,
            "red_block_ratio": red_block_rows / float(max(1, max_rows)),
            "black_block_ratio": black_block_rows / float(max(1, max_rows)),
            "ys_list": ys,
            "centers_list": centers_px,
            "left_seen": left_seen,
            "right_seen": right_seen,
            "single_side": ("left" if left_seen and not right_seen
                            else "right" if right_seen and not left_seen
                            else None),
        }

    # ═══════════════════════════════════════════════════════════
    # Simple bottom midline + assist band
    # ═══════════════════════════════════════════════════════════

    def _bottom_quarter_midline(self, gray, bgr, black_th, track_is_dark,
                                hint_x, lane_width_hint, gray_raw=None):
        base = self._scan_band_midline(
            gray, bgr, black_th, track_is_dark,
            hint_x, lane_width_hint,
            self.bottom_start_ratio, 1.0,
            self.bottom_rows, self.bottom_step,
            gray_raw=gray_raw,
        )
        if base is None:
            return None

        return base

    # ═══════════════════════════════════════════════════════════
    # Two-band cascade detection (hard-coded birdseye rows)
    # ═══════════════════════════════════════════════════════════

    def _detect_two_band_lanes(self, gray, bgr, black_th, track_is_dark,
                                hint_x, lane_width_hint, gray_raw=None):
        """Scan two bands on birdseye: low(350-399), mid(300-349)."""
        band_specs = [
            ("low", self.band_low_y0 / float(self.bird_h),
                    self.band_low_y1 / float(self.bird_h),
             self.band_rows_low, self.band_step_low, self.band_weight_low),
            ("mid", self.band_mid_y0 / float(self.bird_h),
                    self.band_mid_y1 / float(self.bird_h),
             self.band_rows_mid, self.band_step_mid, self.band_weight_mid),
        ]

        results = []
        last_center = hint_x
        last_width = lane_width_hint
        for name, ys, ye, rows, step, weight in band_specs:
            res = self._scan_band_midline(
                gray, bgr, black_th, track_is_dark,
                last_center, last_width,
                ys, ye, rows, step,
                gray_raw=gray_raw,
            )
            if res is None:
                continue
            res["weight"] = weight
            res["band_name"] = name
            results.append(res)
            last_center = res["center_px"]
            last_width = res["lane_width_px"]

        return results

    # ═══════════════════════════════════════════════════════════
    # Bottom center lock (on birdseye)
    # ═══════════════════════════════════════════════════════════

    def _detect_bottom_center_lock(self, gray, bgr, black_th, track_is_dark):
        if not self.bottom_lock_enable:
            return {
                "valid": True,
                "quality": 1.0,
                "pair_ratio": 1.0,
                "center_px": float(self.center_x),
                "center_err_px": 0.0,
                "symmetry_abs_px": 0.0,
                "centers_list": [],
                "center_ys": [],
            }

        img_w = self.bird_w
        img_h = self.bird_h
        img_cx = self.center_x
        x0 = 0
        x1 = img_w - 1
        y_start = int(clamp(self.bottom_lock_start_ratio * img_h, 0, img_h - 1))
        row_step = max(1, self.bottom_lock_step)

        pair_rows = 0
        rows_done = 0
        centers = []
        center_ys = []

        y = y_start
        while y < img_h and rows_done < max(1, self.bottom_lock_rows):
            red_block, black_block = self._detect_row_blocker(
                gray, bgr, y, x0, x1, black_th, track_is_dark
            )
            if red_block or black_block:
                rows_done += 1
                y += row_step
                continue

            runs = self._collect_track_runs_on_row(
                gray, y, x0, x1, black_th, track_is_dark
            )
            chosen = self._choose_pair_center_from_runs(
                runs, img_cx, 0.0, x0, x1
            )
            if chosen is not None:
                pair_rows += 1
                centers.append(float(chosen["center_px"]))
                center_ys.append(y)

            rows_done += 1
            y += row_step

        if rows_done <= 0 or not centers:
            return {
                "valid": False,
                "quality": 0.0,
                "pair_ratio": 0.0,
                "center_px": float(img_cx),
                "center_err_px": 0.0,
                "symmetry_abs_px": float(img_w),
                "centers_list": [],
                "center_ys": [],
            }

        pair_ratio = pair_rows / float(rows_done)
        center_px = float(median(centers))
        center_err_px = center_px - float(img_cx)
        symmetry_abs_px = abs(center_err_px)

        pair_q = clamp(
            (pair_ratio - self.bottom_lock_min_pair_ratio)
            / max(1.0 - self.bottom_lock_min_pair_ratio, 1e-6),
            0.0,
            1.0,
        )
        sym_q = 1.0 - clamp(
            symmetry_abs_px / max(self.bottom_lock_sym_tol_px * 2.0, 1e-6), 0.0, 1.0
        )
        quality = clamp(0.65 * pair_q + 0.35 * sym_q, 0.0, 1.0)
        valid = (
            pair_ratio >= self.bottom_lock_min_pair_ratio
            and symmetry_abs_px <= self.bottom_lock_sym_tol_px
        )

        return {
            "valid": valid,
            "quality": quality,
            "pair_ratio": pair_ratio,
            "center_px": center_px,
            "center_err_px": center_err_px,
            "symmetry_abs_px": symmetry_abs_px,
            "centers_list": centers,
            "center_ys": center_ys,
        }

    # ═══════════════════════════════════════════════════════════
    # Band helpers
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _band_bit(name):
        if name == "low":
            return 0x1
        if name == "mid":
            return 0x2
        return 0

    @staticmethod
    def _single_band_mask(mask):
        return mask in (0x1, 0x2)

    @staticmethod
    def _pick_result_by_band(results, order):
        for name in order:
            for r in results:
                if str(r.get("band_name", "")) == name:
                    return r
        return None

    def _result_quality_weight(self, r):
        pair_ratio = float(r.get("pair_ratio", 0.0))
        single_ratio = float(r.get("single_ratio", 1.0 - pair_ratio))
        q = 0.60 + 0.40 * pair_ratio
        if pair_ratio < self.min_pair_ratio:
            q *= 0.80
        q *= (1.0 - 0.12 * clamp(single_ratio, 0.0, 1.0))
        return clamp(q, 0.20, 1.00)

    # ═══════════════════════════════════════════════════════════
    # Main process entry
    # ═══════════════════════════════════════════════════════════

    def process(self, bgr):
        """Process one BGR frame (cam_w x cam_h).

        Returns:
            dev_px:      lateral deviation in birdseye px (positive = track center right of robot)
            heading_deg: heading angle in degrees
            conf:        confidence 0-1
            vis:         BGR visualization (bird_w x bird_h)
            debug:       diagnostic info dict
        """
        state = self._state
        state["startup_frames"] += 1

        # ── Step 1: Warp to birdseye (single warp, derive gray on birdseye) ──
        bgr_bird = cv2.warpPerspective(bgr, self.M, (self.bird_w, self.bird_h))
        # Custom grayscale on birdseye: max of max(R,G,B) and standard grayscale
        gray_max = np.max(bgr_bird, axis=2)
        gray_std = cv2.cvtColor(bgr_bird, cv2.COLOR_BGR2GRAY)
        gray = np.maximum(gray_max, gray_std)

        # Black hat: suppress wide shadows, enhance thin dark lines → bright
        k31 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        gray_detect = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k31)

        img_w = self.bird_w
        img_h = self.bird_h
        img_cx = self.center_x

        # ── Step 2: Adaptive threshold (Gaussian + Otsu fallback) ──
        # Otsu 全局阈值（保留作为参考）
        otsu_th = self._otsu_threshold(gray_detect)

        # 高斯自适应阈值（主力，对 black-hat 结果操作：线已变亮）
        adaptive_binary = cv2.adaptiveThreshold(
            gray_detect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -8  # blockSize=31, C=-8 (stricter)
        )
        # Black-hat 后线变亮 → THRESH_BINARY 把线判为 255
        adaptive_mask = (adaptive_binary == 255)
        if np.count_nonzero(adaptive_mask) > 100:
            black_th = np.median(gray_detect[adaptive_mask]) + self.th_offset
        else:
            black_th = otsu_th + self.th_offset

        # 限幅
        black_th = clamp(black_th, self.th_min, self.th_max)

        # ── Clean gray_detect: morphology + CC on the detection input ──
        _, binary_clean = cv2.threshold(gray_detect, black_th, 255, cv2.THRESH_BINARY)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, k5, iterations=1)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, k5, iterations=1)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, k5, iterations=1)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, k3, iterations=1)
        # Connected component filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            if area < 300 or h < 80:
                binary_clean[labels == label_id] = 0
        # Apply mask: noise pixels → 0, track pixels keep original value
        gray_detect[binary_clean == 0] = 0

        # ── Step 3: Track color detection ──
        # After black-hat, lines are always bright → track_is_dark=False
        track_is_dark = False

        # ── Startup transient params ──
        startup_active = (
            self.startup_settle_frames > 0
            and state["startup_frames"] < self.startup_settle_frames
        )
        if startup_active:
            conf_min_dyn = self.conf_min * clamp(self.startup_conf_min_scale, 0.20, 1.00)
            min_weight_dyn = self.min_weight * clamp(self.startup_min_weight_scale, 0.20, 1.00)
        else:
            conf_min_dyn = self.conf_min
            min_weight_dyn = self.min_weight

        # ── Scan hint ──
        if startup_active:
            scan_hint_center = float(img_cx)
            scan_hint_width = 0.0
        else:
            scan_hint_center = state["last_lane_center_x"]
            scan_hint_width = state["last_lane_width_px"]

        # ── Run detectors ──
        roi_results = []
        if startup_active and self.startup_force_simple_bottom:
            res = self._bottom_quarter_midline(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray,
            )
            if res is not None and res["conf"] >= conf_min_dyn:
                roi_results.append(res)
            elif self.two_band_mode:
                roi_results = self._detect_two_band_lanes(
                    gray_detect, bgr_bird, black_th, track_is_dark,
                    scan_hint_center, scan_hint_width, gray_raw=gray,
                )
                roi_results = [r for r in roi_results if r["conf"] >= conf_min_dyn]
        elif self.two_band_mode:
            roi_results = self._detect_two_band_lanes(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray,
            )
            roi_results = [r for r in roi_results if r["conf"] >= conf_min_dyn]
        elif self.simple_bottom_mode:
            res = self._bottom_quarter_midline(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray,
            )
            if res is not None and res["conf"] >= conf_min_dyn:
                roi_results.append(res)

        # ── Initialize outputs ──
        base_err_px = 0.0
        base_err_cm = 0.0
        angle_err = 0.0
        far_dist_cm = 0.0
        avg_conf = 0.0
        band_mask = 0
        red_block_score = 0.0
        black_block_score = 0.0
        bottom_pair_ratio = 0.0
        bottom_sym_err_px = 0.0
        center_lock_quality = 1.0
        bottom_lock_valid = True
        left_seen = False
        right_seen = False
        single_line = False
        single_side = None
        near_err_px_pre_lock = 0.0
        far_err_px_saved = 0.0
        curve_px = 0.0
        turn_gate = 0.0

        # ── Bottom center lock ──
        bottom_lock = self._detect_bottom_center_lock(
            gray_detect, bgr_bird, black_th, track_is_dark,
        )
        bottom_pair_ratio = float(bottom_lock.get("pair_ratio", 0.0))
        bottom_sym_err_px = float(bottom_lock.get("center_err_px", 0.0))
        center_lock_quality = float(bottom_lock.get("quality", 0.0))
        bottom_lock_valid = bool(bottom_lock.get("valid", False))

        if (
            self.lock_reacquire_reset
            and bottom_lock_valid
            and (not state["last_bottom_lock_valid"])
        ):
            state["smoothed_err"] *= 0.35
        state["last_bottom_lock_valid"] = bottom_lock_valid

        # ── Filter by total weight ──
        if roi_results:
            score_total = 0.0
            for r in roi_results:
                score_total += r["weight"] * r["conf"] * self._result_quality_weight(r)
            if score_total <= min_weight_dyn:
                roi_results = []

        # ── Red bar detection ──
        if self.red_detect_enable:
            res = self._detect_red_bar(bgr)
            if res is not None:
                state["red_bar_cx"], state["red_bar_cy"], state["red_bar_z_cm"] = res
                state["red_bar_count"] = min(state["red_bar_count"] + 1, self.red_bar_confirm_frames + 1)
                state["red_bar_miss"] = 0
            else:
                state["red_bar_miss"] += 1
                if state["red_bar_miss"] >= self.red_bar_release_frames:
                    state["red_bar_count"] = 0
        red_bar_detected = state["red_bar_count"] >= self.red_bar_confirm_frames
        # While confirmed, hold the last valid position through single-frame dropouts
        red_bar_cx, red_bar_cy, red_bar_z_cm = 0.0, 0.0, 0.0
        red_bar_x_cm = 0.0
        if red_bar_detected:
            red_bar_cx = state["red_bar_cx"]
            red_bar_cy = state["red_bar_cy"]
            red_bar_z_cm = state["red_bar_z_cm"]
            red_bar_x_cm = (red_bar_cx - self.cx_px) / self.fx_px * red_bar_z_cm

        # Defaults for lost-frame path
        ng_exit_z = 0.0
        ng_enter_z = 0.0
        inside_narrow = False
        ratio_out = False
        red_visible = False
        line_visible = False

        # ── Pixel-domain error fusion ──
        if roi_results:
            state["lost_frames"] = 0

            near = self._pick_result_by_band(roi_results, ("low", "mid"))
            if near is None:
                near = min(roi_results, key=lambda r: r["dist_cm"])

            left_seen = bool(near.get("left_seen", False))
            right_seen = bool(near.get("right_seen", False))
            single_line = left_seen != right_seen
            single_side = "left" if left_seen else "right" if right_seen else None

            far = self._pick_result_by_band(roi_results, ("mid", "low"))
            if far is None:
                far = max(roi_results, key=lambda r: r["dist_cm"])

            for r in roi_results:
                bn = str(r.get("band_name", ""))
                band_mask |= self._band_bit(bn)
                red_block_score = max(
                    red_block_score, float(r.get("red_block_ratio", 0.0))
                )
                black_block_score = max(
                    black_block_score, float(r.get("black_block_ratio", 0.0))
                )
            state["last_band_mask"] = band_mask

            near_err_cm = near["center_cm"]
            far_err_cm = far["center_cm"]
            near_err_px = near["center_px"] - img_cx
            far_err_px = far["center_px"] - img_cx
            far_err_px_saved = far_err_px  # raw far error for controller (pre-assist)

            # If near band missing (only mid visible), blend with history
            if str(near.get("band_name", "")) != "low":
                near_err_cm = 0.68 * near_err_cm + 0.32 * state["last_base_err"]
                near_err_px = 0.68 * near_err_px + 0.32 * (
                    state["last_lane_center_x"] - img_cx
                )
            near_err_px_pre_lock = near_err_px  # after historic blend, before lock fusion

            # Shake robust layer
            shake_active = self.robust_enable and (state["shake_active_frames"] > 0)
            alpha_eff = self.robust_alpha_high if shake_active else self.smooth_alpha
            lock_blend_scale = (
                self.robust_bottom_lock_blend_scale if shake_active else 1.0
            )

            # Bottom lock fusion
            if bottom_pair_ratio > 0.0:
                lock_gain = (
                    (self.bottom_lock_blend * lock_blend_scale)
                    * (0.55 + 0.45 * center_lock_quality)
                )
                lock_gain = clamp(lock_gain, 0.0, 0.95)
                near_err_px = (
                    1.0 - lock_gain
                ) * near_err_px + lock_gain * bottom_sym_err_px
                near_err_cm = near_err_px * self.cm_per_px

            far_dist_cm = far["dist_cm"]
            state["last_lane_center_x"] = clamp(
                float(img_cx + near_err_px), 0.0, float(img_w - 1)
            )
            state["last_lane_width_px"] = clamp(
                float(near["lane_width_px"]),
                float(self.min_track_width),
                float(self.max_track_width),
            )

            base_err_cm = near_err_cm
            base_err_px = near_err_px

            use_assist = False
            if self.simple_bottom_mode and ("assist_center_px" in near):
                assist_conf = float(near.get("assist_conf", 0.0))
                assist_delta = abs(
                    float(near["assist_center_px"]) - float(near["center_px"])
                )
                if assist_conf >= max(self.conf_min, 0.28) and assist_delta <= (
                    self.max_center_jump_px * 1.2
                ):
                    use_assist = True

            if use_assist:
                far_err_px = float(near["assist_center_px"]) - img_cx
                far_err_cm = float(near.get("assist_center_cm", near_err_cm))
                far_dist_cm = float(near.get("assist_dist_cm", far_dist_cm))

            # ── Unified heading fit: merge all center points from both bands + bottom lock ──
            all_ys = []
            all_cx = []
            all_w  = []

            if "ys_list" in near and "centers_list" in near:
                yl = near["ys_list"]
                cl = near["centers_list"]
                all_ys.extend(yl)
                all_cx.extend(cl)
                w = float(near.get("pair_ratio", 0.5))
                all_w.extend([w] * len(yl))

            if "ys_list" in far and "centers_list" in far:
                yl = far["ys_list"]
                cl = far["centers_list"]
                all_ys.extend(yl)
                all_cx.extend(cl)
                w = float(far.get("pair_ratio", 0.5))
                all_w.extend([w] * len(yl))

            if "centers_list" in bottom_lock and "center_ys" in bottom_lock:
                yl = bottom_lock["center_ys"]
                cl = bottom_lock["centers_list"]
                if len(yl) >= 2:
                    all_ys.extend(yl)
                    all_cx.extend(cl)
                    lock_q = float(bottom_lock.get("quality", 0.5))
                    all_w.extend([lock_q * 1.5] * len(yl))

            if len(all_ys) >= 3:
                w_arr = np.array(all_w, dtype=np.float64)
                y_arr = np.array(all_ys, dtype=np.float64)
                x_arr = np.array(all_cx, dtype=np.float64)
                w_sum = float(np.sum(w_arr))
                if w_sum > 1e-9:
                    w_arr /= w_sum
                    y_mean = float(np.average(y_arr, weights=w_arr))
                    x_mean = float(np.average(x_arr, weights=w_arr))
                    num = float(np.sum(w_arr * (y_arr - y_mean) * (x_arr - x_mean)))
                    den = float(np.sum(w_arr * (y_arr - y_mean) ** 2))
                    if den > 1e-9:
                        a = num / den
                        angle_err = math.degrees(math.atan(a * self._asp))
                    else:
                        angle_err = 0.5 * (near["angle"] + far["angle"])
                else:
                    a, _ = line_fit(all_ys, all_cx)
                    angle_err = math.degrees(math.atan(a * self._asp))
            else:
                if use_assist:
                    angle_err = 0.5 * (
                        near["angle"]
                        + float(near.get("assist_angle_deg", near["angle"]))
                    )
                else:
                    angle_err = 0.5 * (near["angle"] + far["angle"])

            curve_px = far_err_px - near_err_px

            # ── Narrow gate detection ──
            # Entering: 只看红条距离（<=70cm）。宽度比不作为判据 —— 鸟瞰的横向
            #   比例尺随行变化（20cm 处比标称大 51%，80cm 处才对齐），一条等宽赛道
            #   在低带会比中带凭空宽 11%，而判据要求窄 13%，畸变吃掉绝大部分余量。
            #   且该畸变随相机高度/俯角变化，narrow_gate_exit_ratio 是按老几何调的。
            # Exiting:  red bar close OR start line close
            narrow_gate_detected = False
            narrow_gate_score = 1.0
            narrow_gate_dir = 0  # -1=entering, +1=exiting, 0=none

            # Width ratio (spatial diff between mid and low bands)
            ratio_out = False  # "out" pattern: mid wider than low
            if len(roi_results) >= 2:
                mid_width = float(far.get("lane_width_px", 140.0))
                low_width = float(near.get("lane_width_px", 140.0))
                if mid_width > 0 and low_width > 0:
                    narrow_gate_score = low_width / max(mid_width, 1.0)
                    curve_ok = abs(curve_px) < self.curve_switch_px * 1.3
                    if curve_ok and narrow_gate_score < 1.0 / self.narrow_gate_exit_ratio:
                        ratio_out = True  # mid > low: approaching/entering pattern

            # ── Narrow gate positioning: dual verification (red bar + start line) ──
            # Layout: narrow gate exit --24cm--> red bar --36cm--> start line
            start_line_y = self._detect_start_line(gray_detect, black_th, track_is_dark)
            start_line_z = 0.0
            line_visible = False
            if start_line_y is not None:
                _, start_line_z = self._px_to_ground_cm(float(self.center_x), float(start_line_y))
                line_visible = start_line_z > 0

            red_visible = red_bar_detected and red_bar_z_cm > 0
            red_at_gate = red_visible and red_bar_z_cm <= self.narrow_red_enter_z_cm
            red_close = red_visible and red_bar_z_cm < self.narrow_red_close_z_cm
            line_close = line_visible and start_line_z < self.narrow_line_close_z_cm

            # Entering: 红条进入 70cm 内
            if red_at_gate:
                narrow_gate_detected = True
                narrow_gate_dir = -1

            # Exiting: red bar or start line close to camera bottom
            if red_close or line_close:
                narrow_gate_detected = True
                narrow_gate_dir = 1

            state["start_line_z"] = start_line_z

            ng_exit_z = 0.0
            ng_enter_z = 0.0
            ng_conf = 0

            # Source A: red bar → ng_exit = red_z + 24cm
            ng_from_red = (red_bar_z_cm + 24.0) if (red_bar_detected and red_bar_z_cm > 0) else 0.0
            # Source B: start line → ng_exit = start_line_z - 60cm
            ng_from_line = (start_line_z - 60.0) if start_line_z > 0 else 0.0

            if ng_from_red > 0 and ng_from_line > 0:
                if abs(ng_from_red - ng_from_line) < 15.0:
                    ng_exit_z = 0.5 * (ng_from_red + ng_from_line)
                    ng_conf = 2
                else:
                    ng_exit_z = ng_from_red
                    ng_conf = 1
            elif ng_from_red > 0:
                ng_exit_z = ng_from_red
                ng_conf = 1
            elif ng_from_line > 0:
                ng_exit_z = ng_from_line
                ng_conf = 1

            if ng_exit_z > 0:
                ng_enter_z = max(ng_exit_z - 46.0, 0.0)

            # Cross-validate red bar distance against start line
            if red_bar_detected and start_line_z > 0:
                red_expected = start_line_z - 36.0
                if abs(red_bar_z_cm - red_expected) < 15.0:
                    red_bar_z_cm = 0.5 * (red_bar_z_cm + red_expected)

            # robot z (low band center y=375): ~ 20 + (399-375)*0.150 = 23.6cm
            robot_z = 20.0 + (self.bird_h - 1 - 375) * self.z_per_px
            inside_narrow = ng_enter_z > 0 and robot_z > ng_enter_z and robot_z < ng_exit_z

            avg_conf = sum(
                (r["conf"] * self._result_quality_weight(r)) for r in roi_results
            ) / float(len(roi_results))
            # One visible boundary cannot pair, so the bottom lock is meant to report
            # nothing here. Penalising that is what pushed single-line frames under
            # conf_min and turned them into line loss.
            if not bottom_lock_valid and not single_line:
                avg_conf *= (
                    1.0 - self.bottom_lock_conf_penalty * (1.0 - center_lock_quality)
                )
                avg_conf = clamp(avg_conf, 0.0, 1.0)

            near_norm = near_err_px / max(0.5 * img_w, 1.0)
            far_norm = far_err_px / max(0.5 * img_w, 1.0)
            curve_norm = curve_px / max(0.5 * img_w, 1.0)

            # Dynamic lookahead (弯道时增强前瞻)
            turn_gate = clamp(abs(curve_px) / max(self.curve_switch_px, 1.0), 0.0, 1.0)
            lookahead_dyn = self.pix_lookahead_gain * (0.70 + 0.90 * turn_gate)

            # Pixel-domain error fusion
            fused_err = -near_norm
            fused_err += lookahead_dyn * (-far_norm)
            fused_err += self.pix_curve_gain * (-curve_norm)
            fused_err += self.pix_angle_gain * (-angle_err / 45.0)
            if curve_px < -self.left_curve_outward_px:
                fused_err += self.left_curve_outward_gain * curve_norm

            state["smoothed_err"] = confidence_weighted_ema(
                state["smoothed_err"], fused_err, alpha_eff, avg_conf
            )

            # Curve mode detection (for dual-mode PID)
            _cm = abs(curve_px) >= self.curve_switch_px
            # One visible boundary leaves curve_px nothing to measure: both bands place
            # the same half-width offset off the same run, so their difference is ~0 on
            # precisely the frames the single-line gain exists for. The fitted heading
            # of that one boundary is then the only curve evidence there is.
            if single_line or LineDetector._single_band_mask(band_mask):
                _cm = _cm or abs(angle_err) >= self.curve_angle_deg
            if ((not bottom_lock_valid) and bottom_pair_ratio > 0.0
                    and abs(bottom_sym_err_px) > self.bottom_lock_sym_tol_px):
                _cm = True
            curve_mode = _cm

            # Shake diff RMS tracking
            hist = state["near_err_history"]
            hist.append(float(near_err_px))
            if len(hist) > self.robust_diff_window + 1:
                del hist[0]
            if len(hist) >= 3:
                diffs = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
                rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
                state["diff_rms_px"] = rms
                if rms >= self.robust_diff_rms_trigger_px:
                    state["shake_active_frames"] = self.robust_decay_frames
                elif state["shake_active_frames"] > 0:
                    state["shake_active_frames"] -= 1

            state["last_base_err"] = base_err_cm
            state["last_angle_err"] = angle_err
            state["last_far_dist"] = far_dist_cm

        else:
            # Lost tracking
            state["lost_frames"] += 1
            base_err_cm = state["last_base_err"]
            base_err_px = state["last_lane_center_x"] - img_cx
            angle_err = state["last_angle_err"]
            far_dist_cm = state["last_far_dist"]
            avg_conf = 0.0
            band_mask = state["last_band_mask"]
            curve_mode = False
            narrow_gate_detected = False
            narrow_gate_score = 1.0
            narrow_gate_dir = 0

        # ── Output ──
        dev_px = base_err_px
        heading_deg = angle_err
        conf = clamp(avg_conf, 0.0, 1.0)

        # ── Visualization (on birdseye) ──
        vis = self._build_visualization(
            gray, bgr_bird, roi_results, black_th, track_is_dark,
            dev_px, heading_deg, conf, base_err_px, band_mask,
        )

        # ── Debug info ──
        binary_raw_inv = 255 - binary_clean  # invert for display: black line on white bg
        debug = {
            "bird": gray,
            "binary_raw": binary_raw_inv,
            "binary": binary_clean,
            "black_th": black_th,
            "track_is_dark": track_is_dark,
            "base_err_px": base_err_px,
            "base_err_cm": base_err_cm,
            "angle_err_deg": angle_err,
            "far_dist_cm": far_dist_cm,
            "avg_conf": avg_conf,
            "band_mask": band_mask,
            "red_block_score": red_block_score,
            "black_block_score": black_block_score,
            "red_bar_detected": red_bar_detected,
            "red_bar_z_cm": red_bar_z_cm,
            "red_bar_x_cm": red_bar_x_cm,
            "red_bar_cx": red_bar_cx,
            "red_bar_cy": red_bar_cy,
            "bottom_pair_ratio": bottom_pair_ratio,
            "bottom_sym_err_px": bottom_sym_err_px,
            "bottom_lock_valid": bottom_lock_valid,
            "center_lock_quality": center_lock_quality,
            "lost_frames": state["lost_frames"],
            "startup_frames": state["startup_frames"],
            "diff_rms_px": state["diff_rms_px"],
            "shake_active_frames": state["shake_active_frames"],
            "n_roi_results": len(roi_results),
            "near_err_px": near_err_px_pre_lock,
            "far_err_px": far_err_px_saved,
            "curve_px": curve_px,
            "turn_gate": turn_gate,
            "fused_err": state["smoothed_err"],
            "fused_err_cm": state["smoothed_err"] * self.err_scale_cm,
            "narrow_gate_detected": narrow_gate_detected,
            "narrow_gate_score": narrow_gate_score,
            "narrow_gate_dir": narrow_gate_dir,
            "narrow_ratio_out": ratio_out,
            "narrow_red_visible": red_visible,
            "narrow_line_visible": line_visible,
            "start_line_z": state.get("start_line_z", 0.0),
            "ng_exit_z": ng_exit_z,
            "ng_enter_z": ng_enter_z,
            "inside_narrow": inside_narrow,
            "curve_mode": curve_mode,
            "left_seen": left_seen,
            "right_seen": right_seen,
            "single_line": single_line,
            "single_side": single_side,
        }

        debug["vision_speed_cm_s"] = 0.0
        debug["vision_omega_rad_s"] = 0.0

        return dev_px, heading_deg, conf, vis, debug

    # ═══════════════════════════════════════════════════════════
    # Visualization (on birdseye)
    # ═══════════════════════════════════════════════════════════

    def _build_visualization(self, gray_bird, bgr_bird,
                             roi_results, black_th, track_is_dark,
                             dev_px, heading_deg, conf, base_err_px,
                             band_mask):
        """Overlay detection results on the birdseye image."""
        vis = cv2.cvtColor(gray_bird, cv2.COLOR_GRAY2BGR)

        # Draw two band regions
        band_regions = [
            (self.band_low_y0, self.band_low_y1, (255, 200, 100)),
            (self.band_mid_y0, self.band_mid_y1, (100, 200, 255)),
        ]
        for y0, y1, color in band_regions:
            y0_cl = clamp(y0, 0, self.bird_h - 1)
            y1_cl = clamp(y1, 0, self.bird_h - 1)
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, y0_cl), (self.bird_w - 1, y1_cl), color, -1)
            cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)
            cv2.line(vis, (0, y0_cl), (self.bird_w - 1, y0_cl), color, 1)
            cv2.line(vis, (0, y1_cl), (self.bird_w - 1, y1_cl), color, 1)

        # Bottom lock region
        lock_y0 = int(clamp(self.bottom_lock_start_ratio * self.bird_h, 0, self.bird_h - 1))
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, lock_y0), (self.bird_w - 1, self.bird_h - 1),
                      (0, 100, 100), -1)
        cv2.addWeighted(overlay, 0.06, vis, 0.94, 0, vis)

        # Draw detected track center points
        for r in roi_results:
            cx = int(r.get("center_px", 0))
            if "band_name" in r:
                bn = r["band_name"]
                if bn == "low":
                    approx_y = (self.band_low_y0 + self.band_low_y1) // 2
                elif bn == "mid":
                    approx_y = (self.band_mid_y0 + self.band_mid_y1) // 2
                else:
                    approx_y = self.bird_h // 2
            else:
                approx_y = self.bird_h // 2
            cv2.circle(vis, (cx, approx_y), 5, (0, 255, 255), -1)
            cv2.circle(vis, (cx, approx_y), 7, (0, 180, 180), 1)

        # Center crosshair
        cv2.line(vis, (self.center_x, 0), (self.center_x, self.bird_h - 1),
                 (128, 128, 128), 1)
        cv2.line(vis, (0, self.bird_h // 2), (self.bird_w - 1, self.bird_h // 2),
                 (128, 128, 128), 1)

        # Lateral deviation indicator
        dev_x = int(self.center_x + dev_px)
        cv2.line(vis, (self.center_x, self.bird_h - 20),
                 (self.center_x, self.bird_h - 5), (255, 255, 255), 2)
        cv2.circle(vis, (dev_x, self.bird_h - 12), 5, (0, 255, 0), -1)
        cv2.line(vis, (self.center_x, self.bird_h - 12),
                 (dev_x, self.bird_h - 12), (0, 255, 0), 2)

        # Heading indicator
        arrow_len = 35
        h_rad = math.radians(-heading_deg)  # positive=right curve, but line-fit sign is opposite
        dx = int(arrow_len * math.sin(h_rad))
        dy = -int(arrow_len * math.cos(h_rad))
        arrow_start = (self.center_x, self.bird_h - 40)
        arrow_end = (self.center_x + dx, self.bird_h - 40 + dy)
        cv2.arrowedLine(vis, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.4)

        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"dev={dev_px:+.1f}px  hdg={heading_deg:+.1f}deg  conf={conf:.2f}",
            f"th={black_th}  lost={self._state['lost_frames']}  bmask={band_mask}",
        ]
        for i, text in enumerate(lines):
            y_pos = 16 + i * 18
            cv2.putText(vis, text, (6, y_pos), font, 0.45, (255, 255, 0), 1)

        # Band labels
        label_y = self.bird_h - 6
        cv2.putText(vis, "low", (6, label_y), font, 0.35, (180, 180, 255), 1)
        cv2.putText(vis, "mid", (50, label_y), font, 0.35, (180, 255, 180), 1)

        return vis


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    print("LineDetector V1 (Warp) — self-test")
    ld = LineDetector()
    bgr = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dev, hdg, conf, vis, dbg = ld.process(bgr)
    print(f"OK: dev={dev:.2f}px  hdg={hdg:.2f}deg  conf={conf:.3f}")
    print(f"  lost={dbg['lost_frames']}  n_roi={dbg['n_roi_results']}  black_th={dbg['black_th']}")
    print(f"  vis.shape={vis.shape}")
    print("Test passed!")
