"""V1 = V0's band-scanning detection heuristics + IPM birdseye warp.

Key differences from V0:
  - Warps BGR to 320x400 birdseye at start of process(), then all processing on birdseye
  - No camera LUT — uniform cm_per_px on birdseye
  - Band definitions adapted for 400px birdseye (down 266-398, mid 132-264, up 0-130)
  - No PID/steer/lost controller code — pure vision pipeline
  - No JSON config loading — all params are hardcoded defaults
  - Constructor: V1(cam_w=320, cam_h=240, cam_height_cm=38, cam_vfov_deg=43.6)
"""

import cv2
import numpy as np
import math


# ═══════════════════════════════════════════════════════════════════════
# Utility functions (same as V0)
# ═══════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


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


# ═══════════════════════════════════════════════════════════════════════
# LineDetector V1
# ═══════════════════════════════════════════════════════════════════════

class LineDetector:
    def __init__(self, cam_w=320, cam_h=240, cam_height_cm=38, cam_pitch_deg=45.0, cam_vfov_deg=43.6):
        # ── Camera params ──
        self.cam_w = int(cam_w)
        self.cam_h = int(cam_h)
        self.cam_height = float(cam_height_cm)
        self.cam_pitch = np.radians(cam_pitch_deg)
        self.cam_vfov_deg = float(cam_vfov_deg)

        # ── Birdseye ──
        self.bird_h = 400
        self.bird_w = 320
        self.center_x = self.bird_w // 2  # 160

        # Build IPM matrix (same as V2/V3)
        self.M = self._build_birdseye_matrix(lookahead=(10.0, 80.0))
        self.cm_per_px = self._compute_cm_per_px()
        self.z_per_px = (80.0 - 10.0) / float(self.bird_h - 1)  # vertical cm per px

        # ── Threshold params ──
        self.th_offset = -8
        self.th_min = 25
        self.th_max = 80
        self.dark_margin = 24
        self.track_color_mode = "auto"

        # ── ROI general params ──
        self.min_track_width = 24
        self.max_track_width = 300    # narrow bands, max plausible width
        self.min_pair_ratio = 0.35    # easier pair matching in tight band
        self.min_valid_lines = 2
        self.width_std_max = 20       # tighter band → lower width variance
        self.conf_min = 0.12
        self.min_line_width = 4
        self.max_line_width = 120
        self.lane_width_init_px = 140.0
        self.lane_width_tol_px = 50.0  # tighter band → tighter width tolerance
        self.max_center_jump_px = 45.0  # 50px band, can't jump beyond band height
        self.min_pair_lines = 2

        # ── Simple Bottom Mode ──
        self.simple_bottom_mode = True
        self.bottom_start_ratio = 0.875   # y=350, bottom 1/8
        self.bottom_rows = 5
        self.bottom_step = 2
        self.single_line_conf = 0.30
        self.assist_enable = False

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
        self.red_min_r = 105
        self.red_dom_margin = 28
        self.red_row_ratio = 0.35

        # ── Bottom lock ──
        self.bottom_lock_enable = True
        self.bottom_lock_start_ratio = 0.875  # y=350, bottom 1/8
        self.bottom_lock_rows = 10
        self.bottom_lock_step = 2
        self.bottom_lock_min_pair_ratio = 0.55
        self.bottom_lock_sym_tol_px = 24.0
        self.bottom_lock_blend = 0.72  # stronger lock anchor, less band-scan bias in curves
        self.bottom_lock_conf_penalty = 0.45
        self.bottom_lock_speed_penalty = 0.25
        self.lock_reacquire_reset = True

        # ── Startup ──
        self.startup_settle_frames = 25
        self.startup_speed_scale = 0.55
        self.startup_conf_min_scale = 0.70
        self.startup_min_weight_scale = 0.70
        self.startup_force_simple_bottom = True
        self.startup_lost_bias_free = True

        # ── Fusion params ──
        self.smooth_alpha = 0.72         # narrower bands → more noise → more smoothing
        self.curve_gain = 0.18
        self.angle_gain = 0.15
        self.min_weight = 0.10

        # ── Pixel domain gains（窄带适配：50px band separation, less lookahead）──
        self.pix_lookahead_gain = 0.25   # far band closer, less curvature info
        self.pix_curve_gain = 0.18       # narrower band → weaker curve signal
        self.pix_angle_gain = 0.15       # fewer scanlines → noisier angle
        self.curve_switch_px = 18.0      # ~1/3 of 50px band separation
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
        self._state = {
            "smoothed_err": 0.0,
            "lost_frames": 0,
            "last_base_err": 0.0,
            "last_angle_err": 0.0,
            "last_far_dist": 0.0,
            "last_lane_center_x": float(self.center_x),
            "last_lane_width_px": float(self.lane_width_init_px),
            "track_dark_score": 0,
            "last_band_mask": 0,
            "startup_frames": 0,
            "last_bottom_lock_valid": False,
            "near_err_history": [],
            "shake_active_frames": 0,
            "diff_rms_px": 0.0,
        }

    # ═══════════════════════════════════════════════════════════
    # Birdseye matrix (same as V2/V3: line_detector.py lines 87-133)
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

    def _px_to_ground_cm(self, x, y):
        """Convert birdseye pixel (x, y) to ground cm.
        x_cm: horizontal offset from center (positive = right)
        z_cm: forward distance from robot
        """
        x_cm = (x - self.center_x) * self.cm_per_px
        z_cm = 10.0 + (self.bird_h - 1 - y) * self.z_per_px
        return x_cm, z_cm

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
    # Track color detection
    # ═══════════════════════════════════════════════════════════

    def _detect_track_is_dark(self, gray, black_th):
        if self.track_color_mode == "dark":
            return True
        if self.track_color_mode == "light":
            return False

        h, w = gray.shape
        dark = 0
        light = 0
        step_y = max(1, h // 20)
        step_x = max(1, w // 20)

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                g = int(gray[y, x])
                if g <= black_th:
                    dark += 1
                else:
                    light += 1

        return dark <= light

    # ═══════════════════════════════════════════════════════════
    # Pixel classification
    # ═══════════════════════════════════════════════════════════

    def _pixel_is_track(self, g, black_th, track_is_dark):
        if track_is_dark:
            return g <= max(0, black_th - self.dark_margin)
        return g >= min(255, black_th + self.dark_margin)

    def _pixel_is_red(self, bgr, x, y):
        if not self.red_detect_enable:
            return False
        r = int(bgr[y, x, 2])
        g = int(bgr[y, x, 1])
        b = int(bgr[y, x, 0])
        return (r >= self.red_min_r) and ((r - g) >= self.red_dom_margin) and ((r - b) >= self.red_dom_margin)

    # ═══════════════════════════════════════════════════════════
    # Obstacle detection
    # ═══════════════════════════════════════════════════════════

    def _detect_row_blocker(self, gray, bgr, y, x0, x1, black_th, track_is_dark):
        total = max(1, x1 - x0 + 1)
        track_count = 0
        red_count = 0
        longest_track_run = 0
        cur_run = 0

        for x in range(x0, x1 + 1):
            g = int(gray[y, x])
            is_track = self._pixel_is_track(g, black_th, track_is_dark)
            if is_track:
                track_count += 1
                cur_run += 1
                if cur_run > longest_track_run:
                    longest_track_run = cur_run
            else:
                cur_run = 0

            if self._pixel_is_red(bgr, x, y):
                red_count += 1

        red_ratio = red_count / float(total)
        cover_ratio = track_count / float(total)
        run_ratio = longest_track_run / float(total)

        red_block = red_ratio >= self.red_row_ratio
        black_block = (
            run_ratio >= self.cross_black_run_ratio
            and cover_ratio >= self.cross_black_cover_ratio
        )
        return red_block, black_block

    # ═══════════════════════════════════════════════════════════
    # Run collection
    # ═══════════════════════════════════════════════════════════

    def _collect_track_runs_on_row(self, gray, y, x0, x1, black_th, track_is_dark):
        runs = []
        run_start = -1
        for x in range(x0, x1 + 1):
            g = int(gray[y, x])
            is_track = self._pixel_is_track(g, black_th, track_is_dark)
            if is_track and run_start < 0:
                run_start = x
            elif (not is_track) and run_start >= 0:
                run_end = x - 1
                w = run_end - run_start + 1
                if self.min_line_width <= w <= self.max_line_width:
                    runs.append((run_start, run_end))
                run_start = -1
        if run_start >= 0:
            run_end = x1
            w = run_end - run_start + 1
            if self.min_line_width <= w <= self.max_line_width:
                runs.append((run_start, run_end))
        return runs

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

    def _infer_center_from_single_run(self, run, hint_center, lane_width_hint, x0, x1):
        c = 0.5 * (run[0] + run[1])
        w = max(float(lane_width_hint), float(self.min_track_width))
        img_cx = self.center_x

        cand_left = c + 0.5 * w
        cand_right = c - 0.5 * w

        if abs(cand_left - hint_center) < abs(cand_right - hint_center):
            center = cand_left
        elif abs(cand_left - hint_center) > abs(cand_right - hint_center):
            center = cand_right
        else:
            center = cand_left if c < img_cx else cand_right

        center = clamp(center, x0, x1)
        return {
            "center_px": center,
            "lane_width_px": w,
            "conf": self.single_line_conf,
            "line_mode": 1,
        }

    # ═══════════════════════════════════════════════════════════
    # Band scanning (on birdseye)
    # ═══════════════════════════════════════════════════════════

    def _scan_band_midline(self, gray, bgr, black_th, track_is_dark,
                           hint_x, lane_width_hint,
                           y_start_ratio, y_end_ratio, max_rows, row_step):
        """Scan a band of rows on the birdseye, find midline per row."""
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
            if chosen is None and len(runs) >= 1:
                best_run = self._choose_single_run_near_hint(runs, last_center)
                if best_run is not None:
                    chosen = self._infer_center_from_single_run(
                        best_run, last_center, last_width, x0, x1
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
                if int(chosen.get("line_mode", 1)) >= 2:
                    pair_rows += 1
                else:
                    single_rows += 1
                last_center = center_px
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
        angle = math.degrees(math.atan(a))

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
        }

    # ═══════════════════════════════════════════════════════════
    # Simple bottom midline + assist band
    # ═══════════════════════════════════════════════════════════

    def _bottom_quarter_midline(self, gray, bgr, black_th, track_is_dark,
                                hint_x, lane_width_hint):
        base = self._scan_band_midline(
            gray, bgr, black_th, track_is_dark,
            hint_x, lane_width_hint,
            self.bottom_start_ratio, 1.0,
            self.bottom_rows, self.bottom_step,
        )
        if base is None:
            return None

        if self.assist_enable:
            assist = self._scan_band_midline(
                gray, bgr, black_th, track_is_dark,
                base["center_px"], base["lane_width_px"],
                self.assist_start_ratio, self.assist_end_ratio,
                self.assist_rows, self.assist_step,
            )
            if assist is not None:
                base["assist_center_px"] = assist["center_px"]
                base["assist_center_cm"] = assist["center_cm"]
                base["assist_dist_cm"] = assist["dist_cm"]
                base["assist_angle_deg"] = assist["angle"]
                base["assist_conf"] = assist["conf"]

        return base

    # ═══════════════════════════════════════════════════════════
    # Two-band cascade detection (hard-coded birdseye rows)
    # ═══════════════════════════════════════════════════════════

    def _detect_two_band_lanes(self, gray, bgr, black_th, track_is_dark,
                                hint_x, lane_width_hint):
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
            cv2.THRESH_BINARY, 31, -3  # blockSize=31, C=-3
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
        # Original _detect_track_is_dark call preserved for non-black-hat mode:
        # track_dark_candidate = self._detect_track_is_dark(gray, black_th)
        # if track_dark_candidate:
        #     state["track_dark_score"] = int(
        #         clamp(state["track_dark_score"] + 1, -6, 6)
        #     )
        # else:
        #     state["track_dark_score"] = int(
        #         clamp(state["track_dark_score"] - 1, -6, 6)
        #     )
        # track_is_dark = state["track_dark_score"] >= 0
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
                scan_hint_center, scan_hint_width,
            )
            if res is not None and res["conf"] >= conf_min_dyn:
                roi_results.append(res)
            elif self.two_band_mode:
                roi_results = self._detect_two_band_lanes(
                    gray_detect, bgr_bird, black_th, track_is_dark,
                    scan_hint_center, scan_hint_width,
                )
                roi_results = [r for r in roi_results if r["conf"] >= conf_min_dyn]
        elif self.two_band_mode:
            roi_results = self._detect_two_band_lanes(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width,
            )
            roi_results = [r for r in roi_results if r["conf"] >= conf_min_dyn]
        elif self.simple_bottom_mode:
            res = self._bottom_quarter_midline(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width,
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

        # ── Pixel-domain error fusion ──
        if roi_results:
            state["lost_frames"] = 0

            near = self._pick_result_by_band(roi_results, ("low", "mid"))
            if near is None:
                near = min(roi_results, key=lambda r: r["dist_cm"])

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
                angle_err = 0.5 * (
                    near["angle"]
                    + float(near.get("assist_angle_deg", near["angle"]))
                )
            else:
                angle_err = 0.5 * (near["angle"] + far["angle"])

            curve_px = far_err_px - near_err_px
            avg_conf = sum(
                (r["conf"] * self._result_quality_weight(r)) for r in roi_results
            ) / float(len(roi_results))
            if not bottom_lock_valid:
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

            state["smoothed_err"] = (
                alpha_eff * state["smoothed_err"] + (1.0 - alpha_eff) * fused_err
            )

            # Curve mode detection (for dual-mode PID)
            _cm = abs(curve_px) >= self.curve_switch_px
            if LineDetector._single_band_mask(band_mask):
                _cm = _cm or abs(angle_err) >= 8.0
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
            "curve_mode": curve_mode,
        }

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
    ld = LineDetector(cam_w=320, cam_h=240, cam_height_cm=38, cam_vfov_deg=43.6)
    bgr = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    dev, hdg, conf, vis, dbg = ld.process(bgr)
    print(f"OK: dev={dev:.2f}px  hdg={hdg:.2f}deg  conf={conf:.3f}")
    print(f"  lost={dbg['lost_frames']}  n_roi={dbg['n_roi_results']}  black_th={dbg['black_th']}")
    print(f"  vis.shape={vis.shape}")
    print("Test passed!")
