"""V1 = V0's band-scanning detection heuristics + IPM birdseye warp.

Key differences from V0:
  - Warps BGR to an equal-scale metric birdseye before lane processing
  - Ground LUT has constant cm/pixel; scan regions migrate from legacy ground distances
  - Band definitions adapted for 400px birdseye (down 266-398, mid 132-264, up 0-130)
  - No PID/steer/lost controller code — pure vision pipeline
  - Optional nested config for P1 observation/fusion; camera parameters remain explicit
  - Constructor: V1(cam_w=1280, cam_h=720, cam_height_cm=32.5, cam_vfov_deg=55.876)
"""

import cv2
import numpy as np
import math
import time
from copy import deepcopy

from utils import clamp
from line_preprocess import extract_lane_candidates, sampled_otsu_threshold
from photometric_thresholds import measure, MAX_CHANNEL_REFERENCE
from continuous_lane_heading import HeadingScanConfig, trace_heading


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


def _cfg_get(config, path, default):
    """Read optional nested detector configuration without changing camera geometry."""
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def time_constant_ema(previous, measurement, dt_s, tau_s):
    """EMA with a fixed time constant in seconds; quality gates happen upstream."""
    if not math.isfinite(dt_s) or dt_s < 0 or not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("EMA requires finite dt_s >= 0 and tau_s > 0")
    beta = -math.expm1(-dt_s / tau_s)
    return previous + beta * (measurement - previous)


# ═══════════════════════════════════════════════════════════════════════
# LineDetector V1
# ═══════════════════════════════════════════════════════════════════════

class LineDetector:
    # 低带中心行（y=350~399 的中点）。near_err 量在这一带，厘米换算也按这一行。
    NEAR_BAND_ROW = 375.0

    def __init__(self, cam_w=1280, cam_h=720, cam_height_cm=32.5, cam_pitch_deg=45.0, cam_vfov_deg=55.876,
                 z_calib=None, lane_width_cm=None, config=None):
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

        # Ground-plane orthographic raster: equal cm/pixel on both axes.
        # Keep the legacy physical width and depth, not its distorted aspect.
        if not math.isfinite(self.z_a) or self.z_a <= 0 or not math.isfinite(self.z_b):
            raise ValueError("Ground distance calibration must have a positive finite slope")
        self.bird_h = 400
        self.bird_w = 320
        self.center_x = 160
        self._ipm_lookahead = (20.0, 80.0)
        self.fy_px = self.cam_h / (2.0 * np.tan(np.radians(self.cam_vfov_deg / 2)))
        self.fx_px = self.fy_px
        self.cx_px, self.cy_px = self.cam_w / 2.0, self.cam_h / 2.0
        # Only used to migrate old pixel thresholds/regions into physical units.
        legacy_M = self._build_legacy_birdseye_matrix(self._ipm_lookahead)
        self.M_inv = np.linalg.inv(legacy_M)
        old_near_scale = (self._ground_from_bird_px(161, 375)[0]
                          - self._ground_from_bird_px(160, 375)[0])
        self._legacy_row_z = [self._ground_from_bird_px(160, y)[1] for y in range(400)]
        self.cm_per_px = self.z_a * 60.0 / (self.bird_h - 1)
        self.z_per_px = self.cm_per_px
        self._asp = 1.0
        width_cm = 2 * 80 * np.tan(np.radians(self.cam_vfov_deg / 2)) * self.cam_w / self.cam_h * .7
        self.bird_w = 2 * int(math.ceil(width_cm / (2 * self.cm_per_px))) + 1
        self.center_x = (self.bird_w - 1) // 2
        self._legacy_x_scale = old_near_scale / self.cm_per_px
        self.M = self._build_birdseye_matrix(self._ipm_lookahead)
        self.M_inv = np.linalg.inv(self.M)
        self._build_valid_ground_mask()
        self.NEAR_BAND_ROW = self._migrate_row(375)

        # ── 车道宽锚（横向比例尺的绝对值）──
        self.lane_width_true_cm = (
            float(lane_width_cm) if lane_width_cm else self._lane_width_from_config())
        self.lateral_scale = 1.0
        self.lateral_scale_min = 0.85    # ±20% 之外的事不是模型误差，是量错了
        self.lateral_scale_max = 1.20
        self.lateral_scale_alpha = 0.02  # ~10Hz 下 τ≈5s

        # 逐行地面 LUT 依赖上面那组内参，必须排在它们之后。
        self.near_scale_row = self.NEAR_BAND_ROW
        self._build_ground_lut()
        self._rebuild_err_scale()

        # ── Threshold params ──
        self.preprocess_mode = "legacy"
        self.adaptive_c = -12.0
        self.photometric_mode = "legacy"
        self._photometry = None
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
        self.max_center_jump_px = 60.0  # Effective 132px jump cap; tolerate body sway and abrupt image shifts.
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
        # What these two decide is how much of a single-boundary band's weight survives
        # - and that same product is what the error EMA is updated by, so a low value
        # is a lag on every single-line correction. At 0.30 x 0.42 the frame reached the
        # loop with confidence 0.127, i.e. a 1.75 s time constant against 0.23 s for a
        # paired frame; 0.70 x 0.85 puts it at 0.60 and 0.45 s. The side is read off a
        # constant now, which is what makes the reading worth trusting this much.
        self.single_line_conf = 0.70
        self.one_line_quality = 0.85

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
        self.heading_far_cm = None  # Legacy rows unless the heading entrypoint opts in.
        self.heading_near_cm = None
        self.heading_regions_cm = None
        self._heading_region_rows = None
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
        # Legacy offset threshold retained for old callers; offset is not quality.
        self.bottom_lock_sym_tol_px = 24.0
        self.bottom_lock_blend = 0.72  # stronger lock anchor, less band-scan bias in curves
        self.bottom_lock_conf_penalty = 0.45
        self.lock_reacquire_reset = False  # P1 never shrinks history on lock transitions

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
        # 前瞻/曲率/航向/左弯外推四项之和，最多是近带读数的这个倍数。近带是唯一
        # 直接测量，其余是推断 —— 推断不能反号把测量翻掉。0 = 关。见融合那段。
        self.anticipation_clip = 0.5
        # ── 整条车道拟合（实验，默认关，只出诊断量）──
        # 现在只扫两条 50 行的带（地面 20~35cm），而相机能看到约 105cm。
        # 打开后多扫一整条高带、对中心点做二次拟合，读出两个固定距离处的中心。
        # **不接进控制** —— 先量"远端−近端"在圆弧和直道上分不分得开。
        self.lane_fit_enable = False
        self.lane_fit_rows = 40          # 每一段最多采几行
        self.lane_fit_step = 6
        self.lane_fit_seg = 50           # 每段多少行（滑动窗口的高度）
        self.lane_fit_top_cm = 70.0      # 高带最远扫到地面多少 cm（成对的点扫不了那么远）
        self.lane_fit_near_cm = 25.0     # 读第一个点的地面距离
        self.lane_fit_far_cm = 50.0      # 读第二个点；再远在最紧的弯上读不到（见下）
        self.lane_fit_min_pts = 8        # 少于这个点数就不拟合
        # Optional three-segment ground-path diagnostic. It never changes the
        # legacy low/mid observation or the published steering command.
        self.lane_segments_enable = False
        from steering_config import DEFAULT_SEGMENT_REGIONS_CM
        self.segment_regions_cm = DEFAULT_SEGMENT_REGIONS_CM
        self.curve_smooth_alpha = 0.85   # ~6-frame EMA, for telling a curve from jitter
        self.curve_angle_deg = 8.0       # fitted heading past which one band is a curve
        # 车偏得这么远就强制进弯道模式：这是**转向策略**的门槛，不是"锁可不可信"
        # 的门槛，所以和 bottom_lock_sym_tol_px 分开。两者都从 24.0 起步（等于
        # 拆分前的行为）；以前共用一个常量，2026-10-02 改一个等于同时改两个。
        self.curve_force_sym_tol_px = 24.0
        self.left_curve_outward_gain = 0.35
        self.left_curve_outward_px = 6.0

        # ── Shake robust ──
        self.robust_enable = True
        self.robust_diff_window = 5
        self.robust_diff_rms_trigger_px = 8.0  # narrower band → lower diff tolerance
        self.robust_alpha_high = 0.88
        self.robust_bottom_lock_blend_scale = 1.5
        self.robust_decay_frames = 8

        # P1 observation contract. Units are part of each configuration name.
        def p1(name, default):
            return float(_cfg_get(config, "vision.p1." + name, default))
        self.lock_width_cv_max = p1("lock_width_cv_max", 0.15)
        self.observation_rmse_max_px = p1("observation_rmse_max_px", 8.0)
        self.measurement_quality_min = p1("measurement_quality_min", 0.20)
        self.measurement_max_age_s = p1("measurement_max_age_s", 0.25)
        self.single_width_max_age_s = p1("single_width_max_age_s", 1.0)
        self.single_quality_max = p1("single_quality_max", 0.35)
        self.filter_tau_s = p1("filter_tau_s", 0.18)
        self.shake_filter_tau_s = p1("shake_filter_tau_s", 0.25)
        self.preview_gain = p1("preview_gain", 0.5)
        self.preview_max_cm = p1("preview_max_cm", 5.0)
        self.heading_min_points = int(p1("heading_min_points", 6))
        self.heading_min_span_px = p1("heading_min_span_px", 8.0)
        # Heading only: scan bottom-to-top over the full configured interval.
        self.heading_scan = HeadingScanConfig()
        for value in (self.lock_width_cv_max, self.observation_rmse_max_px,
                      self.measurement_max_age_s, self.single_width_max_age_s,
                      self.filter_tau_s, self.shake_filter_tau_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("P1 scales and time constants must be finite and positive")
        if not (0 < self.measurement_quality_min <= 1 and
                0 <= self.single_quality_max <= 1 and 0 <= self.preview_gain <= 1 and
                math.isfinite(self.preview_max_cm) and self.preview_max_cm >= 0 and
                self.heading_min_points >= 3 and self.heading_min_span_px > 0):
            raise ValueError("Invalid P1 quality, preview or heading configuration")
        # This clock keeps advancing through card-stop snapshots. Accepted tracking
        # times below are restored, so diagnostic frames cannot rejuvenate old seeds.
        self._observation_clock_s = 0.0
        self._last_process_time = None

        self._migrate_metric_sampling()

        # ── Internal state ──
        self._state = self._initial_state()
        self._scan_diagnostics = {}

    def _initial_state(self):
        """Every cross-frame memory, at its as-just-started value."""
        return {
            "smoothed_err": 0.0,  # legacy normalized view of the centimeter EMA
            "smoothed_err_cm": 0.0,
            "last_accepted_tracking_time": None,
            "last_paired_tracking_time": None,
            "paired_width_frames": 0,
            "curve_px_ema": 0.0,
            "lost_frames": 0,
            "last_base_err": 0.0,
            "last_angle_err": 0.0,
            "last_far_dist": 0.0,
            "last_lane_center_x": float(self.center_x),
            "last_lane_width_px": float(self.lane_width_init_px),
            "last_band_mask": 0,
            "heading_rows": [],
            "heading_rows_time": None,
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

    def snapshot_tracking_state(self):
        """Save lane-tracking memory before a frame that may start a card stop.

        Red-bar confirmation and start-line observations remain live while stopped.
        Camera pitch, IPM matrices and ground LUTs belong to current geometry, so
        they are not rolled back. The width-derived lateral scale is tracking
        memory and is saved alongside the state dictionary.
        """
        return {
            "tracking": deepcopy({
                key: value for key, value in self._state.items()
                if not key.startswith("red_") and key != "start_line_z"
            }),
            "lateral_scale": float(self.lateral_scale),
        }

    def restore_tracking_state(self, snapshot):
        """Restore a saved walking seed without restarting or rewinding obstacles.

        Copy on restoration too: a resumed frame must not mutate a snapshot that
        will be reused after another diagnostic frame during the same card stop.
        Recompute the centimeter conversion using the current camera geometry.
        """
        tracking = deepcopy({
            key: value for key, value in snapshot["tracking"].items()
            if not key.startswith("red_") and key != "start_line_z"
        })
        lateral_scale = float(snapshot["lateral_scale"])
        self._state.update(tracking)
        self.lateral_scale = lateral_scale
        self._rebuild_err_scale()

    def reset_state(self):
        """Drop every cross-frame memory, so the next frame starts from scratch.

        Two of these are self-reinforcing and are why a detector that locks onto the
        wrong thing stays locked: `last_lane_center_x` is the next frame's scan hint,
        and `smoothed_err` is an EMA with a long memory. The fallback path also
        republishes `last_base_err` / `last_angle_err` verbatim. A fresh process on
        the same curve tracks fine, so a fresh state should behave the same.
        """
        self._state = self._initial_state()
        self._scan_diagnostics = {}

    # ═══════════════════════════════════════════════════════════
    # Birdseye matrix (IPM: pinhole back-projection of ground plane)
    # ═══════════════════════════════════════════════════════════

    def _build_legacy_birdseye_matrix(self, lookahead):
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

    def _build_birdseye_matrix(self, lookahead):
        """Unclipped ground projection, then ONE metric scale for x and z.

        Source points outside the sensor are valid mathematical coordinates;
        moving them onto the image boundary changes the ground homography.
        """
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        h, fx, fy = self.cam_height, self.fx_px, self.fy_px
        cx, cy = self.cx_px, self.cy_px
        ground_to_camera = np.array([
            [fx, cx * cp, cx * h * sp],
            [0, cy * cp - fy * sp, h * (fy * cp + cy * sp)],
            [0, cp, h * sp]], dtype=np.float64)
        scale = 1.0 / self.cm_per_px
        ground_to_raster = np.array([
            [scale, 0, self.center_x],
            [0, -self.z_a * scale, self.z_a * lookahead[1] * scale],
            [0, 0, 1]], dtype=np.float64)
        return ground_to_raster @ np.linalg.inv(ground_to_camera)

    def _migrate_row(self, old_y):
        z = float(np.interp(old_y, np.arange(400), self._legacy_row_z))
        return int(round(clamp((self._to_true_z(80) - z) / self.cm_per_px, 0, 399)))

    def _build_valid_ground_mask(self):
        source = np.full((self.cam_h, self.cam_w), 255, np.uint8)
        mask = cv2.warpPerspective(source, self.M, (self.bird_w, self.bird_h),
                                   flags=cv2.INTER_NEAREST)
        # Suppress interpolation rims; do not interpret unobserved ground as black line.
        self.ground_valid_mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0

    def _migrate_metric_sampling(self):
        for name in ('band_low_y0', 'band_low_y1', 'band_mid_y0', 'band_mid_y1'):
            setattr(self, name, self._migrate_row(getattr(self, name)))
        self.bottom_start_ratio = self._migrate_row(350) / self.bird_h
        self.bottom_lock_start_ratio = self.bottom_start_ratio
        # Preserve lateral centimeter gates, including the recently relaxed jump.
        for name in ('min_track_width', 'max_track_width', 'width_std_max',
                     'min_line_width', 'max_line_width', 'lane_width_init_px',
                     'lane_width_tol_px', 'max_center_jump_px', 'bottom_lock_sym_tol_px',
                     'curve_switch_px', 'curve_force_sym_tol_px', 'left_curve_outward_px',
                     'robust_diff_rms_trigger_px', 'observation_rmse_max_px'):
            setattr(self, name, getattr(self, name) * self._legacy_x_scale)
        from dataclasses import replace
        c = self.heading_scan
        self.heading_scan = replace(c, bottom_y=self._migrate_row(c.bottom_y),
            top_y=self._migrate_row(c.top_y), fit_top_y=self._migrate_row(c.fit_top_y),
            step_px=max(1, round(abs(self._migrate_row(350)-self._migrate_row(350+c.step_px)))),
            min_line_width_px=max(1, round(c.min_line_width_px*self._legacy_x_scale)),
            min_lane_width_px=c.min_lane_width_px*self._legacy_x_scale,
            max_lane_width_px=c.max_lane_width_px*self._legacy_x_scale,
            initial_width_px=c.initial_width_px*self._legacy_x_scale,
            search_radius_px=c.search_radius_px*self._legacy_x_scale)

    def _to_true_z(self, z_model):
        """相机模型读数 → 地面真值 cm（系数在 cameras.json 的 distance_calib）。"""
        return self.z_a * z_model + self.z_b

    def _ground_from_bird_px(self, x, y):
        """birdseye 像素 → 地面 (x_cm, z_cm)，走完整逆投影。

        (x,y) 先由 M⁻¹ 回到原图，再和地面平面求交。相机坐标下地面就是
        Yc·cosθ + Zc·sinθ = h —— 把 Yc = h·cosθ − wz·sinθ、Zc = h·sinθ + wz·cosθ
        代进去两边都等于 h，所以沿视线 d 走 t = h/(dy·cosθ + sinθ) 就落地。
        红条那条路（_detect_red_bar）用的就是这个，这里只是让鸟瞰坐标共用同一套。
        """
        p = self.M_inv @ np.array([x, y, 1.0], dtype=np.float64)
        dx = (p[0] / p[2] - self.cx_px) / self.fx_px
        dy = (p[1] / p[2] - self.cy_px) / self.fy_px
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        t = self.cam_height / (dy * cp + sp)
        return t * dx, self._to_true_z(t * (cp - dy * sp))

    @staticmethod
    def _lane_width_from_config():
        """赛道真实宽度（cm）。规则是 350mm，落在 cameras.json 方便重测。"""
        try:
            from camera_config import load as load_camera
            width = float(load_camera().get("lane_width_cm", 0.0) or 0.0)
        except Exception:
            width = 0.0
        return width if width > 0.0 else 35.0

    def set_camera_pitch_deg(self, pitch_deg):
        """换掉光轴俯角（安装角 + 机身实时前倾）。

        cam_pitch 只被烤进两个结构：鸟瞰单应 M（和它的逆）和逐行地面 LUT。
        重建这两样，err_scale_cm 跟着 LUT 走。cm_per_px / _asp / pix_angle_gain
        都只依赖内参，与俯角无关，不动。

        为什么要它：2026-10-02 台架对照 —— 车一步没动，只把站姿从直立换成
        后仰，读数当场垮（ang 22→45、curve 0→-24、far -33→-70、近带锁失效），
        之后 lateral_scale 还要花 ~15s 在新几何上重新收敛。停车做动作时机身
        必然后仰，而巡线原来完全不知道这件事（姿态只喂了图卡）。

        走路时不要每帧喂：步态以 1.7Hz 摆，低通滞后值还不如静态安装角。
        调用方按状态决定喂不喂，见 run_policy_vision.py 的 --line-pitch。
        """
        new = math.radians(float(pitch_deg))
        if abs(new - self.cam_pitch) < 1e-9:
            return                      # 大多数帧是这一条，省掉重建
        self.cam_pitch = new
        self.M = self._build_birdseye_matrix(lookahead=self._ipm_lookahead)
        self.M_inv = np.linalg.inv(self.M)
        self._build_valid_ground_mask()
        self._build_ground_lut()
        if self.heading_regions_cm is not None:
            self.set_heading_regions_cm(self.heading_regions_cm)
        elif self.heading_far_cm is not None:
            self.set_heading_distances(self.heading_near_cm, self.heading_far_cm)
        self._rebuild_err_scale()

    def set_heading_far_cm(self, distance_cm):
        self.set_heading_distances(self.heading_near_cm, distance_cm)

    def set_heading_regions_cm(self, regions_cm):
        """Select eight evenly spaced image rows INSIDE each ground interval.

        The near lock uses the same rows. Validate before replacing geometry;
        this adds no per-frame scans and never connects segment regions.
        """
        from steering_config import validate_heading_regions
        regions_cm = validate_heading_regions(regions_cm)
        if regions_cm is None:
            raise ValueError('heading regions must contain two intervals')
        rows = []
        for lo,hi in regions_cm:
            eligible = np.flatnonzero((self._lut_z_cm >= lo) & (self._lut_z_cm < hi))
            if len(eligible) < 8 or hi > max(self._lut_z_cm):
                raise ValueError('each heading region must contain at least eight calibrated image rows')
            rows.append(tuple(int(v) for v in eligible[np.linspace(0,len(eligible)-1,8,dtype=int)]))
        low,mid = rows
        if mid[-1] >= low[0]:
            raise ValueError('heading far region must be above the near region')
        self.band_low_y0,self.band_low_y1 = low[0],low[-1]
        self.band_mid_y0,self.band_mid_y1 = mid[0],mid[-1]
        self.heading_regions_cm = regions_cm
        self._heading_region_rows = tuple(rows)
        self.heading_near_cm = float(np.median(self._lut_z_cm[list(low)]))
        self.heading_far_cm = float(np.median(self._lut_z_cm[list(mid)]))
        self.near_scale_row = float(np.median(low))
        self.bottom_lock_start_ratio = (low[0]+.5)/self.bird_h
        self._rebuild_err_scale()

    def set_heading_distances(self, near_cm, far_cm):
        """Move measured bands atomically; None keeps the original near geometry.

        Eight rows per band are retained. The near lock and pixel scale follow
        a configured near point, while the separate narrow-gate reference stays put.
        """
        def band(distance, rows, step):
            distance = float(distance)
            if (not math.isfinite(distance) or distance <= 0
                    or not min(self._lut_z_cm) <= distance <= max(self._lut_z_cm)):
                raise ValueError('heading observation distance is outside the calibrated image')
            centre = int(np.argmin(np.abs(self._lut_z_cm-distance)))
            span = (rows-1)*step
            first, last = centre-span//2, centre-span//2+span
            if first < 0 or last >= self.bird_h:
                raise ValueError('heading observation band does not fit in the image')
            return first, last

        low = (self._migrate_row(350), self._migrate_row(399)) if near_cm is None else band(near_cm, self.band_rows_low, self.band_step_low)
        mid = band(far_cm, self.band_rows_mid, self.band_step_mid)
        lock_last = low[0] + (self.bottom_lock_rows-1)*self.bottom_lock_step
        if mid[1] + 8 > low[0] or lock_last >= self.bird_h:
            raise ValueError('heading far band must be separated from near by >=8 rows; near lock must fit')
        self.band_low_y0, self.band_low_y1 = low
        self.band_mid_y0, self.band_mid_y1 = mid
        self.heading_near_cm = None if near_cm is None else float(near_cm)
        self.heading_far_cm = float(far_cm)
        self.heading_regions_cm = self._heading_region_rows = None
        self.near_scale_row = self.NEAR_BAND_ROW if near_cm is None else .5*(low[0]+low[1])
        self.bottom_lock_start_ratio = (low[0]+.5)/self.bird_h
        self._rebuild_err_scale()

    def _build_ground_lut(self):
        """Inverse-projection LUT, checked against the metric raster in tests.

        The corrected raster has constant x/z scale. Camera pose/calibration
        errors can still distort physical ground; this LUT cannot fix them.
        """
        ys = np.arange(self.bird_h, dtype=np.float64)

        def ground_at(xs):
            points = self.M_inv @ np.stack([xs, ys, np.ones_like(ys)])
            dx = (points[0] / points[2] - self.cx_px) / self.fx_px
            dy = (points[1] / points[2] - self.cy_px) / self.fy_px
            cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
            t = self.cam_height / (dy * cp + sp)
            return t * dx, t * (cp - dy * sp)

        left, _ = ground_at(np.full(self.bird_h, self.center_x - 20.0))
        right, _ = ground_at(np.full(self.bird_h, self.center_x + 20.0))
        _, z_model = ground_at(np.full(self.bird_h, float(self.center_x)))
        self._lut_cm_per_px = (right - left) / 40.0
        self._lut_z_cm = self._to_true_z(z_model)

    def cm_per_px_at(self, y):
        """该行的横向厘米/像素，含车道宽锚的缩放。"""
        base = float(np.interp(
            float(y), np.arange(self.bird_h), self._lut_cm_per_px))
        return base * self.lateral_scale

    def z_cm_at(self, y):
        """该行的前方地面距离 cm。"""
        return float(np.interp(
            float(y), np.arange(self.bird_h), self._lut_z_cm))

    def _rebuild_err_scale(self):
        """fused_err 是无量纲的 near_err_px/(0.5*bird_w)，而消费者（PID 增益、
        STEP_LEN_CM、steer_full_scale_cm）全按厘米标定，所以换算要一起发布。"""
        self.err_scale_cm = 0.5 * self.bird_w * self.cm_per_px_at(self.near_scale_row)

    def _update_lateral_scale(self, near):
        """拿低带量到的车道宽，把横向比例尺锚到赛道的真实宽度上。

        逐行 LUT 修的是"把透视当线性"，剩下的是相机姿势本身 —— 物理重拟给
        h=30.45/θ=41.55°（配置写的是 32.5/45），横向因此差 ±7%。这部分没有干净的
        几何答案，只能靠赛道上一条已知宽度的东西。标定照片里量到 146.5px、按 LUT
        折 32.8cm，对规则 350mm 差 6%，正好在这个量级。

        夹在 ±20% 是有意的：窄门只有 240mm，真让它跟进去，比例尺会被抬 46%。
        """
        width_px = float(near.get("lane_width_px", 0.0) or 0.0)
        if width_px <= 0 or float(near.get("pair_ratio", 0.0)) < 0.45:
            return
        measured_cm = width_px * self.cm_per_px_at(self.near_scale_row)
        if measured_cm <= 1.0:
            return
        target = clamp(self.lateral_scale * self.lane_width_true_cm / measured_cm,
                       self.lateral_scale_min, self.lateral_scale_max)
        self.lateral_scale += self.lateral_scale_alpha * (target - self.lateral_scale)
        self._rebuild_err_scale()

    def _px_to_ground_cm(self, x, y):
        """Convert birdseye pixel (x, y) to ground cm.
        x_cm: horizontal offset from center (positive = right)
        z_cm: forward distance from robot
        """
        y = clamp(float(y), 0.0, float(self.bird_h - 1))
        return ((x - self.center_x) * self.cm_per_px_at(y), self.z_cm_at(y))

    # ═══════════════════════════════════════════════════════════
    # Otsu adaptive threshold
    # ═══════════════════════════════════════════════════════════

    def _otsu_threshold(self, gray):
        """Manual Otsu — same algorithm as V0 (not cv2.THRESH_OTSU)."""
        return sampled_otsu_threshold(gray)

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

    def _collect_track_runs_on_row(self, gray, y, x0, x1, black_th, track_is_dark, diagnostics=None):
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
        if diagnostics is not None:
            diagnostics['raw_runs'] = diagnostics.get('raw_runs', 0) + len(widths)
            diagnostics['line_width_rejected_runs'] = diagnostics.get('line_width_rejected_runs', 0) + int((~valid).sum())
        return list(zip(rises[valid].tolist(), falls[valid].tolist()))

    def _centroid_pair_center(self, gray_raw, y, hint_center, lane_width_hint,
                              x0, x1, diagnostics=None):
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
        min_contrast = (self._photometry.difference(self.centroid_min_contrast)
                        if self._photometry is not None else self.centroid_min_contrast)
        if diagnostics is not None:
            diagnostics['centroid_contrast_min'] = min_contrast
        if peak < min_contrast:
            if diagnostics is not None:
                diagnostics['centroid_low_contrast_rows'] = diagnostics.get('centroid_low_contrast_rows', 0) + 1
            return None

        thr = max(min_contrast, 0.25 * peak)
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
            if diagnostics is not None:
                diagnostics['centroid_insufficient_dips_rows'] = diagnostics.get('centroid_insufficient_dips_rows', 0) + 1
            return None

        want = (lane_width_hint if lane_width_hint >= self.min_track_width * 2
                else self.lane_width_init_px)
        best = None
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                lane_w = segs[b] - segs[a]
                if not (self.min_track_width <= lane_w <= self.max_track_width):
                    if diagnostics is not None:
                        diagnostics['centroid_width_range_rejects'] = diagnostics.get('centroid_width_range_rejects', 0) + 1
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

    def _choose_pair_center_from_runs(self, runs, hint_center, lane_width_hint, x0, x1, diagnostics=None):
        if len(runs) < 2:
            if diagnostics is not None:
                diagnostics['pair_insufficient_runs_rows'] = diagnostics.get('pair_insufficient_runs_rows', 0) + 1
            return None

        best = None
        best_score = 1e9
        for i in range(len(runs)):
            wi = runs[i][1] - runs[i][0] + 1
            if wi < 5:  # too thin to be a real track edge, noise
                if diagnostics is not None:
                    diagnostics['pair_thin_edge_skips'] = diagnostics.get('pair_thin_edge_skips', 0) + 1
                continue
            li = 0.5 * (runs[i][0] + runs[i][1])
            for j in range(i + 1, len(runs)):
                wj = runs[j][1] - runs[j][0] + 1
                if wj < 5:
                    if diagnostics is not None:
                        diagnostics['pair_thin_edge_skips'] = diagnostics.get('pair_thin_edge_skips', 0) + 1
                    continue
                rj = 0.5 * (runs[j][0] + runs[j][1])
                lane_w = rj - li
                if not self.min_track_width <= lane_w <= self.max_track_width:
                    if diagnostics is not None:
                        diagnostics['pair_width_range_rejects'] = diagnostics.get('pair_width_range_rejects', 0) + 1
                if self.min_track_width <= lane_w <= self.max_track_width:
                    if lane_width_hint > 0:
                        max_width_err = max(48.0, self.lane_width_tol_px * 1.6)
                        if abs(lane_w - lane_width_hint) > max_width_err:
                            if diagnostics is not None:
                                diagnostics['pair_width_hint_rejects'] = diagnostics.get('pair_width_hint_rejects', 0) + 1
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

    def _row_at_cm(self, z_cm):
        """地面距离 z_cm 落在哪一行。_lut_z_cm 是按行给的真实地面距离（非线性）。"""
        return int(clamp(np.searchsorted(-self._lut_z_cm, -float(z_cm)),
                         0, self.bird_h - 1))

    def _detect_lane_fit(self, gray, bgr, black_th, track_is_dark,
                         hint_x, lane_width_hint, gray_raw=None):
        """扫一整条高带（近端到远端），拟合中心线，从曲线上读两个固定距离的点。

        **只产诊断量，不改任何控制输出。** 要回答一个问题：把视野从 20~35cm
        拉到 20~70cm 之后，"远端中心 − 近端中心"这个量，在圆弧上和在直道上分
        得开吗？分得开就能拿它判直道/弯道，把转弯的基础转速做成已知的常数 v/R
        （0.2 m/s 下 0.258 rad/s），而不是每帧去估 —— 这才是"已知赛道形状"
        真正能给的东西。
        """
        # 一段一段往上扫（滑动窗口），**相邻窗口重叠一半**。两个坑都在这几行里：
        #
        # 1. `_scan_band_midline` 的 y 是**递增**的，而调用处 y_start=y_lo（远）、
        #    y_end=y_hi（近），所以它是**由远及近**扫，`centers_list[0]` 是段内
        #    **最远**那行。原来每段整个往上挪一格（`y_hi = y_lo`），新段的第一行
        #    就比任何测过的行还远一整段，而种子是上一段最远那行（差着 50 行）；
        #    偏偏新段第一行 `centers_px` 还是空的，连续性闸
        #    （`_scan_band_midline` 的 `if len(centers_px) > 0`）被跳过，只剩那个
        #    很松的绝对门。实车 91 帧里 49 帧就是这样死在第 3 段、`top` 卡在
        #    34~36cm（正好是中带顶边 row 300）。重叠一半之后，新段的远端落在上一段
        #    已经扫过的行里，种子可以取到一个**真正的测量值**。
        # 2. 一整段都是单线盲推（mode 1）时，它报的 `lane_width_px` 是段内中值 ——
        #    在没有成对行的情况下那是编出来的宽度，回灌给下一段会把配对门一起带歪。
        #    成对行不够就不回灌，沿用上一次的。
        y_top = max(1, self._row_at_cm(self.lane_fit_top_cm))
        ys_all, cx_all, modes_all, widths_all = [], [], [], []
        center, width = hint_x, lane_width_hint
        win_step = max(1, self.lane_fit_seg // 2)
        y_hi = self.bird_h - 1
        diagnostic_index = 0
        while True:
            y_lo = max(y_top, y_hi - self.lane_fit_seg)
            if y_lo >= y_hi:
                break
            # 种子取已测点里行号最接近 y_lo（新段第一行）的那个。用**全部**点，
            # 含单线盲推的：它只负责把搜索窗领到大致位置，连续性闸会兜住；
            # 拟合才只认成对的点。
            if ys_all:
                nearest = min(range(len(ys_all)), key=lambda i: abs(ys_all[i] - y_lo))
                center = float(cx_all[nearest])
            res = self._scan_band_midline(
                gray, bgr, black_th, track_is_dark, center, width,
                y_lo / float(self.bird_h), y_hi / float(self.bird_h),
                self.lane_fit_rows, self.lane_fit_step, gray_raw=gray_raw,
                diagnostics=self._new_scan_diagnostics(
                    f'fit_{diagnostic_index}', diagnostic_only=True))
            diagnostic_index += 1
            if res is not None:
                yl = res.get("ys_list", [])
                cl = res.get("centers_list", [])
                ml = res.get("modes_list", [])
                wl = res.get("widths_list", [])
                if yl:
                    ys_all.extend(yl)
                    cx_all.extend(cl)
                    modes_all.extend(ml if len(ml) == len(yl) else [1] * len(yl))
                    widths_all.extend(wl if len(wl) == len(yl) else [res["lane_width_px"]] * len(yl))
                    if len(ml) == len(yl) and sum(m >= 2 for m in ml) >= max(3, len(yl) // 2):
                        width = float(res["lane_width_px"])
            if y_lo <= y_top:
                break
            y_hi = y_lo + win_step
        ys = np.asarray(ys_all, dtype=np.float64)
        cx = np.asarray(cx_all, dtype=np.float64)
        widths = np.asarray(widths_all, dtype=np.float64)
        keep = np.asarray(modes_all, dtype=np.int32) >= 2
        # 只留"两条边界都真的看见"的行。单线盲推（_infer_center_from_single_run）
        # 是按**画面正中**判边、再横挪半个车道得出来的，弯道远端外侧线跑出鸟瞰图
        # 之后它能差 130px —— 一个这样的点就够把整条二次曲线拖歪：合成 R=77.6cm
        # 的圆弧上，top_cm 从 60 放到 65（多收进 row 102~148 那批单线点），
        # far=65cm 的读数从 −120px 变成 +115px，符号都反了。
        ys, cx, widths = ys[keep], cx[keep], widths[keep]
        # 扫到多远：真正要看的诊断量。上面那些行没出点时，这里会明显偏小。
        top_cm = float(self._lut_z_cm[int(ys.min())]) if len(ys) else None
        out = {"fit_pts": int(len(ys)), "fit_pair_pts": int(keep.sum()),
               "fit_top_cm": top_cm}
        if self.lane_segments_enable:
            from lane_segments import describe_lane_segments
            out.update(describe_lane_segments(
                ys, cx, widths, self._lut_z_cm,
                self._lut_cm_per_px * self.lateral_scale,
                self.center_x, self.lane_width_true_cm, regions_cm=self.segment_regions_cm))
        if len(ys) < self.lane_fit_min_pts or ys.max() - ys.min() < 40.0:
            return out
        coeff = np.polyfit(ys, cx, 2)
        row_near = self._row_at_cm(self.lane_fit_near_cm)
        row_far = self._row_at_cm(self.lane_fit_far_cm)
        # 只在拟合数据**覆盖到**的那一段里取值。2026-10-03 第一版没加这个：
        # 某帧 row 240 以上一个点都没有，却在 row 102（65cm）处求值 —— 拿
        # 240~399 行的数据外推 138 行，二次曲线飞出 +210px 这种不可能的中心。
        if row_near <= ys.max():
            out["fit_near_px"] = float(np.polyval(coeff, row_near)) - self.center_x
        if row_far >= ys.min():
            out["fit_far_px"] = float(np.polyval(coeff, row_far)) - self.center_x
        if "fit_near_px" in out and "fit_far_px" in out:
            out["fit_curve_px"] = out["fit_far_px"] - out["fit_near_px"]
            out["fit_ok"] = True
        return out

    def _clip_anticipation(self, near_term, anticipation):
        """把推断项夹在近带读数的 anticipation_clip 倍以内。

        近带是唯一直接测量，推断（前瞻带 / 曲率 / 拟合航向 / 左弯外推）只有
        补充的份，没有翻案的份 —— 近带报 42px 偏差时，推断加起来最多把它削掉
        一半，削不成零、更翻不了号。0 关掉。
        """
        if self.anticipation_clip <= 0.0:
            return anticipation
        limit = self.anticipation_clip * abs(near_term)
        return clamp(anticipation, -limit, limit)

    def _new_scan_diagnostics(self, name, *, diagnostic_only=False):
        """Current-frame scalar evidence, separate from rollback tracking state."""
        d = dict(attempted=True, diagnostic_only=diagnostic_only,
                 rows_scanned=0, run_candidate_rows=0, candidate_rows=0,
                 candidate_pair_rows=0, accepted_rows=0, paired_rows=0, single_rows=0,
                 centroid_pair_rows=0, no_candidate_rows=0,
                 raw_runs=0, line_width_rejected_runs=0,
                 pair_insufficient_runs_rows=0, pair_thin_edge_skips=0,
                 pair_width_range_rejects=0, pair_width_hint_rejects=0,
                 centroid_low_contrast_rows=0, centroid_insufficient_dips_rows=0,
                 centroid_width_range_rejects=0,
                 line_width_min_px=self.min_line_width, line_width_max_px=self.max_line_width,
                 pair_edge_min_px=5, track_width_min_px=self.min_track_width,
                 track_width_max_px=self.max_track_width,
                 continuity_reject_rows=0, center_jump_reject_rows=0,
                 red_block_rows=0, black_block_rows=0,
                 pair_fraction=None, pair_support_ratio=None,
                 pair_ratio_min=self.bottom_lock_min_pair_ratio,
                 width_cv=None, width_cv_max=self.lock_width_cv_max, width_cv_pass=None,
                 fit_rmse_px=None, fit_rmse_max_px=self.observation_rmse_max_px,
                 fit_residual_pass=None, width_hint_error_px=None,
                 width_hint_tolerance_px=max(48.0, self.lane_width_tol_px * 1.6),
                 width_hint_pass=None, confidence=None, confidence_min=None,
                 observation_quality=None, measurement_quality_min=self.measurement_quality_min,
                 single_width_age_s=None, single_width_max_age_s=self.single_width_max_age_s,
                 single_width_frames=None, weighted_score=None,
                 reject_stage='scan', reject_reason='pending')
        self._scan_diagnostics[name] = d
        return d

    @staticmethod
    def _reject_scan(d, stage, reason):
        d.update(reject_stage=stage, reject_reason=reason)

    def _diag_for_result(self, result):
        name = result.get('band_name', 'simple')
        d = result.get('_scan_diagnostics')
        if d is None:
            d = self._scan_diagnostics.get(name)
        if d is None:
            d = self._new_scan_diagnostics(name)
        elif 'attempted' not in d:
            defaults = self._new_scan_diagnostics(name)
            for key, value in defaults.items():
                d.setdefault(key, value)
        self._scan_diagnostics[name] = d
        result['_scan_diagnostics'] = d
        return d

    def _passes_band_confidence(self, result, minimum):
        d = self._diag_for_result(result)
        d.update(confidence=float(result['conf']), confidence_min=minimum)
        accepted = result['conf'] >= minimum
        if not accepted:
            self._reject_scan(d, 'confidence', 'confidence')
        return accepted

    def _scan_band_midline(self, gray, bgr, black_th, track_is_dark,
                           hint_x, lane_width_hint,
                           y_start_ratio, y_end_ratio, max_rows, row_step,
                           gray_raw=None, sample_rows=None, diagnostics=None):
        """Scan a band of rows on the birdseye, find midline per row.

        gray_raw: 未做黑帽/掩膜处理的鸟瞰灰度，只给质心兜底用。gray 上的线是
        「亮峰压在 0 背景上」且被连通域掩膜削过，质心需要的是「灰底上的暗凹陷」。
        """
        d = diagnostics if diagnostics is not None else {}
        d.update(rows_scanned=0, run_candidate_rows=0, candidate_rows=0,
                 candidate_pair_rows=0, no_candidate_rows=0, centroid_pair_rows=0,
                 continuity_reject_rows=0, center_jump_reject_rows=0)
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
        modes = []
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
        scan_rows = range(y_start,y_end+1,row_step) if sample_rows is None else sample_rows
        for y in scan_rows[:max_rows]:
            red_block, black_block = self._detect_row_blocker(
                gray, bgr, y, x0, x1, black_th, track_is_dark
            )
            if red_block:
                red_block_rows += 1
                rows_done += 1
                continue
            if black_block:
                black_block_rows += 1
                rows_done += 1
                continue

            runs = self._collect_track_runs_on_row(
                gray, y, x0, x1, black_th, track_is_dark, diagnostics=d
            )
            d['run_candidate_rows'] += bool(runs)
            chosen = self._choose_pair_center_from_runs(
                runs, last_center, last_width, x0, x1, diagnostics=d
            )
            # 阈值配不成对时先试质心：它实打实量出两条线的位置，
            # 信息量高于下面的单线盲推（后者只测到一条线，另一条按线宽硬挪）。
            if chosen is None and gray_raw is not None:
                chosen = self._centroid_pair_center(
                    gray_raw, y, last_center, last_width, x0, x1, diagnostics=d
                )
            if chosen is None and len(runs) >= 1:
                best_run = self._choose_single_run_near_hint(runs, last_center)
                if best_run is not None:
                    chosen = self._infer_center_from_single_run(
                        best_run, last_width, x0, x1
                    )

            if chosen is None:
                d['no_candidate_rows'] += 1
            if chosen is not None:
                d['candidate_rows'] += 1
                mode = int(chosen.get("line_mode", 1))
                d['candidate_pair_rows'] += mode >= 2
                d['centroid_pair_rows'] += mode == 3
                center_px = chosen["center_px"]
                lane_w = chosen["lane_width_px"]
                # 空间连续性：和上一行的 center 比，跳跃太大说明是噪声
                if len(centers_px) > 0:
                    jump = abs(center_px - last_center)
                    if jump > max(30.0, lane_w * 1.2):
                        d['continuity_reject_rows'] += 1
                        rows_done += 1
                        continue
                if abs(center_px - last_center) > (self.max_center_jump_px * 2.2):
                    d['center_jump_reject_rows'] += 1
                    rows_done += 1
                    continue
                x_cm, z_cm = self._px_to_ground_cm(center_px, y)
                mode = int(chosen.get("line_mode", 1))
                centers_px.append(center_px)
                centers_cm.append(x_cm)
                lane_widths.append(lane_w)
                ys.append(y)
                zs_cm.append(z_cm)
                modes.append(mode)
                conf_sum += chosen["conf"]
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

        d.update(rows_scanned=rows_done, accepted_rows=len(centers_px),
                 paired_rows=pair_rows, single_rows=single_rows,
                 red_block_rows=red_block_rows, black_block_rows=black_block_rows,
                 pair_fraction=pair_rows / float(max(1, pair_rows + single_rows)),
                 pair_support_ratio=pair_rows / float(max(1, rows_done)),
                 min_accepted_rows=3, center_jump_max_px=self.max_center_jump_px * 2.2)
        if len(centers_px) < 3:
            reason = ('rows_blocked' if rows_done and red_block_rows + black_block_rows == rows_done
                      else 'no_candidates' if not d['candidate_rows']
                      else 'center_jump' if not centers_px
                      else 'insufficient_rows')
            self._reject_scan(d, 'scan', reason)
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
        d.update(confidence=conf, width_std_px=width_std,
                 width_std_max_px=self.width_std_max)
        self._reject_scan(d, 'scan', 'candidate')

        return {
            "_scan_diagnostics": d,
            "center_cm": center_cm,
            "center_px": center_px,
            "dist_cm": dist_cm,
            "lane_width_px": lane_width_px,
            "weight": 1.0,
            "angle": angle,
            "conf": conf,
            "pair_ratio": pair_ratio,
            "scan_rows": rows_done,
            "pair_support_ratio": pair_rows / float(max(1, rows_done)),
            "single_ratio": single_ratio,
            "red_block_ratio": red_block_rows / float(max(1, max_rows)),
            "black_block_ratio": black_block_rows / float(max(1, max_rows)),
            "ys_list": ys,
            "centers_list": centers_px,
            "modes_list": modes,
            "widths_list": lane_widths,
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
            gray_raw=gray_raw, diagnostics=self._new_scan_diagnostics('simple'),
        )
        if base is None:
            return None

        return base

    # ═══════════════════════════════════════════════════════════
    # Two-band cascade detection (hard-coded birdseye rows)
    # ═══════════════════════════════════════════════════════════

    def _detect_two_band_lanes(self, gray, bgr, black_th, track_is_dark,
                                hint_x, lane_width_hint, gray_raw=None):
        """Scan the near band and the configurable far observation band."""
        band_specs = [
            ("low", (self.band_low_y0 + 0.5) / float(self.bird_h),
                    (self.band_low_y1 + 0.5) / float(self.bird_h),
             self.band_rows_low, self.band_step_low, self.band_weight_low),
            # Half-pixel margin preserves integer rows through int(ratio * height).
            ("mid", (self.band_mid_y0 + 0.5) / float(self.bird_h),
                    (self.band_mid_y1 + 0.5) / float(self.bird_h),
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
                gray_raw=gray_raw, diagnostics=self._new_scan_diagnostics(name),
                **({'sample_rows': self._heading_region_rows[0 if name == 'low' else 1]}
                   if self._heading_region_rows is not None else {}),
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

    def _paired_geometry(self, ys, centers, widths, hint_width, diagnostics=None):
        """Pair geometry quality, independent of distance from the image centre."""
        if len(ys) < 3 or len(ys) != len(centers) or len(widths) != len(ys):
            if diagnostics is not None:
                self._reject_scan(diagnostics, 'geometry', 'insufficient_geometry_rows')
            return False, 0.0, float("inf"), float("inf")
        a, b = line_fit(ys, centers)
        residual = math.sqrt(sum((x - (a * y + b)) ** 2
                                 for y, x in zip(ys, centers)) / len(ys))
        ground_widths = [w * self.cm_per_px_at(y) for y, w in zip(ys, widths)]
        width_mean = sum(ground_widths) / len(ground_widths)
        width_cv = stdev(ground_widths) / max(width_mean, 1e-6)
        width_err = abs(float(median(widths)) - hint_width) if hint_width > 0 else 0.0
        width_tol = max(48.0, self.lane_width_tol_px * 1.6)
        valid = (width_cv <= self.lock_width_cv_max and
                 residual <= self.observation_rmse_max_px and width_err <= width_tol)
        if diagnostics is not None:
            diagnostics.update(width_cv=width_cv, fit_rmse_px=residual,
                               width_hint_error_px=width_err, width_hint_tolerance_px=width_tol,
                               width_cv_pass=width_cv <= self.lock_width_cv_max,
                               fit_residual_pass=residual <= self.observation_rmse_max_px,
                               width_hint_pass=width_err <= width_tol)
            if not valid:
                reason = ('width_cv' if width_cv > self.lock_width_cv_max
                          else 'fit_residual' if residual > self.observation_rmse_max_px
                          else 'width_hint')
                self._reject_scan(diagnostics, 'geometry', reason)
        quality = (math.exp(-0.5 * (width_cv / self.lock_width_cv_max) ** 2) *
                   math.exp(-0.5 * (residual / self.observation_rmse_max_px) ** 2))
        return valid, quality if valid else 0.0, width_cv, residual

    def _detect_bottom_center_lock(self, gray, bgr, black_th, track_is_dark):
        d = self._new_scan_diagnostics('bottom_lock')
        empty = {"valid": False, "quality": 0.0, "pair_ratio": 0.0,
                 "center_px": float(self.center_x), "center_err_px": 0.0,
                 "symmetry_abs_px": 0.0, "centers_list": [], "center_ys": [],
                 "width_cv": 0.0, "fit_rmse_px": 0.0}
        if not self.bottom_lock_enable:
            d['attempted'] = False
            self._reject_scan(d, 'not_run', 'disabled')
            return empty
        centers, ys, widths = [], [], []
        rows_done = 0
        y = int(clamp(self.bottom_lock_start_ratio * self.bird_h, 0, self.bird_h - 1))
        hint_width = float(self._state["last_lane_width_px"])
        hint_center = float(self._state["last_lane_center_x"])
        lock_rows = (range(y,self.bird_h,max(1,self.bottom_lock_step))[:max(1,self.bottom_lock_rows)]
                     if self._heading_region_rows is None else self._heading_region_rows[0])
        for y in lock_rows:
            red, black = self._detect_row_blocker(gray, bgr, y, 0, self.bird_w - 1,
                                                   black_th, track_is_dark)
            d['red_block_rows'] += bool(red)
            d['black_block_rows'] += bool(black and not red)
            if not red and not black:
                runs = self._collect_track_runs_on_row(gray, y, 0, self.bird_w - 1,
                                                       black_th, track_is_dark, diagnostics=d)
                d['run_candidate_rows'] += bool(runs)
                chosen = self._choose_pair_center_from_runs(
                    runs, hint_center, hint_width, 0, self.bird_w - 1, diagnostics=d)
                if chosen is None:
                    d['no_candidate_rows'] += 1
                if chosen is not None:
                    centers.append(float(chosen["center_px"]))
                    widths.append(float(chosen["lane_width_px"]))
                    ys.append(y)
            rows_done += 1
        d.update(rows_scanned=rows_done, candidate_rows=len(centers),
                 candidate_pair_rows=len(centers), accepted_rows=len(centers),
                 paired_rows=len(centers), pair_fraction=1. if centers else 0.,
                 pair_support_ratio=len(centers) / float(max(1, rows_done)))
        if not centers:
            reason = ('rows_blocked' if rows_done and d['red_block_rows'] + d['black_block_rows'] == rows_done
                      else 'no_candidates')
            self._reject_scan(d, 'scan', reason)
            return empty
        ratio = len(centers) / float(rows_done)
        valid, quality, width_cv, residual = self._paired_geometry(
            ys, centers, widths, hint_width, diagnostics=d)
        if valid and ratio < self.bottom_lock_min_pair_ratio:
            self._reject_scan(d, 'pairing', 'pair_support_ratio')
        valid = valid and ratio >= self.bottom_lock_min_pair_ratio
        if valid:
            self._reject_scan(d, 'accepted', 'accepted')
        center = float(median(centers))
        return {"valid": valid, "quality": ratio * quality if valid else 0.0,
                "pair_ratio": ratio, "center_px": center,
                "center_err_px": center - self.center_x,
                "symmetry_abs_px": abs(center - self.center_x),
                "centers_list": centers, "center_ys": ys,
                "width_cv": width_cv, "fit_rmse_px": residual}

    def _qualified_band(self, result):
        """Accept paired geometry, or a single edge supported by recent paired width."""
        d = self._diag_for_result(result)
        r = dict(result)
        ys = r.get("ys_list", [])
        centers = r.get("centers_list", [])
        modes = r.get("modes_list", [])
        widths = r.get("widths_list", [r["lane_width_px"]] * len(ys))
        if not (len(ys) == len(centers) == len(modes) == len(widths)) or len(ys) < 3:
            self._reject_scan(d, 'scan', 'insufficient_rows')
            return None
        paired = [i for i, mode in enumerate(modes) if mode >= 2]
        d.update(accepted_rows=len(ys), paired_rows=len(paired), single_rows=len(ys)-len(paired),
                 pair_fraction=len(paired)/len(ys),
                 pair_support_ratio=len(paired)/max(1, int(r.get('scan_rows', len(ys)))))
        if len(paired) / len(ys) >= self.bottom_lock_min_pair_ratio:
            if len(paired) / max(1, int(r.get("scan_rows", len(ys)))) < self.bottom_lock_min_pair_ratio:
                self._reject_scan(d, 'pairing', 'pair_support_ratio')
                return None
            py, px, pw = ([values[i] for i in paired] for values in (ys, centers, widths))
            valid, geometry_q, cv, residual = self._paired_geometry(
                py, px, pw, float(self._state["last_lane_width_px"]), diagnostics=d)
            if not valid:
                return None
            # Inferred single-edge points never define a paired centre or heading.
            r.update(ys_list=py, centers_list=px, widths_list=pw,
                     modes_list=[2] * len(py), center_px=float(median(px)),
                     center_cm=float(median([self._px_to_ground_cm(x, y)[0]
                                            for y, x in zip(py, px)])),
                     lane_width_px=float(median(pw)), observation_paired=True,
                     observation_quality=float(r["conf"]) * geometry_q,
                     observation_rmse_px=residual, observation_width_cv=cv)
        else:
            last_pair = self._state["last_paired_tracking_time"]
            supported = (self._state["paired_width_frames"] >= 2 and last_pair is not None and
                         self._observation_clock_s - last_pair <= self.single_width_max_age_s)
            single_side = bool(r.get("left_seen")) != bool(r.get("right_seen"))
            d.update(single_width_frames=self._state['paired_width_frames'],
                     single_width_age_s=(self._observation_clock_s-last_pair if last_pair is not None else None))
            if paired or not supported or not single_side:
                reason = ('pair_fraction' if paired
                          else 'single_width_history_missing' if self._state['paired_width_frames'] < 2 or last_pair is None
                          else 'single_width_history_stale' if not supported
                          else 'single_side_ambiguous')
                self._reject_scan(d, 'pairing' if paired else 'single_support', reason)
                return None
            a, b = line_fit(ys, centers)
            residual = math.sqrt(sum((x - a*y - b)**2 for y,x in zip(ys,centers))/len(ys))
            d.update(fit_rmse_px=residual, fit_residual_pass=residual <= self.observation_rmse_max_px)
            if residual > self.observation_rmse_max_px:
                self._reject_scan(d, 'geometry', 'fit_residual')
                return None
            r.update(observation_paired=False,
                     observation_quality=min(self.single_quality_max, float(r["conf"]) * 0.5),
                     observation_rmse_px=residual, observation_width_cv=0.0)
        d['observation_quality'] = r['observation_quality']
        if r["observation_quality"] >= self.measurement_quality_min:
            self._reject_scan(d, 'accepted', 'accepted')
            return r
        self._reject_scan(d, 'measurement_quality', 'measurement_quality')
        return None

    def trusted_heading_points(self, results, bottom_lock):
        """Return sorted (bird rows, centre columns) from qualified paired evidence.

        Preserve the P1 deduplication contract: a band point takes precedence over
        a lock at the same row, and only an enabled, valid lock contributes.
        """
        points = {}
        for r in results:
            if r.get("observation_paired", False):
                for y, x in zip(r["ys_list"], r["centers_list"]):
                    points[float(y)] = float(x)
        if self.bottom_lock_enable and bottom_lock.get("valid", False):
            for y, x in zip(bottom_lock["center_ys"], bottom_lock["centers_list"]):
                # Band points have already been accepted; a lock does not overwrite them.
                points.setdefault(float(y), float(x))
        ys = sorted(points)
        return ys, [points[y] for y in ys]

    def _fit_trusted_heading(self, results, bottom_lock):
        """Legacy pixel heading kept unchanged for existing callers and diagnostics."""
        ys, xs = self.trusted_heading_points(results, bottom_lock)
        if len(ys) < self.heading_min_points:
            return 0.0, False, 0.0
        if ys[-1] - ys[0] < self.heading_min_span_px:
            return 0.0, False, 0.0
        a, b = line_fit(ys, xs)
        rmse = math.sqrt(sum((x-a*y-b)**2 for y,x in zip(ys,xs))/len(ys))
        valid = rmse <= self.observation_rmse_max_px
        return math.degrees(math.atan(a * self._asp)) if valid else 0.0, valid, rmse

    def _fit_ground_line(self, gx, gz):
        """Least-squares x(z) over ground points; positive heading is left.

        None when the points cannot define a direction. Shared by the paired
        control heading and the single-edge loss fallback.
        """
        z_mean, x_mean = float(np.mean(gz)), float(np.mean(gx))
        dz = gz - z_mean
        denominator = float(np.dot(dz, dz))
        if denominator <= 1e-9:
            return None
        slope = float(np.dot(dz, gx - x_mean)) / denominator
        intercept = x_mean - slope * z_mean
        residual = float(np.sqrt(np.mean((gx - (slope * gz + intercept)) ** 2)))
        if not all(math.isfinite(v) for v in (slope, intercept, residual)):
            return None
        return slope, intercept, residual

    def _fit_ground_control_heading(self, results, bottom_lock):
        """Fit actual per-row ground (x,z) coordinates; positive heading is left.

        These are calibrated-camera estimates, including the current lateral scale,
        not measured robot yaw. Retain the same trusted points and pixel quality
        gate as P1. Ground residual and depth span are independent audit fields.
        """
        ys, xs = self.trusted_heading_points(results, bottom_lock)
        out = {"heading_control_deg": 0.0, "heading_control_valid": False,
               "heading_control_reject_reason": "no_paired_points",
               "heading_control_rmse_cm": 0.0, "heading_control_z_span_cm": 0.0,
               "heading_control_points": len(ys), "heading_control_slope_dx_dz": 0.0,
               "heading_control_intercept_cm": 0.0,
               "heading_control_pixel_rmse_px": 0.0,
               "heading_control_source": "ground_x_z"}
        if not ys:
            return out
        ground = np.asarray([self._px_to_ground_cm(x, y) for y, x in zip(ys, xs)],
                            dtype=np.float64)
        if not np.all(np.isfinite(ground)):
            out['heading_control_reject_reason'] = 'nonfinite_ground_coordinates'
            return out
        gx, gz = ground[:, 0], ground[:, 1]
        out["heading_control_z_span_cm"] = float(np.ptp(gz))
        _legacy, pixel_valid, pixel_rmse = self._fit_trusted_heading(results, bottom_lock)
        out["heading_control_pixel_rmse_px"] = pixel_rmse
        if not pixel_valid:
            out['heading_control_reject_reason'] = ('insufficient_points' if len(ys)<self.heading_min_points
                else 'insufficient_span' if ys[-1]-ys[0]<self.heading_min_span_px else 'pixel_residual')
            return out
        fit = self._fit_ground_line(gx, gz)
        if fit is None:
            out['heading_control_reject_reason'] = 'degenerate_ground_fit'
            return out
        slope, intercept, residual = fit
        out.update(heading_control_deg=-math.degrees(math.atan(slope)),
                   heading_control_valid=True, heading_control_rmse_cm=residual,
                   heading_control_slope_dx_dz=slope,
                   heading_control_intercept_cm=intercept, heading_control_reject_reason='accepted')
        return out

    def _fit_single_edge_heading(self, near):
        """Direction of the one boundary still in view, as a loss fallback.

        用近带里"单边 + 近期配对宽度"推出来的中心点做地面直线拟合：这些点平行
        于那条边界，方向就是单线的方向。只喂丢线兜底（丢掉配对几何时沿着它
        继续走），不写 heading_control_*，也不参与 curve_mode / preview ——
        配对几何的契约不变。
        """
        out = {"single_edge_valid": False, "single_edge_heading_deg": 0.0,
               "single_edge_side": near.get("single_side"),
               "single_edge_rmse_cm": 0.0, "single_edge_z_span_cm": 0.0,
               "single_edge_points": 0, "single_edge_near_cm": 0.0,
               "single_edge_z_cm": 0.0}
        ys = [float(y) for y in near.get("ys_list", [])]
        xs = [float(x) for x in near.get("centers_list", [])]
        out["single_edge_points"] = len(ys)
        if len(ys) < self.heading_min_points or max(ys) - min(ys) < self.heading_min_span_px:
            return out
        ground = np.asarray([self._px_to_ground_cm(x, y) for y, x in zip(ys, xs)],
                            dtype=np.float64)
        if not np.all(np.isfinite(ground)):
            return out
        gx, gz = ground[:, 0], ground[:, 1]
        out["single_edge_z_span_cm"] = float(np.ptp(gz))
        fit = self._fit_ground_line(gx, gz)
        if fit is None:
            return out
        slope, _intercept, residual = fit
        out.update(single_edge_valid=True,
                   single_edge_heading_deg=-math.degrees(math.atan(slope)),
                   single_edge_rmse_cm=residual,
                   single_edge_near_cm=float(near["center_cm"]),
                   single_edge_z_cm=float(near["dist_cm"]))
        return out

    def _derive_narrow_gate(self, red_detected, red_z_cm, start_z_cm):
        """Obstacle-only events stay live independently of lane tracking quality."""
        red_visible = red_detected and red_z_cm > 0
        line_visible = start_z_cm > 0
        direction = -1 if red_visible and red_z_cm <= self.narrow_red_enter_z_cm else 0
        if ((red_visible and red_z_cm < self.narrow_red_close_z_cm) or
                (line_visible and start_z_cm < self.narrow_line_close_z_cm)):
            direction = 1
        from_red = red_z_cm + 24.0 if red_visible else 0.0
        from_line = start_z_cm - 60.0 if line_visible else 0.0
        exit_z = 0.0
        if from_red > 0 and from_line > 0:
            exit_z = (0.5 * (from_red + from_line)
                      if abs(from_red - from_line) < 15.0 else from_red)
        elif from_red > 0:
            exit_z = from_red
        elif from_line > 0:
            exit_z = from_line
        enter_z = max(exit_z - 46.0, 0.0) if exit_z > 0 else 0.0
        if red_detected and start_z_cm > 0:
            expected = start_z_cm - 36.0
            if abs(red_z_cm - expected) < 15.0:
                red_z_cm = 0.5 * (red_z_cm + expected)
        robot_z = self.z_cm_at(self.NEAR_BAND_ROW)
        return {"narrow_gate_detected": direction != 0, "narrow_gate_dir": direction,
                "narrow_red_visible": bool(red_visible), "narrow_line_visible": line_visible,
                "ng_exit_z": exit_z, "ng_enter_z": enter_z,
                "inside_narrow": enter_z > 0 and enter_z < robot_z < exit_z,
                "red_bar_z_cm": red_z_cm}

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
        """How far to trust this band's centre, from its paired fraction.

        pair_ratio cannot tell "one boundary is out of frame" from "the pair failed to
        lock": pair_ratio + single_ratio is 1 by construction, so the old pair_ratio
        penalty fired only on the legitimate single-boundary band and never on the case
        it was written for. Three factors there came to 0.42 on exactly the frames the
        side rule now handles deliberately, and the same product is the weight the
        error EMA is updated by.
        """
        pair_ratio = float(r.get("pair_ratio", 0.0))
        return clamp(self.one_line_quality
                     + (1.0 - self.one_line_quality) * pair_ratio, 0.20, 1.00)

    # ═══════════════════════════════════════════════════════════
    # Main process entry
    # ═══════════════════════════════════════════════════════════

    def process(self, bgr, *, dt=None):
        """Process one BGR frame, with optional elapsed dt in seconds.

        Omitted dt uses monotonic time (first frame 0.1s). Only current qualified
        geometry sets measurement_valid; historical error is exposed for diagnosis
        and never represents a fresh lane observation. Positive near/preview are
        image-right; positive fused error is a left steering demand.

        Returns:
            dev_px:      lateral deviation in birdseye px (positive = track center right of robot)
            heading_deg: trusted paired heading in degrees (positive = lane ahead left)
            conf:        confidence 0-1
            vis:         BGR visualization (bird_w x bird_h)
            debug:       diagnostic info dict
        """
        now = time.monotonic()
        if dt is None:
            dt = 0.1 if self._last_process_time is None else now - self._last_process_time
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("process dt must be finite seconds >= 0")
        self._last_process_time = now
        self._observation_clock_s += dt
        self._scan_diagnostics = {}
        for name in ('simple', 'low', 'mid', 'bottom_lock'):
            d = self._new_scan_diagnostics(name)
            d['attempted'] = False
            self._reject_scan(d, 'not_run', 'not_run')
        state = self._state
        previous_time = state["last_accepted_tracking_time"]
        prior_age = (self._observation_clock_s - previous_time
                     if previous_time is not None else float("inf"))
        state["startup_frames"] += 1

        # Match the current camera frame to the archived auto-exposure moments
        # before the unchanged IPM, thresholding, and color-detection pipeline.
        self._photometry = measure(np.max(bgr, axis=2), self.photometric_mode,
                                   MAX_CHANNEL_REFERENCE)
        bgr = self._photometry.match_image(bgr)

        # ── Step 1: Warp to birdseye (single warp, derive gray on birdseye) ──
        bgr_bird = cv2.warpPerspective(bgr, self.M, (self.bird_w, self.bird_h))
        # Neutral padding prevents the missing sensor area becoming a dark edge.
        valid = self.ground_valid_mask
        fill = np.median(bgr_bird[valid], axis=0) if np.any(valid) else np.array([255]*3)
        bgr_bird[~valid] = fill
        # Custom grayscale on birdseye: max of max(R,G,B) and standard grayscale
        gray_max = np.max(bgr_bird, axis=2)
        gray_std = cv2.cvtColor(bgr_bird, cv2.COLOR_BGR2GRAY)
        gray = np.maximum(gray_max, gray_std)
        img_w = self.bird_w
        img_h = self.bird_h
        img_cx = self.center_x
        gray_detect, binary_clean, black_th, preprocess_debug = extract_lane_candidates(
            gray, self.preprocess_mode, self.th_offset, self.th_min, self.th_max,
            photometry=self._photometry, adaptive_c=self.adaptive_c)
        gray_detect[~valid] = 0
        binary_clean[~valid] = 0
        preprocess_debug["ipm_metric_cm_per_px"] = self.cm_per_px
        preprocess_debug["ipm_valid_fraction"] = float(np.mean(valid))
        preprocess_debug.update(self._photometry.diagnostics())
        preprocess_debug["centroid_min_contrast_effective"] = self._photometry.difference(
            self.centroid_min_contrast)

        # ── Step 3: Track color detection ──
        # After black-hat, lines are always bright → track_is_dark=False
        track_is_dark = False

        previous_heading_time = state["heading_rows_time"]
        previous_heading_age = (self._observation_clock_s - previous_heading_time
                                if previous_heading_time is not None else float("inf"))
        heading_trace = trace_heading(
            binary_clean,
            lambda y: self._collect_track_runs_on_row(
                gray_detect, y, 0, self.bird_w - 1, black_th, track_is_dark),
            self._px_to_ground_cm, self.cm_per_px_at,
            previous_rows=state["heading_rows"], previous_age_s=previous_heading_age,
            config=self.heading_scan)
        if heading_trace["valid"] and heading_trace["paired_rows"] >= 2:
            state["heading_rows"] = [r for r in heading_trace["rows"]
                                     if r["source"] == "paired"]
            state["heading_rows_time"] = self._observation_clock_s
        elif previous_heading_age > self.heading_scan.history_max_age_s:
            state["heading_rows"] = []
            state["heading_rows_time"] = None

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
            if res is not None and self._passes_band_confidence(res, conf_min_dyn):
                roi_results.append(res)
            elif self.two_band_mode:
                roi_results = self._detect_two_band_lanes(
                    gray_detect, bgr_bird, black_th, track_is_dark,
                    scan_hint_center, scan_hint_width, gray_raw=gray,
                )
                roi_results = [r for r in roi_results if self._passes_band_confidence(r, conf_min_dyn)]
        elif self.two_band_mode:
            roi_results = self._detect_two_band_lanes(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray,
            )
            roi_results = [r for r in roi_results if self._passes_band_confidence(r, conf_min_dyn)]
        elif self.simple_bottom_mode:
            res = self._bottom_quarter_midline(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray,
            )
            if res is not None and self._passes_band_confidence(res, conf_min_dyn):
                roi_results.append(res)

        # ── 整条车道拟合（--lane-fit，只出诊断量）──
        lane_fit = {}
        if self.lane_fit_enable:
            lane_fit = self._detect_lane_fit(
                gray_detect, bgr_bird, black_th, track_is_dark,
                scan_hint_center, scan_hint_width, gray_raw=gray)

        # Qualify the bands independently; a failed lock is never smuggled in as
        # the same unvalidated near candidate or as inferred heading points.
        roi_results = [q for r in roi_results if (q := self._qualified_band(r)) is not None]

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
        fused_err_raw = 0.0
        turn_gate = 0.0
        lock_gain = 0.0
        measurement_quality = 0.0
        measurement_valid = False
        heading_valid = False
        heading_rmse_px = 0.0
        heading_control = self._fit_ground_control_heading([], {})
        single_edge = {"single_edge_valid": False, "single_edge_heading_deg": 0.0,
                       "single_edge_side": None, "single_edge_rmse_cm": 0.0,
                       "single_edge_z_span_cm": 0.0, "single_edge_points": 0,
                       "single_edge_near_cm": 0.0, "single_edge_z_cm": 0.0}
        preview_valid = False
        preview_error_cm = 0.0
        lookahead_z_cm = 0.0
        near_z_cm = 0.0

        # ── Bottom center lock ──
        bottom_lock = self._detect_bottom_center_lock(
            gray_detect, bgr_bird, black_th, track_is_dark,
        )
        bottom_pair_ratio = float(bottom_lock.get("pair_ratio", 0.0))
        bottom_sym_err_px = float(bottom_lock.get("center_err_px", 0.0))
        center_lock_quality = float(bottom_lock.get("quality", 0.0))
        bottom_lock_valid = bool(bottom_lock.get("valid", False))

        state["last_bottom_lock_valid"] = bottom_lock_valid

        # ── Filter by total weight ──
        score_total = 0.0
        lane_reject_reason = 'no_qualified_band'
        if roi_results:
            for r in roi_results:
                score = r["weight"] * r["conf"] * self._result_quality_weight(r)
                score_total += score
                self._diag_for_result(r)['weighted_score'] = score
            if score_total <= min_weight_dyn:
                lane_reject_reason = 'total_weight'
                for r in roi_results:
                    self._reject_scan(self._diag_for_result(r), 'total_weight', 'total_weight')
                roi_results = []
            else:
                lane_reject_reason = 'accepted'

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

        # Start-line sensing remains live even on frames with no trusted lane.
        start_line_y = self._detect_start_line(gray_detect, black_th, track_is_dark)
        start_line_z = 0.0
        if start_line_y is not None:
            _, start_line_z = self._px_to_ground_cm(float(self.center_x), float(start_line_y))
        state["start_line_z"] = start_line_z

        # Defaults for lost-frame path
        ng_exit_z = 0.0
        ng_enter_z = 0.0
        inside_narrow = False
        ratio_out = False
        red_visible = False
        line_visible = False

        # ── Pixel-domain error fusion ──
        if roi_results and heading_trace["valid"]:
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

            if near.get("observation_paired", False):
                self._update_lateral_scale(near)

            near_err_cm = near["center_cm"]
            far_err_cm = far["center_cm"]
            near_err_px = near["center_px"] - img_cx
            far_err_px = far["center_px"] - img_cx
            far_err_px_saved = far_err_px  # raw far error for controller (pre-assist)

            near_err_px_pre_lock = near_err_px

            # Shake robust layer
            shake_active = self.robust_enable and (state["shake_active_frames"] > 0)
            lock_blend_scale = (
                self.robust_bottom_lock_blend_scale if shake_active else 1.0
            )

            # Bottom lock fusion
            if self.bottom_lock_enable and bottom_lock_valid:
                lock_gain = (
                    (self.bottom_lock_blend * lock_blend_scale)
                    * center_lock_quality
                )
                lock_gain = clamp(lock_gain, 0.0, 0.95)
                near_err_px = (
                    1.0 - lock_gain
                ) * near_err_px + lock_gain * bottom_sym_err_px
                lock_ys = bottom_lock.get("center_ys", [])
                lock_centers = bottom_lock.get("centers_list", [])
                if lock_ys and len(lock_ys) == len(lock_centers):
                    lock_cm = float(median([self._px_to_ground_cm(x, y)[0]
                                            for y, x in zip(lock_ys, lock_centers)]))
                else:
                    lock_cm = bottom_sym_err_px * self.cm_per_px_at(self.near_scale_row)
                near_err_cm = (1.0 - lock_gain) * near_err_cm + lock_gain * lock_cm

            near_z_cm = float(near["dist_cm"])
            far_dist_cm = far["dist_cm"]
            state["last_lane_center_x"] = clamp(
                float(img_cx + near_err_px), 0.0, float(img_w - 1)
            )
            if near.get("observation_paired", False):
                state["paired_width_frames"] += 1
                state["last_paired_tracking_time"] = self._observation_clock_s
                state["last_lane_width_px"] = clamp(
                    float(near["lane_width_px"]),
                    float(self.min_track_width),
                    float(self.max_track_width),
                )

            base_err_cm = near_err_cm
            base_err_px = near_err_px

            # Internal X(Z) slope is positive toward image right. Both published
            # heading interfaces historically use positive toward image left.
            angle_err = -heading_trace["heading_right_deg"]
            heading_valid = True
            heading_rmse_px = (heading_trace["residual_cm"] /
                               self.cm_per_px_at(self.NEAR_BAND_ROW))
            x_ref, slope, z_ref = heading_trace["fit"]
            heading_control = {
                "heading_control_deg": angle_err,
                "heading_control_valid": True,
                "heading_control_reject_reason": "accepted",
                "heading_control_rmse_cm": heading_trace["residual_cm"],
                "heading_control_z_span_cm": heading_trace["span_cm"],
                "heading_control_points": heading_trace["observed_rows"],
                "heading_control_slope_dx_dz": slope,
                "heading_control_intercept_cm": x_ref - slope * z_ref,
                "heading_control_pixel_rmse_px": heading_rmse_px,
                "heading_control_source": "ground_x_z",
            }
            if self.lane_segments_enable:
                lane_fit["fit_seg_anchored"] = False
                if (near.get("observation_paired", False)
                        and lane_fit.get("fit_seg0_valid")):
                    seg_z = lane_fit["fit_seg0_z_cm"]
                    seg_x = lane_fit["fit_seg0_x_cm"]
                    seg_h = math.radians(lane_fit["fit_seg0_heading_deg"])
                    predicted_near = seg_x - (near_z_cm - seg_z) * math.tan(seg_h)
                    disagreement = abs(predicted_near - near_err_cm)
                    lane_fit["fit_seg_near_disagreement_cm"] = disagreement
                    lane_fit["fit_seg_anchored"] = disagreement <= 6.0
                lane_fit['near_observation_paired'] = bool(near.get('observation_paired', False))
                lane_fit['near_observation_quality'] = float(near.get('observation_quality', 0.))
            if single_line:
                single_edge = self._fit_single_edge_heading(near)
            preview_valid = (near is not far and
                             near.get("observation_paired", False) and
                             far.get("observation_paired", False) and heading_valid)
            if preview_valid:
                preview_error_cm = clamp(self.preview_gain * (far_err_cm - near_err_cm),
                                         -self.preview_max_cm, self.preview_max_cm)
                lookahead_z_cm = near_z_cm + self.preview_gain * (far_dist_cm - near_z_cm)
            else:
                lookahead_z_cm = near_z_cm

            curve_px = far_err_px - near_err_px
            turn_gate = clamp(abs(curve_px) / max(self.curve_switch_px, 1.0), 0.0, 1.0)
            # Smoothed alongside the raw value. A straight's curve_px jitters several
            # px either way and its instantaneous magnitude reaches past any threshold
            # that a real curve (9-14 px) also reaches - which is why curve_mode as a
            # one-frame test came out anti-correlated with curvature. The jitter has no
            # mean and a curve does, so the average is what separates them.
            state["curve_px_ema"] = (
                self.curve_smooth_alpha * state["curve_px_ema"]
                + (1.0 - self.curve_smooth_alpha) * curve_px
            )

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

            measurement_quality = float(near["observation_quality"])
            avg_conf = measurement_quality
            measurement_valid = True
            # near_error_cm is positive when the measured lane centre is right.
            # fused_error is a correction demand: positive means steer left.
            fused_err_cm = -near_err_cm - preview_error_cm
            fused_err_raw = fused_err_cm / self.err_scale_cm
            tau_s = self.shake_filter_tau_s if shake_active else self.filter_tau_s
            if previous_time is None or prior_age > self.measurement_max_age_s:
                state["smoothed_err_cm"] = fused_err_cm
            else:
                state["smoothed_err_cm"] = time_constant_ema(
                    state["smoothed_err_cm"], fused_err_cm, dt, tau_s)
            state["last_accepted_tracking_time"] = self._observation_clock_s

            # Curve mode detection (for dual-mode PID)
            _cm = abs(curve_px) >= self.curve_switch_px
            # Legacy curve mode remains available to explicit continuous callers.
            # Only a trusted paired heading can add angular evidence in P1.
            if single_line or LineDetector._single_band_mask(band_mask):
                _cm = _cm or abs(angle_err) >= self.curve_angle_deg
            # Continuous-mode compatibility uses trusted near geometry only;
            # rejected lock offsets have no authority over mode or bias.
            if abs(base_err_px) > self.curve_force_sym_tol_px:
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
            angle_err = 0.0  # Historical heading is never a current observation
            far_dist_cm = state["last_far_dist"]
            avg_conf = 0.0
            band_mask = state["last_band_mask"]
            curve_mode = False
            narrow_gate_detected = False
            narrow_gate_score = 1.0
            narrow_gate_dir = 0

        if not measurement_valid:
            heading_control["heading_control_reject_reason"] = (
                heading_trace["reason"] if not heading_trace["valid"]
                else "near_measurement_invalid")
            heading_control["heading_control_points"] = heading_trace["observed_rows"]
            heading_control["heading_control_z_span_cm"] = heading_trace["span_cm"]

        obstacle = self._derive_narrow_gate(red_bar_detected, red_bar_z_cm, start_line_z)
        red_bar_z_cm = obstacle["red_bar_z_cm"]
        narrow_gate_detected = obstacle["narrow_gate_detected"]
        narrow_gate_dir = obstacle["narrow_gate_dir"]
        red_visible = obstacle["narrow_red_visible"]
        line_visible = obstacle["narrow_line_visible"]
        ng_exit_z = obstacle["ng_exit_z"]
        ng_enter_z = obstacle["ng_enter_z"]
        inside_narrow = obstacle["inside_narrow"]

        accepted_time = state["last_accepted_tracking_time"]
        measurement_age_s = (self._observation_clock_s - accepted_time
                             if accepted_time is not None else float("inf"))
        measurement_stale = measurement_age_s > self.measurement_max_age_s

        # Only the legacy normalized view depends on today's lateral scale.
        # The filter memory and published centimeter error keep their physical unit.
        state["smoothed_err"] = state["smoothed_err_cm"] / self.err_scale_cm

        # ── Output ──
        dev_px = base_err_px
        heading_deg = angle_err
        conf = clamp(avg_conf, 0.0, 1.0)

        # ── Visualization (on birdseye) ──
        vis = self._build_visualization(
            gray, bgr_bird, roi_results, black_th, track_is_dark,
            dev_px, heading_deg, conf, base_err_px, band_mask,
            heading_trace,
        )

        # ── Debug info ──
        binary_raw_inv = 255 - binary_clean  # invert for display: black line on white bg
        scan_debug = {f'scan_{name}_{key}': value
                      for name, values in self._scan_diagnostics.items()
                      for key, value in values.items()}
        reject_details = '|'.join(
            f"{name}:{d['reject_stage']}:{d['reject_reason']}"
            for name, d in self._scan_diagnostics.items()
            if name != 'bottom_lock' and d['attempted'] and not d['diagnostic_only']
            and d['reject_reason'] != 'accepted')
        debug = {
            **preprocess_debug, **scan_debug,
            "lane_total_weight": score_total,
            "lane_min_weight_active": min_weight_dyn,
            "lane_conf_min_active": conf_min_dyn,
            "lane_reject_reason": lane_reject_reason,
            "lane_reject_details": reject_details,
            "heading_min_points": self.heading_min_points,
            "heading_min_span_px": self.heading_min_span_px,
            "heading_rmse_max_px": self.observation_rmse_max_px,
            "bird": gray,
            "bird_color": bgr_bird,
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
            "bottom_lock_weight": lock_gain,
            "bottom_width_cv": float(bottom_lock.get("width_cv", 0.0)),
            "bottom_fit_rmse_px": float(bottom_lock.get("fit_rmse_px", 0.0)),
            "measurement_valid": measurement_valid,
            "measurement_quality": measurement_quality,
            "measurement_age_s": measurement_age_s,
            "measurement_max_age_s": self.measurement_max_age_s,
            "measurement_stale": measurement_stale,
            "filter_dt_s": dt,
            "near_error_cm": base_err_cm if measurement_valid else 0.0,
            "near_z_cm": near_z_cm,
            "heading_deg": angle_err,
            "heading_valid": heading_valid and measurement_valid,
            "heading_fit_rmse_px": heading_rmse_px,
            "heading_confidence": heading_trace["confidence"] if measurement_valid else 0.0,
            "heading_fit_residual": heading_trace["residual_cm"],
            "heading_observed_rows": heading_trace["observed_rows"],
            "heading_inferred_rows": heading_trace["inferred_rows"],
            "heading_span_cm": heading_trace["span_cm"],
            "heading_right_deg": heading_trace["heading_right_deg"],
            "heading_rows": heading_trace["rows"],
            **heading_control,
            **single_edge,
            "preview_error_cm": preview_error_cm,
            "preview_valid": preview_valid and measurement_valid,
            "lookahead_z_cm": lookahead_z_cm,
            "center_lock_quality": center_lock_quality,
            "lost_frames": state["lost_frames"],
            "startup_frames": state["startup_frames"],
            "diff_rms_px": state["diff_rms_px"],
            "shake_active_frames": state["shake_active_frames"],
            "n_roi_results": len(roi_results),
            # 整条车道拟合的诊断量（--lane-fit）。只读，不参与任何控制。
            **lane_fit,
            "near_err_px": near_err_px_pre_lock,
            "far_err_px": far_err_px_saved,
            "curve_px": curve_px,
            "curve_px_smooth": state["curve_px_ema"],
            "turn_gate": turn_gate,
            "fused_err": state["smoothed_err"],
            "fused_err_raw": fused_err_raw,
            "fused_err_cm": state["smoothed_err_cm"],
            "lateral_scale": self.lateral_scale,
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
                             band_mask, heading_trace):
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

        # Actual sampled rows: cyan=paired, orange=single-edge, gray=gap prediction.
        for row in heading_trace["rows"]:
            color = ((255, 255, 0) if row["source"] == "paired" else
                     (0, 165, 255) if row["source"] in ("left", "right") else
                     (150, 150, 150))
            y = row["y"]
            for key in ("left_x", "right_x"):
                x = row[key]
                if x is not None and 0 <= x < self.bird_w:
                    cv2.circle(vis, (int(round(x)), y), 2, color, -1)
            x = row["center_x"]
            if x is not None and 0 <= x < self.bird_w:
                cv2.drawMarker(vis, (int(round(x)), y), color,
                               cv2.MARKER_CROSS, 7, 1)
        if heading_trace["valid"]:
            x_ref, slope, z_ref = heading_trace["fit"]
            points = []
            for row in heading_trace["rows"]:
                y = row["y"]
                if y < self.heading_scan.fit_top_y:
                    continue
                ground_x = x_ref + slope * (self.z_cm_at(y) - z_ref)
                x = self.center_x + ground_x / self.cm_per_px_at(y)
                if 0 <= x < self.bird_w:
                    points.append((int(round(x)), y))
            if len(points) >= 2:
                cv2.polylines(vis, [np.asarray(points, np.int32)], False,
                              (255, 0, 255), 2)

        # Center crosshair
        cv2.line(vis, (int(self.center_x), 0), (self.center_x, self.bird_h - 1),
                 (128, 128, 128), 1)
        cv2.line(vis, (0, self.bird_h // 2), (self.bird_w - 1, self.bird_h // 2),
                 (128, 128, 128), 1)

        # Lateral deviation indicator
        dev_x = int(self.center_x + dev_px)
        cv2.line(vis, (int(self.center_x), self.bird_h - 20),
                 (self.center_x, self.bird_h - 5), (255, 255, 255), 2)
        cv2.circle(vis, (dev_x, self.bird_h - 12), 5, (0, 255, 0), -1)
        cv2.line(vis, (int(self.center_x), self.bird_h - 12),
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
            f"rowhdg={-heading_trace['heading_right_deg']:+.1f} "
            f"valid={int(heading_trace['valid'])} c={heading_trace['confidence']:.2f}",
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
