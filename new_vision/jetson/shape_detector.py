"""ShapeDetector — 2026新规则几何图卡识别（6种几何图形）。

圆形/五角星/正方形/菱形/十字形/三角形，10cm×10cm白底黑线卡，带矩形外边框。
v2 找框重构：线宽选择性预处理（blackhat核9 + 各向异性闭 + 笔画宽过滤）
+ Hough双族候选 + 量化验证（闭合度/线宽/环内含量）+ 帧间IoU跟踪。

场景：图卡平贴白色有污渍地面，赛道粗黑线（2cm）干扰。
粗线被 blackhat 核9 抑制 + 笔画宽[1.5,7]px 拒绝；细框线（0.5cm→3-5px）保留。
逐操作审阅巡线管线（line_detector_v1_warp.py 预处理链 887-910行）：
  blackhat→保留（核31→9）；adaptive→保留；black_th二次阈值→废弃（细线瓶颈）；
  close5→保留（桥接断口，改用1×5/5×1各向异性）；open5/open3→废弃（磨细线）；
  CC面积/高度过滤→废弃（删细线CC），改笔画宽判据。

接口与 QRDetector 保持一致：update(bgr) → (action_number, dbg) 或 (None, None)。
动作映射（与2025二维码1-6对应）：
  圆形=1举左手 五角星=2举右手 正方形=3抬左腿 菱形=4抬右腿 十字形=5举双手 三角形=6摇头
"""
import math
import os
import cv2
import time
from collections import deque
import numpy as np

from camera_config import load as _load_camera, to_model_z

_CAM = _load_camera()


# 固定工作分辨率（参数标定基准，与YOLO方案A一致）
WORK_W = 960
WORK_H = 540


class ShapeDetector:
    def __init__(
        self,
        stable_frames=3,        # 连续确认帧数
        cooldown_ms=3200,       # 发送冷却
        roi_ratio=1.0,          # 检测ROI：画面下roi_ratio区域（默认全图）
        debug=True,
    ):
        self.stable_frames = stable_frames
        self.cooldown_ms = cooldown_ms
        self.roi_ratio = roi_ratio
        self.debug = debug

        # ── 找框参数（集中管理）──
        self.cfg = {
            # 线宽选择性预处理
            "bh_kernel": 9,          # 线宽选择性：只增强<核的细线（找框目标）
            "adaptive_block": 31,    # 自适应阈值窗口（与巡线一致）
            "adaptive_c": -16,       # 阈值偏移
            "stroke_min": 1.0,       # 笔画宽下限px（distanceTransform中位半径×2）
            "stroke_max": 7.0,       # 笔画宽上限px（拒巡线）
            # Hough候选
            "hough_thresh": 20,      # HoughLinesP投票阈值
            "min_line": 12,          # 线段最小长度（远距56px框的边）
            "max_gap": 10,           # 线段拼接最大间距（断线桥接）
            "topk": 4,               # 每族取前K条线组合
            "lsd_min_len": 10,       # LSD线段最短长度（LSD补HoughLinesP短边盲区）
            "corner_gap": 15,        # 角点通道：交点到线段近端端点容差（远卡断口碎片差12px，10误杀）
            # 几何闸门
            "min_w": 30,             # 框最小宽（960×540，图卡56px@1.3m）
            "min_h": 12,             # 框最小高（56×14@1.3m）
            "aspect_min": 1.0,       # 宽高比下限（真实wh最低1.9；1.0兜住正视图/极端姿态，由其他验证把关）
            "aspect_max": 5.0,
            "area_min": 550,         # 面积下限（__init__ 按相机几何覆盖）
            # 面积只做便宜的预筛，真正判"是不是 10cm 卡"的是 _square_on_ground
            # （地面反投影 + 边长/边比/对角比/直角），那一套与视角和朝向无关。
            # 20000 的问题在于它比卡片自己还小：用相机模型正投影标准 10cm 卡实测，
            # 外接框面积（960×540）随朝向急剧膨胀 ——
            #   距离   正对     偏30°    偏45°
            #   14cm   26144    44100    47306
            #   18cm   20698    35295    37976
            #   20cm   18942    31941    34300   ← 停车区
            #   27cm   13328    22646    24186   ← 停车点
            # 于是 20000 会把"偏 30° 且近于 30cm"的卡全判掉，而这正是
            # --card-trigger-frac 0.5 停车之后卡片所在的距离。
            # 55000 覆盖到 12cm/偏45°（53110，再近卡片就掉出画面下沿了）。
            "area_max": 55000,
            # 相机几何（config/cameras.json，用于按距离算图卡像素面积下限）
            "cam_height_cm": _CAM["mount_height_cm"],
            "cam_pitch_deg": _CAM["pitch_deg"],
            "cam_vfov_deg": _CAM["vfov_deg"],
            "max_dist_cm": float(os.environ.get("SHAPE_MAX_DIST_CM", "80.0")),
            # 最远识别距离：面积下限由该距离的图卡投影面积决定（env 可调）
            # 触发距离：图卡进到这个距离内才允许发动作指令（拦场地误检）
            "trigger_dist_cm": float(os.environ.get("SHAPE_TRIGGER_DIST_CM",
                                                    "40.0")),
            "ang_min": 40,           # quad内角范围（度）；远桶GT实测38.6-143.5°
            "ang_max": 150,          # 原135/45误杀远桶透视压扁+旋转卡
            # 地面方形判据：4 角点反投影到地面后，图卡必须是个边长 10cm 的正方形。
            # 在透视空间量"四边等长/边接近水平"是错的 —— 卡片平躺 + 相机 45°
            # 斜视时，真实投影边比随距离从 1.5 涨到 2.4（越远越大）。反投影消掉
            # 视角后，真实图卡实测边比 1.20、对角比 1.15、角差 11°；
            # 无卡视频 448 个候选里边比 p10 就有 2.84。
            "square_side_min_cm": 6.0,       # 地面边长范围（图卡黑框外缘约 9.9cm）
            "square_side_max_cm": 14.0,
            "square_side_ratio_max": 1.35,   # 地面最长/最短边比
            "square_diag_ratio_max": 1.22,   # 地面两对角线比
            "square_angle_tol_deg": 18.0,    # 地面内角偏离 90° 的上限
            # 验证阈值（边必须几乎全在线上）
            # n_samples=16 → 每边只有 15 个采样点。原来的 0.95 要求命中 15/15，
            # 即 4 边 60 个点一个不漏 —— 真实图卡实测最差边只有 0.67。
            # 视角无关性交给上面 _square_on_ground 把关，闭合度只做像素级确认，
            # 放宽到 0.65 后 11 张真卡全检出、188 帧无卡视频仍 0 假阳性。
            "closure_total": 0.75,   # 4边采样命中率均值
            "closure_edge": 0.65,    # 单边最低命中率
            "sample_band": 1,        # 采样带半宽px（真框边几乎全在线上）
            "n_samples": 16,         # 每边采样点数
            "max_gap_frac": 0.35,    # 单边最长连续断口占总采样点比例上限
            "inner_ratio": (0.02, 0.8),  # warp后中心区图形线占比（五角星5边实测0.72）
            "warp_size": 200,
            "warp_inset": 0.14,      # warp向内收缩比例（外框环不进warp）
            "track_iou": 0.5,        # 帧间续锁IoU
            # 框顶边必须落到画面这个高度以下才算"够近"。1/3 = 下 2/3。
            # 触发只看这一条：找框本身已经把 10cm 地面方形 + max_dist_cm 卡住了，
            # 形状分类和车道位置不再参与"要不要停"。
            "presence_top_frac": 1.0 / 3.0,
            # 存在信号兜底（_presence_cue）：模糊/稍远时外框被运动模糊打断，
            # 四边凑不齐 → 整个找框通道为 0。这里改用"闭合环"结构：
            # 图卡 = 一圈细暗边框围出的亮纸面，纸面上还有暗图形；
            # 巡线/白色道线被 blackhat 打成实心粗笔画，没有大块亮孔。
            # 只喂 dbg["presence"]，不参与分类。
            #
            # ink 阈值化：必须用大核 RECT blackhat（不是 _binary_selective 的
            # 细线选择性核9）。运动模糊把 0.5cm 边框糊成 4-8px 粗、低对比的
            # 条带，核9 + 笔画宽[1.5,7] 直接把它滤掉——1314 全片 0 命中就是
            # 这个原因。核31 对"比核细的暗结构"一律响应，模糊边框与巡线都在。
            "cue_bh_kernel": 31,       # blackhat 核（RECT，O(1)，约0.6ms）
            "cue_ink_thresh": 12,      # 暗于局部最大值多少算墨
            "cue_close": 9,            # 闭运算核：把糊断的边框连成闭合环
            # 结构闸门（连通域 = 卡的外轮廓）
            "cue_side_min": 30,        # 外接矩形短边下限 px
            "cue_side_max": 340,       # 长边上限（再大不是 10cm 卡）
            "cue_aspect_max": 2.6,     # 长边/短边上限
            "cue_fill_min": 0.30,      # 连通域面积 / 外接矩形面积
            "cue_edge_margin": 4,      # 上/左/右被画面切掉的不算完整卡
            "cue_inner_min": 0.005,    # 环内亮孔占比下限（必须有纸面）
            "cue_ring_max": 20.0,      # 环厚（面积/周长）上限，拒实心粗笔画
            "cue_core_gray_min": 100.0,  # 亮孔灰度下限
            "cue_contrast_min": 4.0,   # 亮孔 − 环 灰度下限
            # 时间累积：抖动是步态频率的周期运动，单帧判定必然断续。
            # 近 N 帧里出现 M 次算"卡在前面"；再看最近 R 帧里至少有一次命中，
            # 把尾随段压到 R-1 帧，不然卡走了 presence 还挂着就是假阳性。
            "cue_hist_n": 9,
            "cue_hist_m": 3,
            "cue_hist_recent": 3,
        }

        # 动作映射: shape_name -> action_number (1-6)
        self.action_map = {
            "circle": 1,     # 举左手
            "pentagon": 2,   # 举右手
            "square": 3,     # 抬左腿
            "diamond": 4,    # 抬右腿
            "cross": 5,      # 举双手
            "triangle": 6,   # 摇头
        }

        # 状态
        self.candidate = None
        self.candidate_count = 0
        self.last_send_ms = None
        self.first_candidate_ms = None
        self.last_quad = None       # 帧间跟踪锁
        self.track = []             # 候选框历史 (t_ms, cx, cy, box_w)，形状切换时清空
        self.lane_offset_cm = None  # 机身相对赛道中心的横向偏差cm，update() 传入
        # 触发闩锁：发过指令后须先"检测不到图卡"若干帧才重新武装，
        # 否则动作结束→恢复巡线时同一张卡还在视野内，会被反复触发停车
        self.armed = True
        self.miss_count = 0
        self._geom_rejects = {}
        # 已通过连续确认的形状（"可信 flag"）。识别阶段全图累积，与位置无关；
        # 触发阶段才查位置。图卡离开或触发后清除。
        self.trusted = None
        self._lsd = None            # LSD 检测器（复用，创建有开销）
        # 存在信号时间窗（per-frame 命中 0/1）与最近一次 cue 框
        self._cue_hist = deque(maxlen=self.cfg["cue_hist_n"])
        self._cue_box = None

        # ── 面积下限：按相机几何算（图卡在 max_dist_cm 处的投影像素面积）──
        self.cfg["area_min"] = int(self._card_area_at_dist(
            self.cfg["max_dist_cm"]))

        # ── 触发门槛：按相机几何算（图卡进到 trigger_dist_cm 时的框宽/中心y）──
        self.cfg["trigger_box_w"], self.cfg["trigger_y"] = \
            self._triggers_at_dist(self.cfg["trigger_dist_cm"])

        # ── Hu 矩模板（辅助判据，不改判定树；缺失则跳过）──
        self.hu_templates = {}
        self.last_hu = None
        try:
            hu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "shape_hu_templates.npz")
            with np.load(hu_path) as f:
                self.hu_templates = {k: f[k] for k in f.files}
        except Exception:
            self.hu_templates = {}

    def _triggers_at_dist(self, z_cm):
        """触发距离 → (最小框宽px, 最小框中心y)。

        与 _card_area_at_dist 同一 pinhole 模型：图卡宽 10cm 投影为
        fx·10/zc，故像素宽可反推距离。距离越小 → 框越大、y 越靠下。
        返回的门槛即"图卡刚好进到 z_cm 时"的值，可作 >= 判据。
        """
        h = self.cfg["cam_height_cm"]
        th = math.radians(self.cfg["cam_pitch_deg"])
        vfov = math.radians(self.cfg["cam_vfov_deg"])
        fy = WORK_H / (2.0 * math.tan(vfov / 2.0))
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * WORK_W / WORK_H)
        fx = WORK_W / (2.0 * math.tan(hfov / 2.0))
        z = max(1.0, to_model_z(z_cm))
        zc = h * math.sin(th) + z * math.cos(th)
        y_c = WORK_H / 2.0 + fy * (h * math.cos(th) - z * math.sin(th)) / zc
        return fx * 10.0 / zc, y_c

    def _card_area_at_dist(self, z_cm):
        """图卡（10cm×10cm 平放地面）在水平距离 z_cm 处的工作图像素面积。

        pinhole 模型（工作图 960×540 内参）：
          v(z') = cy + fy·(h·cosθ − z'·sinθ)/(h·sinθ + z'·cosθ)
        图卡近/远边（z∓5cm）投影出垂直跨度；水平跨度 = fx·10/Zc。
        面积随距离急剧下降（近处大、远处小），用于按距离设面积门槛。
        """
        h = self.cfg["cam_height_cm"]
        th = math.radians(self.cfg["cam_pitch_deg"])
        vfov = math.radians(self.cfg["cam_vfov_deg"])
        fy = WORK_H / (2.0 * math.tan(vfov / 2.0))
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * WORK_W / WORK_H)
        fx = WORK_W / (2.0 * math.tan(hfov / 2.0))
        z_cm = to_model_z(z_cm)

        def v_of(z):
            z = max(1.0, z)
            yc = h * math.cos(th) - z * math.sin(th)
            zc = h * math.sin(th) + z * math.cos(th)
            return WORK_H / 2.0 + fy * yc / zc, zc

        v_near, _ = v_of(z_cm - 5.0)
        v_far, _ = v_of(z_cm + 5.0)
        dv = abs(v_near - v_far)
        _, zc_c = v_of(z_cm)
        du = fx * 10.0 / max(zc_c, 1.0)
        return dv * du

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def update(self, bgr_or_gray, lane_offset_cm=None):
        """返回 (action_number, debug_dict) 或 (None, None)。

        输入归一化：任意分辨率 → resize到960×540（参数标定基准，
        YOLO方案A同款）——参数与分辨率解耦；输出quad坐标映射回原图。

        lane_offset_cm: 机身相对赛道中心的横向偏差（cm，正值=赛道中心在
        画面右侧）。用于把图卡位置约束在赛道两条边线内；不传则按画面中央算。
        """
        self.lane_offset_cm = lane_offset_cm
        if len(bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_or_gray
        h0, w0 = gray.shape[:2]
        if self.roi_ratio < 1.0:
            self._roi_y0 = int(h0 * (1.0 - self.roi_ratio))
            roi = gray[self._roi_y0:, :]
        else:
            self._roi_y0 = 0
            roi = gray
        # 等比缩放到工作图（不足处补黑边）——避免非等比拉伸导致图卡变形
        rh, rw = roi.shape[:2]
        s = min(WORK_W / w0, WORK_H / rh)
        nw = max(1, int(round(w0 * s)))
        nh = max(1, int(round(rh * s)))
        roi_s = cv2.resize(roi, (nw, nh))
        if (nw, nh) == (WORK_W, WORK_H):
            gray = roi_s
        else:
            gray = np.zeros((WORK_H, WORK_W), np.uint8)
            gray[:nh, :nw] = roi_s
        # 映射回原图：x_orig = x_work·(w0/nw)，y_orig = y_work·(rh/nh) + roi_y0
        self._scale_x = w0 / nw
        self._scale_y = rh / nh

        # S1 线宽选择性二值化（线=白255）
        binary = self._binary_selective(gray)
        dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # S2 候选生成
        hsegs, lsegs = self._detect_segments(binary)
        quads = self._hough_quads(binary, hsegs)
        quads += self._lsd_quads(binary, lsegs)
        quads += self._corner_quads(binary, hsegs + lsegs)
        quads += self._cc_quads(binary)

        # S3 验证 + 评分
        # 性能：候选可能数百个（视频帧纹理），先轻量几何预筛（纯数值，
        # 不采样不warp），通过的才做完整验证（采样+DT+warp200）——
        # 实测512候选完整验证2.5s → 预筛后剩几十个
        self._geom_rejects = {}
        best, best_score, scores = None, 0.0, []
        # Which stage killed the box: no candidate at all means the binarization
        # gave nothing to work with, candidates that never reach _verify_quad mean
        # the geometry gate is too tight, and an empty scores list with candidates
        # means the verification thresholds rejected everything.
        quad_total, quad_geom = len(quads), 0
        for q in quads:
            if not self._geom_ok(q):
                continue
            quad_geom += 1
            r = self._refine_quad(binary, q)
            # refine 偶尔会把本来合格的框推坏（远处小卡的角点通道），此时退回原框
            q = r if self._geom_ok(r) else q
            v = self._verify_quad(binary, dt, q)
            if v is not None:
                score, closure = v
                scores.append((round(score, 3), closure))
                # 帧间跟踪：与上一帧候选IoU高的加分
                if self.last_quad is not None:
                    iou = self._poly_iou(q, self.last_quad)
                    if iou > self.cfg["track_iou"]:
                        score += 0.3
                if score > best_score:
                    best, best_score = q, score
        if best is not None:
            self.last_quad = best
        else:
            self.last_quad = None

        shape = None
        dbg = {"card_found": best is not None, "roi_y0": self._roi_y0,
               "roi_ratio": self.roi_ratio, "scores": scores[:8],
               "quad_total": quad_total, "quad_geom": quad_geom,
               "geom_rejects": self._geom_rejects,
               "binary": binary, "gray": gray,
               "presence_box": None, "presence_box_work": None,
               "presence_cue": 0.0, "presence_cy_frac": None}

        if best is not None:
            self._cue_hist.append(1)   # 找框成功 = 强存在证据，时间窗记命中
            warp = self._warp_card(binary, best)
            dbg["warp"] = warp
            shape = self._classify(warp, dbg)
            # quad映射回原图分辨率（找框在960×540上做，含 ROI 偏移）
            q_orig = best.astype(np.float32) * np.array(
                [self._scale_x, self._scale_y], np.float32)
            q_orig[:, 1] += self._roi_y0
            dbg["quad"] = q_orig
            dbg["quad_work"] = best  # 工作图(960×540)坐标，用于叠加在二值图上
            dbg["closure"] = best_score
            # 存在信号：只看"有没有一个够近的框"，不看它是什么形状。
            # 停下来的决定用它；是什么形状等停稳了再分类。
            dbg["box_top_work"] = float(best[:, 1].min())
            dbg["presence"] = bool(
                self.armed
                and dbg["box_top_work"] >= WORK_H * self.cfg["presence_top_frac"])
            # 框质心在画面纵向的位置（0=顶, 1=底）。调用方拿它当"够近了"的闸门：
            # 存在信号只说明前面有卡，质心压到下方才说明真的走到跟前了。
            ys = best[:, 1]
            dbg["presence_cy_frac"] = float(ys.min() + ys.max()) * 0.5 / WORK_H
        else:
            # 这一帧没有合格框，就是没有图卡 —— 不做兜底猜测。
            # （兜底拿最大连通域的外接矩形硬判形状，既没有触发权，
            #  产生的分类结果也只会污染统计）
            dbg["fallback"] = True
            shape = None
            # 找框全线为 0，才退到"细环 + 亮纸面"的存在信号兜底（省几毫秒，
            # 也避免两条通道给出不一致的框）。命中进时间窗，累积够了才认。
            box, cue_score = self._presence_cue(gray)
            self._cue_hist.append(1 if box is not None else 0)
            if box is not None:
                bx, by, bw, bh_ = box
                qw = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh_],
                               [bx, by + bh_]], np.float32)
                q_orig = qw * np.array([self._scale_x, self._scale_y],
                                       np.float32)
                q_orig[:, 1] += self._roi_y0
                self._cue_box = (qw.astype(np.int32), q_orig.astype(np.int32),
                                 cue_score)
            # Gate one - the stop decision, while the robot is still walking - is the
            # cue, not the box: the shake on the move breaks the card's border into
            # two or three strokes and no quad ever closes, so a box-only gate never
            # fires. Gate two, after the stop, is the quad and the classifier.
            dbg["presence"] = bool(self.armed and self._cue_confirmed())
            if dbg["presence"] and self._cue_box is not None:
                # 用最近一次命中框做可视化：累积确认期间框不闪
                dbg["presence_box_work"], dbg["presence_box"], \
                    dbg["presence_cue"] = self._cue_box
                ys = self._cue_box[0][:, 1]
                dbg["presence_cy_frac"] = float(ys.min() + ys.max()) * 0.5 / WORK_H
                dbg["cue_cy_frac"] = dbg["presence_cy_frac"]

        if shape is None:
            # 这一帧没检出图卡（框没进门槛，或抖动导致漏检）。不清零候选计数
            # —— 中间漏一两帧不该推翻已有的连续证据。连续 4 次都检不出，
            # 才认为卡已离开：重置计数并重新武装。
            self.miss_count += 1
            if self.miss_count >= 4:
                self.candidate = None
                self.candidate_count = 0
                self.armed = True
                self.trusted = None      # 卡已离开，撤回可信标记
            dbg["shape"] = None
            return None, dbg
        self.miss_count = 0

        dbg["shape"] = shape
        return self._confirm(shape, dbg)

    # ═══════════════════════════════════════════════════════════
    # 存在信号兜底（不要求闭合四边形）
    # ═══════════════════════════════════════════════════════════

    def _presence_cue(self, gray):
        """模糊图卡的"存在"兜底信号：一圈细暗边框围出的亮纸面。

        找框四通道全为 0 时（外框被运动模糊打断成 2-3 段，闭合度上不去），
        卡在画面里仍然是明确可见的 —— 只是"糊了"。这里不拼四边形，只认
        结构：blackhat(核31) → 墨 → 闭运算 → 连通域，要求连通域是
        「细环 + 大块亮孔」：环厚（墨面积/周长）小、亮孔占外接矩形可观、
        亮孔比环亮。图卡天然满足；巡线/白色道线是实心粗笔画，环厚超标；
        画面边缘被切掉的连通域不是"完整一张卡"，直接拒。

        返回 ((x, y, w, h), score) 或 (None, 0.0)。判据全部来自 cfg["cue_*"]。
        """
        c = self.cfg
        bh = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (c["cue_bh_kernel"],) * 2))
        ink = (bh > c["cue_ink_thresh"]).astype(np.uint8) * 255
        m = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (c["cue_close"],) * 2))
        y_lo = WORK_H * c["presence_top_frac"]
        em = c["cue_edge_margin"]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            smax, smin = max(w, h), min(w, h)
            if smin < c["cue_side_min"] or smax > c["cue_side_max"]:
                continue
            if smax > c["cue_aspect_max"] * smin:
                continue
            if y + h < y_lo:            # 只认下半屏（够近）
                continue
            # 上/左/右被切掉 → 真实尺寸未知，不给触发权。下边缘不拦：
            # 卡到了脚前本来就会从画面下沿出去，那正是最该停的时候。
            if x < em or y < em or x + w > WORK_W - em:
                continue
            if cv2.contourArea(cnt) < c["cue_fill_min"] * w * h:
                continue
            mo = np.zeros((h, w), np.uint8)
            cv2.drawContours(mo, [cnt - (x, y)], -1, 1, -1)
            inner = (mo > 0) & (m[y:y + h, x:x + w] == 0)
            if not inner.any():
                continue                 # 实心块：没有亮纸面
            inner_frac = float(inner.sum()) / float(w * h)
            if inner_frac < c["cue_inner_min"]:
                continue
            na = int((mo > 0).sum())
            ring = (na - int(inner.sum())) / max(cv2.arcLength(cnt, True), 1.0)
            if ring > c["cue_ring_max"]:
                continue                 # 环太厚 = 粗笔画，不是卡边框
            gs = gray[y:y + h, x:x + w]
            cg = float(gs[inner].mean())
            sg = float(gs[(mo > 0) & (~inner)].mean())
            if cg < c["cue_core_gray_min"] or cg - sg < c["cue_contrast_min"]:
                continue
            score = inner_frac * (cg - sg)
            if score > best_score:
                best_score, best = score, (int(x), int(y), int(w), int(h))
        return best, best_score

    def _cue_confirmed(self):
        """存在信号时间累积：近 N 帧有 M 帧命中，且最近 R 帧内还有命中。

        抖动按步态频率周期性丢帧，单帧判定必然断续；累积到 M 帧才认，中间
        漏帧不撤销。要求最近 R 帧内仍有命中是为了把尾随段压到 R-1 帧 ——
        否则卡走出画面后 presence 还会挂几帧，那些帧就是假阳性。
        """
        c = self.cfg
        n = len(self._cue_hist)
        if sum(self._cue_hist) < c["cue_hist_m"]:
            return False
        r = min(c["cue_hist_recent"], n)
        return any(self._cue_hist[i] for i in range(n - r, n))

    # ═══════════════════════════════════════════════════════════
    # S1 线宽选择性二值化
    # ═══════════════════════════════════════════════════════════

    def _binary_selective(self, gray):
        """复用YOLO预处理方案（shape_preprocess）+ 找框保留环节。

        YOLO方案：blackhat(31) → adaptive(31,-12) → close(3×3)；
        不反转（YOLO要白底黑线，找框要黑底白线=线白）。
        找框保留：各向异性闭（桥接竖边断点）、笔画宽过滤（核31会
        增强2cm巡线，必须按线宽拒掉）、细长度过滤（拒圆斑污渍）。"""
        kbh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (self.cfg["bh_kernel"],
                                         self.cfg["bh_kernel"]))
        gd = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kbh)

        binary = cv2.adaptiveThreshold(
            gd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.cfg["adaptive_block"], self.cfg["adaptive_c"])

        # close(3×3)（YOLO方案形态学）
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)

        # 各向异性闭：1×5竖桥接远距竖边断点，5×1横补角
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kv)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kh)

        binary = self._stroke_width_filter(binary)
        return self._elongation_filter(binary)

    def _elongation_filter(self, binary, area_max=60.0):
        """小CC细长度过滤：面积<area_max且长宽比<2的CC删除（圆斑污渍）。

        只过滤小面积：图形环（圆形/方形/三角）bbox长宽比≈1但面积≥80px²
        （最小卡51px的图形环），外框环段面积更大；污渍圆斑直径1-7px
        面积≤38px²——面积上限+长宽比双闸区分线/斑，不误删图形。
        向量化（stats数组numpy操作，无Python逐CC循环——视频帧CC数百）。"""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n <= 1:
            return np.zeros_like(binary)
        w = stats[1:, cv2.CC_STAT_WIDTH]
        h = stats[1:, cv2.CC_STAT_HEIGHT]
        a = stats[1:, cv2.CC_STAT_AREA]
        ratio = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
        bad = (a < area_max) & (ratio < 2.0)
        keep = np.ones(n, bool)
        keep[0] = False  # 背景
        keep[1:][bad] = False
        lut = np.zeros(n, np.uint8)
        lut[keep] = 255
        return lut[labels]

    def _stroke_width_filter(self, binary):
        """笔画宽过滤：CC内DT中位半径×2∈[1.5,7]px（拒巡线、留细框线）。

        一次 lexsort（label 主键、DT 次键）后各组 DT 已有序，中位按下标直取，
        偶数个取中间两个均值（与 np.median 逐位一致）。原实现逐 CC 调
        np.median，视频帧 CC 可达数千时是全链最大单项。"""
        dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n <= 1:
            return np.zeros_like(binary)
        lo, hi = self.cfg["stroke_min"], self.cfg["stroke_max"]
        lab_v = labels.ravel()
        dt_v = dt.ravel()
        valid = lab_v > 0
        lab_s = lab_v[valid]
        dt_s = dt_v[valid]
        counts = np.bincount(lab_s, minlength=n)
        order = np.lexsort((dt_s, lab_s))
        lab_o = lab_s[order]
        dt_o = dt_s[order]
        starts = np.searchsorted(lab_o, np.arange(n))
        mid_lo = starts + np.maximum(counts - 1, 0) // 2
        mid_hi = starts + counts // 2
        med = np.where(counts > 0,
                       (dt_o[mid_lo] + dt_o[mid_hi]) * 0.5, 0.0)
        keep = (med >= lo / 2) & (med <= hi / 2)
        keep[0] = False
        lut = np.zeros(n, np.uint8)
        lut[keep] = 255
        return lut[labels]

    # ═══════════════════════════════════════════════════════════
    # S2 候选生成
    # ═══════════════════════════════════════════════════════════

    def _detect_segments(self, binary):
        """HoughLinesP + LSD 线段一次检出，供各候选通道共用。"""
        c = self.cfg
        hough_segs = []
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                                c["hough_thresh"],
                                minLineLength=c["min_line"],
                                maxLineGap=c["max_gap"])
        if lines is not None:
            hough_segs = [tuple(int(v) for v in ln) for ln in lines[:, 0]]
        lsd_segs = []
        if self._lsd is None:
            self._lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        det = self._lsd.detect(binary)[0]
        if det is not None:
            lsd_segs = [tuple(int(v) for v in s[0]) for s in det]
        return hough_segs, lsd_segs

    def _hough_quads(self, binary, segs=None):
        """Hough主通道：线段按角度分横/竖族，极值线组合求交点成quad。"""
        c = self.cfg
        if segs is None:
            segs, _ = self._detect_segments(binary)
        horiz, vert = self._split_families(segs, c["min_line"])
        return self._quads_from_families(horiz, vert)

    @staticmethod
    def _split_families(segs, min_len, gray_band=18.0):
        """线段按角度分横/竖族；45°边界灰色带内的线双族收录。

        segs: [(x1,y1,x2,y2), ...] → (horiz, vert)，各为 [(坐标,线段),...]。
        坐标：横族用y中值（上下边），竖族用x中值（左右边）。
        """
        horiz, vert = [], []
        for x1, y1, x2, y2 in segs:
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < min_len:
                continue
            theta = abs(np.degrees(np.arctan2(dy, dx)))
            if theta > 90.0:
                theta = 180.0 - theta
            # theta∈[0,90]：0=水平 90=垂直；45°附近灰色带双族收录
            if abs(theta - 45.0) > gray_band:
                if theta < 45.0:
                    horiz.append(((y1 + y2) / 2.0, (x1, y1, x2, y2)))
                else:
                    vert.append(((x1 + x2) / 2.0, (x1, y1, x2, y2)))
            else:
                horiz.append(((y1 + y2) / 2.0, (x1, y1, x2, y2)))
                vert.append(((x1 + x2) / 2.0, (x1, y1, x2, y2)))
        return horiz, vert

    def _quads_from_families(self, horiz, vert):
        """横/竖族极值线组合成quad（top/bottom/left/right各取topk）。"""
        c = self.cfg
        if len(horiz) < 2 or len(vert) < 2:
            return []
        horiz.sort()
        vert.sort()
        top_cands = horiz[:c["topk"]]
        bot_cands = horiz[-c["topk"]:]
        lft_cands = vert[:c["topk"]]
        rgt_cands = vert[-c["topk"]:]
        quads = []
        for yt, l_top in top_cands:
            for yb, l_bot in bot_cands:
                if yb - yt < c["min_h"]:
                    continue
                for xl, l_lft in lft_cands:
                    for xr, l_rgt in rgt_cands:
                        if xr - xl < c["min_w"]:
                            continue
                        q = self._quad_from_lines(l_top, l_bot, l_lft, l_rgt)
                        if q is not None:
                            quads.append(q)
        return quads

    def _lsd_quads(self, binary, segs=None):
        """LSD辅助通道：确定性线段检测，补HoughLinesP短边漏检（远距断线）。

        HoughLinesP是概率算法（内部RNG），20-40px短边时好时坏；
        LSD确定性检出所有细线段，双族组合成quad。
        """
        c = self.cfg
        if segs is None:
            _, segs = self._detect_segments(binary)
        horiz, vert = self._split_families(segs, c["lsd_min_len"])
        return self._quads_from_families(horiz, vert)

    def _corner_quads(self, binary, segs=None, gap=None, min_len=6):
        """角点4环通道：线段端点近交成角点，4条段闭环成quad。

        远桶（卡<80px）外框断裂成碎片且卡不在画面极值处，Hough/LSD
        的"每族取极值线"组合被噪声线干扰（59/115失败样本）。本通道
        直接找"两线段交点在各自近端端点gap内"的角点，再取横上/横下
        两段共同垂直伙伴构成4环——只生成有端点支撑的quad，与卡位置
        无关。角点本身由线段端点簇确定，碎片越多角点证据越强。
        """
        if gap is None:
            gap = self.cfg["corner_gap"]
        if segs is None:
            hs, ls = self._detect_segments(binary)
            segs = hs + ls
        # 去重（网格量化O(n)）+ 长度过滤 + 数量上限
        # 上限原因：视频帧纹理线段可达6000+条，O(n²)求交爆炸
        # （实测f0: Hough 2355+LSD 3677 → 3600万次求交卡死分钟级）；
        # 图卡框线15-50px比纹理线长，按长度取top200足够。
        seen_keys = set()
        uniq = []
        for s in segs:
            if np.hypot(s[2]-s[0], s[3]-s[1]) < min_len:
                continue
            key = (s[0] // 8, s[1] // 8, s[2] // 8, s[3] // 8)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            uniq.append(s)
        uniq.sort(key=lambda s: -(np.hypot(s[2]-s[0], s[3]-s[1])))
        segs = [tuple(float(v) for v in s) for s in uniq[:200]]
        N = len(segs)
        if N < 4:
            return []
        angs = []
        ex, ey = [], []
        for (x1, y1, x2, y2) in segs:
            th = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
            angs.append(th if th <= 90 else 180 - th)
            ex.append((x1, x2))
            ey.append((y1, y2))
        # 角点：近垂直对且交点在近端端点gap内。
        # 向量化算 N×N 交点与端点距（N 上限200，矩阵 320KB 可接受），
        # 逐对纯 Python 版本在此处有 2 万次迭代 / 百万次 min-max 调用。
        S = np.asarray(segs, np.float64)
        sx1, sy1, sx2, sy2 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
        A = np.asarray(angs, np.float64)
        DA = np.abs(A[:, None] - A[None, :])
        DA = np.minimum(DA, 180.0 - DA)
        X1, Y1, X2, Y2 = sx1[:, None], sy1[:, None], sx2[:, None], sy2[:, None]
        U1, V1, U2, V2 = sx1[None, :], sy1[None, :], sx2[None, :], sy2[None, :]
        denom = (X1 - X2) * (V1 - V2) - (Y1 - Y2) * (U1 - U2)
        ok = ((DA >= 35) & (DA <= 145)) & (np.abs(denom) >= 1e-9)
        t = ((X1 - U1) * (V1 - V2) - (Y1 - V1) * (U1 - U2)) / np.where(ok, denom, 1.0)
        px = X1 + t * (X2 - X1)
        py = Y1 + t * (Y2 - Y1)
        d1 = np.minimum(np.hypot(px - X1, py - Y1), np.hypot(px - X2, py - Y2))
        d2 = np.minimum(np.hypot(px - U1, py - V1), np.hypot(px - U2, py - V2))
        ii, jj = np.nonzero(np.triu(ok & (d1 <= gap) & (d2 <= gap), 1))
        partners = [set() for _ in range(N)]
        for a_, b_ in zip(ii.tolist(), jj.tolist()):
            partners[a_].add(b_)
            partners[b_].add(a_)
        h_idx = [i for i in range(N) if angs[i] <= 62]
        v_idx = [i for i in range(N) if angs[i] >= 28]
        v_set = set(v_idx)
        ymin = [min(a, b) for a, b in ey]
        ymax = [max(a, b) for a, b in ey]
        exs = [a + b for a, b in ex]
        quads, seen = [], []
        for a in range(len(h_idx)):
            for b in range(a+1, len(h_idx)):
                i, j = h_idx[a], h_idx[b]
                # 上下边对：两段y投影须分离
                if min(ymax[i], ymax[j]) - max(ymin[i], ymin[j]) > 0:
                    continue
                common = partners[i] & partners[j]
                if len(common) < 2:
                    continue
                cl = sorted(common)
                for m in range(len(cl)):
                    for n in range(m+1, len(cl)):
                        p, q = cl[m], cl[n]
                        if p not in v_set or q not in v_set:
                            continue
                        if exs[p] > exs[q]:
                            p, q = q, p
                        qf = self._quad_from_lines(segs[i], segs[j], segs[p], segs[q])
                        if qf is None:
                            continue
                        x, y, w, h = cv2.boundingRect(qf.astype(np.int32))
                        if w < 24 or h < 10:
                            continue
                        area = cv2.contourArea(qf)
                        if area < 300 or area > 40000:
                            continue
                        if not self._quad_convex(qf):
                            continue
                        dup = any(self._quad_close(qf, s2) for s2 in seen)
                        if dup:
                            continue
                        seen.append(qf)
                        quads.append(qf)
        return quads

    @staticmethod
    def _quad_convex(q):
        """凸度：面积/凸包面积≥0.85。远距1-2px测点噪声会让正确quad
        一个角呈微凹（叉积符号翻转），严格符号判据误杀，改用面积比。"""
        pts = q.astype(np.float32)
        hull_area = cv2.contourArea(cv2.convexHull(pts))
        if hull_area <= 1e-6:
            return False
        return cv2.contourArea(pts) / hull_area >= 0.85

    @staticmethod
    def _quad_close(a, b, tol=8.0):
        """两quad是否重复：a的4角到b的最近角距离均值≤tol。

        不用bbox-IoU（斜卡bbox重叠大但多边形不重叠，会误判重复）。
        纯 Python 算术而非 numpy：本函数在候选去重里被调数万次，
        4×2 小数组上的 numpy 调用开销远大于计算本身。
        """
        pa = a.ravel().tolist() if isinstance(a, np.ndarray) else list(a)
        pb = b.ravel().tolist() if isinstance(b, np.ndarray) else list(b)
        d = 0.0
        for i in range(0, 8, 2):
            px, py = pa[i], pa[i + 1]
            best = 1e18
            for j in range(0, 8, 2):
                dx = pb[j] - px
                dy = pb[j + 1] - py
                dd = dx * dx + dy * dy
                if dd < best:
                    best = dd
            d += math.sqrt(best)
        return d * 0.25 <= tol

    @staticmethod
    def _line_intersect(l1, l2):
        (x1, y1, x2, y2), (x3, y3, x4, y4) = l1, l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def _quad_from_lines(self, top, bot, lft, rgt):
        tl = self._line_intersect(lft, top)
        tr = self._line_intersect(rgt, top)
        br = self._line_intersect(rgt, bot)
        bl = self._line_intersect(lft, bot)
        if any(p is None for p in (tl, tr, br, bl)):
            return None
        return np.array([tl, tr, br, bl], np.float32)

    def _cc_quads(self, binary):
        """CC辅助通道：连通域四边形拟合（近卡冗余，远距断线时失效）。"""
        c = self.cfg
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if not (3 <= len(approx) <= 6):
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w < c["min_w"] or h < c["min_h"]:
                continue
            area = cv2.contourArea(cnt)
            if not (c["area_min"] <= area <= c["area_max"]):
                continue
            box = cv2.boxPoints(rect)
            quads.append(box)
        return quads

    # ═══════════════════════════════════════════════════════════
    # S3 量化验证
    # ═══════════════════════════════════════════════════════════

    def _refine_quad(self, binary, quad, band=8):
        """逐边垂直平移±band，取沿线支撑最多的位置对准框线。

        Hough/LSD线段是框线的中心线且可能偏离数px（远距卡3-16px），
        直接把边滑到框线中心，warp后图形不与外框粘连。
        """
        h, w = binary.shape
        q = quad.astype(np.float32)
        edges = []
        for e in range(4):
            p1, p2 = q[e], q[(e + 1) % 4]
            d = p2 - p1
            L = np.hypot(*d)
            if L < 1e-6:
                return quad
            n = np.array([-d[1], d[0]]) / L
            npts = max(10, int(L * 0.6))
            base_t = np.linspace(0.0, 1.0, npts)
            best_off, best_sup = 0, -1
            for off in range(-band, band + 1):
                q1 = p1 + n * off
                q2 = p2 + n * off
                xs = q1[0] + base_t * (q2[0] - q1[0])
                ys = q1[1] + base_t * (q2[1] - q1[1])
                xi = np.clip(xs.astype(int), 0, w - 1)
                yi = np.clip(ys.astype(int), 0, h - 1)
                sup = int(np.count_nonzero(binary[yi, xi]))
                sup += int(np.count_nonzero(binary[np.clip(yi + 1, 0, h - 1), xi]))
                sup += int(np.count_nonzero(binary[np.clip(yi - 1, 0, h - 1), xi]))
                sup += int(np.count_nonzero(binary[yi, np.clip(xi + 1, 0, w - 1)]))
                sup += int(np.count_nonzero(binary[yi, np.clip(xi - 1, 0, w - 1)]))
                if sup > best_sup:
                    best_sup, best_off = sup, off
            edges.append((p1 + n * best_off, p2 + n * best_off, d / L))
        corners = []
        for i in range(4):
            a, _, da = edges[(i - 1) % 4]
            b, _, db = edges[i]
            denom = da[0] * db[1] - da[1] * db[0]
            if abs(denom) < 1e-9:
                return quad
            t = ((b[0] - a[0]) * db[1] - (b[1] - a[1]) * db[0]) / denom
            corners.append(a + da * t)
        refined = np.stack(corners)
        # 外扩（仅对过小quad）：quad边来自线段中心交点，比GT外框
        # 小1-6px/边，远卡相对损失大被area闸门拒；逐边沿外法向滑到
        # 外框外缘。支撑判据=采样点带内命中占比≥0.6（外框线全长覆盖；
        # 图形内笔画只局部斜穿，占比低，不会停错）。已正大的quad跳过。
        if cv2.contourArea(refined) < self.cfg["area_min"] * 1.5:
            ctr = refined.mean(axis=0)
            edges_out = []
            for e in range(4):
                p1, p2 = refined[e], refined[(e + 1) % 4]
                d = p2 - p1
                L = np.hypot(*d)
                if L < 1e-6:
                    break
                n = np.array([-d[1], d[0]]) / L
                mid = (p1 + p2) / 2.0
                if np.dot(n, mid - ctr) < 0:
                    n = -n
                npts = max(10, int(L * 0.6))
                base_t = np.linspace(0.0, 1.0, npts)
                # 外框线是quad外法向最外侧的全长线：扫描全程取最后一个
                # 全长覆盖位置（内侧图形笔画只局部覆盖，不会入选）
                best_off = 0
                for off in range(1, 16):
                    q1 = p1 + n * off
                    q2 = p2 + n * off
                    xs = q1[0] + base_t * (q2[0] - q1[0])
                    ys = q1[1] + base_t * (q2[1] - q1[1])
                    xi = np.clip(xs.astype(int), 0, w - 1)
                    yi = np.clip(ys.astype(int), 0, h - 1)
                    # 只取线本身命中（±1带会让线停在框外2-3px，线宽闸门dt=0拒）
                    if np.count_nonzero(binary[yi, xi]) >= npts * 0.5:
                        best_off = off
                edges_out.append((p1 + n * best_off, p2 + n * best_off, d / L))
            if len(edges_out) == 4:
                out_corners = []
                for i in range(4):
                    a, _, da = edges_out[(i - 1) % 4]
                    b, _, db = edges_out[i]
                    denom = da[0] * db[1] - da[1] * db[0]
                    if abs(denom) < 1e-9:
                        break
                    t = ((b[0] - a[0]) * db[1] - (b[1] - a[1]) * db[0]) / denom
                    out_corners.append(a + da * t)
                if len(out_corners) == 4:
                    expanded = np.stack(out_corners)
                    if cv2.contourArea(expanded) > cv2.contourArea(refined):
                        refined = expanded
        # 精调回退：远距短线卡（高16-30px）边滑到图形内平行线时，
        # 角点求交会爆开（41×27→291×427）。回退保持原quad，验证闸门把关。
        if self._poly_iou(refined, quad) < 0.4:
            return quad
        return refined

    def _quad_to_ground(self, quad):
        """四边形 4 角点反投影到地面平面，返回 [(X_cm, Z_cm), ...]。

        针孔模型的反解（与 _triggers_at_dist 同一套）：
            a = (v - cy)/fy
            Z = h(cosθ - a·sinθ) / (a·cosθ + sinθ)
            X = ((u - cx)/fx) · (h·sinθ + Z·cosθ)
        """
        c = self.cfg
        h = c["cam_height_cm"]
        th = math.radians(c["cam_pitch_deg"])
        vfov = math.radians(c["cam_vfov_deg"])
        fy = WORK_H / (2.0 * math.tan(vfov / 2.0))
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * WORK_W / WORK_H)
        fx = WORK_W / (2.0 * math.tan(hfov / 2.0))
        cx, cy = WORK_W / 2.0, WORK_H / 2.0
        pts = []
        for u, v in quad:
            a = (float(v) - cy) / fy
            den = a * math.cos(th) + math.sin(th)
            if abs(den) < 1e-6:
                return None
            z = h * (math.cos(th) - a * math.sin(th)) / den
            if z <= 1.0 or z > c["max_dist_cm"] * 2.0:
                return None
            pts.append((((float(u) - cx) / fx)
                        * (h * math.sin(th) + z * math.cos(th)), z))
        return pts

    def _square_on_ground(self, quad):
        """图卡在地面平面上必须是个正方形 —— 视角无关的判据。

        透视空间里量"四边等长"是在拿常数硬凑视角，见 cfg 里 square_* 的说明。
        """
        c = self.cfg
        g = self._quad_to_ground(quad)
        if g is None:
            return False
        sides = [math.dist(g[i], g[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 1e-6:
            return False
        if min(sides) < c["square_side_min_cm"] or max(sides) > c["square_side_max_cm"]:
            return False
        if max(sides) / min(sides) > c["square_side_ratio_max"]:
            return False
        d1 = math.dist(g[0], g[2])
        d2 = math.dist(g[1], g[3])
        if max(d1, d2) / max(min(d1, d2), 1e-6) > c["square_diag_ratio_max"]:
            return False
        for i in range(4):
            p0, p1, p2 = g[(i - 1) % 4], g[i], g[(i + 1) % 4]
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            n1 = math.hypot(v1[0], v1[1])
            n2 = math.hypot(v2[0], v2[1])
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2 + 1e-9)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if abs(ang - 90.0) > c["square_angle_tol_deg"]:
                return False
        return True

    def _reject(self, reason):
        """Tally why _geom_ok turned a candidate down; dbg carries the totals."""
        self._geom_rejects[reason] = self._geom_rejects.get(reason, 0) + 1
        return False

    def _geom_ok(self, quad):
        """轻量几何预筛（纯数值，不采样）：面积/宽高/宽高比/内角/地面方形。"""
        c = self.cfg
        q = quad.astype(np.float32)
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        # 外接框面积，与 area_min 的推导口径一致：_card_area_at_dist 算的是
        # "图卡在画面上占的外接矩形"。换成 contourArea 会小 ~20%（斜置四边形），
        # 结果把最远那批卡（面积正好卡在门槛上）全判掉。
        if not (c["area_min"] <= w * h <= c["area_max"]):
            return self._reject("area_min" if w * h < c["area_min"] else "area_max")
        if w < c["min_w"] or h < c["min_h"]:
            return self._reject("min_wh")
        aspect = max(w, h) / max(min(w, h), 1.0)
        if not (c["aspect_min"] <= aspect <= c["aspect_max"]):
            return self._reject("aspect")
        for i in range(4):
            p1 = q[(i - 1) % 4]
            p2 = q[i]
            p3 = q[(i + 1) % 4]
            v1 = p1 - p2
            v2 = p3 - p2
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if not (c["ang_min"] <= ang <= c["ang_max"]):
                return self._reject("angle")
        return True if self._square_on_ground(q) else self._reject("ground")

    def _verify_quad(self, binary, dt, quad):
        """返回 (score, closure) 或 None。闭合度/线宽/环内含量（几何闸门在_geom_ok）。"""
        c = self.cfg
        q = quad.astype(np.float32)

        # 闭合度：边采样 ±band 带内命中 + 最长连续断口
        ns = c["n_samples"]
        band = c["sample_band"]
        hits_per_edge = []
        widths = []
        for e in range(4):
            p1 = q[e]
            p2 = q[(e + 1) % 4]
            hits = 0
            gap = best_gap = 0
            for k in range(1, ns):
                t = k / ns
                px = int(round(p1[0] + t * (p2[0] - p1[0])))
                py = int(round(p1[1] + t * (p2[1] - p1[1])))
                y0 = max(0, py - band)
                y1 = min(binary.shape[0], py + band + 1)
                x0 = max(0, px - band)
                x1 = min(binary.shape[1], px + band + 1)
                if np.any(binary[y0:y1, x0:x1] > 0):
                    hits += 1
                    gap = 0
                    if (y0 < py < y1 and x0 < px < x1):
                        widths.append(dt[py, px])
                else:
                    gap += 1
                    best_gap = max(best_gap, gap)
            hits_per_edge.append(hits / (ns - 1))
            # 单边最长连续断口限制（真框实测 0-2/40，假框断口长）
            if best_gap > c["max_gap_frac"] * (ns - 1):
                return None
        closure = float(np.mean(hits_per_edge))
        if closure < c["closure_total"] or min(hits_per_edge) < c["closure_edge"]:
            return None

        # 线宽一致性
        if widths:
            med_w = 2.0 * float(np.median(widths))
            if not (c["stroke_min"] <= med_w <= c["stroke_max"]):
                return None

        # 环内含量（warp后中心区图形线占比）
        warp = self._warp_card(binary, q)
        ws = c["warp_size"]
        margin = int(ws * 0.15)
        inner = warp[margin:ws - margin, margin:ws - margin]
        ratio = np.count_nonzero(inner) / inner.size
        r_lo, r_hi = c["inner_ratio"]
        if not (r_lo <= ratio <= r_hi):
            return None

        # score：闭合度为主 + 线宽/几何符合加分
        score = closure
        if widths and 1.5 <= 2.0 * np.median(widths) <= 7.0:
            score += 0.2
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        aspect = max(w, h) / max(min(w, h), 1.0)
        if 1.5 <= aspect <= 4.0:
            score += 0.1
        return score, closure

    # ═══════════════════════════════════════════════════════════
    # 单应性矫正
    # ═══════════════════════════════════════════════════════════

    def _warp_card(self, binary, quad):
        """4角点 → warp_size×warp_size 正视图。

        角排序修复：绕质心按atan2排序后旋转到左上起点（旧版sum/diff假设
        凸四边形完整，缺角/木纹合并时排序错乱）。
        quad来自Hough线段中心线→先沿中心外扩约半个线宽，避免外框
        线被裁到warp边缘（否则RETR_LIST拿不到完整外框环）。
        """
        ws = self.cfg["warp_size"]
        q = quad.astype(np.float32)
        c = q.mean(axis=0)
        # 先外扩4px：quad可能落在框线内缘（远距1-2px细线，精调后
        # 内外缘难分），外扩到框外缘
        v = q - c
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        q = c + v / norms * (norms + 4.0)
        # 再向内收缩：外框环（卡边5%+框线5%）不进warp，图形不会与环
        # 粘连（远距卡透视压缩后图形-环间隙只有2-4px，warp后必并）。
        q = c + (q - c) * (1.0 - self.cfg["warp_inset"])
        cx, cy = q.mean(axis=0)
        ang = np.arctan2(q[:, 1] - cy, q[:, 0] - cx)
        order = np.argsort(ang)
        qs = q[order]
        # 旋转起点：使第一个角点是"左上"（x+y最小，与dst映射一致）
        start = int(np.argmin(qs.sum(axis=1)))
        qs = np.roll(qs, -start, axis=0)
        dst = np.float32([[0, 0], [ws - 1, 0], [ws - 1, ws - 1], [0, ws - 1]])
        M = cv2.getPerspectiveTransform(qs, dst)
        # INTER_NEAREST：二值图线性插值会把1-2px细笔画磨断（软边被
        # 阈值吃掉），最近邻保持笔画完整
        return cv2.warpPerspective(binary, M, (ws, ws),
                                   flags=cv2.INTER_NEAREST)

    @staticmethod
    def _poly_iou(q1, q2):
        """两个四边形的IoU（用轮廓相交近似）。"""
        p1 = q1.astype(np.int32).reshape(-1, 1, 2)
        p2 = q2.astype(np.int32).reshape(-1, 1, 2)
        r1 = cv2.boundingRect(p1)
        r2 = cv2.boundingRect(p2)
        x = max(r1[0], r2[0])
        y = max(r1[1], r2[1])
        w = min(r1[0] + r1[2], r2[0] + r2[2]) - x
        h = min(r1[1] + r1[3], r2[1] + r2[3]) - y
        inter = max(0, w) * max(0, h)
        a1 = cv2.contourArea(p1)
        a2 = cv2.contourArea(p2)
        return inter / max(a1 + a2 - inter, 1e-6)

    # ═══════════════════════════════════════════════════════════
    # 形状分类（纯 CV 规则法）
    # ═══════════════════════════════════════════════════════════

    def _classify(self, warp, dbg):
        self.last_hu = None
        shape = self._classify_shape(warp)
        dbg["shape_rules"] = shape
        dbg["hu_best"] = self.last_hu[0] if self.last_hu else None
        dbg["hu_dist"] = self.last_hu[1] if self.last_hu else None
        return shape

    def _classify_shape(self, warp):
        """先直接分类；失败且检测到边缘粘连（田字形）时裁边重试。"""
        shape = self._classify_shape_once(warp)
        if shape is not None:
            return shape
        if not self._touches_edge(warp):
            return None
        # 十字臂与外框粘连成"田"字 → 裁掉边缘 6% 切断粘连
        m = int(warp.shape[0] * 0.06)
        w2 = warp.copy()
        w2[:m, :] = 0
        w2[-m:, :] = 0
        w2[:, :m] = 0
        w2[:, -m:] = 0
        return self._classify_shape_once(w2)

    def _touches_edge(self, warp):
        """最大连通域是否接触图像边缘（粘连/外框残留的特征）。"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        w = cv2.morphologyEx(warp, cv2.MORPH_CLOSE, kernel)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        w = cv2.dilate(w, k3, iterations=1)
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(w, 8)
        if n <= 1:
            return False
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, ww, hh, _a = stats[i]
        H, W = w.shape
        return bool(x == 0 or y == 0 or x + ww >= W or y + hh >= H)

    def _classify_shape_once(self, warp):
        # 远距卡warp后笔画1-2px且有2-10px断口：close(5,5)+dilate(1px)桥接
        # 笔画碎片（五角星臂/十字臂/三角形边）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        warp = cv2.morphologyEx(warp, cv2.MORPH_CLOSE, kernel)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        warp = cv2.dilate(warp, k3, iterations=1)
        contours, _ = cv2.findContours(warp, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # 参考=卡大小（warp边长）：inset-warp后外框环不进warp，最大
        # 轮廓就是图形本体，以它为参考会排除图形自身的内边界。
        # bbox≤0.85×warp挡环残留（quad偏时环残片bbox≈0.8-0.9×warp，
        # 但其质心偏边，居中闸门也会排除）。
        size_ref = warp.shape[0]
        area_ref = float(size_ref * size_ref)
        cw, ch = warp.shape[1], warp.shape[0]
        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if a >= area_ref * 0.92:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            if (abs(cx - cw / 2) > cw * 0.15
                    or abs(cy - ch / 2) > ch * 0.15):
                continue
            ix, iy, iw, ih = cv2.boundingRect(c)
            if max(iw, ih) > size_ref * 0.95:
                continue
            cands.append((a, c, (ix, iy, iw, ih)))
        if not cands:
            return None
        # 以最大候选为主体，并集与其bbox重叠（含10px余量）的碎片/内边界：
        # 五角星/三角形的细边碎片（星臂与主体分离数px）并入主体；
        # 远离主体的轮廓（环残留等）不并，避免锯齿导致凸度虚高。
        cands.sort(key=lambda t: -t[0])
        _, main, (mx, my, mw, mh) = cands[0]
        mask = np.zeros_like(warp)
        cv2.drawContours(mask, [main], -1, 255, -1)
        for a, c, (ix, iy, iw, ih) in cands[1:]:
            if (ix < mx + mw + 10 and ix + iw > mx - 10
                    and iy < my + mh + 10 and iy + ih > my - 10):
                cv2.drawContours(mask, [c], -1, 255, -1)
        blobs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not blobs:
            return None
        blob = max(blobs, key=cv2.contourArea)
        if cv2.contourArea(blob) < 30 * 30:
            return None
        return self._classify_contour(blob)

    def _classify_shape_full(self, binary, dbg=None):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            a = cv2.contourArea(c)
            if a > best_area:
                best_area = a
                best = c
        if best is None or best_area < 30 * 30:
            return None
        if dbg is not None:
            # 兜底路径也要交框指标，否则触发门槛无从判断
            x, y, w, h = cv2.boundingRect(best)
            dbg["quad_work"] = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        return self._classify_contour(best)

    def _hu_match(self, contour):
        """Hu 矩模板匹配（辅助信息）：返回 (最接近类名, 距离) 或 None。

        注意：Hu 矩旋转不变 → 方形/菱形理论同距，仅作参考，不参与判定。
        """
        if not self.hu_templates:
            return None
        best = None
        for name, tpl in self.hu_templates.items():
            d = cv2.matchShapes(contour, tpl, cv2.CONTOURS_MATCH_I1, 0.0)
            if best is None or d < best[1]:
                best = (name, float(d))
        return best

    def _classify_contour(self, contour):
        self.last_hu = self._hu_match(contour)
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return None
        area = cv2.contourArea(contour)
        if area < 30 * 30:
            return None
        # 顶点数用粗epsilon（0.035），凹口数用细epsilon（0.02）——
        # 粗epsilon会把五角星的凹口合并掉（0.035×长周长≈17px）
        approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
        n_vertices = len(approx)
        # 凹顶点数：顶点转向与多边形环绕方向相反即凹。
        # 五角星5个凹口，十字4个，圆形/方形/菱形/三角形0个。
        n_conc = 0
        if n_vertices >= 4:
            fine = cv2.approxPolyDP(contour, 0.02 * peri, True)
            pts = fine[:, 0].astype(np.float32)
            nf = len(pts)
            sarea = 0.0
            for i in range(nf):
                p, q = pts[i], pts[(i + 1) % nf]
                sarea += p[0] * q[1] - q[0] * p[1]
            # 转向=入边(b-a)到出边(c-b)的叉积；凹顶点转向与环绕方向相反
            for i in range(nf):
                a = pts[(i - 1) % nf]
                b = pts[i]
                c = pts[(i + 1) % nf]
                cross = ((b[0] - a[0]) * (c[1] - b[1])
                         - (b[1] - a[1]) * (c[0] - b[0]))
                if cross * sarea < 0:
                    n_conc += 1
        # 圆度 4πA/P²（不依赖轮廓点密度——CHAIN_APPROX_SIMPLE 会把直线
        # 轮廓压成拐点，使"半径变异系数"对多边形失真）：
        # 圆=1.00、正方形/菱形=0.785、等边三角=0.605、五角星≈0.5、十字更小
        circularity = 4.0 * math.pi * area / max(peri * peri, 1e-6)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        fill_rect = area / max(rw * rh, 1e-6)
        ang = abs(rect[2]) % 90.0
        ang = min(ang, 90.0 - ang)
        # 凹形：十字 vs 五角星。十字臂细（fillR<0.42）；五角星凹口
        # 深且占满外接矩形（fillR 0.42-0.52）。凹口3个+fillR<0.60
        # 兜住厚臂/非对称十字（自测卡十字fillR≈0.56）
        if n_conc >= 3:
            if fill_rect < 0.42:
                return "cross"
            if n_conc >= 5:
                return "pentagon"
            if fill_rect < 0.60:
                return "cross"
            return "pentagon"
        # 凸形
        if n_vertices == 3:
            return "triangle"
        # 圆形：fill≈π/4=0.785（方形/菱形≈1.0、三角≈0.55）+ 圆度>0.8
        # 圆度用 4πA/P²（不受轮廓点密度影响）；minAreaRect 对圆的角度
        # 不确定，必须先于"菱形(ang≥25)"判定
        if 0.72 <= fill_rect <= 0.87 and circularity >= 0.80:
            return "circle"
        # 菱形 = 旋转45°的方形（minAreaRect 角度≈45°）
        if ang >= 25.0:
            return "diamond"
        if fill_rect >= 0.84:
            return "square"
        return "triangle"

    # ═══════════════════════════════════════════════════════════
    # 确认 + 冷却（与QRDetector一致）
    # ═══════════════════════════════════════════════════════════

    def _approach_score(self):
        """框中心y的下移趋势（0~1）：机器人前进时图卡从画面上方往下走。

        只作软加分记录，不参与拦截——图卡一进画面就很近时无历史，不因此扣分。
        """
        if len(self.track) < 3:
            return 0.0
        seg = self.track[-8:]
        span = seg[-1][0] - seg[0][0]
        if span <= 0:
            return 0.0
        px_per_s = (seg[-1][2] - seg[0][2]) * 1000.0 / span
        return max(0.0, min(1.0, px_per_s / 20.0))

    def _confirm(self, shape, dbg):
        now = int(time.time() * 1000)
        if dbg.get("fallback"):
            # 兜底路径的"框"是最大连通域的外接矩形，不是图卡外框，
            # 拿它算距离/位置没有意义（空场地上的大色块也能凑出大 bbox）。
            # 只在图像里显示，不给触发权。
            return None, dbg
        # ── 阶段① 识别累积：全图范围，只看形状，不看位置 ──
        if shape == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = shape
            self.candidate_count = 1
            self.first_candidate_ms = now
            self.track = []
            if self.debug:
                print(f"  [shape] NEW {shape}")
        if self.candidate_count >= self.stable_frames:
            self.trusted = shape        # 连续确认通过 → 打上可信 flag

        # 候选框指标（工作图坐标）
        qw = dbg.get("quad_work")
        cx = box_w = box_top = None
        if qw is not None:
            qw = np.asarray(qw, np.float32)
            x0, x1 = float(qw[:, 0].min()), float(qw[:, 0].max())
            y0, y1 = float(qw[:, 1].min()), float(qw[:, 1].max())
            cx, box_w, box_top = (x0 + x1) * 0.5, x1 - x0, y0
            cy = (y0 + y1) * 0.5
            self.track.append((now, cx, cy, box_w))
            if len(self.track) > 32:
                self.track = self.track[-32:]

        # ── 阶段② 触发判定：只看位置，形状由 flag 保证 ──
        if self.trusted is None or self.trusted != shape:
            return None, dbg
        if cx is None or not self.armed:
            return None, dbg

        # 位置闸门只剩一条：框顶边落到画面下 2/3（= 顶边过了上 1/3 线）。
        y_mid = WORK_H * self.cfg["presence_top_frac"]
        dbg["box_top"] = box_top
        dbg["y_mid"] = y_mid
        if box_top < y_mid:
            return None, dbg

        # 车道偏移照算，只记录不拦截。找框本身已经把"10cm 地面方形 + max_dist_cm"
        # 卡住了；机器人停稳后卡就在正前方，再拿 lane_offset_cm 去拦只会漏检。
        # px_per_cm 由框宽反推（图卡物理宽 10cm），无需外部传像素尺度。
        px_per_cm = box_w / 10.0
        lane_cx = WORK_W / 2.0 + (self.lane_offset_cm or 0.0) * px_per_cm
        half = 17.5 * px_per_cm         # 赛道半宽 17.5cm
        off = abs(cx - lane_cx)
        dbg["lane_offset_px"] = off
        dbg["lane_half_px"] = half

        # 软加分（仅记录，不拦截）
        dbg["bonus_center"] = 1.0 - off / max(half, 1e-6)
        dbg["bonus_approach"] = self._approach_score()

        ready = (self.last_send_ms is None
                 or (now - self.last_send_ms) >= self.cooldown_ms)
        if ready:
            action = self.action_map[shape]
            self.last_send_ms = now
            self.candidate_count = 0
            self.armed = False
            self.miss_count = 0
            self.trusted = None
            latency = now - (self.first_candidate_ms or now)
            dbg["action"] = action
            dbg["latency_ms"] = latency
            if self.debug:
                print(f"  [shape] >>> SEND action={action} ({shape}) "
                      f"latency={latency}ms  w={box_w:.0f} "
                      f"top={box_top:.0f}/{y_mid:.0f} "
                      f"off={off:.0f}/{half:.0f}")
            return action, dbg
        return None, dbg


# ═══════════════════════════════════════════════════════════
# 自测：合成图卡验证分类
# ═══════════════════════════════════════════════════════════

def _make_card(shape_name, size=200):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cx, cy = size // 2, size // 2
    cv2.rectangle(img, (10, 10), (size - 10, size - 10), (0, 0, 0), 3)
    c = (0, 0, 0)
    if shape_name == "circle":
        cv2.circle(img, (cx, cy), size // 3, c, 3)
    elif shape_name == "triangle":
        pts = np.array([[cx, cy - size // 3], [cx - size // 3, cy + size // 4],
                        [cx + size // 3, cy + size // 4]], np.int32)
        cv2.polylines(img, [pts], True, c, 3)
    elif shape_name == "square":
        cv2.rectangle(img, (cx - size // 3, cy - size // 3),
                      (cx + size // 3, cy + size // 3), c, 3)
    elif shape_name == "diamond":
        pts = np.array([[cx, cy - size // 3], [cx - size // 3, cy],
                        [cx, cy + size // 3], [cx + size // 3, cy]], np.int32)
        cv2.polylines(img, [pts], True, c, 3)
    elif shape_name == "pentagon":
        r_outer = size // 3
        r_inner = r_outer * 0.45
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + np.pi * i / 5
            r = r_outer if i % 2 == 0 else r_inner
            pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, c, 3)
    elif shape_name == "cross":
        w = size // 25
        arm = size // 3
        pts = [
            [cx - w, cy - arm], [cx + w, cy - arm],
            [cx + w, cy - w],   [cx + arm, cy - w],
            [cx + arm, cy + w], [cx + w, cy + w],
            [cx + w, cy + arm], [cx - w, cy + arm],
            [cx - w, cy + arm], [cx - arm, cy + w],
            [cx - arm, cy - w], [cx - w, cy - w],
        ]
        cv2.polylines(img, [np.array(pts, np.int32)], True, c, 3)
    return img


if __name__ == "__main__":
    names = ["circle", "pentagon", "square", "diamond", "cross", "triangle"]
    sd = ShapeDetector(stable_frames=1, cooldown_ms=0, debug=False)
    ok = 0
    for name in names:
        card = _make_card(name)
        # 真实场景：图卡贴地45°俯视 → 纵向压缩0.55，放白底场景
        card = cv2.resize(card, (200, int(200 * 0.55)))
        canvas = np.ones((540, 960, 3), dtype=np.uint8) * 240
        canvas[200:200 + card.shape[0], 380:380 + card.shape[1]] = card
        action, dbg = sd.update(canvas)
        expect = sd.action_map[name]
        got = dbg.get("shape")
        status = "OK" if got == name else f"FAIL(got {got})"
        if got == name:
            ok += 1
        print(f"  {name:>10} -> {str(got):>10}  action={action}  [{status}]")
    print(f"Result: {ok}/6")
