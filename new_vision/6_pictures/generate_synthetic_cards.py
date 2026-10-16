"""CV合成训练数据生成器 v2 — 6种几何图卡（白场地场景 + 正视图两种模式）。

模拟真实比赛：10cm图卡平贴白色有污渍地面，45°俯视（透视+纵向压缩），
赛道粗黑线干扰，光照不均。数据多样性升级（子代理几何推导）：
- 线宽 5%（size//20，规则0.5cm）；旧版2.5%是bug
- 图卡尺寸 scale 0.12-0.40（占帧5-16%@640，覆盖远距；30%样本<0.20）
- 透视：梯形 + 整体纵向压缩 h/w=0.3-0.6（真实0.25-0.55）
- 旋转 ±30°；图形相对外框偏移 ±12%
- 场景：白底(220-250) + 污渍斑点 + 低频斑块 + 梯形黑线（巡线）

模式：
1. 场景模式（默认）：白场地场景图 + YOLO标注（图卡bbox）+ 外框4角GT
   → 找框验证/训练
2. 正视图模式（--frontal）：96×96二值图卡 + 增强（腐蚀/膨胀断线等）
   + 负样本（--negatives）→ 96×96 分类样本（6类+背景）

用法:
    python generate_synthetic_cards.py --count 500 --out synthetic_dataset
    python generate_synthetic_cards.py --frontal --count 2000 --negatives 2000 --out frontal_dataset
"""
import argparse
import multiprocessing
import os
import random
import sys
import time as _time
import cv2
import numpy as np

_JETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jetson")
if _JETS not in sys.path:
    sys.path.insert(0, _JETS)
from shape_detector import ShapeDetector

# 6种形状: (名字, 类别id, 动作号)
SHAPES = [
    ("circle", 0, 1),      # 举左手
    ("pentagon", 1, 2),    # 举右手
    ("square", 2, 3),      # 抬左腿
    ("diamond", 3, 4),     # 抬右腿
    ("cross", 4, 5),       # 举双手
    ("triangle", 5, 6),    # 摇头
]

YAML = """names:
  0: circle
  1: pentagon
  2: square
  3: diamond
  4: cross
  5: triangle
"""

LINE_FRAC = 0.05      # 线宽 = 5% 图卡尺寸（规则0.5cm/10cm）
OFFSET_FRAC = 0.08    # 圆形/五角星/十字最大偏移（外框安全）

# 各形状最大偏移：外框内缘0.45-半框线0.025=0.425，减去图形半线宽0.025和
# 图形外沿半径（圆心0.28/0.308），留≥0.04间隙（8px@200warp，防止
# warp插值+闭运算把图形与外框粘连成同一轮廓）：
#   circle/pentagon/cross(臂1.0r): 0.425-0.025-0.28-0.04 = 0.08
#   square/diamond/triangle(1.1r): 0.425-0.025-0.308-0.04 = 0.055
# 旧版统一0.12：1.1r图形0.308+0.025+0.12=0.453>0.45压到外框线，
# warp后图形与外框粘连成一个轮廓，分类无法区分。规则要求图形不压框。


def _offset_frac(shape_idx):
    return OFFSET_FRAC if shape_idx in (0, 1, 4) else 0.055


def render_card(shape_idx, size=256, offset_frac=None):
    """渲染图卡：白底+黑图形+外框。图形中心偏移±offset_frac（不压框）。

    返回 (灰度图, 外框4角卡坐标[N,2])。
    """
    if offset_frac is None:
        offset_frac = _offset_frac(shape_idx)
    img = np.ones((size, size), dtype=np.uint8) * 255
    lw = max(2, int(size * LINE_FRAC))
    c = 0

    # 外框（5%-95%）
    cv2.rectangle(img, (int(size * 0.05), int(size * 0.05)),
                  (int(size * 0.95), int(size * 0.95)), c, lw)

    # 图形偏移
    dx = random.uniform(-offset_frac, offset_frac) * size
    dy = random.uniform(-offset_frac, offset_frac) * size
    cx, cy = size // 2 + dx, size // 2 + dy
    r = size * 0.28

    if shape_idx == 0:  # circle
        cv2.circle(img, (int(cx), int(cy)), int(r), c, lw)
    elif shape_idx == 1:  # pentagon 五角星
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + np.pi * i / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append([int(cx + rr * np.cos(ang)), int(cy + rr * np.sin(ang))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, c, lw)
    elif shape_idx == 2:  # square
        s = r * 1.1
        cv2.rectangle(img, (int(cx - s), int(cy - s)),
                      (int(cx + s), int(cy + s)), c, lw)
    elif shape_idx == 3:  # diamond
        pts = np.array([[cx, cy - int(r * 1.1)], [cx - int(r * 1.1), cy],
                        [cx, cy + int(r * 1.1)], [cx + int(r * 1.1), cy]], np.int32)
        cv2.polylines(img, [pts], True, c, lw)
    elif shape_idx == 4:  # cross — 一条横线+一条竖线（规则）
        # 臂长1.0r：0.28+0.12偏移=0.40<0.45外框，臂尖不压框
        # （原1.2r=0.336+0.12=0.456>0.45，臂尖与外框粘连，warp后
        # 形状与外框合并成一个轮廓，分类拿不到inner）
        arm = int(r * 1.0)
        cv2.line(img, (int(cx - arm), int(cy)), (int(cx + arm), int(cy)), c, lw)
        cv2.line(img, (int(cx), int(cy - arm)), (int(cx), int(cy + arm)), c, lw)
    elif shape_idx == 5:  # triangle
        pts = np.array([[cx, cy - int(r * 1.1)], [cx - int(r * 1.1), cy + int(r * 0.9)],
                        [cx + int(r * 1.1), cy + int(r * 0.9)]], np.int32)
        cv2.polylines(img, [pts], True, c, lw)

    quad = np.array([[0.05, 0.05], [0.95, 0.05],
                     [0.95, 0.95], [0.05, 0.95]], np.float32) * size
    return img, quad


def _apply_M(pts, M):
    """pts(N,2) 应用变换 M(2,3)或(3,3)。"""
    pts = np.asarray(pts, np.float32)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]])
    ones = np.ones((len(pts), 1), np.float32)
    hom = np.hstack([pts, ones]) @ M.T
    return hom[:, :2] / hom[:, 2:3]


def _crop_to_card(img):
    ys, xs = np.where(img < 250)
    if len(ys) == 0:
        return img, (0, 0)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (xs.min(), ys.min())


def _build_background(img_w, img_h, interference="full"):
    """白场地背景（无卡）：白底 + 污渍 + 斑块(full) + 梯形黑线(full)。"""
    scene = np.ones((img_h, img_w), dtype=np.uint8) * random.randint(220, 250)
    if interference in ("full", "light"):
        _add_stains(scene)
        if interference == "full" and random.random() < 0.7:
            low_res = np.random.randn(max(2, img_w // 24), max(2, img_h // 24)).astype(np.float32)
            patch = cv2.resize(low_res, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            patch = patch / max(patch.std(), 1e-6) * random.uniform(15, 35)
            scene = np.clip(scene.astype(np.float32) + patch, 0, 255).astype(np.uint8)
        if interference == "full" and random.random() < 0.6:
            cx_line = img_w // 2
            half_bottom = random.uniform(img_w * 0.15, img_w * 0.25)
            half_top = random.uniform(img_w * 0.04, img_w * 0.10)
            y_far, y_near = int(img_h * 0.15), img_h
            lw = random.randint(9, 18)  # 巡线投影10-20px@960
            for y in range(y_far, y_near, 2):
                t = (y - y_far) / max(1, y_near - y_far)
                half = half_top + (half_bottom - half_top) * t
                cv2.line(scene, (int(cx_line - half), y), (int(cx_line - half), y), 0, lw)
                cv2.line(scene, (int(cx_line + half), y), (int(cx_line + half), y), 0, lw)
    return scene


def _add_stains(scene, n=None, val_range=(60, 200)):
    """污渍斑点：随机灰斑/暗点（模拟白色地面污渍）。"""
    h, w = scene.shape
    if n is None:
        n = random.randint(15, 60)
    lo, hi = val_range
    for _ in range(n):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(2, 12)
        val = random.randint(lo, hi)
        cv2.circle(scene, (x, y), r, val, -1)
    return scene


def compose_scene(card, quad, img_w, img_h, interference="full",
                  conservative=False):
    """图卡+角点 → 白场地场景。返回 (scene_bgr, yolo_line, quad_line)。

    变换链（与角点同步）：缩放 → 旋转(±30°) → 梯形透视 → 纵向压扁(0.3-0.6)
    → 平移入场景。场景：白底 + 污渍 + 斑块(full) + 梯形黑线(full)。

    conservative=True（重度保守，用户确认）：生成"显著"样本——变形幅度
    收窄（scale 0.30-0.45、旋转±10°、压扁0.45-0.6、透视10-20%概率减半、
    图形偏移±0.04），用于改善类间混淆（菱形↔圆）。
    """
    card_h, card_w = card.shape

    # 1. 缩放：卡宽51-115px（占帧5-12%，对应真实0.4-1.3m的图卡投影；
    #    30%样本<0.28 → 偏远）。0.12-0.40的域外小卡（<5%）线宽<1.5px
    #    会被预处理过滤，且h<min_h被几何闸门拒——去掉。
    if conservative:
        scale = random.uniform(0.30, 0.45)
    elif random.random() < 0.3:
        scale = random.uniform(0.20, 0.28)
    else:
        scale = random.uniform(0.28, 0.45)
    new_w = int(card_w * scale)
    new_h = int(card_h * scale)
    card = cv2.resize(card, (new_w, new_h))
    quad = quad * (new_w / card_w)

    # 2. 放大画布居中（padding 40%）→ 旋转不裁角
    pad = int(max(new_w, new_h) * 0.4)
    canvas_w, canvas_h = new_w + 2 * pad, new_h + 2 * pad
    big = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    big[pad:pad + new_h, pad:pad + new_w] = card
    quad = quad + pad

    # 3. 旋转 ±30°（整卡旋转：外框+图形一体）
    ang = random.uniform(-10, 10) if conservative else random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((canvas_w // 2, canvas_h // 2), ang, 1.0)
    big = cv2.warpAffine(big, M, (canvas_w, canvas_h),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    quad = _apply_M(quad, M)
    card, (ox, oy) = _crop_to_card(big)
    quad = quad - (ox, oy)
    new_h, new_w = card.shape

    # 4. 梯形透视（85%概率）：远端收缩10-35% + 上移
    p_persp = 0.5 if conservative else 0.85
    if random.random() < p_persp:
        if conservative:
            dx_far = random.uniform(new_w * 0.10, new_w * 0.20)
            dy_far = random.uniform(0, new_h * 0.04)
        else:
            dx_far = random.uniform(new_w * 0.10, new_w * 0.35)
            dy_far = random.uniform(0, new_h * 0.08)
        src = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
        dst = np.float32([[dx_far, dy_far], [new_w - dx_far, dy_far],
                          [new_w, new_h], [0, new_h]])
        M2 = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(card, M2, (new_w, new_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
        quad = _apply_M(quad, M2)
        card, (ox, oy) = _crop_to_card(warped)
        quad = quad - (ox, oy)
        new_h, new_w = card.shape

    # 5. 纵向压扁（俯视压缩，h/w=0.3-0.6，真实0.25-0.55）
    ratio = random.uniform(0.45, 0.6) if conservative else random.uniform(0.3, 0.6)
    card = cv2.resize(card, (new_w, max(2, int(new_h * ratio))))
    quad[:, 1] *= ratio
    new_h = card.shape[0]

    # 6. 场景：白底 + 污渍 + 斑块 + 梯形黑线
    scene = _build_background(img_w, img_h, interference)

    # 7. 随机位置：画面中部偏下（图卡贴地面）
    x0 = random.randint(0, max(1, img_w - new_w))
    y_min = int(img_h * 0.25)
    y_max = max(y_min + 1, int(img_h * 0.60))
    y0 = random.randint(y_min, min(y_max, max(1, img_h - new_h)))
    scene[y0:y0 + new_h, x0:x0 + new_w] = card
    quad = quad + (x0, y0)

    # 8. 全局光照/对比度 + 噪声 + 模糊
    brightness = random.uniform(0.85, 1.1)
    contrast = random.uniform(0.9, 1.1)
    scene = cv2.convertScaleAbs(scene, alpha=contrast, beta=(brightness - 1) * 128)
    noise_amp = random.uniform(2, 5) if interference != "clean" else random.uniform(1, 2)
    noise = np.random.normal(0, noise_amp, scene.shape)
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.25 and interference != "clean":
        scene = cv2.GaussianBlur(scene, (3, 3), 0)

    # 9. GT
    scene_bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    xs, ys = quad[:, 0], quad[:, 1]
    cx, cy = xs.mean(), ys.mean()
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    yolo_line = f"{cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
    quad_n = quad / np.array([img_w, img_h], np.float32)
    quad_line = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in quad_n)
    return scene_bgr, yolo_line, quad_line


# ═══════════════════════════════════════════════════════════
# 正视图模式（96×96 二值图卡 + 增强 + 负样本）
# ═══════════════════════════════════════════════════════════

FRONTAL_SIZE = 96

# 闭环模式：训练数据 = 真实找框管线矫正后的输出（流程对齐）。
# 合成场景 → ShapeDetector找框 → 预测quad矫正96×96 → 保存。
# 负样本 = 无卡场景的误检框矫正图。
_CLOSED_LOOP = True


def _augment_frontal(card):
    """随机增强模拟部署噪声：断线(腐蚀/膨胀)、旋转、缩放、平移、cutout、噪声。"""
    img = card.copy()
    if random.random() < 0.3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, k, iterations=1)  # 线变细/断
    if random.random() < 0.3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, k, iterations=1)  # 线变粗
    if random.random() < 0.4:
        ang = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((48, 48), ang, 1.0)
        img = cv2.warpAffine(img, M, (96, 96), borderValue=255)
    if random.random() < 0.3:
        s = random.uniform(0.95, 1.05)
        img = cv2.resize(img, (int(96 * s), int(96 * s)))
        if s > 1.0:
            st = (img.shape[0] - 96) // 2
            img = img[st:st + 96, st:st + 96]
        else:
            img = cv2.copyMakeBorder(img, (96 - img.shape[0]) // 2,
                                     96 - img.shape[0] - (96 - img.shape[0]) // 2,
                                     (96 - img.shape[1]) // 2,
                                     96 - img.shape[1] - (96 - img.shape[1]) // 2,
                                     cv2.BORDER_CONSTANT, value=255)
    if random.random() < 0.3:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (96, 96), borderValue=255)
    if random.random() < 0.3:  # cutout：污渍盖线（黑底白线下填背景色0）
        for _ in range(random.randint(1, 2)):
            cx = random.randint(10, 86)
            cy = random.randint(10, 86)
            s = random.randint(4, 8)
            cv2.rectangle(img, (cx, cy), (cx + s, cy + s), 0, -1)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _threshold_noise(sigma, k):
    """高斯纹理 → 平滑(σ=sigma控制空间频率) → 归一化 → 阈值k → 黑底白斑。

    sigma小=细碎噪点（传感器噪声），sigma大=低频污渍斑（白场地污渍）。
    k越大亮斑占比越低：k=0.3≈38%，k=1.5≈7%。白占比超45%时压回45%
    （正样本实测上限46%）——高亮度不能成为"背景"特征，类边界必须
    是结构而非亮度。"""
    tex = np.random.randn(96, 96).astype(np.float32)
    blur = cv2.GaussianBlur(tex, (0, 0), sigma)
    blur = blur / (blur.std() + 1e-6)
    img = np.zeros((96, 96), np.uint8)
    img[blur > k] = 255
    if img.mean() > 115:  # 白占比>45%：保留blur值最高的45%
        img[blur <= np.quantile(blur, 0.55)] = 0
    return img


def _line_residue():
    """线残影：1-6段短亮线（60%近横/竖——找框按h/v族检线）+ 细斑叠加。"""
    img = np.zeros((96, 96), np.uint8)
    for _ in range(random.randint(1, 6)):
        x1, y1 = random.randint(0, 95), random.randint(0, 95)
        ln = random.randint(8, 48)
        if random.random() < 0.6:  # 近横/竖
            if random.random() < 0.5:
                x2 = int(np.clip(x1 + ln * random.choice([-1, 1]), 0, 95))
                y2 = int(np.clip(y1 + random.randint(-4, 4), 0, 95))
            else:
                y2 = int(np.clip(y1 + ln * random.choice([-1, 1]), 0, 95))
                x2 = int(np.clip(x1 + random.randint(-4, 4), 0, 95))
        else:  # 任意角
            a = random.uniform(0, np.pi)
            x2 = int(np.clip(x1 + ln * np.cos(a), 0, 95))
            y2 = int(np.clip(y1 + ln * np.sin(a), 0, 95))
        cv2.line(img, (x1, y1), (x2, y2), 255, random.randint(1, 3))
    return img


def _frontal_negative():
    """负样本（类别6=背景）：无卡区域矫正图 ≈ 无意义噪声场（暗背景亮结构）。

    推理时负样本的真实来源：找框在无卡场地误检 → _warp_card 矫正。误检
    区域是白场地的污渍/线残影/噪声经 _binary_selective 后的输出——不规则
    噪点场，不是"半张卡/空框"这类结构化假正样本（它们看起来像卡，会教
    模型学到"卡形结构→背景"的错误边界，真卡被部分遮挡时被误杀）。
    设计：
    - 极性必须与正样本一致（暗背景亮结构）：误检 warp 与正样本出自同一
      二值管线（blackhat后线=白背景=黑）；亮背景负样本会让模型靠"均值
      亮度"捷径分类，推理时误检warp（暗背景）全被当正类
    - 大部分二值化（管线输出是二值的），少量灰度高斯噪声防"锐度"捷径
      （输入可灰度，正样本下采样也有软边+增广噪声，不是纯二值）
    - 变体覆盖误检内容：灰度高斯噪点 / 二值细斑 / 低频污渍斑 / 线残影
      / 纯黑空场
    """
    kind = random.random()
    if kind < 0.18:  # 灰度高斯噪声（不二值）：传感器噪声
        img = np.clip(np.random.normal(0, random.uniform(15, 50), (96, 96)),
                      0, 255).astype(np.uint8)
    elif kind < 0.45:  # 二值化高频细斑
        img = _threshold_noise(random.uniform(0.3, 1.2), random.uniform(0.4, 1.4))
    elif kind < 0.70:  # 二值化低频污渍斑
        img = _threshold_noise(random.uniform(2.5, 8.0), random.uniform(0.3, 1.3))
    elif kind < 0.93:  # 线残影 + 可叠加细斑
        img = _line_residue()
        if random.random() < 0.6:
            img = cv2.bitwise_or(
                img, _threshold_noise(random.uniform(0.3, 1.0),
                                      random.uniform(0.8, 1.5)))
    else:  # 纯黑：空白场地矫正图
        img = np.zeros((96, 96), np.uint8)
    if random.random() < 0.4:  # 模拟正样本下采样的软边
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 0.8))
    img = _augment_frontal(img)
    frac = (img > 128).mean()
    if frac > 0.5:  # 增广后白占比>50%（正样本实测上限46%）：随机抽掉部分白像素
        img[(img > 128) & (np.random.rand(96, 96) < (frac - 0.5) / frac)] = 0
    return img


def _render_pristine_card(shape_idx, size=192):
    """近原图卡（黑底白线）：外框margin随机(3-10%)、图形轻微偏移、
    可缺一条边。用户要求：外框位置不固定、一条边可有可无。"""
    img = np.zeros((size, size), dtype=np.uint8)  # 黑底
    m = int(size * random.uniform(0.03, 0.10))    # 外框margin随机
    lw = max(2, int(size * 0.05))
    cv2.rectangle(img, (m, m), (size - 1 - m, size - 1 - m), 255, lw)

    cx = size // 2 + random.uniform(-0.05, 0.05) * size
    cy = size // 2 + random.uniform(-0.05, 0.05) * size
    r = size * 0.28
    if shape_idx == 0:  # circle
        cv2.circle(img, (int(cx), int(cy)), int(r), 255, lw)
    elif shape_idx == 1:  # pentagon
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + np.pi * i / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append([int(cx + rr * np.cos(ang)), int(cy + rr * np.sin(ang))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, 255, lw)
    elif shape_idx == 2:  # square
        s = r * 1.1
        cv2.rectangle(img, (int(cx - s), int(cy - s)),
                      (int(cx + s), int(cy + s)), 255, lw)
    elif shape_idx == 3:  # diamond
        pts = np.array([[cx, cy - int(r * 1.1)], [cx - int(r * 1.1), cy],
                        [cx, cy + int(r * 1.1)], [cx + int(r * 1.1), cy]], np.int32)
        cv2.polylines(img, [pts], True, 255, lw)
    elif shape_idx == 4:  # cross
        arm = int(r * 1.0)
        cv2.line(img, (int(cx - arm), int(cy)), (int(cx + arm), int(cy)), 255, lw)
        cv2.line(img, (int(cx), int(cy - arm)), (int(cx), int(cy + arm)), 255, lw)
    elif shape_idx == 5:  # triangle
        pts = np.array([[cx, cy - int(r * 1.1)], [cx - int(r * 1.1), cy + int(r * 0.9)],
                        [cx + int(r * 1.1), cy + int(r * 0.9)]], np.int32)
        cv2.polylines(img, [pts], True, 255, lw)

    # 断边：35%概率缺一条边（抹掉中间50-80%）
    if random.random() < 0.35:
        edge = random.randint(0, 3)
        seg = random.uniform(0.5, 0.8)
        c0 = int(size * (0.5 - seg / 2))
        c1 = int(size * (0.5 + seg / 2))
        if edge == 0:
            cv2.rectangle(img, (c0, m - lw), (c1, m + lw), 0, -1)
        elif edge == 1:
            cv2.rectangle(img, (size - 1 - m - lw, c0),
                          (size - 1 - m + lw, c1), 0, -1)
        elif edge == 2:
            cv2.rectangle(img, (c0, size - 1 - m - lw),
                          (c1, size - 1 - m + lw), 0, -1)
        else:
            cv2.rectangle(img, (m - lw, c0), (m + lw, c1), 0, -1)
    return img


def _worker_pristine(task):
    """近原图正样本：黑底白线理想卡 → 旋转≤10° → 位移 → 开闭+少噪声。

    用户要求补充1/10"非常显著"样本——数据里有不少造得歪的，需要一批
    几乎就是原图的清晰样本（外框位置随机、可缺一边、偏转<10°）。
    """
    shape_idx, cls_id, name, i, im_dir, lb_dir = task
    card = _render_pristine_card(shape_idx, size=192)
    if random.random() < 0.6:  # 偏转 ±10°
        ang = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((96, 96), ang, 1.0)
        card = cv2.warpAffine(card, M, (192, 192), flags=cv2.INTER_CUBIC,
                              borderValue=0)
    card = cv2.resize(card, (FRONTAL_SIZE, FRONTAL_SIZE))
    if random.random() < 0.8:  # 轻微位移 ±2px
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        card = cv2.warpAffine(card, M, (FRONTAL_SIZE, FRONTAL_SIZE),
                              borderValue=0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.random() < 0.5:  # 开运算（消细刺）或闭运算（补断口）
        card = cv2.morphologyEx(card, cv2.MORPH_OPEN, k)
    else:
        card = cv2.morphologyEx(card, cv2.MORPH_CLOSE, k)
    if random.random() < 0.3:  # 轻微噪声
        noise = np.random.normal(0, 5, card.shape)
        card = np.clip(card.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    fname = f"pristine_{name}_{i:05d}"
    cv2.imencode(".png", card)[1].tofile(os.path.join(im_dir, fname + ".png"))
    with open(os.path.join(lb_dir, fname + ".txt"), "w") as f:
        f.write(f"{cls_id}\n")
    return ("ok", name, i)


def gen_pristine(count_per_class, out_dir, jobs=1, classes=None):
    """近原图正样本：高显著清晰样本，每类补充。"""
    if classes is None:
        classes = list(range(len(SHAPES)))
    batch = _time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    out = os.path.join(base_dir, f"batch_{batch}")
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    tasks = [(s, cls, name, i, images_dir, labels_dir)
             for s, (name, cls, act) in enumerate(SHAPES) if s in classes
             for i in range(count_per_class)]
    done = 0
    with multiprocessing.Pool(jobs) as pool:
        for _ in pool.imap_unordered(_worker_pristine, tasks, chunksize=16):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)}")
    print(f"Done: {len(tasks)} pristine images -> {out}")
    return out


_SD_WORKER = None


def _get_sd():
    """每个worker进程一个ShapeDetector（lazy，避免重复构造）。

    关键：cv2.setNumThreads(1)——多进程并行时若不关OpenCV内部线程，
    每进程64线程×N进程=线程爆炸（实测48进程CPU仅22.5%，全在等锁）。"""
    global _SD_WORKER
    if _SD_WORKER is None:
        cv2.setNumThreads(1)
        _SD_WORKER = ShapeDetector()
    return _SD_WORKER


def _quality_gate(warp96):
    """质量闸：过滤找框不准的样本（防用户删的三类问题）。

    注意：warp_card 内缩14%——外框环不进warp，所以不能检查外框，
    改为检查**图形本身**：
    - 白占比≤50%：防图形线粘连成大片白
    - 中心60%区域线占比≥5%：图形存在且居中
    - 图形质心与中心距离<15px：防"只识别图卡一部分"（如左下角）
      →矫正后图形残缺/偏角"""
    white = warp96 > 127
    if white.mean() > 0.5:
        return False
    h, w = warp96.shape
    c0h, c1h = int(h * 0.2), int(h * 0.8)
    c0w, c1w = int(w * 0.2), int(w * 0.8)
    if white[c0h:c1h, c0w:c1w].mean() < 0.05:
        return False
    ys, xs = np.where(white)
    if len(ys) == 0:
        return False
    cy, cx = float(ys.mean()), float(xs.mean())
    if np.hypot(cx - w / 2, cy - h / 2) > 15.0:
        return False
    return True


def _worker_frontal(task):
    shape_idx, cls_id, name, i, img_w, img_h, inter, conservative, im_dir, lb_dir = task
    sd = _get_sd()
    off = 0.04 if conservative else None
    card, quad = render_card(shape_idx, size=256, offset_frac=off)
    scene, _, _ = compose_scene(card, quad, img_w, img_h, inter, conservative)
    act, dbg = sd.update(scene)
    pred = dbg.get("quad")
    if pred is None:
        return ("miss", name, i)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    binary = sd._binary_selective(gray)
    warp = sd._warp_card(binary, pred)
    warp96 = cv2.resize(warp, (FRONTAL_SIZE, FRONTAL_SIZE))
    if conservative and not _quality_gate(warp96):
        return ("miss", name, i)  # 质量闸：找框不准样本丢弃
    if random.random() < 0.4:
        warp96 = _augment_frontal(warp96)
    fname = f"{name}_{i:05d}"
    cv2.imencode(".png", warp96)[1].tofile(os.path.join(im_dir, fname + ".png"))
    with open(os.path.join(lb_dir, fname + ".txt"), "w") as f:
        f.write(f"{cls_id}\n")
    return ("ok", name, i)


def _worker_negative(task):
    i, img_w, img_h, inter, im_dir, lb_dir = task
    sd = _get_sd()
    scene = _build_background(img_w, img_h, inter)
    scene_bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    act, dbg = sd.update(scene_bgr)
    pred = dbg.get("quad")
    if pred is None:
        img = _frontal_negative()
        real = False
    else:
        gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        binary = sd._binary_selective(gray)
        warp = sd._warp_card(binary, pred)
        img = cv2.resize(warp, (FRONTAL_SIZE, FRONTAL_SIZE))
        real = True
    fname = f"neg_{i:05d}"
    cv2.imencode(".png", img)[1].tofile(os.path.join(im_dir, fname + ".png"))
    with open(os.path.join(lb_dir, fname + ".txt"), "w") as f:
        f.write("6\n")
    return real


def _worker_scene(task):
    shape_idx, cls_id, name, i, img_w, img_h, inter, im_dir, lb_dir, qd_dir = task
    card, quad = render_card(shape_idx, size=256)
    scene, yolo_line, quad_line = compose_scene(card, quad, img_w, img_h, inter)
    fname = f"{name}_{i:05d}"
    cv2.imencode(".jpg", scene)[1].tofile(os.path.join(im_dir, fname + ".jpg"))
    with open(os.path.join(lb_dir, fname + ".txt"), "w") as f:
        f.write(f"{cls_id} {yolo_line}\n")
    with open(os.path.join(qd_dir, fname + ".txt"), "w") as f:
        f.write(quad_line + "\n")
    return True


def gen_frontal(count_per_class, negatives, out_dir, img_w=960, img_h=540,
                interference="full", jobs=1, classes=None,
                conservative=False):
    """闭环模式（多进程）：训练数据 = 真实管线矫正输出（流程对齐）。

    正类：合成场景 → ShapeDetector找框 → 预测quad矫正96×96 → 保存。
         找框失败的场景跳过（真实管线同样拿不到该样本）。
    负类：无卡场景（白+污渍+线）→ 误检quad矫正 → 负样本；无误检时用
         _frontal_negative() 兜底。
    conservative=True：重度保守变形（显著样本）+ 质量闸过滤。
    """
    if classes is None:
        classes = list(range(len(SHAPES)))
    batch = _time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    out = os.path.join(base_dir, f"batch_{batch}")
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    total = count_per_class * len(classes)
    tasks = [(s, cls, name, i, img_w, img_h, interference, conservative,
              images_dir, labels_dir)
             for s, (name, cls, act) in enumerate(SHAPES) if s in classes
             for i in range(count_per_class)]
    found = done = 0
    with multiprocessing.Pool(jobs) as pool:
        for res in pool.imap_unordered(_worker_frontal, tasks, chunksize=8):
            done += 1
            if res[0] == "ok":
                found += 1
            if done % 1000 == 0:
                print(f"  {done}/{total}")
    if total:
        print(f"  找框成功率: {found}/{total} ({found/total*100:.0f}%)")

    neg_tasks = [(i, img_w, img_h, interference, images_dir, labels_dir)
                 for i in range(negatives)]
    neg_real = 0
    with multiprocessing.Pool(jobs) as pool:
        for real in pool.imap_unordered(_worker_negative, neg_tasks, chunksize=16):
            neg_real += real
    print(f"  负样本真实误检: {neg_real}/{negatives}")
    print(f"Done: {found + negatives} images -> {out}")
    return out


def gen_scene(count_per_class, out_dir, img_w, img_h, interference,
              jobs=1, classes=None):
    if classes is None:
        classes = list(range(len(SHAPES)))
    batch = _time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    out = os.path.join(base_dir, f"batch_{batch}")
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    quads_dir = os.path.join(out, "quads")
    for d in (images_dir, labels_dir, quads_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(out, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out}\ntrain: images\nval: images\n{YAML}")

    total = count_per_class * len(classes)
    tasks = [(s, cls, name, i, img_w, img_h, interference,
              images_dir, labels_dir, quads_dir)
             for s, (name, cls, act) in enumerate(SHAPES) if s in classes
             for i in range(count_per_class)]
    done = 0
    with multiprocessing.Pool(jobs) as pool:
        for _ in pool.imap_unordered(_worker_scene, tasks, chunksize=8):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total}")
    print(f"Done: {total} images -> {out}")
    print(f"  图片: {images_dir}\n  标注: {labels_dir}（YOLO bbox）\n  "
          f"外框角点GT: {quads_dir}（4角归一化）")


def main():
    parser = argparse.ArgumentParser(description="CV合成几何图卡数据 v2")
    parser.add_argument("--count", type=int, default=500, help="每类数量")
    parser.add_argument("--out", type=str, default="synthetic_dataset")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "light", "clean"])
    parser.add_argument("--size", type=int, default=960, help="场景宽（默认960）")
    parser.add_argument("--frontal", action="store_true",
                        help="正视图模式（96×96 二值图卡）")
    parser.add_argument("--negatives", type=int, default=0,
                        help="正视图模式负样本数量（类别6）")
    parser.add_argument("--jobs", type=int, default=1, help="并行进程数")
    parser.add_argument("--classes", type=str, default=None,
                        help="只生成指定类别，逗号分隔（如 2,3,4,5）")
    parser.add_argument("--pristine", action="store_true",
                        help="近原图正样本模式（高显著，位移+开闭，不做透视）")
    parser.add_argument("--conservative", action="store_true",
                        help="重度保守变形+质量闸（显著样本，改善类间混淆）")
    args = parser.parse_args()

    classes = None
    if args.classes:
        classes = [int(c) for c in args.classes.split(",")]

    if args.pristine:
        gen_pristine(args.count, args.out, args.jobs, classes)
    elif args.frontal:
        gen_frontal(args.count, args.negatives, args.out,
                    args.size, int(args.size * 0.5625), args.mode,
                    args.jobs, classes, args.conservative)
    else:
        gen_scene(args.count, args.out, args.size, int(args.size * 0.5625),
                  args.mode, args.jobs, classes)


if __name__ == "__main__":
    main()
