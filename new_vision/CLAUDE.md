# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboCup Humanoid competition (2026规则) — a biped robot follows a black line (white background) on a ~6m closed-loop track, detects **geometric shapes** (圆形/五角星/正方形/菱形/十字形/三角形) printed on 10cm×10cm cards to trigger actions, and crosses a red obstacle bar. Chinese-language project.

**2026规则变化**（详见 `docs/新赛事规则2026.md`）：
- 二维码识别 → **平面几何形状识别**（6种图形，10cm×10cm，线粗0.5cm，白底带外框）
- 识别图卡后需**完全停止步行**做动作维持3秒以上
- 行进竞速最高分 60 → **200**
- 场地/赛道/限宽门尺寸不变（3660×2440mm，614cm，350mm，限宽门240mm）

**两个视觉平台：**
- **Jetson Nano** (实车主力): USB camera 1280×720, VFOV=55.876°, 高度32.5cm, 俯角45°. `jetson/`
- **OpenMV H7+** (嵌入式备选): 320×240, `openmv/`
- **Webots R2025a** (仿真): `Webots/controllers/`

## 四个核心视觉输出 (Four Core Visual Outputs)

| # | 输出 | 实现位置 | 说明 |
|---|------|---------|------|
| 1 | **红条检测** | `jetson/line_detector_v1_warp.py::_detect_red_bar()` | 原图判红→最低行→精确反投影距离 |
| 2 | **几何图卡识别** | `jetson/shape_detector.py` | 找框+纯CV规则分类，替代原QR |
| 3 | **窄门信号** | `jetson/line_detector_v1_warp.py` | mid/low band宽度空间差分 + 起跑线/红条双验证 |
| 4 | **转向+偏角** | `jetson/line_detector_v1_warp.py` | two-band扫描 + bottom lock + 统一heading fit → fused_err |

## Key Architecture

### Jetson Vision (主力实车方案, CPython)
- `jetson/line_detector_v1_warp.py` — **核心**: IPM鸟瞰变换 + 巡线 + 红条(原图判红+精确反投影) + 窄门
- `jetson/shape_detector.py` — 图卡找框 + 分类（纯 CV 规则法）
- `jetson/run_robot.py` — 机器人控制器: 协议V2 LINE_CTRL + 一步前瞻（弯道切线补偿）
- `jetson/run_real_car.py` — 测试车控制器: PID + 差速 + 串口(7字节帧带XOR校验)
- `jetson/vision_main.py` — 调试Demo: 四窗口可视化 + 图卡 + 红条 + 窄门
- `jetson/gpu_vision/` — Jetson GPU 加速版（cv2.cuda 预处理，CPU 自动回退）

### OpenMV Camera (retired platform — 仅协议源保留)
- `openmv/main_webots_aligned.py` / `main1.py` — MicroPython 参考（已退役）
- `openmv/protocol_v2.py` — protocol V2 source（与 jetson/ Webots 三端同步）

### Webots Simulation (CPython)
- `Webots/worlds/Robocup.wbt` — simulation world (相机高0.40m, FOV 0.855rad)
- `Webots/controllers/line_follow_transfer/line_follow_transfer.py` — main controller (Supervisor for camera shake + lens)
- `Webots/controllers/line_follow_transfer/uart_sink.py` — pluggable byte sink (file/UDP/multi)
- `Webots/controllers/line_follow_transfer/protocol_v2.py` — same protocol, CPython compat

### Configuration (single source of truth)
- **`line_follow_params.json`** (repo root) — ALL parameters: camera, ROI, PID, thresholds, shake, protocol, QR. Both Webots and OpenMV scripts hard-code a search path pointing here. Do NOT casually move it.
- `config/field/场地参数基线.json` — field CAD baseline dimensions + 2026规则字段
- `config/presets/*.json` — read-only golden snapshots (cp to root to use)
- `config/README.md` — details on config layering
- 规则文档: `docs/新赛事规则2026.md`（2026新规则文字版），`rule.pdf`（2025旧规则历史参考）

### Communication Protocol V2
- Frame: `0x55 0xAA VER MSG FLAGS SEQ TS_MS(4) LEN PAYLOAD CRC16(2)`
- CRC: CCITT-FALSE (poly=0x1021)
- Key messages: heartbeat (0x01, 10Hz), line_ctrl (0x02, 10Hz), QR_event (0x10), robot_state (0x81)
- line_ctrl payload: mode_u8, route_u8 (1=go, 2=left, 3=right, 4=slight-left), conf_u8, lost_u8, ex_mm_i16, ang_cdeg_i16, v/w reserved
- Spec: `docs/vision_main_protocol_v2.md`, `docs/motor_protocol_v2_p1.md`

### V1 Line Following Algorithm (Jetson)
- IPM: 1280×720 BGR → 透视变换 → 320×400 birdseye, z=20~80cm
- cm_per_px≈0.332, z_per_px≈0.150, _asp≈2.21 (非方形像素)
- 预处理: custom_gray → blackhat(31×31) → 自适应阈值(GAUSSIAN,31,C=-8) → 形态学(3级close+open) → 连通域过滤
- Two-band检测: low band(y=350-399, z≈20-27cm) + mid band(y=300-349, z≈27-35cm)
- Bottom lock: y=350-399区域对称性验证配对
- 统一heading fit: 加权最小二乘(all points), atan(a×_asp)→物理角度
- 误差融合: near_err + lookahead + curve_slope + angle_err → fused_err
- 红条: birdseye全图BGR判红(R≥105, R>G+28, R>B+28) → 质心 → _px_to_ground_cm
- 窄门: 进入=宽度比例out模式+红条可见+起跑线可见(三重确认); 离开=红条/起跑线靠近阈值
- 几何布局: 窄门出口 --24cm--> 红条 --36cm--> 起跑线(60cm)

### 图卡识别 (CV找框 + 纯CV规则分类, 已部署)
- 6种几何图形: 圆形/五角星/正方形/菱形/十字形/三角形
- 管线: 找框(960×540 二值化四通道候选→几何验证) → 单应矫正200×200 → 规则法分类
- 数据: `6_pictures/generate_synthetic_cards.py` 闭环合成（场景→找框→矫正）
- 动作映射: 圆形=举左手, 五角星=举右手, 正方形=抬左腿, 菱形=抬右腿, 十字形=举双手, 三角形=左右摇头
- 识别后需完全停止步行3秒（run_robot.py 图卡状态机）

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/auto_tune_webots_params.py` | Black-box parameter search via Webots (--trials, --run-seconds, --apply-best) |
| `scripts/run_shake_eval.py` | Run shake evaluation in Webots |
| `scripts/evaluate_line_follow_log.py` | Grade telemetry log quality |
| `scripts/iterate_line_follow_params.py` | Analyze telemetry to recommend param changes |
| `scripts/calibrate_camera_opencv.py` | Camera intrinsic/extrinsic calibration |
| `scripts/check_camera_consistency.py` | Check calibration sanity |
| `scripts/apply_calibration_to_line_follow_params.py` | Save calibration to params JSON |

## Important Workflows

### Simulation loop
1. Edit `line_follow_params.json` (ROI, PID, thresholds, shake, etc.)
2. Launch Webots world `Webots/worlds/Robocup.wbt` (reads params at startup)
3. Controller writes log lines to stdout + optionally to `generated/` via env `LINE_FOLLOW_LOG_FILE`
4. Run `scripts/evaluate_line_follow_log.py` or `scripts/iterate_line_follow_params.py` on logs

### Parameter auto-tuning
```
python scripts/auto_tune_webots_params.py --trials 24 --run-seconds 16 --apply-best
```

### Run Jetson vision demo
```
python jetson/vision_main.py
```
Shows 4 windows: Original / Warp(birdseye) / Adaptive(binary) / Close+Fit
Status bar: FPS, speed, red bar z, narrow gate, deviation, heading, confidence, shape

### Run real car controller
```
python jetson/run_real_car.py
```
PID + 4-wheel differential drive + serial output. Keys: Q=quit, S=toggle serial.

### Run robot controller (Jetson GPU 版)
```
python jetson/gpu_vision/run_robot.py --headless
```
详见 `jetson/gpu_vision/README.md`（probe/compare_cpu_gpu 验收）。

## Running Tests
```
python -m pytest tests/test_protocol_v2.py -v
python tests/test_protocol_v2.py
```

## Conventions
- All parameters loaded via `_cfg_get(SHARED_CFG, "dotted.path", default)` — JSON config drives behavior, not hardcoded constants (except the _cfg_get itself)
- Telemetry logs use regex `TELEMETRY_RE` for structured parsing (shared between auto-tune, evaluate, iterate scripts)
- Two parallel codebases share algorithm logic: MicroPython (openmv) + CPython (Webots controller). Keep `protocol_v2.py` in sync between `openmv/` and `Webots/controllers/`
- V1 detector (Jetson) and Webots controller are independent codebases — algorithm ideas shared but implementations differ
- OpenMV runs on grayscale (no red detection) unless `red_detect_on_grayscale` is explicitly enabled
