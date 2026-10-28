# Jetson + USB 摄像头 视觉方案（2026 规则）

主线：**几何图卡识别（找框 → 纯 CV 规则分类）+ 巡线 + 红条距离**。本地 Windows
调试，最终跑 Jetson（GPU 加速版见 `gpu_vision/`）。

## 运行

```powershell
.venv\Scripts\python jetson\vision_main.py      # 调试 Demo：巡线+红条+图卡 4 窗口
.venv\Scripts\python jetson\run_robot.py        # 机器人控制器（协议 V2 LINE_CTRL）
.venv\Scripts\python jetson\run_real_car.py     # 测试车控制器（4 轮速帧）
```

`.bat` 双击启动对应脚本。Q/ESC 退出。

## 目录

```
jetson/
  line_detector_v1_warp.py  # 巡线（IPM 鸟瞰 + 红条 + 窄门）
  shape_detector.py         # 图卡找框 + 分类（纯 CV 规则法）
  run_robot.py              # 机器人入口（协议 V2 + 一步前瞻）
  run_real_car.py           # 测试车入口
  vision_main.py            # 调试 Demo
  protocol_v2.py            # 协议 V2（与 openmv/ Webots 三端同步）
  gpu_vision/               # Jetson GPU 加速版（cv2.cuda + CUDA 推理）
```

## GPU 版（Jetson）

```powershell
python jetson/gpu_vision/run_robot.py --headless
python jetson/gpu_vision/probe.py            # 板子环境探测
```

详见 `jetson/gpu_vision/README.md`。

## 归档

2025 规则（QR）代码、YOLO 训练产物（权重/曲线/训练脚本）归档在
`archive/legacy_2026/`。图形识别 demo（`shape_demo.py`，含纯 CV 方案）
在 jetson/ 下。
