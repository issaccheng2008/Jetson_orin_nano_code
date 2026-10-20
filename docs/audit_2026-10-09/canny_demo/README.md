# 连续移动片段的赛道候选对照

[65.5 秒 MP4](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_moving_65s.mp4) · [8 秒 GIF](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/preview_8s.gif) · [代表画面：丢线](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_frame_590.png) · [代表画面：线缆](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_frame_1040.png) · [代表画面：末段真实卡片](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_frame_1150.png)

从 `camera_commands3.avi` 保留源录像 **32.7–71.0、79.6–87.7、98.7–117.8 秒** 三个连续窗口，共 655 帧，10 fps，65.5 秒。画面底部持续显示源录像时间、源帧号、播放时间和片段号；片段之间有时间跳转。录像 134.2 秒中，明确持续经过赛道标记的片段约 58.8 秒，运动窗口内的丢线、线缆、人员遮挡、偏离赛道及停止转换均保留。

五列从左到右为：原视频、鸟瞰原灰度、当前 `contrast` 候选、Canny 原始边缘、Canny 填充候选。四个鸟瞰面板均为同一 320×400 坐标。原灰度使用鸟瞰图 RGB 通道最大值，与已有回放脚本一致，没有为了展示而拉伸亮度。`contrast` 与 Canny 直接调用相应候选提取函数；白色表示边缘或候选像素。

**Canny 原始边缘包含同一条黑胶带两侧的轮廓，不能把它们当成两条车道。** 填充候选依据局部暗纹及相反梯度支持恢复黑色笔画，不填满环内部，也不能证明白色对象就是赛道。车道身份仍需几何配对；此视频没有运行配对、转向控制或电机输出。

源帧 **543–558（54.3–55.8 秒）** 完全相同，此处保留并标明“原录像冻结”。源帧 **1178–1212（117.8–121.2 秒）** 也完全相同，位于本演示窗口之外；该段旧 HUD 仍显示 `vx=+0.20`，因此 HUD 速度不能作为实际行走真值。运动判断依据连续经过赛道标记与画面起伏，实际机器人步态仍为推断，不能排除镜头转动或人员干预。

固定标定为相机高度 32.5 cm、俯角 45°、垂直视场角 55.876°、赛道宽 35 cm，与已有标定文件一致；未重建实际姿态。左列 HUD 属于录制时的历史输出，不是本次候选算法计算结果。视频可见三角形、圆形、方形状卡片，但不能建立全部六类卡片真值或分类成功率。该离线展示也不能证明实车控制效果或 Jetson 实时性能。

每个代表帧同时保存 `bird_gray`、`contrast`、`canny_edges`、`canny_filled` 原尺寸 PNG，保留精确像素。MP4 与 GIF 仅供观看。全部源文件摘要、标定、655 个源帧到输出帧的映射及候选像素统计见 [manifest.json](manifest.json)；人工视觉分段与信心说明见 [motion_segments.json](motion_segments.json)。GIF 是源录像 51.0–59.0 秒的连续缩略片段（包含原录像冻结），每两帧取一帧，按原时长播放。

已完整解码核对 MP4 的 655 帧、1920×640 尺寸及 65.5 秒时长；GIF 为 40 帧、8 秒，各代表候选 PNG 仅含 0/255。`source_modules_sha256` 记录渲染启动时实际加载的源码快照；随后 Canny 分支接入和归一化质心诊断改变了两个模块的文件摘要，但本演示调用的 `contrast` 分支和鸟瞰矩阵未变。后续摘要差异单独记录，不把旧渲染快照称为最终 HEAD。

复现（依赖 `numpy`、OpenCV、Pillow、`imageio-ffmpeg`，中文字体可通过 `--font` 指定）：

```bash
/tmp/jetson-loss-check-venv/bin/python scripts/render_canny_comparison.py \
  --data-dir '/mnt/d/用户/Lenovo/桌面/Robocup' \
  --output docs/audit_2026-10-09/canny_demo
```

脚本只读取原视频；不访问相机、串口、电机、远端设备。复现需要原视频和同一版本的候选代码，默认分段文件是本目录的 `motion_segments.json`，也可用 `--segments` 指定。
