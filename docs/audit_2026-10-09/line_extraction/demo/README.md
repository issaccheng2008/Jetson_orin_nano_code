# camera_commands3：最终二值化 demo

直接使用已有实车录像逐帧处理，调用当前生产的`extract_lane_candidates()`。白色像素为最终保留候选，与检测器debug中的`binary`定义相同，尚未证明候选都是赛道线。

## 打开观看

- [24秒对照demo](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/comparison_24s.mp4)：依次使用原录像10～18、38～46、77～85秒，包含清晰画面、模糊片段和弯道；片段以原速度播放，显示原录像时间。
- [完整134.2秒对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/comparison_full.mp4)：全部1342帧，无删帧、加速或插入合成线。
- [完整新版最终二值视频](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/binary_contrast_full.mp4)：320×400，黑底白线，独立查看最终候选。
- [4秒GIF预览](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/preview.gif)：原录像78.5～82.5秒，预览下采样为5fps；查看细线请打开MP4或PNG。

对照视频从左到右：原始相机画面、原始鸟瞰、旧版legacy、新版contrast。角落显示实际阈值及最终候选像素数。原画面中的vx/wz叠层来自录制时的旧运行，不代表本demo重新执行的命令。

### 精确像素快照

| 录像时间 | 完整对照 | 新版二值PNG |
|---|---|---|
| 13.4秒 | [对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/comparison_frame_134.png) | [二值](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/binary_frame_134.png) |
| 40.2秒 | [对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/comparison_frame_402.png) | [二值](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/binary_frame_402.png) |
| 80.5秒 | [对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/comparison_frame_805.png) | [二值](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/line_extraction/demo/binary_frame_805.png) |

PNG保留精确0/255像素；MP4/GIF用于展示。已核对三个MP4的帧数、fps、末帧可解码；80.5秒二值MP4解码值仅0/255，和对应PNG一致。

## 几何与范围

原录像960×540、10fps。按现有标定透视映射到320×400，固定安装俯角45°；没有该录像的逐帧IMU重建，不宣称恢复了录制时实际姿态。显示用缩放不参与提取，未裁掉状态栏后拉伸。完整录像保留原有重复帧和后段非赛道内容。

输入视频、校准参数、处理模块哈希见[manifest.json](manifest.json)。亮度暗、反光抹掉的笔画及遮挡仍可漏检，场地边缘/刻度也可能成为额外候选。

## 复现

在有OpenCV4.x、NumPy、Pillow和imageio-ffmpeg的离线工具环境中：

```bash
python scripts/render_lane_binary_demo.py \
  --video '/mnt/d/用户/Lenovo/桌面/Robocup/camera_commands3.avi' \
  --output /tmp/lane-binary-demo \
  --font /mnt/c/Windows/Fonts/msyh.ttc
```

`--font`可替换为本机其他中文TTF/TTC字体。工具不访问相机、串口或电机；原录像只读，不要求在机器人生产环境安装这些导出依赖。
