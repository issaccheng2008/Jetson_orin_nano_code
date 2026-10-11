# 固定曝光图像到自动曝光参考图像的 LUT 标定

这两个工具只采集并分析灰阶板照片，导出固定设置图像到自动设置参考图像的 LUT。它们不自动安装映射，也不启动巡线、图卡、控制器、串口或电机。灰阶板只能检查灰阶和中性色响应，不能标定任意颜色。

## 最短流程

先固定相机、灰阶板和灯光，关闭其他占用相机的程序。两组都使用自动白平衡，只需明确提供固定曝光；可以在命令行传入，也可以等程序提示后输入，不会猜默认值：

```bash
python3 new_vision/jetson/capture_photometric_pairs.py \
  --camera-index 0 --width 1280 --height 720 \
  --fixed-exposure-ms <实测值> \
  --output-dir calibration
```

程序会查询并显示驱动支持的控制模式、范围和初始值：Linux 使用 V4L2，Windows 使用 DirectShow 的 `IAMCameraControl` / `IAMVideoProcAmp`。输入 `0` 开启驱动确认支持的自动曝光和自动白平衡；稳定后保存 10 张原始解码帧 PNG。输入 `1` 设置固定曝光，同时开启并读回确认自动白平衡，不写固定色温；稳定后保存另 10 张。两组都会检查自动白平衡漂移。输入 `q` 会尝试恢复启动时可恢复的曝光/白平衡设置，并把结果写入 `metadata.json`。相机采集在线程中持续运行，终端等待输入时也不会停在旧帧上。

### Windows PowerShell 直接拍摄

在 Windows PowerShell 中进入仓库目录，使用 **Windows 安装的 Python**（WSL 中的 Python 和包不共用）：

```powershell
python -m pip install numpy opencv-python
python new_vision/jetson/capture_photometric_pairs.py --camera-index 0 --width 1280 --height 720 --fixed-exposure-ms 30 --allow-exposure-rounding --output-dir calibration
```

然后输入 `0` → `1` → `q`。Windows 不需要 `v4l2-ctl`、WSL 或 USB 转接；控制适配仅使用 Python 标准库 `ctypes`。摄像头编号采用 Windows DirectShow 枚举顺序，与 Linux 的编号可能不同。

Windows 的标准 [DirectShow 曝光单位是 log2 秒](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolproperty)，常见可选值包含 15.625 ms、31.25 ms 等。上面的 `--allow-exposure-rounding` 明确允许按驱动范围/步长选择最接近 30 ms 的档位，通常为 31.25 ms；请求值和实际读回值都会显示并分别记录。若必须精确 30 ms，去掉此参数；不支持时程序会拒绝拍摄，而不会把 31.25 ms 记录成 30 ms。自动/手动支持情况来自驱动 `GetRange`，设置后同时核对驱动 `Get` 返回的数值和模式；自动模式的数值允许随场景变化。

标定时序和阈值集中在 `CaptureStabilityConfig`：至少丢弃 3 秒过渡帧，随后观察中央 ROI 是否连续稳定 1.5 秒，最长等待 25 秒；稳定或等待超时后，每隔约 0.5 秒取一张，共 10 张。ROI 为画面横向 25–75%、纵向 35–75%，检查中位亮度以及 R/G、B/G。按“先拍完”模式，稳定性超时、组内亮度/色比/自动参数漂移超限仅提示并记录，不阻止拍摄另一组。`capture_complete` 和 `quality_qualified` 分别记录拍摄完成与质量达标状态；照片拍全不代表质量检查通过，拟合阶段的检查保持原样。控制设置失败、相机无法取帧或文件保存失败仍会停止相应拍摄。每张图记录可读取的相机控制值，未知值写 `unknown`。

默认用 OpenCV 窗口点击自动组 `auto/000.png` 的板面四角，顺序为左上、右上、右下、左下。程序会检查两组和组内图像是否移动；检测到移动时会要求重新选择对应图像的四角。无 GUI 时传坐标 JSON：

```json
{
  "auto_corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "fixed_corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "by_file": {
    "fixed/003.png": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  }
}
```

坐标以原照片左上角为原点，单位为像素。只有发现某张图相对组参考移动时，才需要在 `by_file` 中为该图提供新四角。

```bash
python3 new_vision/jetson/fit_photometric_mapping.py calibration/<session> \
  --corners-json corners.json --no-gui
```

默认前 8 张成对照片拟合，后 2 张做同一相机/场景下的重复性检查。输出目录 `photometric_mapping/` 包含 `lut_bgr.npy`、标准灰度 `lut_gray.npy`、巡线 max(B,G,R) 专用的 `lut_max.npy`、`mapping.json`、逐块误差 CSV/JSON、采样框覆盖图、映射曲线和固定原图/映射图/自动参考图对照。GUI 模式还会打开采样框预览供核对；无 GUI 时可查看保存的 `sampling_overlay.png`。后两张不是独立场景的泛化验证。

额外照片或视频可使用同一个 LUT 验证，不会重新拟合：

```bash
python3 new_vision/jetson/fit_photometric_mapping.py calibration/<session> \
  --corners-json corners.json --no-gui \
  --apply-to track.png card.jpg track_video.avi
```

## Python 接口与生产接入（2026-10-11 已接入）

底层接口不变：

```python
from photometric_lut import load_mapping, apply_mapping

mapping = load_mapping("calibration/<session>/photometric_mapping")
mapped_bgr = apply_mapping(frame_bgr, mapping)  # 选择 B/G/R 三通道 LUT
mapped_gray = apply_mapping(gray, mapping)      # 选择标准灰度 LUT
```

BGR 和灰度是两个互斥入口；一次图像处理只能调用一条 LUT 路径。生产巡线与图卡入口的 Normalization 已替换为该 LUT（`--photometric-mode lut`，现为 `run_policy_vision.py` 默认）：

- 巡线入口在 IPM 后对 `max(B,G,R)` 单通道使用从同一像素域拟合的 `lut_max`（[line_detector_v1_warp.py:1895](/mnt/d/用户/lenovo/桌面/github%20repository/jetson_orin_nano_code/new_vision/jetson/line_detector_v1_warp.py:1895)）；图卡入口在裁剪前对标准灰度使用 `lut_gray`（[shape_detector.py:405](/mnt/d/用户/lenovo/桌面/github%20repository/jetson_orin_nano_code/new_vision/jetson/shape_detector.py:405)）。两条入口分别取对应像素域，不叠加映射。
- 映射目录默认 `new_vision/config/photometric_mapping/`（已从 session `20261011_130034_658597` 部署），可用 `--photometric-lut-dir` 或 `PHOTOMETRIC_LUT_DIR` 覆盖；模式可用 `PHOTOMETRIC_MODE` 覆盖。`lut` 与 `normalize` 都使用这套 LUT；若需旧的统计量阈值适配，应明确选择 `threshold`。`legacy` 仍是旧的逐帧均值/方差图像匹配。
- lut 模式下旧的整帧均值/标准差匹配与基于帧统计量的阈值补偿全部停用（阈值原样传递、`scale=1`），不存在叠加；缺失映射文件时启动即报错，不会静默回退。
- 本 session 中，用前 8 张拟合、后 2 张同场景照片核对：标准灰度块 MAE 49.69→2.14 DN；max(B,G,R) 灰阶块 MAE 49.76→2.12 DN。这只是灰阶板同场景重复性检查，不代表真实赛道、图卡或强彩色像素的恢复质量。
- 回退：`--photometric-mode legacy` 恢复旧的整帧匹配。

**相机前置条件**：该 LUT 只在相机处于标定时的固定设置下有效 —— 手动曝光 ≈31.25 ms（V4L2 0.1 ms 步进写 31.2）+ 自动白平衡。B 命令已带 `--camera-exposure-mode manual --camera-exposure-ms 31.2 --camera-white-balance-mode auto`；不设定这些相机参数直接跑 LUT 会得到错误亮度。

本机离线拟合只需 Python、NumPy；交互点选四角、JPEG 和视频支持需要 OpenCV。拍摄支持 Windows 原生 Python + DirectShow，或 Jetson/Linux + USB 摄像头 + `v4l2-ctl`。2026-10-11 的 20 张标定照片由用户实测拍摄（Windows + DirectShow；自动组有亮度漂移，`quality_qualified=false`，拟合与逐帧对照已核实并接受）。离线测试已覆盖：光度相关 37 项、入口集成 16 项、全套 613 项通过（另有 2 个与本改动无关的环境性失败：存档图片路径、GBK 编码读取）。新的 LUT 映射尚未在 Jetson 实车相机（V4L2 固定曝光）上验证。

