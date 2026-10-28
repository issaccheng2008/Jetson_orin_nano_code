# 显式CUDA图像后端交接

2026-10-10。用户已授权独立CUDA环境与重视觉算子迁移；本子任务只做本机视觉实现，没有SSH安装、开摄像头或连接电机。实际Nano安装与离线验证由主代理执行。

## 接口与接入

`new_vision/jetson/cuda_vision.py`：

```python
from cuda_vision import create_vision_backend
backend = create_vision_backend('cuda', allow_fallback=False)
line.vision_backend = backend
shape.vision_backend = backend
print(backend.diagnostics())
```

`run_policy_vision.py` 新参数 `--vision-device cpu|cuda`，默认CPU；仅显式 `--vision-cuda-fallback` 才允许启动或运行故障时全后端退CPU。失败的严格CUDA探测在打开相机、创建connector前抛错。CPU回退警告、requested/actual设备、实际完成CUDA调用累计数、启用算子与保留CPU原因均可见。元数据记录该后端及 `cuda_vision.py` source SHA256。

`morphology(image, operations)` 支持 uint8单通道，`operations=[(cv2.MORPH_...,kernel), ...]`。每一条连续链单次upload/download，中途保留GpuMat；按operation/核内容缓存filter。`gray`/`resize`/`warp`是可选CUDA路径；在启动探测像素不一致或绑定缺失时，明确列入CPU保留路径。后台对象用于一个视觉循环，不应跨并发stream共享filter；[OpenCV官方CUDA过滤器文档](https://docs.opencv.org/4.x/dc/d66/group__cudafilters.html)提供工厂接口并要求每stream独立filter。

- 巡线：IPM与灰度用后端可选路径；31 ELLIPSE blackhat、legacy close/open/close/open链接CUDA。原adaptive Gaussian、median/Otsu、C按标准差映射、CC/统计与行几何均保留CPU。
- 图卡：原图标准灰度/resize可选；9 ELLIPSE BH、3/1×5/5×1 close链、ink close、旧cue BH/close接CUDA；DT/CC/lexsort、Hough/LSD/候选组合、地面几何和精准轮廓模块仍CPU。
- brokenhelper：保留当前96×96 card-plane与全部判断/门槛；**已有**两种BH及按band的边支持缓存继续有效，缓存的BH/close/大图dilate接CUDA。纸面/marking/nearest-ink warp等CPU规则没有替换。

[OpenCV CUDA warping文档](https://docs.opencv.org/4.x/db/d29/group__cudawarping.html)支持对应warp/resize接口，但接口存在不代表当前构建与CPU舍入一致，因此启动探测必须使用实际CUDA输出。

## 像素与故障守门

启动必须真实得到CUDA设备并运行uint8 morphology，逐像素比CPU：核9/31 ELLIPSE、31/51 RECT BH；legacy形态学链；卡框各向异性close链；RECT close5/9/15/19/25；边支持dilate5/7/9/11。固定随机灰度、0/255二值、0/1墨迹和图像边界一起覆盖。任何必需形态学像素差异都拒绝CUDA启动，不能静默换识别证据。

可选gray/resize/非平凡perspective warp只有探测完全一致才启用，否则保留CPU并记录原因。该小图启动探测不是完整数据验收，真实照片/视频仍需比对。自适应Gaussian/C取整、源曝光统计没有尝试用近似CUDA实现。

运行中CUDA异常：默认抛 `VisionCudaError`；明确许可fallback时发RuntimeWarning、sticky切CPU、debug实际设备变cpu。该故障会被原主循环的清理路径处理，不启动额外控制进程。

## 本机验证状态

- 新CUDA测试11项通过：CPU默认、不触碰GPU探测；严格/显式fallback；required morphology像素差异；optional color差异保留CPU；单链一上传下载和filter复用；运行故障不可隐藏/明确fallback；legacy+C与曝光变化；双检测器/破框cue输出保留；CLI默认和probe发生在camera/connector前。
- CUDA测试使用fake GpuMat/filter验证控制和数据路径，**不代表真实GPU或速度**。
- 原line_preprocess7、card_presence8、photometric6、line_geometry17、card_appearance7通过。
- policy_vision128项，126通过；2项在本机sandbox创建UDP socket时PermissionError，未出现功能断言失败。此限制不来自CUDA改动。
- 本机真实 `create_vision_backend('cuda')` 明确报 `OpenCV reports no usable CUDA device`；未伪称已在GPU加速。

## 主代理板上离线验收

在隔离目录与独立环境中先只导入并探测，不运行robot入口：

```bash
PYTHONPATH=new_vision/jetson python -c 'from cuda_vision import create_vision_backend; print(create_vision_backend("cuda").diagnostics())'
```

随后相同已保存实拍帧，CPU与CUDA分别独立检测器：先比较gray/blackhat/二值/ink、quad/存在/分类/控制几何，再测暖机后的整条line.process、shape.update与各stage/实际CUDA累计调用。读取启动日志中的enabled_ops与retained_cpu，不能只根据requested=cuda报告已加速。backend的download累计计时包含GPU同步等待，**不是纯传输耗时**。

必须用当前实际camera config/source hashes，31六类、额外WIN三角、人工80帧和独立背景/反光负例验收。确认不降低原正确输出与不新增错类后，再由用户授权的实车测试核实实际摄像头排队、端到端延迟和停车距离。
