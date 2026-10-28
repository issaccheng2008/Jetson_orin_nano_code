# 离线视觉 CPU/CUDA 对照交接

2026-10-10，脚本冻结 SHA256：`5d83d25efaa3d568f55d24bdb7e42593d31c3f6e2dec2f0a3f59ffefbd3184b8`。

## 文件与运行

- `scripts/benchmark_vision_devices.py`：独立构造 LineDetector/ShapeDetector，只读取图像和普通视频文件。
- `tests/test_benchmark_vision_devices.py`：5 个测试，覆盖像素与前景差异、分类/几何变动、视频均匀采样、重复状态与 fake 对照、CUDA 缺失报告/退出码。
- 不导入机器人入口，不打开相机、UDP、串口。没有新增依赖或运行代码修改。

在 Nano 安装的 CUDA OpenCV Python 环境中，从仓库根运行：

```sh
python scripts/benchmark_vision_devices.py \
  --input-dir /path/to/isolated-testdata \
  --output-dir /path/to/vision-device-report \
  --max-images 20 --repeats 5 --warmup 1 --require-cuda
```

默认参数与现有入口默认值相同：`line=legacy`、`shape=selective`、`photometric=normalize`。实际使用 contrast 巡线时加 `--line-preprocess contrast`；Canny 模式也需显式传入对应参数。相机配置从当前 `cameras.json` active/profile 读取，拒绝静默回退 default；输出保存完整参数和配置及源码 SHA256。

`--devices cpu` 仅测 CPU。默认同时尝试 cpu/cuda，严格 CUDA 不自动回退；显式 `--cuda-fallback` 允许回退但报告 status=fallback。`--require-cuda` 要求 actual=cuda 且测量段真实 calls>0，否则保存报告后退出2。CPU 执行失败也退出2。CUDA 缺失时仍保存 CPU 报告。

## 测量约定

文件共享 max-images 预算；视频获配帧均匀分布到完整时长。样本按 source/frame 排序，每个 source、每轮新建同配置检测器；CPU 与 CUDA 使用同一份解码后的 BGR。视频 line dt 使用帧号/FPS，shape wall time 固定回放；真实 perf_counter 保留。预热完整回放后才统计 repeats，后端滤波器缓存继续保留。

输出 `report.json`、`report.md`、`opencv_build.txt`，其中：

- line.binary、shape.binary、shape.ink 逐帧严格值差/前景差比例，以及所有比较像素加权差比例。
- shape/action 分类、presence、quad/box/cy 与 line 的距离/航向/质量/有效性变化，附数值最大绝对变化。
- line 总耗时、warp/gray/morphology、preprocess 和 remainder 的 warm p50/p95；remainder 包含几何、光度与可视化。
- shape 原有 preprocess/proposals/verification/classify/presence/total 以及外层调用耗时；操作后端耗时也单独保留。
- 请求/实际设备、enabled_ops、retained_cpu、回退原因、测量段 CUDA calls/transfer-ms、OpenCV 模块路径/构建信息/线程数、CUDA DeviceInfo 与原始像素哈希。

计时包含同步下载和调试可视化；backend 操作与检测器阶段有嵌套，不能相加。backend transfer-ms 包含 download 等待 GPU 完成，不能当成纯数据搬运成本。backend 最终累计计数包含预热；独立 measured_cuda_calls/transfer_ms 已扣除预热。此回放用于设备结果一致性和耗时；视频子采样的确认窗与实际机器人完整帧流不同，图卡准确率沿用独立 GT 验收。

## 本地验证

命令：`PYTHONPYCACHEPREFIX=/tmp/vision-benchmark-pycache /tmp/jetson-loss-check-venv/bin/python -m unittest discover -s tests -p test_benchmark_vision_devices.py -v`，5/5通过。

fake backend 使用 CPU 参考算子：6 个测量 calls（3 连续帧×2轮）比较全部 binary、分类、presence 和几何均为0变化。这仅验证比较器与状态回放。

真实输入使用 OpenCV4.14.0/NumPy2.5.3，usb_main 相机配置。本机 CUDA device count=0，所有实际运行均 cpu=ok/cuda=error；calls_compared=0，尚无实 GPU 速度或一致性结论。

| 本地输入 | 测量 calls | line p50/p95 ms | shape p50/p95 ms |
|---|---:|---:|---:|
| 6card 原始实拍6张：三1、五3、十3、圆6、正6、菱4 | 18 | 129.612 / 146.913 | 106.922 / 163.436 |
| 背景1张 + old_star 视频帧0/57/114/172 | 15 | 113.157 / 120.691 | 79.764 / 100.902 |

完整证据保存在相邻目录 `vision_benchmark_local_stills/`、`vision_benchmark_local_video/`；后者 source 是 `/tmp` 的原数据 symlink，像素哈希已持久化。两份报告均匹配上面的冻结脚本 SHA256。桌面耗时仅说明脚本成功执行；Nano 上仍需独立运行同一份文件。

板上验收查看 `devices.cuda.status=ok`、实际 GPU DeviceInfo（Orin SM87）、measured_cuda_calls>0、retained_cpu 原因和 comparison_summary/binary_summary。GPU 加速判断使用同板 warm CPU/CUDA 各阶段与整体耗时；形态学用了 GPU 本身不能证明整个视觉循环加速。
