# Nano 视觉与策略模型 CPU/GPU 对照

测试日期：2026-10-10。设备：Orin Nano，`192.168.1.131`。所有运行均为离线图像/观测回放；没有打开 STM32 串口，也没有使能电机。

## 环境与范围

- 视觉：独立 Python 环境 `/home/isaac/venvs/robot_gpu_20261010`，OpenCV 4.14.0，CUDA 设备可用数为 1。输入是两段真机巡线视频、两段真机图卡视频和六张实拍图卡图片。抽样 24 帧，预热 1 轮、测量 3 轮，共 72 对 CPU/CUDA 调用。
- 模型：使用板上当前 `config/button_start.env` 指定的 `policy_49_max.onnx`（49→12）和 `policy-one-foot-standing_old.onnx`（46→12），SHA 与工作区文件匹配。CPU 为 ONNX Runtime 1.27，GPU 为 Orin 上构建的 TensorRT 10.16.2 FP32/no-TF32 引擎，CUDA runtime 13.2。
- 板上没有找到真实 `POLICY_OBS_CSV`，所以模型对照用 256 条脚本生成的固定合成输入，seed=`20261010`，预热 100 次、重复 3 轮。结果只说明数值执行和推理延时，不代表真实观测分布或实车闭环表现。

## 视觉结果

| 处理阶段 | CPU p50 / p95 | CUDA p50 / p95 | CUDA相对耗时 |
|---|---:|---:|---:|
| 巡线完整 `line_total_ms` | 80.47 / 125.92 ms | 127.09 / 163.17 ms | 1.58× 慢 |
| 巡线预处理 `line_preprocess_ms` | 14.44 / 14.81 ms | 25.00 / 42.12 ms | 1.73× 慢 |
| 图卡完整 `shape_total_ms` | 153.34 / 214.16 ms | 401.63 / 446.88 ms | 2.62× 慢 |
| 图卡预处理 `shape_preprocess_ms` | 29.20 / 35.36 ms | 50.12 / 57.43 ms | 1.72× 慢 |
| 图卡 presence | 21.42 / 24.64 ms | 256.16 / 267.18 ms | 约 12.0× 慢 |

像素和语义比较：

- 巡线二值图：9,216,000 像素中 **0 个不同**；所有 72 次巡线输出字段一致。
- 图卡二值图：37,324,800 像素中 354 个不同（约 0.00095%）；图卡 ink 图 0 个不同。
- 卡片类别改变 0/72，presence 判定改变 0/72；但图卡几何字段有 9/72 次改变，涉及 `quad`、`box_top_work`、`presence_cy_frac`。其中一个样本的四边形顶点最大差约 182 px。这说明类别相同并不足以证明停车触发几何也一致。
- CUDA 路径实际被调用 993 次。当前版本只启用了灰度转换和形态学；warp 与 resize 因像素探针不一致而保留 CPU。实际输入上，形态学仍有少量输出像素差异。

结论：当前 OpenCV CUDA 实现没有加速实车视觉流水线；巡线和图卡端到端延迟都更高，图卡定位几何还发生变化。**继续保持 CPU 默认，不在比赛流程中启用视觉 CUDA 路径。** 若继续优化，应先针对 `shape_presence` 逐算子剖析，减少小算子反复上传/下载和同步，再做像素与卡片距离回归。

## 策略模型结果

| 模型 | CPU p50 / p95 | TensorRT GPU p50 / p95 | GPU相对耗时 | CPU/GPU 最大动作误差 |
|---|---:|---:|---:|---:|
| `policy_49_max.onnx` 49→12 | 0.106 / 0.117 ms | 0.248 / 1.143 ms | 2.34× 慢 | 1.43e-6 |
| `policy-one-foot-standing_old.onnx` 46→12 | 0.107 / 0.118 ms | 0.231 / 0.962 ms | 2.16× 慢 | 8.34e-7 |

每个模型比较 768 个输入（256 条×3）。两模型均 0 条超出 `atol=rtol=1e-5`；映射到关节目标后的最大误差分别为 `3.58e-7 rad` 和 `2.09e-7 rad`。TensorRT CUDA event 显示单次模型计算 p50 约 0.107–0.108 ms，但 H2D/D2H 与主机同步使完整 GPU 调用中位数达到 0.231–0.248 ms。50 Hz 的 20 ms 周期下两边都远低于预算，但 GPU 没有性能收益；当前这两个小模型建议继续走 CPU ONNX Runtime。

## 文件与复现证据

Nano 保留完整工作副本和日志：

- `/home/isaac/records/gpu_setup_20261010/vision_replay_v2/`
- `/home/isaac/records/gpu_setup_20261010/policy_replay/`

板端原始报告与逐样本 CSV 已另存于本地临时目录 `/tmp/nano_gpu_board_run_20261010/`。TensorRT 引擎绑定 Nano 的 GPU UUID，不应拷到其他设备使用。
