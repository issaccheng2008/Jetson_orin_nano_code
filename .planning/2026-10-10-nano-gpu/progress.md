# 进展

- 用户最新指示由只读调查提升为实际安装与迁移，保留旧环境。
- 复用planning-with-files工作流，三个此前获授权的gpt6.1-sol high代理分别处理视觉、模型、兼容资料；远程安装仅root执行。
- 新地址已重新核对：Orin Nano、L4T R39.2 / CUDA 13.2；旧项目环境与系统包保留。
- 2026-10-10 实测完成：隔离 OpenCV CUDA 环境能调用 GPU，但实拍视觉端到端耗时变慢，部分图卡几何字段变化；继续 CPU 默认。
- 49维 walking 与 46维 one-foot 模型均在 Nano 构建 TensorRT FP32 引擎并离线对照。GPU 结果数值接近 CPU，但完整调用更慢；继续 CPU ONNX Runtime。
- 无真实 `POLICY_OBS_CSV`，模型测试明确使用固定合成输入，不作为实车闭环结论。全过程未打开串口、未使能电机。
- 详细数据见 `nano_gpu_benchmark_results_zh.md`；原始报告留在 Nano `records/gpu_setup_20261010/{vision_replay_v2,policy_replay}`。
