# 当前证据

此前192.168.74.140实测：L4T39.2、Ubuntu24.04、Python3.12.3、CUDA13.2、cuDNN9.20、TensorRT10.16.2；系统OpenCV4.6缺CUDA；项目venv ORT1.27仅CPU等provider；四环境均无torch。新地址必须重新确认，不用旧地址状态冒充本次结果。

本地视觉重项主要960×540图卡BH/morphology、Hough/LSD和Python候选组合；320×400巡线IPM+31椭圆BH。只有密集图像算子适合先GPU，轮廓与候选组合暂CPU。CUDA13.2与旧OpenCV构建存在官方兼容问题，必须冻结包含修复的配对源码；不照抄JetPack6/CUDA12安装教程。

2026-10-10，`192.168.1.131` 板端复测：隔离环境中的 OpenCV 4.14 CUDA 设备可用。24个实拍/视频抽样、3轮共72对比较里，巡线二值图完全一致，但完整巡线p50由80.47ms升至127.09ms；图卡处理p50由153.34ms升至401.63ms。图卡类别/presence一致，仍有9/72次几何字段变化和354/37,324,800个二值像素差异，最大四边形顶点偏差约182px。当前CUDA路径不应投入比赛运行。

当前比赛启动配置中的 `policy_49_max.onnx` 与 `policy-one-foot-standing_old.onnx` 均已建立 TensorRT 10.16.2 FP32/no-TF32引擎。因为板端无真实 `POLICY_OBS_CSV`，每模型以256条固定合成观测、100次预热、3轮回放；768个结果全通过1e-5容差，但GPU完整调用p50分别0.248ms和0.231ms，CPU分别0.106ms和0.107ms。模型规模太小，当前不适合为速度切到TensorRT。
