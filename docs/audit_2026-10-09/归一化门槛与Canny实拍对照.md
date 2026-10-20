# 归一化门槛与 Canny 实拍对照

日期：2026-10-09。原版对照提交：`3260323`。本次读取本地实拍，未连接机器人。

## 1. 默认改动：用均值和标准差换算门槛

采用 `z = (I - μ) / σ` 的标准化定义，参见 [StandardScaler 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。不采用先前提议的两个 0.5 加权项。

保留旧门槛在旧画面里的标准差位置，在当前帧原灰度中换算：

```text
s = max(σ当前, 4) / σ参考
亮度门槛 T当前 = μ当前 + s × (T参考 − μ参考)
明暗差/黑帽响应门槛 D当前 = s × D参考
```

它等价于把当前灰度映射为 `I参考 = μ参考 + (I−μ当前)/s` 后，用旧门槛判断。实现直接换算门槛，避免先映射到 uint8 导致截断或饱和；原始画面与颜色检测保持原像素。`σ` 的底限 4 灰度级是工程保护，避免近乎均匀的画面将噪声无限放大，属于标准化以外的显式限制。

统计区域是原图 `x=25%～75%、y=35%～75%` 的地面区域，不含录像底部 HUD，不含 IPM 黑色填边。每帧重新测量；形状检测在裁剪、缩放、补边前统计。该区域仍可能含胶带、图卡、阴影和遮挡，因此统计量不是曝光计读数。

旧参考来自 4 段 9 月 19 日原始视频的 43 个有效采样帧，每帧等权；新视频沿用已审查的 45 个有效地面样本，描述性对比如下：

| 同一地面 ROI | 灰度均值 μ | 帧内标准差 σ 的平均值 |
|---|---:|---:|
| 旧画面参考，BGR2GRAY | 152.58 | 20.67 |
| `camera_commands3.avi` | 64.59 | 15.04 |

均值/标准差采用 [已存档逐视频统计](line_extraction/photometry_summary.csv) 聚合；示例以平均统计量代入，实际程序使用各帧统计量。`s≈0.728` 时：

| 图卡判据 | 原门槛 | 当前示例门槛 |
|---|---:|---:|
| 纸面灰度最低值 | 100 | 26.3 |
| 黑帽墨迹响应 | 12 | 8.7 |
| 纸面与边框明暗差 | 4 | 2.9 |
| cue 原灰度分数门槛 | 1.2 | 0.87 |
| selective 二值化 `C` | −16 | −11.6 |

纸面门槛中的均值项用于抵消整体变暗；黑帽响应和明暗差中的均值已经相减消掉，只缩放标准差。不能把这两种门槛混用同一亮度加权公式。

### 精准识别的 C 有一个经照片验证的限制

OpenCV 自适应二值化用局部均值减去 `C` 得到门槛，见 [官方阈值教程](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)。这里 C 为负值，所以绝对值越大，墨迹要求越强。

实际使用 `C当前 = C旧 × min(1, s)`：对比度下降时补偿；图卡近距大黑图形把画面标准差抬高时，不提高原有拒绝门槛。最初直接使用完整比例，在 31 张旧照片中造成正确数 27→24、出现 4 个错类；采用这个限制后恢复 27 个正确、零错类。这个限制来自实拍回归，不能描述为标准化公式本身。

精准识别的 ring 通道继续使用原灰度 Otsu，不人为指定一个固定亮度门槛。分类仍使用 selective binary 与 ring ink 的并集，保留图形内部拓扑。框尺寸、地面方形验证、闭合度、时间累计、停车和投票流程没有在本次光度处理中修改。

### 模糊 cue 与巡线质心

模糊 cue 的纸面、墨迹、明暗差和最终分数同步换算，避免只降纸面门槛却仍被其它固定灰度门槛挡住。输出 `presence_cue`/控制台 `cue=` 改为**参考对比度单位**，计算为 `cue_raw/s`；仍与参考门槛 1.2 比较。`cue_score_raw` 与 `cue_score_threshold_raw` 另行记录，不可直接把新 `cue=` 与旧日志原灰度分数混为一列。

巡线原灰度质心兜底的最低凹陷对比度由固定 5 改成 `5 × s`。该检测器用 RGB 最大通道，因此独立使用同批旧 ROI 的参考 `μ=154.52、σ=20.55`；不将 BGR2GRAY 参考直接套在不同灰度定义上。线宽、角度、横向位置、时间滤波和转向档位参数不随亮度缩放。巡线正常候选仍默认已有的 `contrast` 预处理。

## 2. Canny 两版做了什么

巡线与图卡各有独立 Canny 候选函数；采用小高斯平滑、梯度、非极大值抑制和双门槛连通，流程依据 [OpenCV Canny 说明](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)。本项目额外加入相反梯度配对与黑帽暗纹支持，输出被两侧边缘支持的**填充笔画**。一条黑胶带的两个边缘不会直接被当成两条赛道边界。

巡线用 33 像素黑帽尺度、配对搜索半径 24；图卡用 11 像素尺度、半径 7，并拒绝中位笔画宽大于 7 像素的连通域。仅做 3 像素小闭运算，不补全大断口。

梯度门槛：`high=max(12, 6×噪声估计, 1.3×梯度P90)`，`low=0.4×high`；暗纹门槛：`max(3, 3.5×噪声估计, 1.3×黑帽P70)`。这些是项目实验参数，分别对应梯度证据、噪声与连通性，并非均值/标准差混权。Canny 独立比较原灰度，不依赖默认门槛换算或 CLAHE。

## 3. 实拍结果与可看演示

[65.5 秒连续活动片段视频](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_moving_65s.mp4) · [8 秒预览](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/preview_8s.gif) · [丢线画面](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_frame_590.png) · [晚段真实卡片画面](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/canny_demo/comparison_frame_1150.png)。演示五列是原画面、原鸟瞰灰度、contrast、Canny 原始边缘、Canny 填充候选。

只取源视频 32.7–71.0、79.6–87.7、98.7–117.8 秒，保留困难过程；没有把全部 134.2 秒算为运动数据。源录像存在冻结，旧 HUD 非零速度不是运动证明，详见 [分段说明](canny_demo/README.md)。

4 个活动标注帧 536/670/805/1073 的稀疏中心线对照：

| 候选提取 | 中心线覆盖，4px 容差 | 候选靠近车道的比例，10px 容差 |
|---|---:|---:|
| legacy | 77.0% | 100.0% |
| 默认 contrast | 96.9% | 88.3% |
| Canny | 98.2% | 67.9% |

覆盖提高同时带来更多无关细笔画，不能把更多白像素或更高覆盖直接等同于更好巡线。标注为视觉估计的稀疏中心线，误差通常 2～4px，不是专家逐像素真值。

31 张原始六类实拍照片的 Canny 对照：

| 图卡路径 | 找到框 | 分类 | 正确 | 错类 |
|---|---:|---:|---:|---:|
| 原版 | 28 | 27 | 27 | 0 |
| 仅 selective 改 Canny，ring 保持 Otsu | 28 | 26 | 19 | 7 |
| selective 与 ring 都改 Canny | 13 | 12 | 6 | 6 |

当前 `--shape-preprocess canny` 对应第三行。它改变细环和内部图形拓扑，当前不能用作比赛默认。默认仍为 selective，加上述归一化门槛补偿。详细结果与图片放在本报告相邻的 `canny_evaluation/`、`photometric_cards/`。

### 默认归一化图卡路径的单独回归

| 31 张照片处理 | 原版正确 / 错类 | 默认归一化正确 / 错类 |
|---|---:|---:|
| 原始实拍像素 | 27 / 0 | 27 / 0 |
| 原图乘 0.6 | 24 / 3 | 26 / 2 |
| 原图乘 0.4 | 24 / 3 | 26 / 1 |

后两行是同场景像素乘法模拟，不是新光照实拍；仍有错类，不能宣称暗图全部解决。每张重建检测器，不以旧卡历史帮助下一张。全图对照包含全部失败样本：[原始 31 图](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/photometric_cards/allphotos_gain1.0.jpg)、[0.6 模拟](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/photometric_cards/allphotos_gain0.6.jpg)、[0.4 模拟](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/photometric_cards/allphotos_gain0.4.jpg)；[逐图 CSV](photometric_cards/photo_scores.csv) 与 [报告](photometric_cards/report.json) 可复核。

真实新视频帧 1130/1220/1230 中，原版没有 cue，归一化恢复出口候选，参考分数分别 4.197/4.183/4.037；帧 1150 新旧都检测分类为 square。前三帧独立检测器没有累积历史，`presence` 还未确认，**候选过分数门槛不等于已触发停车**。这些帧没有完整六类真值，不计入分类正确率。[真实晚段对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/7016d44590b06545a9e412d02bdd13d5138a0258/docs/audit_2026-10-09/photometric_cards/camera_commands3_1130.jpg)；另外的 `背景.png` 负例三版均无检测或分类。

## 4. 使用与日志

在终端 B 原命令末尾添加或保持默认：

```bash
--line-preprocess contrast --shape-preprocess selective --photometric-mode normalize
```

巡线 Canny 单独观看/回放选择 `--line-preprocess canny`；图卡实验选择 `--shape-preprocess canny`。恢复旧固定光度门槛选择 `--photometric-mode legacy`，它不会自动切换巡线候选；完整原提取对照需要同时写 `--line-preprocess legacy --shape-preprocess selective`。

按钮配置变量：`LINE_PREPROCESS`、`SHAPE_PREPROCESS`、`PHOTOMETRIC_MODE`，见 `config/button_start.env.example`。统计常量与公式位于 `new_vision/jetson/photometric_thresholds.py`，没有新增人为均值/标准差权重。

`records/line_telemetry/<run>/line_frames.jsonl` 的 `measurement` 记录：

- 巡线：`photometric_mean/std`、`photometric_contrast_scale`、`centroid_min_contrast_effective`。
- 图卡检测帧：`card_photometric_mean/std`、`card_photometric_contrast_scale`、`card_cue_core_gray_threshold`、`card_cue_ink_threshold`、`card_cue_contrast_threshold`、`card_cue_score_threshold_raw`、`card_cue_score_raw`、`card_shape_adaptive_c`。
- `card_cue_evaluated` 区分“有精准框所以无需跑 cue”与“确实跑了 cue”；`card_cue_score_reference`、`card_cue_score_threshold_reference` 提供同单位出口比较；`card_shape_adaptive_c_applied` 标明该帧是否真的使用 C（Canny 不使用）。
- Canny：高低梯度门槛、噪声估计、边缘和候选像素数；大图数组不进入 JSONL。

控制台 `[shape]` 同时显示 `cue`、`raw`、`photo=均值/标准差`、`scale`。图卡隔帧运行，未检测帧不把上一帧字段重新写成新数据。

## 5. 验证范围

归一化算式、增益和偏置一致性、暗纸面 cue、空画面、近距 C 限制有针对性测试；Canny 有 10 项图像测试，包含不同方向、亮暗背景、空场噪声、缺边与六种图形。已有视觉、几何、按钮配置和本机 UDP 回归检查通过；首轮 UDP 套接字被沙箱限制，已在允许本机 socket 的环境补跑。

空地噪声探针覆盖均值 25/65/160、噪声标准差 0/1/3/8，12 例 cue 均未命中。这不能替代包含线缆、反光、场外目标的实拍假阳性统计。当前标准化也不能恢复运动模糊丢失的空间细节，无法由这些离线图片证明停车距离、投票率或完成比赛的概率。

## 6. 发布记录

功能与演示提交 `016ea8f` 已通过 WSL SSH 推送到 GitHub `policy49flitter`。推送前合并队友的 `38f2aba` 近位置恢复提交；本次候选对照冻结在 `3260323`，未把队友控制改动混入候选提取指标。合并后 82 项相关回归全部通过。记录发布状态的后续文档提交不改变运行代码。
