# 旧版基础巡线：只归一化自适应二值化 C

2026-10-09。按当前要求，把比赛默认提取路径恢复为 `legacy`，在旧版流程上换算 C。先前的 `contrast` / `canny` 可显式选用。

## 流程和公式

原图最大 RGB 通道灰度 → 原鸟瞰映射 → 黑帽 31×31 → 自适应阈值 31×31 → 用候选灰度估计最终门槛 → 旧版形态学 → 旧版连通域筛选。

设旧参考画面的均值/标准差为 μ₀、σ₀，当前地面 ROI 为 μ、σ。标准化的等价灰度为：

```
I_ref = μ₀ + σ₀ * (I - μ) / max(σ, 4)
s = max(σ, 4) / σ₀ = sqrt(方差比)  # 分子设噪声下限
C_current = C_reference * s
```

自适应二值化比较 `I > local_mean - C`。均值偏移在相减时抵消，所以 C 只乘标准差比，不能把两种均值/方差随意加权。黑帽同样消除加性亮度偏移。没有把原图归一化、提亮或裁剪到新的像素范围。

巡线实际使用最大 RGB 通道，旧参考为 **μ₀=154.5171、σ₀=20.5473**；图卡 BGR2GRAY 的参考值不同，不能混用。当前统计仍取原图 x25%～75%、y35%～75% 的地面 ROI，避开已有视频 HUD 和鸟瞰黑边。该 ROI 的空间标准差也受线、卡、影子等占比影响，并不等同于相机曝光或纯线对比度。

参考 C 默认 **−12**。例如当前标准差为 15，实际 C 约 **−8.76**；参考曝光/对比度恢复时回到 −12。C 越接近零，局部弱响应越容易进入候选。本次巡线采用完整比例；图卡已有 C 的单向补偿规则独立保留。

## 启动与调整

终端 B 在原命令中加入：

```bash
--line-preprocess legacy --photometric-mode normalize --line-adaptive-c -12
```

三个选项也是新的默认。`--photometric-mode legacy` 可恢复固定 C，但同时恢复该选项管辖的图卡/质心固定门槛。仅比较 C 时，使用本报告的离线脚本，避免混入其他门槛变化。

按钮配置：

```bash
LINE_PREPROCESS=legacy
PHOTOMETRIC_MODE=normalize
LINE_ADAPTIVE_C=-12
```

**已安装的 `config/button_start.env` 覆盖脚本默认**。原配置若写着 `LINE_PREPROCESS=contrast`，必须显式改成 `legacy` 才走本路径；旧配置缺少 `LINE_ADAPTIVE_C` 时默认 −12。

需要更宽松时，可先小幅把参考 C 从 −12 调到 −10；噪声进入明显增加时往 −14 调。C 是估计最终门槛之前的一道候选筛选，**不是最终门槛本身**。最终门槛依然由候选响应中位数或 Otsu、`th_offset=-12`、范围 `[25,80]` 决定；本次没有同步修改这些值或形态学/连通域要求。

记录中新增 `preprocess_adaptive_c_reference`、`preprocess_adaptive_c_effective`，结合 `photometric_mean/std`、`photometric_contrast_scale` 和 `black_th` 可以确认实际执行的门槛。

## 已有实拍验证

固定同一标定、同一原始画面、同一人工中心线标注，比较固定 C 与只换算 C。

| 数据 | 线中心覆盖率：固定 → 归一化 | 候选容差精度：固定 → 归一化 |
|---|---:|---:|
| 暗版 `camera_commands3`，7 帧 | 85.76% → **89.92%** | 99.01% → 98.66% |
| 四段旧录像，7 帧 | 36.76% → 37.61% | 100% → 100% |
| 合计 14 帧 | 61.10% → 63.59% | 99.25% → 98.97% |

中心线容差 4 鸟瞰像素、候选容差 10 像素；忽略人工标注的遮挡/不确定区域。这是同一批开发录像上的像素统计，不能当作比赛成功率或独立测试。OpenCV 4.14.0、NumPy 2.5.3；标定和源码版本见 [report.json](lane_c_normalization/report.json)。

[暗版帧 805 三列对照](https://github.com/issaccheng2008/Jetson_orin_nano_code/blob/dd8228a2e08744712f4a12ee9d7690159448bbe7/docs/audit_2026-10-09/lane_c_normalization/camera_commands3_805.png)：左原鸟瞰，中固定 C，右归一化 C。来源提交中保存全部 17 张对照；当前分支仅保留文本报告和指标，3 张严重模糊/遮挡帧不纳入精度统计。

复现：

```bash
python scripts/evaluate_lane_c_normalization.py \
  --data-dir /path/to/Robocup \
  --output docs/audit_2026-10-09/lane_c_normalization
```

## 仍然存在的限制

C 归一化可补偿近似线性曝光/对比度变化，无法重建反光饱和、运动模糊、缺失像素或严重色彩非线性。旧版开运算和“面积≥300、高度≥80”的连通域门槛仍会删掉细线与横向弯道；固定最终下限 25 也会挡低对比响应。这些不足没有因为换算 C 消失。本次按照旧版基础小改落地，没有据此宣称新版局部增强不再有价值。
