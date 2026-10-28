# 当前视觉计算热点与 GPU 迁移依据

2026-10-09，只读审查当前工作区；运行代码没有修改，没有安装软件。下面的 GPU 项是按计算结构提出的候选，尚未核实 Nano 的 OpenCV CUDA 构建、可用 API、功耗模式或吞吐。README 提及 `gpu_vision/`，但当前仓库 `rg --files` 未找到该目录；不能据此认定已有 GPU 路径。

## 当前执行链与尺寸

主入口 `new_vision/jetson/run_policy_vision.py` 在同一视觉循环先 `LineDetector.process(frame)`，再按条件 `ShapeDetector.update(frame)`。图卡耗时会延长这次整帧巡线输出到达时间，不能只用平均每帧摊销解释控制时延。

- 基础巡线：每个有效摄像头帧运行；当前默认 `line_preprocess=legacy`、`photometric_mode=normalize`。原图 BGR 经一次 IPM 得到 **320×400×3** 鸟瞰；巡线二值化、行扫描和几何拟合主要在 **320×400** 上。
- 原图归一化统计：输入原尺寸；统计 ROI 是 y=35%–75%、x=25%–75%（20%面积），巡线使用原图 max-channel，图卡使用原图标准灰度，二者参考常量不同。**legacy 巡线只把 adaptive C 按当前标准差/参考标准差映射**，没有默认 CLAHE。不要在加速时换成之前的 contrast/Canny 路径。
- 图卡：原图灰度转 ROI/等比缩放+必要补边为 **960×540**，所有完整找框操作在此尺寸。每次新图卡分类仅处理 **200×200** 的矫正图。
- 图卡调用周期：默认 `SHAPE_EVERY=2`、`CARD_EVERY_STOPPED=2`；停车时用后者，走路有 card_flag 时用两个周期最小值；倾斜等待阶段不检测。周期是**已处理帧数**，不是固定墙钟频率。
- QR：若启用且当前尚未扫描完成，每 `qr_every=5` 处理帧尝试；是另外一个可阻塞项，应在整循环分解里独立统计。
- 红条：`red_detect_enable=True` 时，巡线每帧在**原尺寸**做 HSV、两个 inRange、OR；若红像素数够才连通域统计。不能从320×400鸟瞰大小估算整条巡线成本。

源码入口：`line_detector_v1_warp.py:1519`、`line_preprocess.py:39`、`photometric_thresholds.py:50`、`shape_detector.py:381`、`run_policy_vision.py:1177`。

## 操作清单与主要热点

| 路段 | 尺寸与调用次数 | 当前重操作 | 处理建议 |
|---|---|---|---|
| 巡线 IPM/灰度 | 原图→320×400，每帧 | warpPerspective；鸟瞰 np.max、cvtColor、np.maximum；原图 np.max；源 ROI mean/std | 密集操作可组合 GPU；先避免无用整帧统计中间图 |
| 巡线 legacy 二值化 | 320×400，每帧 | 31×31 **ELLIPSE** blackhat；31 adaptive Gaussian；选中像素 median（不足100时采样 Otsu 256级Python扫描）；阈值；5×5 close/open/close，3×3 open；CC+标签 LUT | GPU 候选是前面的连续密集处理；CC/统计/行扫描仍需CPU证据；保留当前内核和C映射 |
| 巡线行扫描与拟合 | 默认low8行、mid8行、bottom-lock10行，每行最多320px | 行段循环、run配对、单边/质心回退；小点集回归/地面映射 | CPU更合适；先观察分支/重复行扫描，别为几十个点启动GPU |
| 可选 lane-fit | 仅参数启用时 | 多滑窗行扫描（每窗最多40行），polyfit小点集 | 独立统计，不能将默认巡线与开启诊断后的耗时混合 |
| 图卡 selective | 960×540，每次图卡调用 | 9×9 ELLIPSE BH、31 adaptive；3×3 close、1×5/5×1 close；DT/CC按宽度过滤；np.lexsort前景像素组内精确中位数；再CC细长度过滤 | 密集预处理可GPU；lexsort/标签归约要整链衡量，目前CC已有向量LUT，不是旧逐CC全图median |
| 图卡额外预处理 | 960×540，每次调用 | 宽度过滤已DT一次，update对过滤后mask再DT一次；总共2次CC（宽度、长度），`_cc_quads`另外使用外轮廓；另原灰度Gaussian5+Otsu+RECT5 close | 两次DT输入不同，不能直接删除第二次；减少副本/内核构建与CPU/GPU来回 |
| 图卡线段生成 | 960×540，每次调用 | HoughLinesP与LSD各一次，输出Python tuple列表 | 潜在主要耗时；现有CPU算法输出需保留。任何GPU替代Hough参数/输出都要重跑准确率；不能假定LSD有同等API |
| Hough/LSD框组合 | 线段列表，每次调用 | 每族topk=4，上/下/左/右4重组合，各通道至多4⁴=256（两通道约512） | **CPU候选组合**；先跨通道重复quad缓存或验证复用，保留不同ring来源语义 |
| 角点四环 | 去重后最多200条线，每次调用 | 多个200×200 float64广播矩阵；partner sets；水平线对×共同垂线对循环；quad与seen逐个比较 | 前半已向量化且小矩阵；后半仍组合增长/去重热点。优先空间索引、批量数值/C++化，单纯GPU图像预处理不会消除此成本 |
| 框几何/精修/验证 | 与候选数相关 | 每quad多次小numpy运算；geom过关后 refine；边缘采样16点/边、±3px小窗口np.any；部分quad warp200×200；地面校准几何 | CPU适合；先减少重复同一quad验证，不要放宽ground闸来换速度 |
| 精确内图形 | 找到best才调用，200×200 | 灰度线性warp、percentile、Otsu、RETR_TREE、轮廓面积/矩/层次；fallback旧morph/CC/轮廓 | 优先保留CPU；体量小，GPU上传/同步可能比计算贵 |
| closed_ring cue | 只有没有合格best才调用，960×540 | RECT31 BH、close9、轮廓；局部mask/纸墨均值 | 失败路径可能比成功分类路径更重；与brokenhelper共享BH响应有机会 |
| broken-border cue | closed_ring无框或低分时，960×540 | **4套** BH/阈值/close/contours，核31/31/51/51；close15/19/19/25；每候选_edge_support重新全图dilate；局部多次掩码、erode、均值 | 存在保真可缓存重复响应/支持mask；GPU候选是密集BH+close，主体轮廓/拒绝理由仍CPU |
| 调试输出 | 每process仍构造320×400视图 | _build_visualization灰度→BGR、多个overlay copy/addWeighted、文本绘制；binary invert | headless目前只不imshow，**不会自动省去detector内部视图构造**。若测出明显成本再增加仅无消费者时跳过的选项 |

## 已确认的等价计算优化机会

1. **brokenhelper BH响应缓存**：四组 `(31,15,12)、(31,19,16)、(51,19,12)、(51,25,8)` 只有31与51两种 BH。先分别算各一次，再按不同阈值/close复用响应，可以保留四路全部结构证据。旧closed_ring cue之前也已经算过31的BH；若函数间显式传递响应，可进一步合并。这不是删除尺度或减少触发尝试。
2. **边支持膨胀缓存**：`card_presence._edge_support` 对每候选做全960×540的dilate，再取160个采样点。`band=max(2,min(5,int(.035*short)))` 只可能2/3/4/5；同一 observed ink mask 内按band缓存，最多4个支持图，避免候选数倍膨胀。这不改变原始边证据、最长缺口、分数门槛。
3. **巡线源ROI先裁再max**：目前 `measure(np.max(bgr,axis=2))` 先处理整幅输入，measure只用中央20%源像素。可先截取精确同一整数边界ROI再max/mean/std，保持参考常量、STD_FLOOR与C映射完全相同，少算80%源max。接口改动须避免measure再裁一次。
4. **鸟瞰灰度冗余**：`gray_max=max(R,G,B)`，标准加权gray不会超过通道最大值；`maximum(gray_max,gray_std)` 等于gray_max。可在真实像素回归确认后省掉标准灰度和maximum；保留当前max-channel统计参考。
5. **图卡重复quad复用**：约512个Hough/LSD候选中可能有大量重复/近重复。先对完全相同的有序坐标和通道来源缓存几何结果，再探索保留精度的近重复空间索引。不能不分来源去掉ring框，因为ring与非ring的验证/精修方式不同。
6. **缓存固定OpenCV核与LUT/缓冲区**：很多每次调用重建getStructuringElement/ones，并分配完整mask、labels、float DT与广播矩阵；适合预分配复用。必须避免debug字典跨帧引用被覆盖造成记录错误。

前三项最容易做到输出完全等价。减少找框频率、直接降960×540分辨率、删LSD/角点/尺度属于行为变化，会改变反应时间与细框证据，需要另外准确率及物理延迟验证，不能当作等价优化。

## CPU只读微测证据（不用于承诺Nano速度）

当前已恢复的Python环境，OpenCV 4.14.0.94/NumPy 2.5.3，`cv2.setNumThreads(1)`。三代表实拍各3次；图卡每次独立检测器；巡线构造时间不计入process。未锁CPU频率/整机负载，三次样本不足以估计P95。只用于判断热点位置。

| 输入 | 原尺寸 | shape preprocess | proposals | verify | classify | presence | shape total | line process |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 背景.png | 1280×720 | 21.2–30.5ms | 35.2–38.4ms | 0.45–1.28ms | 0 | 6.8–22.5ms | 66.6–91.8ms | 31.4–47.0ms |
| 五3.jpg | 1920×1080 | 21.8–39.7ms | 33.4–46.4ms | 1.05–2.87ms | 0.78–13.14ms | 0 | 75.4–85.0ms | 95.5–110.1ms |
| WIN20_32_48 | 1920×1080 | 22.1–26.6ms | 36.6–51.4ms | 11.6–21.3ms | 0.67–2.00ms | 0 | 76.9–100.1ms | 80.0–101.3ms |

背景仍产生515候选，几何通过0；五3产生498/通过1；WIN48产生444/通过4。已证实“没有卡”不等于轻量调用。首次精确分类13ms后降到约1ms，包含冷启动/首次导入因素，不应把第一帧直接当持续分类成本。

WIN48一次cProfile约138k函数调用（profile会加开销）：`_corner_quads`累积约44ms；`_detect_segments`约20ms，其中LSD约10ms/Hough约8ms；3次`_refine_quad`约20ms；2016次`_quad_close`约10ms；2281次`_quad_from_lines`。**这说明候选与小对象Python调用成本是明确瓶颈，不能用“全都搬GPU”解释解决。**

## 现有计时的范围与盲区

`shape_preprocess_ms` 从update开始：BGR转灰度、源统计、ROI/resize、selective及DT、ink都在其中。`shape_proposals_ms` 包含Hough/LSD、所有候选生成；`shape_verification_ms`包含几何、精修、验证。`shape_classify_ms`包含新appearance/分类以及必要旧分类fallback；`shape_presence_ms`包含旧cue和必要brokenhelper。

- 分类分支 **在计时开始前** 已经做 `bitwise_or(binary,ink)` 和旧 `_warp_card`；故分类stage不覆盖这两项。
- `shape_total_ms` 在 `_confirm`/最终返回前记录，不包含动作确认及可能控制台打印。
- stage之和可能小于total：debug生成、坐标映射、状态维护与上述warp不在单独stage内。
- brokenhelper目前只有candidate/reject计数，没有分尺度BH/close/contours与_edge_support耗时。若它在Nano的presence里占多数，下一步再细分。
- 巡线目前没有对应各stage timing；应分原图统计/IPM/legacy提取/行几何/红条/可视化，而不是只看process总时间。
- 还应测camera read/排队、二维码、日志/编码、UDP交付与整个循环。从传感器曝光到命令应用的端到端延迟比单一内核吞吐更有意义。

## GPU候选与暂留CPU的边界

**值得在实际Nano上测的GPU块**：原图/960×540输入的颜色转换、resize/IPM、BH/close/threshold、HSV/inRange/掩码操作。它们是规则、密集、并行的图像运算；应把连续操作保留在设备上，以流/可复用buffer串起来，在轮廓/行扫描必须读CPU之前只下载最终灰度/掩码一次。核类型、边界模式、灰度/HSV单位、阈值比较符号与自适应C必须逐项对齐当前CPU输出。

**暂留CPU**：RETR_TREE层次/轮廓几何、ground校准、小点集拟合、时间窗/确认/动作状态、200×200最终分类；稀疏候选组合/四环去重更适合先CPU算法或编译型实现。即使CUDA能加速某个单项，也不能假设替换算法在细框/模糊边界上的结果一致。

**需端到端判定**：adaptiveThreshold的精确Gaussian/边界语义、DT/CC统计、分标签median/lexsort、LSD/Hough等当前行为。目标不是检查到函数名字就迁移；如果中间每步必须下载再CPU处理，设备同步与小批量launch可能吞掉收益。Jetson共享物理内存也不意味着所有Python/NumPy/GpuMat操作自动零拷贝或异步。

## SSH硬件检查后如何形成具体实施计划

先记录板型/JetPack/L4T、CUDA和OpenCV版本/构建信息、`cv2.cuda`设备数与具体函数、Python来自系统或venv、功耗/频率/温度、摄像头后端/真实帧率/缓存。只查询环境，先不安装或改系统。README历史GPU说明不作为证据。

然后在同一保存原图和摄像头实际运行上，分别记录CPU基线与保真缓存版、可用GPU连续块版：暖机后P50/P95/P99、每阶段、整循环、实际设备同步、输入帧年龄/被处理帧编号、RSS/CPU/GPU使用与温度。保持相同source config、legacy+C归一化、图卡几何/门槛和时间窗。以31六类、WIN11、人工80视频帧及背景反光负例确认准确率不退化；最后再在实际行走验证处理时延与停车距离，不能用桌面时间代替。
