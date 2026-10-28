# 精准图卡内图形提取：开发验证与集成接口

2026-10-09，基于当前 ShapeDetector/selective/normalize 与仓库实际 cameras.json。测试比较的是**同一当前找框与预处理输出上的旧/新图形提取路径**，不作为完整历史版本检测器对照。旧列显式调用 `_classify(dbg['warp'], {})`，没有把集成后 `update()` 的新输出误标为旧结果。测试没有更改找框、动作映射或原轮廓分类阈值。

## 方案

`card_appearance(gray_work, quad_work, size=200, expand_px=4)` 对整张卡的原始工作灰度图做线性插值透视矫正，再在卡 ROI 内 Otsu 二值化。使用 RETR_TREE 的墨/纸交替层次取外框→纸面→内图形的墨外轮廓，不通过固定 inset 裁掉内图形，也不把整帧地板分布作为卡内墨阈值的依据。

完整外框不足时，只恢复另有闭合中央孔的独立轮廓。孔必须位于中央、完全闭合、具有合理面积和范围。外框的单一大纸面孔不能作为此恢复路径；多个同等面积图形拒绝提取。粘连轮廓只有保留了闭合中央内孔才可恢复。返回 `symbol=None` 时提供 reason，可追踪为缺外框、无分离内图形、多个主体或低对比度。

此模块不产生动作，也不绕过原卡框几何验证。原找框失败时不调用它，因此未解决所有漏框。

## 集成

找到 `best` 后，在 `dbg['gray']` / 当前工作灰度与 `best` 上调用。若 `appearance.symbol` 存在，调用现有 `_classify_contour(appearance.symbol)`；若该结果为 None，调用原 `_classify(warp, dbg)`。可靠新结果覆盖原结果；缺足够拓扑证据保留原提取路径。这一组合保留原图所有 27 张正确输出。

`appearance.diagnostics()` 给出 scalar reason/contrast/threshold/frame_count。`appearance.gray` / `.binary` 可保存核查。原 warp 仍可保留调试，但不能声称它是新轮廓的来源。

## 实测结果

按逐张新建检测器评分；曝光仿真使用 `uint8(clip(original*gain))`，不代表真实暗光噪声/运动模糊。

| 数据 | 原正确/错类/无结果 | 新提取单独 | 新提取优先+原路径回退 |
| --- | --- | --- | --- |
| 31原实拍 | 27 / 0 / 4 | 25 / 0 / 6 | **28 / 0 / 3** |
| 31实拍 gain .6 | 26 / 2 / 3 | 25 / 0 / 6 | **28 / 0 / 3** |
| 31实拍 gain .4 | 26 / 1 / 4 | 25 / 0 / 6 | **27 / 0 / 4** |
| 另11张人工确认的WIN三角实拍 | 2 / 4 / 5 | 6 / 0 / 5 | **6 / 0 / 5** |

WIN四个错类 pentagon/diamond/diamond/circle 被恢复成 triangle；原五张无框仍无结果。原31新增正2正确分类；gain .6纠正三1 cross→triangle 与正3 cross→square；gain .4纠正三1。

这些都是共享开发验证，未称独立盲测或100%解决。连续视频验收交由主代理集成后使用80帧独立人工ROI作准确框匹配和分类评分；本报告不包含实车或Nano运行验收。

## 性能与复现

当前CPU环境 OpenCV 4.14.0.94，NumPy 2.5.3，单线程。90个有框调用的额外 appearance 时间 mean 0.83ms、median 0.58ms、P95 1.64ms、max 2.95ms。该时间不含原找框和分类，不能换算为Nano实时性。

`PYTHONPYCACHEPREFIX=/tmp/jetson-loss-check-pycache /tmp/jetson-loss-check-venv/bin/python .planning/2026-10-09-card-recognition/precision_evaluate.py`

逐图数据 `precision_scores.csv`；控制台证据 `precision_evaluation.txt`（旧、新、组合三列）；`precision_metadata.json` 记录实际 `_CAM`、完整检测配置、source/config SHA256、git HEAD 与实际 Python/OpenCV/NumPy/平台版本；七项合成结构测试通过 `python -m unittest discover -s tests -p test_card_appearance.py`。测试覆盖近框三角保留、曝光变化、空框不作图形、连接内孔恢复、奇偶纸层不重复、破框轮廓内孔恢复与角点重排。
